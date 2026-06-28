# MULDE: How It Works — A Full Technical Deep Dive

**MULDE** (Multiscale Log-Density Estimation via Denoising Score Matching) is a video anomaly detection method published at CVPR 2024.  
It works by learning a model of what **normal** video looks like — and flagging anything that deviates from that model as anomalous.

> [!NOTE]
> This document covers all stages of the MULDE pipeline: feature extraction, neural network training via Denoising Score Matching, multiscale evaluation, and GMM-based anomaly scoring.

---

## Overview: The Two-Stage Pipeline

```
STAGE 1 — Feature Extraction
  Raw Video Frames
       │
       ▼
  Hiera-L (Video Transformer)
  [16 frames per clip, stride 4]
       │
       ▼
  1152-dim Feature Vectors (one per frame)
       │
       ▼
  Standardize using training set mean/std
       │
       ▼
  Standardized Feature Vectors [N_frames, 1152]

────────────────────────────────────────────────

STAGE 2 — MULDE Training + GMM Scoring
  Standardized Features
       │
       ├─── Neural Network Training (Denoising Score Matching)
       │         └── Learns f_θ(x, σ): log-density at any noise scale
       │
       ├─── Multiscale Evaluation (after training)
       │         └── Evaluates at L=16 fixed σ levels → 16-dim signature per frame
       │
       └─── GMM Fitting + Anomaly Scoring
                 └── Fits GMM on training signatures
                 └── Scores test frames by negative log-likelihood
```

---

## Part 1: Feature Extraction with Hiera-L

### What is Hiera-L?

Hiera-L is a **Video Transformer** model developed by Meta AI. It was pretrained on **Kinetics-400** — a large dataset containing 400 categories of human actions (running, cooking, jumping, swimming, etc.). The training task was **action classification**: given 16 consecutive video frames, predict which action category is happening.

After pretraining, the final classification layer is discarded. What remains is the model's **internal representation** — a compressed, abstract encoding of what is happening in the video clip. This is the 1152-dimensional feature vector that MULDE uses.

### What do the 1152 numbers encode?

The 1152 values are **not** raw pixel values, edge detectors, or simple color histograms. They are deep, abstract, learned representations that collectively encode:

| Concept | Description |
|:---|:---|
| **Body configuration** | Whether a person is standing, crouching, running, or gesturing |
| **Temporal motion** | Direction, speed, and acceleration of motion across 16 frames |
| **Scene context** | Background structure, crowd density, indoor/outdoor cues |
| **Object interactions** | Whether someone is carrying an object, touching another person, etc. |
| **Action semantics** | The high-level action type, learned from Kinetics-400 categories |

No single number encodes one concept. The 1152 values work **collectively** like a fingerprint — the pattern of all values together describes the action.

### Why 16 frames?

Hiera-L processes **16 frames centered on each target frame**, sampled with stride 4. This temporal window is essential because:

- A single frame captures only **appearance** (what something looks like at one instant).
- 16 frames capture **motion** (how things are changing over time).

A person with raised arms in a single frame could be stretching or throwing a punch. 16 frames over time reveal which it actually is.

### Why does Hiera-L help anomaly detection?

Hiera-L was trained purely on normal human activities. As a result, its internal representations form clusters:

- **Normal actions** (walking, standing, talking) → similar, tightly clustered feature vectors
- **Abnormal actions** (fighting, falling, skateboarding) → feature vectors that deviate far from the normal cluster

MULDE exploits this property: it learns the density of normal feature vectors and flags anything that lies outside that density as anomalous.

### Standardization

Before training, all feature vectors are standardized using the mean and standard deviation computed over all **training** frames only:

$$x_{\text{std}} = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

After standardization, normal training features hover near **zero** with unit variance. The same statistics are applied to test features at inference time.

---

## Part 2: The MULDE Neural Network

### Architecture

The MULDE network is an MLP (Multi-Layer Perceptron) wrapped in a `ScoreOrLogDensityNetwork`. It maps a feature vector and a noise level to a single scalar:

```
Input: [x₁, x₂, ..., x₁₁₅₂, σ]   → 1153-dim vector
          (features)       (noise level)
                │
    ┌───────────────────────┐
    │  Linear (1153 → 4096) │
    │  GELU Activation      │
    ├───────────────────────┤
    │  Linear (4096 → 4096) │
    │  GELU Activation      │
    ├───────────────────────┤
    │  Linear (4096 → 1)    │
    └───────────────────────┘
                │
    Output: f_θ(x, σ)   → 1 scalar (unnormalized log-density)
```

### The Role of σ as a Conditioning Input

The scalar noise level $\sigma$ is concatenated directly to the 1152 feature dimensions before entering the network. Though $\sigma$ is a single number, it **conditions the entire network** through the weights of the first layer.

For each hidden neuron $j$:
$$h_j = \text{GELU}\left( \sum_{i=1}^{1152} W_{j,i} \cdot x_i + W_{j,1153} \cdot \sigma + b_j \right)$$

The weight $W_{j,1153}$ connects the scalar $\sigma$ to every one of the 4096 hidden neurons. Through training, the network learns to use $\sigma$ as a **dial that shifts its behavior**:

- At **high $\sigma$** (large noise): the network reasons about coarse, global structure of the data manifold.
- At **low $\sigma$** (small noise): the network reasons about fine-grained, local boundaries of the manifold.

This allows a single network to learn a **continuous family of density functions** across the entire range of noise scales $\sigma \in [10^{-3}, 1.0]$.

---

## Part 3: Why Score Matching Instead of Direct Density Estimation?

### The Fundamental Problem

If we want to model the probability density of normal frames $p(x)$, the ideal model would be:

$$p_\theta(x) = \frac{e^{f_\theta(x)}}{Z_\theta}$$

Where $Z_\theta = \int e^{f_\theta(x)} \, dx$ is a **normalization constant** (the partition function).

In 1152 dimensions, computing $Z_\theta$ requires integrating over all possible combinations of 1152-dimensional space. This is **completely intractable** — there is no closed-form solution, and numerical approximations are too expensive.

### The Solution: Work with Gradients

Taking the logarithm of $p_\theta(x)$ and then the gradient with respect to $x$:

$$\log p_\theta(x) = f_\theta(x) - \underbrace{\log Z_\theta}_{\text{constant w.r.t. } x}$$

$$\nabla_x \log p_\theta(x) = \nabla_x f_\theta(x) - \underbrace{\nabla_x \log Z_\theta}_{= 0}$$

Because $Z_\theta$ does not depend on $x$, its gradient with respect to $x$ is exactly zero. By working with the **gradient of the log-density** (called the **score function**), the intractable partition function disappears entirely from the equations.

### The Score as a Vector Field

The score $\nabla_x \log p_\theta(x)$ is a vector field over the 1152-dimensional feature space. At every point $x$, it is a 1152-dimensional vector that:

- **Points toward** regions of higher probability density (i.e., toward the normal data manifold)
- **Has larger magnitude** the steeper the density slope is

```
                         ←←←←←
          ░░░░░░░░░░░░░░←    ←←←←←
          ░  NORMAL   ░← ●   ←←←←←←
          ░  MANIFOLD ░←      ←←←←←←
          ░░░░░░░░░░░░░░←      ←←←
                         ←←←
         All arrows point toward the high-density region
         (the cluster of normal training features)
```

Once the network learns this vector field correctly, the scalar output $f_\theta(x, \sigma)$ will naturally be:
- **High (less negative)** for normal frames near the training manifold
- **Very low (very negative)** for anomalous frames far from the training manifold

---

## Part 4: Denoising Score Matching (DSM) Training

### Core Idea

We want the network to learn the score function $\nabla_x \log p(x)$ without knowing $p(x)$ directly. The key insight of **Denoising Score Matching** is:

> If we deliberately corrupt a clean normal sample $x$ by adding Gaussian noise, the exact score (gradient of the log-density) of the corrupted distribution has a simple, closed-form expression — it is just a vector pointing from the noisy sample back to the clean original.

This gives us free supervision signals to train the network.

### The Training Loop (One Batch)

#### Step 1: Sample σ Log-Uniformly

For each frame $x$ in the batch, a noise level is sampled log-uniformly:
$$\sigma \sim \text{LogUniform}(10^{-3},\ 1.0)$$

Each frame in the batch gets its own independently sampled $\sigma$.

> [!IMPORTANT]
> **Why random σ during training?** We want the network to learn a **continuous function** of $\sigma$ over the entire range $[10^{-3}, 1.0]$. Sampling randomly ensures that across epochs, the network sees every frame paired with every noise scale. This is more efficient than evaluating all 16 fixed scales per frame per batch.

#### Step 2: Perturb the Features

Gaussian noise scaled by $\sigma$ is added to the clean standardized features:
$$x_{\text{noisy}} = x + \underbrace{\epsilon \cdot \sigma}_{\text{noise}}, \quad \epsilon \sim \mathcal{N}(0, I)$$

This displaces the feature vector away from the normal data manifold.

#### Step 3: Forward Pass — Compute the Log-Density

The noisy features are concatenated with their $\sigma$ and fed through the MLP:
$$f_\theta(x_{\text{noisy}}, \sigma) \in \mathbb{R} \quad \text{(one scalar)}$$

#### Step 4: First Backprop — Compute the Predicted Score

Using PyTorch's `autograd` with `create_graph=True`, we compute the gradient of $-f_\theta$ with respect to the **input features** (not the weights):
$$\text{score\_pred} = \nabla_{x_{\text{noisy}}} [-f_\theta(x_{\text{noisy}}, \sigma)] \in \mathbb{R}^{1152}$$

This requires `create_graph=True` because we need PyTorch to remember **how** the score was computed — so that the second backpropagation (updating weights) can flow through it.

#### Step 5: Compute the Target Score

The target is the analytically known gradient of the noisy distribution:
$$\text{score\_target} = -\frac{\text{noise}}{\sigma^2} = -\frac{\epsilon}{\sigma}$$

This vector points directly from the noisy sample $x_{\text{noisy}}$ back to the clean sample $x$. It tells the network: *"the normal data manifold is in this direction."*

#### Step 6: Compute the DSM Loss

The loss measures how different the predicted score is from the target:
$$\mathcal{L}_{\text{DSM}} = \frac{1}{2} \cdot \sigma^2 \cdot \left\| \text{score\_pred} + \frac{\text{noise}}{\sigma^2} \right\|^2$$

The weighting by $\sigma^2$ balances the training across different noise scales — high-$\sigma$ losses would otherwise dominate because large noise produces large gradients.

#### Step 7: Regularization

A regularization term is added to prevent $f_\theta$ from drifting to $\pm\infty$:
$$\mathcal{L}_{\text{reg}} = \frac{\beta}{2} \cdot f_\theta(x, \sigma)^2$$

This penalizes excessively large log-density values for clean training features.

#### Step 8: Second Backprop — Update Network Weights

The total loss $\mathcal{L} = \mathcal{L}_{\text{DSM}} + \mathcal{L}_{\text{reg}}$ is backpropagated all the way through the score computation into the MLP weights. The optimizer (Adam with $\beta = (0.5, 0.9)$, $\text{lr} = 10^{-4}$) updates $\theta$.

Over 1000 epochs, the network learns a vector field where, at every point in 1152-dimensional space, the gradient arrows point toward the normal training data manifold.

---

## Part 5: Multiscale Evaluation (After Training)

### Training vs. Testing σ Selection

| Phase | σ selection | Reason |
|:---|:---|:---|
| **Training** | One random σ per frame per batch | Efficient; forces the network to learn a continuous function of σ across the full range |
| **Testing** | L=16 fixed, linearly spaced σ levels | Produces a deterministic, reproducible fingerprint per frame |

### Computing the Multiscale Signature

After training, the network weights are **frozen**. For every test frame (and every training frame, for GMM fitting), the clean standardized feature vector $x$ is passed through the network at $L=16$ fixed noise levels:

$$\sigma_1 = 0.001,\ \sigma_2 = 0.067,\ \sigma_3 = 0.134,\ \dots,\ \sigma_{16} = 1.0$$

For each $\sigma_i$:
$$z_i = f_\theta(x, \sigma_i)$$

The result is a **16-dimensional signature vector**:
$$\mathbf{z} = [f_\theta(x, \sigma_1),\ f_\theta(x, \sigma_2),\ \dots,\ f_\theta(x, \sigma_{16})] \in \mathbb{R}^{16}$$

### Why 16 Scales Capture More Than Any Single Scale

Different types of anomalies manifest at different noise scales:

| Anomaly Type | Low σ (fine detail) | High σ (coarse structure) |
|:---|:---|:---|
| Subtle abnormal gesture | Very anomalous — detected | Looks normal — missed |
| Running in a walking zone | Slightly anomalous | Very anomalous — detected |
| Sudden fall | Anomalous at all scales | Anomalous at all scales |

By combining all 16 scales into one vector, MULDE captures anomalies that would be missed if only a single scale were used.

---

## Part 6: GMM Fitting and Anomaly Scoring

### When is the GMM Fitted?

**Exactly once, after the neural network training is completely finished.** The GMM is not part of the training loop. It is a separate, one-time fitting step that takes only seconds.

```
Neural Network Training (1000 epochs, GPU, hours)
            │
            ▼  (weights frozen)
Compute 16-dim signatures for ALL training frames
            │
            ▼
GMM Fitting on training signatures (CPU, seconds)
            │
            ▼
Score test frames using fitted GMM
```

### What the GMM Learns

The GMM is fitted on the 16-dim signature vectors of **normal training frames only**. It learns:

- **Mean vector** $\boldsymbol{\mu}$: The center of the cluster(s) of normal signatures — what a "typical normal frame" looks like across all 16 noise scales.
- **Covariance matrix** $\Sigma$: The shape, spread, and orientation of the cluster — how much the signatures naturally vary and how the 16 dimensions correlate.

MULDE tries GMMs with **1, 3, and 5 components** (full covariance). Multiple components capture multimodal normal behavior — for example, "walking" and "standing still" may form two distinct clusters.

### Why Not K-Means?

| Property | K-Means | GMM |
|:---|:---|:---|
| Output | Distance (unitless) | Probability density (calibrated likelihood) |
| Cluster shape | Assumes spherical (equal spread in all dimensions) | Learns elliptical shape via covariance matrix |
| Multiple behaviors | Cannot model gaps between clusters correctly | Each component has its own mean + covariance |
| Anomaly scoring | $\|z - \text{center}\|^2$ | $-\log p(\mathbf{z})$ (negative log-likelihood) |

K-Means ignores the fact that some dimensions of the 16-dim signature vary more than others. A point that deviates greatly in a naturally high-variance dimension may be perfectly normal, while a point that deviates slightly in a naturally zero-variance dimension may be highly anomalous. The GMM covariance matrix captures this distinction exactly — this is called the **Mahalanobis distance**.

### Anomaly Scoring

The anomaly score for a test frame with signature $\mathbf{z}$ is the **negative log-likelihood** under the fitted GMM:

$$\text{Anomaly Score} = -\log p_{\text{GMM}}(\mathbf{z})$$

Where the GMM log-likelihood is:

$$\log p_{\text{GMM}}(\mathbf{z}) = \log \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(\mathbf{z} \mid \boldsymbol{\mu}_k, \Sigma_k)$$

The key term driving the score is the **Mahalanobis distance** from the test signature to the nearest GMM component:

$$(\mathbf{z} - \boldsymbol{\mu}_k)^T \Sigma_k^{-1} (\mathbf{z} - \boldsymbol{\mu}_k)$$

- **Normal frame**: $\mathbf{z}$ is close to a GMM component center → high likelihood → low anomaly score ✅
- **Abnormal frame**: $\mathbf{z}$ is far from all GMM component centers → low likelihood → high anomaly score 🚨

---

## Part 7: Final Output

After GMM scoring, the pipeline produces a **frame-level anomaly score** for every test frame in every video. These raw scores are serialized into `mulde_scores.pkl` — an ordered dictionary keyed by video ID:

```python
{
    "video_id_1": {
        "frame_indices": [...],     # frame numbers
        "anomaly_scores": [...],    # raw GMM NLL scores (unsmoothed)
        "labels": [...],            # ground-truth 0/1 per frame
    },
    "video_id_2": { ... },
    ...
}
```

> [!NOTE]
> MULDE intentionally exports **unsmoothed** raw scores. Temporal Gaussian smoothing (if needed) is applied later in `fusion.py`, which can search for the optimal smoothing σ as part of the late fusion pipeline.

---

## Complete Pipeline at a Glance

```
Raw Video Frames
       │
       ▼
┌──────────────────────────────────────┐
│ Hiera-L Feature Extraction           │
│ - 16 frames per clip, stride 4       │
│ - Pretrained on Kinetics-400         │
│ - Output: 1152-dim per frame         │
└──────────────────┬───────────────────┘
                   │
                   ▼
           Standardize Features
           (using training mean/std)
                   │
       ┌───────────┴───────────┐
       │                       │
       ▼                       ▼
  TRAINING PHASE          EVALUATION PHASE
  (1000 epochs)           (once, after training)
       │                       │
  Sample random σ         Use 16 fixed σ levels
  Add Gaussian noise       (no noise added)
  Predict score vector          │
  Compare to target        Compute 16-dim
  Update MLP weights       signature per frame
       │                       │
       │ (weights frozen)  ┌───┴───────────────────┐
       └───────────────────►  GMM Fitting           │
                              (on training sigs)    │
                              └────────────────────►│
                                                    │
                                           GMM Scoring
                                           (negative log-likelihood)
                                                    │
                                                    ▼
                                            mulde_scores.pkl
                                      (frame-level anomaly scores)
```

---

## Key Hyperparameters

| Parameter | Value | Role |
|:---|:---|:---|
| `INPUT_DIM` | 1152 | Hiera-L feature dimension |
| `UNITS` | [4096, 4096] | MLP hidden layer sizes |
| `LR` | 1e-4 | Adam learning rate |
| `EPOCHS` | 1000 | Training duration |
| `BETA` | 0.1 | Regularization weight |
| `SIGMA_LOW` | 1e-3 | Minimum noise scale |
| `SIGMA_HIGH` | 1.0 | Maximum noise scale |
| `L` | 16 | Number of fixed evaluation σ levels |
| `ADAM_BETAS` | (0.5, 0.9) | Adam momentum parameters |
| `GMM_COMPONENTS` | [1, 3, 5] | GMM component counts to try |
| `GMM_COVARIANCE` | full | Full covariance matrix per component |
