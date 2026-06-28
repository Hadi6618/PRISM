# STG-NF: How It Works — A Full Technical Deep Dive

**STG-NF** (Spatio-Temporal Graph Normalizing Flows) is a pose-based video anomaly detection method. Instead of looking at raw pixel appearance, it models the **dynamics of human body skeletons** over time. It learns what normal body motion patterns look like and flags any unusual skeletal motion as anomalous.

> [!NOTE]
> This document covers all stages of the STG-NF pipeline: pose extraction and tracking, graph construction, the GCN-Flow model architecture, how the training loss (error) is computed, and how it scores anomalies at test time.

---

## Overview: The Two-Stage Pipeline

```
STAGE 1 — Pose Extraction & Tracking
  Raw Video Frames
       │
       ▼
  YOLOX-X Detector  →  Human Bounding Boxes
       │
       ▼
  FastPose (ResNet152)  →  17 COCO Keypoints per person per frame
       │                    (x, y, confidence) for each joint
       ▼
  PoseFlow (OSNet ReID)  →  Persistent Person Tracking IDs
       │
       ▼
  Raw AlphaPose JSON  →  Format Conversion
       │
       ▼
  Tracked Person JSON: {person_idx → {frame_idx → {keypoints, scores}}}

────────────────────────────────────────────────────────────────────

STAGE 2 — STG-NF Training + Scoring
  Tracked Keypoints
       │
       ▼
  Spatio-Temporal Graph (STG)
  [Spatial Edges: bones within a frame]
  [Temporal Edges: same joint across adjacent frames]
       │
       ▼
  Graph Convolutional Network (GCN) Feature Extraction
       │
       ▼
  Normalizing Flows (bijective transformations)
       │
       ▼
  Log-Likelihood Score per Frame
       │
       ▼
  Gaussian Temporal Smoothing (σ=3.0) → stgnf_scores.pkl
```

---

## Part 1: Pose Extraction — What the Model Sees

### The Input: Skeleton Keypoints, Not Pixels

STG-NF completely ignores raw pixel appearance. It only cares about the **positions of body joints** over time. Each frame is represented as a set of 17 keypoints in the COCO format:

| Joint Index | Body Part |
|:---|:---|
| 0 | Nose |
| 1, 2 | Left Eye, Right Eye |
| 3, 4 | Left Ear, Right Ear |
| 5, 6 | Left Shoulder, Right Shoulder |
| 7, 8 | Left Elbow, Right Elbow |
| 9, 10 | Left Wrist, Right Wrist |
| 11, 12 | Left Hip, Right Hip |
| 13, 14 | Left Knee, Right Knee |
| 15, 16 | Left Ankle, Right Ankle |

Each keypoint is a triplet: $(x, y, \text{confidence})$.

### YOLOX-X: Person Detection

YOLOX-X is a state-of-the-art object detector. For each frame, it produces **bounding boxes** around all visible humans. These bounding boxes are passed to FastPose.

### FastPose (ResNet152): Keypoint Estimation

FastPose is given each cropped human bounding box region and predicts the exact pixel coordinates of all 17 body joints within that region. This gives us the skeleton pose for each person in each frame.

### PoseFlow (OSNet ReID): Person Tracking

Without tracking, we would only know "there is a person in this frame with this pose." We would not know **which person** it is across frames. PoseFlow assigns a persistent `idx` (tracking ID) to each person across the video sequence, linking poses across time.

The output is the raw AlphaPose JSON — a flat list of detection records, one per person per frame:
```json
[
  {"image_id": "0001.jpg", "idx": 0, "keypoints": [...], "score": 0.92},
  {"image_id": "0001.jpg", "idx": 1, "keypoints": [...], "score": 0.88},
  {"image_id": "0002.jpg", "idx": 0, "keypoints": [...], "score": 0.95},
  ...
]
```

### Format Conversion

The raw JSON is restructured into a **hierarchical tracked schema** that STG-NF can process:

```python
# Before: flat list indexed by frame
[{"image_id": "0001.jpg", "idx": 0, "keypoints": [...]}, ...]

# After: nested by person, then by frame
{
  "0": {           # person idx = 0
    "0001": {      # frame idx = 0001
      "keypoints": [x1, y1, c1, x2, y2, c2, ...],  # 17 joints × 3 values = 51 numbers
      "scores": [c1, c2, ...]
    },
    "0002": { ... }
  },
  "1": {           # person idx = 1
    "0001": { ... },
    ...
  }
}
```

This restructuring is essential because STG-NF needs to trace the **trajectory of each person** across frames — not just detect who is in each frame.

---

## Part 2: Spatio-Temporal Graph Construction

### What is a Graph?

A graph is a mathematical structure with **nodes** (points) connected by **edges** (links). STG-NF represents human pose sequences as a graph where:
- **Nodes** = body joint positions (the 17 keypoints)
- **Edges** = relationships between joints (defined below)

### Spatial Edges: Within a Single Frame

Spatial edges encode the **anatomical structure** of the human body. They connect joints that are physically linked by bones:

```
         Nose
          │
    L.Eye ○ ○ R.Eye
          │
   L.Ear ○   ○ R.Ear
          │
  L.Shoulder ─── R.Shoulder
      │                 │
   L.Elbow         R.Elbow
      │                 │
   L.Wrist         R.Wrist
      │
   L.Hip ──────── R.Hip
      │                 │
   L.Knee          R.Knee
      │                 │
   L.Ankle        R.Ankle
```

These edges force the model to reason about **how joints relate to each other** in space — the relative angles, distances, and orientations of limbs.

### Temporal Edges: Across Adjacent Frames

Temporal edges connect the **same joint** across consecutive frames. For example:
- Left Wrist in frame $t$ → Left Wrist in frame $t+1$
- Left Wrist in frame $t+1$ → Left Wrist in frame $t+2$
- ...

These edges capture **motion trajectories** — how each joint moves over time. This is what distinguishes:
- A person walking (slow, rhythmic wrist/ankle motion)
- A person running (fast, large-amplitude motion)
- A person fighting (sudden, erratic joint displacements)

### The Combined Graph

For a sequence of $T$ frames and $J = 17$ joints, the full spatio-temporal graph has:
- **Nodes**: $T \times J$ total (17 joints × number of frames)
- **Spatial edges**: $J_{\text{bones}}$ per frame (roughly 16 anatomical connections)
- **Temporal edges**: $J$ per frame transition ($17 \times (T-1)$ total)

The entire pose sequence of one person is now encoded as a single structured graph.

---

## Part 3: The STG-NF Model

STG-NF combines two components that work together:
1. **Graph Convolutional Network (GCN)** — extracts features from the graph
2. **Normalizing Flows** — transforms those features into a probability score

### Component 1: Spatio-Temporal GCN

A Graph Convolutional Network (GCN) is a neural network that operates on graph-structured data. Unlike a standard CNN (which operates on a regular pixel grid), a GCN aggregates information from each node's **neighbors** as defined by the graph edges.

For each joint node, the ST-GCN operation computes:
$$h_v^{(l+1)} = \text{GELU}\left( W^{(l)} \cdot \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} h_u^{(l)} \right)$$

Where:
- $h_v^{(l)}$ = feature vector of node $v$ at layer $l$
- $\mathcal{N}(v)$ = neighbors of node $v$ (connected via spatial or temporal edges)
- $W^{(l)}$ = learnable weight matrix

**What this means in practice**: each joint learns a representation that incorporates information from its anatomical neighbors (spatial) and its own trajectory over time (temporal). After several GCN layers, each joint has a feature vector that encodes both its local pose context and its motion dynamics.

The output of the GCN is a **flattened feature vector** representing the entire spatio-temporal pose sequence of one person. This is the input to the Normalizing Flow.

### Component 2: Normalizing Flows — The Core of Anomaly Scoring

#### The Fundamental Idea

A Normalizing Flow is a sequence of **invertible transformations** (bijections) that maps a complex, unknown distribution (in our case, the distribution of normal human poses) into a simple, known distribution (a standard Gaussian $\mathcal{N}(0, I)$).

```
Normal Pose Distribution          Standard Gaussian
(complex, unknown shape)          (simple, known)

      [GCN Features]    ─── Flow ───►    z ~ N(0, I)
      (hard to model)   ◄── Inverse ──   (easy to sample)
```

#### How the Flow Computes Log-Likelihood

The key mathematical result that makes Normalizing Flows work is the **change-of-variables formula**:

$$\log p(x) = \log p_z(f(x)) + \log \left| \det \frac{\partial f(x)}{\partial x} \right|$$

Where:
- $x$ = the GCN feature vector (pose representation)
- $f(x) = z$ = the transformed latent vector (after flowing through all transformations)
- $p_z(z) = \mathcal{N}(z; 0, I)$ = the simple Gaussian base distribution
- $\left| \det \frac{\partial f}{\partial x} \right|$ = the Jacobian determinant, which accounts for how the transformation stretches or compresses the space

The **log-likelihood** of a pose is then:
$$\log p(x) = \underbrace{-\frac{1}{2} \|f(x)\|^2 - \frac{d}{2}\log(2\pi)}_{\text{log-likelihood of } z \text{ under Gaussian}} + \underbrace{\log \left| \det J_f \right|}_{\text{volume correction}}$$

> [!IMPORTANT]
> This is the key advantage of Normalizing Flows over MULDE's approach: **the log-likelihood is computed exactly**, without approximations, because the Jacobian determinant of each transformation layer is analytically tractable (designed to be so). There is no intractable partition function.

#### What the Flow Layers Look Like

The flow consists of multiple stacked **coupling layers** (e.g., Glow or RealNVP style). Each coupling layer:
1. Splits the feature vector into two halves: $[x_A, x_B]$
2. Computes scale and shift parameters from one half: $s, t = \text{MLP}(x_A)$
3. Transforms the other half: $x_B' = x_B \cdot e^s + t$
4. The Jacobian determinant is simply $\sum s$ (diagonal, easy to compute)
5. Outputs $[x_A, x_B']$ and passes to the next layer

After $K$ such layers, the output $z = f(x)$ ideally follows $\mathcal{N}(0, I)$ if $x$ is a normal pose.

---

## Part 4: How the Error is Computed in Training

### The Training Objective: Maximize Log-Likelihood

STG-NF is trained on **normal poses only**. The training objective is simply:

$$\mathcal{L} = -\mathbb{E}_{x \sim p_{\text{data}}} [\log p_\theta(x)]$$

In plain terms: **minimize the negative log-likelihood** of normal training poses. This means the model is trained to assign the **highest possible probability** to the normal training data.

Expanding using the change-of-variables formula:

$$\mathcal{L} = -\mathbb{E}_{x} \left[ -\frac{1}{2} \|f_\theta(x)\|^2 + \log \left| \det J_{f_\theta}(x) \right| \right]$$

$$= \mathbb{E}_{x} \left[ \frac{1}{2} \|f_\theta(x)\|^2 - \log \left| \det J_{f_\theta}(x) \right| \right]$$

### What the Two Terms Mean

| Term | Formula | What it penalizes |
|:---|:---|:---|
| **Gaussian NLL** | $\frac{1}{2} \|z\|^2$ | Penalizes mapped $z$ vectors that are far from the origin (not Gaussian-looking) |
| **Log-Jacobian** | $-\log \|\det J\|$ | Penalizes transformations that collapse volume too aggressively |

### Step-by-Step Training Loop

**Step 1: Build the spatio-temporal graph** for a batch of normal pose sequences.

**Step 2: Pass through the GCN** to get a feature vector $x$ for each person-clip.

**Step 3: Pass through the Normalizing Flow** $f_\theta$:
- Forward pass transforms $x \to z$
- Simultaneously accumulates the log-Jacobian determinant $\sum_k \log |\det J_k|$ across all $K$ flow layers

**Step 4: Compute the loss**:
$$\mathcal{L} = \frac{1}{2}\|z\|^2 - \sum_{k=1}^{K} \log |\det J_k|$$

**Step 5: Backpropagation** — gradients flow through the Jacobian terms and back through the coupling layers into the GCN weights. Both the GCN and the Flow layers are trained jointly.

**Step 6: Optimizer step** — weights are updated to make the model more likely to map future normal poses close to the Gaussian origin.

### Why No Target Needed?

Unlike MULDE (which computes a target score vector to compare against), STG-NF has **no explicit target**. The model simply learns to push the latent representations $z$ of all normal training poses toward $\mathcal{N}(0, I)$. The Gaussian itself is the implicit "target" — we want $\|z\|^2$ to be small (close to origin) for normal data.

---

## Part 5: Testing — Anomaly Scoring

After training, the model weights are frozen.

### For a Test Pose Sequence

1. Build the spatio-temporal graph from the tracked keypoints.
2. Pass through the GCN → feature vector $x$.
3. Pass through the Normalizing Flow → latent $z$ and log-Jacobian.
4. Compute the **normality score** (log-likelihood):
$$\text{normality\_score} = -\frac{1}{2}\|z\|^2 + \log |\det J|$$

5. Negate to get the **anomaly score**:
$$\text{anomaly\_score} = -\text{normality\_score} = \frac{1}{2}\|z\|^2 - \log |\det J|$$

### Why Negation?

The model outputs a **normality score** — high values mean "this looks normal." For downstream anomaly detection (where high score = more anomalous), we flip the sign. This is the convention used throughout the pipeline.

### Multiple Persons per Frame

If multiple tracked persons are present in a frame, STG-NF computes one anomaly score per person. The **frame-level score** is the **minimum normality score** across all persons in that frame — i.e., the most anomalous person drives the frame's anomaly score.

### Temporal Smoothing

After computing raw per-frame anomaly scores, a **1D Gaussian smoothing filter** is applied along the time axis with $\sigma = 3.0$:
$$\text{score\_smoothed}[t] = \sum_{\tau} \text{score\_raw}[\tau] \cdot G_{\sigma}(t - \tau)$$

This suppresses isolated noisy spikes and enforces temporal consistency — an anomaly that persists over several frames produces a wider, more reliable peak than a single-frame detection artifact.

---

## Part 6: MULDE vs. STG-NF — Where They Diverge on Log-Density

Both models use log-density as their core concept, but they arrive at it through completely different paths. This is the most important conceptual distinction between the two methods.

| Property | MULDE | STG-NF |
|:---|:---|:---|
| **Input modality** | Raw frame appearance (Hiera-L features) | Body skeleton keypoints |
| **What is learned** | Gradient of log-density (score function) | Direct log-likelihood via bijective mapping |
| **Partition function** | Bypassed entirely using score matching | Does not exist — Normalizing Flows compute exact likelihoods |
| **Why gradients are needed** | The normalization constant $Z_\theta$ is intractable in 1152-dim space | Not needed — the change-of-variables formula gives exact density |
| **Network output** | $f_\theta(x, \sigma)$: scalar log-density estimate at noise level $\sigma$ | $z = f_\theta(x)$: the latent vector after flow transformation |
| **How log-density is computed** | Evaluated at 16 fixed $\sigma$ levels → 16-dim signature | Directly: $-\frac{1}{2}\|z\|^2 + \log |\det J|$ |
| **Post-processing for anomaly score** | GMM fitted on 16-dim signatures → NLL | Negate log-likelihood directly |
| **Temporal smoothing** | Applied in fusion.py (optional, searched) | Applied in score export ($\sigma = 3.0$, fixed) |
| **Training target** | Explicit: target score vector $-\frac{\text{noise}}{\sigma^2}$ | Implicit: push all normal poses toward $\mathcal{N}(0, I)$ |
| **Training data required** | Frame-level visual features (normal training frames) | Tracked skeleton sequences (normal training clips) |

### The Core Difference: Approximate vs. Exact Likelihood

**MULDE** cannot compute the true log-density $\log p(x)$ because:
$$\log p(x) = f_\theta(x) - \underbrace{\log Z_\theta}_{\text{intractable in 1152-dim}}$$

It therefore learns the **gradient** $\nabla_x \log p(x)$ instead, and uses the shape of that gradient at multiple noise scales to build an indirect signature for the GMM.

**STG-NF** uses a mathematical trick (invertible transformations with tractable Jacobians) to compute the **exact** $\log p(x)$:
$$\log p(x) = \log p_z(f(x)) + \log |\det J_f(x)|$$

Both terms on the right are directly computable — there is no intractable normalization constant.

> [!TIP]
> In practice, MULDE captures **appearance anomalies** (something visually unusual in the frame) while STG-NF captures **motion anomalies** (someone moving in an unusual skeletal pattern). The ensemble fusion of both streams is more powerful than either alone because a person could move normally but appear unusual (e.g., wearing a costume), or appear visually normal but move abnormally (e.g., running in a no-running zone).

---

## Part 7: Final Output

The STG-NF export script serializes the smoothed frame-level anomaly scores into `stgnf_scores.pkl`:

```python
{
    "scores_by_video": {
        "01_0014": {
            "frame_indices": [0, 1, 2, ...],         # frame numbers
            "anomaly_scores": [0.12, 0.15, 3.87, ...], # smoothed, negated log-likelihood
            "labels": [0, 0, 1, ...],                # ground-truth 0=normal, 1=anomaly
        },
        "01_0015": { ... },
        ...
    },
    "micro_auc": 0.859,
    "num_videos": 107
}
```

---

## Key Hyperparameters

| Parameter | Value | Role |
|:---|:---|:---|
| `Keypoints` | 17 (COCO format) | Joint positions per person per frame |
| `Detector` | YOLOX-X | Human bounding box detection |
| `Pose Estimator` | FastPose (ResNet152) | 256×192 resolution |
| `Tracker` | PoseFlow (OSNet-x0.25 ReID) | Cross-frame person identity |
| `GCN layers` | Multiple ST-GCN blocks | Spatio-temporal feature extraction |
| `Flow layers` | Stacked coupling layers | Invertible bijective transformations |
| `Training loss` | $-\log p_\theta(x)$ = NLL | Maximize likelihood of normal training poses |
| `Training data` | Normal training clips only | Only normal behavior is seen during training |
| `Smoothing σ` | 3.0 | Gaussian filter on per-frame anomaly scores |
| `Output` | `stgnf_scores.pkl` | Per-video, per-frame anomaly scores |
