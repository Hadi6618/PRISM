from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class EncoderOutput:
    latent: Tensor
    skip1: Tensor
    skip2: Tensor
    skip3: Tensor


def conv_relu(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )


class PaperEncoder(nn.Module):
    """Three conv + max-pool blocks from Section 3.5.

    The latent representation is 16 activation maps of size 8x8 for 64x64
    inputs. Skip tensors are the convolution outputs before pooling, so they
    spatially match the decoder conv outputs after each upsampling stage.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = conv_relu(in_channels, 32)
        self.conv2 = conv_relu(32, 32)
        self.conv3 = conv_relu(32, 16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> EcoderOutput:
        skip1 = self.conv1(x)          # B x 32 x 64 x 64
        x = self.pool(skip1)           # B x 32 x 32 x 32
        skip2 = self.conv2(x)          # B x 32 x 32 x 32
        x = self.pool(skip2)           # B x 32 x 16 x 16
        skip3 = self.conv3(x)          # B x 16 x 16 x 16
        latent = self.pool(skip3)      # B x 16 x 8 x 8
        return EncoderOutput(latent=latent, skip1=skip1, skip2=skip2, skip3=skip3)


class ResidualDecoder(nn.Module):
    """Decoder branch with residual summation skip connections.

    The paper says encoder features are summed with corresponding decoder
    activations, following ResNet-style residual addition instead of U-Net
    concatenation. This decoder therefore upsamples 8->16, 16->32 and 32->64,
    sums matching encoder maps at each scale, and maps the final hidden tensor
    to the requested output channel count.
    """

    def __init__(self, out_channels: int, output_activation: str = "sigmoid"):
        super().__init__()
        self.up1_conv = conv_relu(16, 16)
        self.up2_conv = conv_relu(16, 32)
        self.up3_conv = conv_relu(32, 32)
        self.out = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        if output_activation not in {"sigmoid", "identity"}:
            raise ValueError(f"Unsupported output activation: {output_activation}")
        self.output_activation = output_activation

    def forward(self, encoded: EncoderOutput) -> Tensor:
        x = F.interpolate(encoded.latent, scale_factor=2, mode="nearest")
        x = self.up1_conv(x)
        x = x + encoded.skip3

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.up2_conv(x)
        x = x + encoded.skip2

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.up3_conv(x)
        x = x + encoded.skip1

        x = self.out(x)
        if self.output_activation == "sigmoid":
            x = torch.sigmoid(x)
        return x


class AppearanceCAE(nn.Module):
    """Appearance CAE with standard, adversarial and segmentation decoders."""

    def __init__(self):
        super().__init__()
        self.encoder = PaperEncoder(in_channels=1)
        self.decoder = ResidualDecoder(out_channels=1, output_activation="sigmoid")
        self.adversarial_decoder = ResidualDecoder(out_channels=1, output_activation="sigmoid")
        self.segmentation_decoder = ResidualDecoder(out_channels=1, output_activation="identity")

    def encode(self, x: Tensor) -> EncoderOutput:
        return self.encoder(x)

    def reconstruct(self, x: Tensor) -> tuple[Tensor, EncoderOutput]:
        encoded = self.encode(x)
        return self.decoder(encoded), encoded

    def segment(self, encoded: EncoderOutput) -> Tensor:
        return self.segmentation_decoder(encoded)

    def adversarial_reconstruct(self, x: Tensor) -> tuple[Tensor, EncoderOutput]:
        encoded = self.encode(x)
        return self.adversarial_decoder(encoded), encoded

    def forward(self, x: Tensor) -> dict[str, Tensor | EncoderOutput]:
        encoded = self.encode(x)
        return {
            "encoded": encoded,
            "reconstruction": self.decoder(encoded),
            "segmentation_logits": self.segmentation_decoder(encoded),
        }


class MotionCAE(nn.Module):
    """Motion CAE for either backward or forward optical flow."""

    def __init__(self):
        super().__init__()
        self.encoder = PaperEncoder(in_channels=2)
        self.decoder = ResidualDecoder(out_channels=2, output_activation="sigmoid")
        self.adversarial_decoder = ResidualDecoder(out_channels=2, output_activation="sigmoid")

    def encode(self, x: Tensor) -> EncoderOutput:
        return self.encoder(x)

    def reconstruct(self, x: Tensor) -> tuple[Tensor, EncoderOutput]:
        encoded = self.encode(x)
        return self.decoder(encoded), encoded

    def adversarial_reconstruct(self, x: Tensor) -> tuple[Tensor, EncoderOutput]:
        encoded = self.encode(x)
        return self.adversarial_decoder(encoded), encoded

    def forward(self, x: Tensor) -> dict[str, Tensor | EncoderOutput]:
        encoded = self.encode(x)
        return {
            "encoded": encoded,
            "reconstruction": self.decoder(encoded),
        }


class BinaryClassifier(nn.Module):
    """Five-conv classifier with encoder-to-classifier residual skip.

    The first three conv + max-pool blocks match the CAE encoder. The 8x8x16
    activation map after the third max-pool is summed with the CAE encoder
    latent map, exactly as described in Section 3.5.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = conv_relu(in_channels, 32)
        self.conv2 = conv_relu(32, 32)
        self.conv3 = conv_relu(32, 16)
        self.conv4 = conv_relu(16, 16)
        self.conv5 = conv_relu(16, 16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(16 * 2 * 2, 128)
        self.out = nn.Linear(128, 2)

    def forward(self, difference: Tensor, encoder_latent: Tensor) -> Tensor:
        x = self.pool(self.conv1(difference))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = x + encoder_latent
        x = self.pool(self.conv4(x))
        x = self.pool(self.conv5(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x), inplace=True)
        return self.out(x)


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)

