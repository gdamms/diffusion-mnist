"""
Autoencoder model for MNIST.
Can be used for latent diffusion.
"""

import torch
import torch.nn as nn


class AEModule(nn.Module):
    """Base class for autoencoder modules (encoder and decoder)."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward method.")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        raise NotImplementedError("Subclasses must implement encode method.")

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to image space."""
        raise NotImplementedError("Subclasses must implement decode method.")


class Autoencoder(AEModule):
    """
    Convolutional Autoencoder for MNIST images.

    Args:
        input_channels: Number of input image channels (1 for MNIST)
        latent_channels: Number of channels in latent space
    """

    def __init__(self, input_channels: int = 1, latent_channels: int = 1):
        super().__init__()
        self.input_channels = input_channels
        self.latent_channels = latent_channels

        # Encoder: 28x28 -> 8x8
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 14x14

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 7x7

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, latent_channels, kernel_size=3, padding=1),
            nn.ReLU(),

            # 7x7 -> 8x8 (no activation - latent space should be unconstrained)
            nn.Conv2d(latent_channels, latent_channels, kernel_size=2, padding=1),
        )

        # Decoder: 8x8 -> 28x28
        self.decoder = nn.Sequential(
            # 8x8 -> 7x7
            nn.Conv2d(latent_channels, latent_channels, kernel_size=2, padding=0),
            nn.ReLU(),

            nn.Conv2d(latent_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 14x14
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # 28x28
            nn.ReLU(),

            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, input_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full autoencoder forward pass."""
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to image space."""
        return self.decoder(z)
