"""
UNet model for MNIST diffusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")
from src.config import IMG_SIZE, NB_CHANNEL, NB_LABEL, DIFFU_STEPS


class SelfAttention(nn.Module):
    """Self-attention module for UNet."""

    def __init__(self, nb_channels: int, nb_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(nb_channels, nb_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, c, w, h = x.shape
        x = x.view(-1, c, w * h)
        x = x.permute(2, 0, 1)
        x, _ = self.attention(x, x, x)
        x = x.permute(1, 2, 0)
        x = x.view(-1, c, w, h)
        return x


class UNetMNIST(nn.Module):
    """
    UNet architecture for MNIST diffusion model.

    Inputs:
        xt: image at step t (NB_CHANNEL x IMG_SIZE x IMG_SIZE)
        t: step number (1)
        vec: one-hot vector of the label (NB_LABEL)

    Output:
        Predicted noise or denoised image (NB_CHANNEL x IMG_SIZE x IMG_SIZE)
    """

    def __init__(self, use_attention: bool = False):
        super().__init__()
        self.use_attention = use_attention

        # Encoder for timestep t
        self.encodet = nn.Linear(1, IMG_SIZE * IMG_SIZE)

        # Encoder for label vector
        self.encodevec = nn.Linear(NB_LABEL, IMG_SIZE * IMG_SIZE)

        # UNet encoder (2 extra channels for t and vec)
        self.conv1 = nn.Conv2d(NB_CHANNEL + 2, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.maxpool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.maxpool2 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)

        # Optional attention layers
        if use_attention:
            self.att1 = SelfAttention(128, 8)
            self.att2 = SelfAttention(256, 8)
            self.att3 = SelfAttention(256, 8)

        # UNet decoder
        self.upconv1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv8 = nn.Conv2d(128, 128, 3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv10 = nn.Conv2d(64, 64, 3, padding=1)

        # Output layer
        self.conv11 = nn.Conv2d(64, NB_CHANNEL, 3, padding=1)

    def forward(self, xt: torch.Tensor, t: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of UNet.

        Args:
            xt: Noisy image at timestep t [B, C, H, W]
            t: Timestep [B, 1]
            vec: Label one-hot vector [B, NB_LABEL]

        Returns:
            Predicted noise or denoised image [B, C, H, W]
        """
        # Encode timestep and label
        t_enc = F.relu(self.encodet(t / DIFFU_STEPS))
        t_enc = t_enc.view(-1, 1, IMG_SIZE, IMG_SIZE)

        vec_enc = F.relu(self.encodevec(vec))
        vec_enc = vec_enc.view(-1, 1, IMG_SIZE, IMG_SIZE)

        # Concatenate input with embeddings
        x = torch.cat((xt, t_enc, vec_enc), dim=1)

        # Encoder path
        x1 = F.relu(self.conv1(x))
        x1 = F.relu(self.conv2(x1))

        x2 = self.maxpool1(x1)
        x2 = F.relu(self.conv3(x2))
        x2 = F.relu(self.conv4(x2))

        # Bottleneck
        x3 = self.maxpool2(x2)
        if self.use_attention:
            x3 = self.att1(x3)
        x3 = F.relu(self.conv5(x3))
        if self.use_attention:
            x3 = self.att2(x3)
        x3 = F.relu(self.conv6(x3))
        if self.use_attention:
            x3 = self.att3(x3)

        # Decoder path with skip connections
        x4 = self.upconv1(x3)
        x4 = torch.cat((x4, x2), dim=1)
        x4 = F.relu(self.conv7(x4))
        x4 = F.relu(self.conv8(x4))

        x5 = self.upconv2(x4)
        x5 = torch.cat((x5, x1), dim=1)
        x5 = F.relu(self.conv9(x5))
        x5 = F.relu(self.conv10(x5))

        # Output
        out = self.conv11(x5)

        return out
