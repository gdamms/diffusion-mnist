"""
Utility functions for MNIST diffusion.
"""

import os
import io
import numpy as np
import torch
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import CHECKPOINT_DIR, PLOTS_DIR


def ensure_dirs():
    """Create necessary directories if they don't exist."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a tensor to a numpy image array.

    Args:
        tensor: Image tensor [C, H, W]

    Returns:
        Numpy array [H, W, C] normalized to [0, 1]
    """
    img = tensor.clone().detach().cpu().numpy().transpose(1, 2, 0)
    img -= img.min()
    img /= img.max() + 1e-8
    return img


def tensor_to_images(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a batch of tensors to numpy image arrays.

    Args:
        tensor: Batch of image tensors [B, C, H, W]

    Returns:
        Numpy array [B, H, W, C] normalized to [0, 1]
    """
    img = tensor.clone().detach().cpu().numpy().transpose(0, 2, 3, 1)
    img -= np.min(img, axis=(1, 2, 3), keepdims=True)
    img /= np.max(img, axis=(1, 2, 3), keepdims=True) + 1e-8
    return img


def figure_to_image(figure: go.Figure) -> np.ndarray:
    """
    Convert a plotly figure to a numpy image array.

    Args:
        figure: Plotly figure

    Returns:
        Numpy array of the figure image
    """
    buf = io.BytesIO()
    figure.write_image(buf, format='png')
    buf.seek(0)
    image = np.array(Image.open(buf))
    return image


def save_checkpoint(model: torch.nn.Module, filename: str):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        filename: Filename (will be saved in CHECKPOINT_DIR)
    """
    ensure_dirs()
    path = os.path.join(CHECKPOINT_DIR, filename)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, filename: str) -> torch.nn.Module:
    """
    Load model checkpoint.

    Args:
        model: Model architecture to load weights into
        filename: Filename (loaded from CHECKPOINT_DIR)

    Returns:
        Model with loaded weights
    """
    path = os.path.join(CHECKPOINT_DIR, filename)
    model.load_state_dict(torch.load(path, weights_only=True))
    return model


def save_plot(figure: go.Figure, filename: str):
    """
    Save a plotly figure to the plots directory.

    Args:
        figure: Plotly figure to save
        filename: Filename (will be saved in PLOTS_DIR)
    """
    ensure_dirs()
    path = os.path.join(PLOTS_DIR, filename)
    figure.write_image(path)
