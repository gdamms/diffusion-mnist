"""
Sampling and evaluation script for trained diffusion model.
Generate samples from a trained model and visualize results.
"""

from models import UNetMNIST
from src.utils import ensure_dirs, tensor_to_image, load_checkpoint
from src.dataloader import get_mnist_dataset
from src.diffusion import p_xt_1_xt_x0_pred, forward_diffusion, q_xt_x0
from src.config import (
    DEVICE, DIFFU_STEPS, NB_CHANNEL, IMG_SIZE, NB_LABEL,
    CHECKPOINT_DIR, PLOTS_DIR
)
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from rich.progress import track

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_samples(
    model: torch.nn.Module,
    n_samples: int = 10,
    labels: list[int] | None = None,
) -> torch.Tensor:
    """
    Generate samples from the diffusion model.

    Args:
        model: Trained UNet model
        n_samples: Number of samples per class (or total if labels provided)
        labels: Optional list of specific labels to generate

    Returns:
        Generated samples tensor [N, C, H, W]
    """
    model.eval()

    if labels is None:
        # Generate samples for all classes
        labels = list(range(NB_LABEL)) * n_samples

    n_total = len(labels)

    with torch.no_grad():
        # Start from pure noise
        x = torch.randn(n_total, NB_CHANNEL, IMG_SIZE, IMG_SIZE, device=DEVICE)

        # Create one-hot label vectors
        vec = torch.tensor(labels, device=DEVICE)
        vec = torch.nn.functional.one_hot(vec, num_classes=NB_LABEL).to(dtype=torch.float32)

        # Reverse diffusion process
        for t in track(range(DIFFU_STEPS, 0, -1), description="Generating samples"):
            t_tensor = torch.tensor([[t]] * n_total, device=DEVICE, dtype=torch.float32)
            x = p_xt_1_xt_x0_pred(model, x, t_tensor, vec)

        # Normalize to [0, 1]
        x = x * 0.5 + 0.5
        x = x.clamp(0, 1)

    return x


def visualize_forward_diffusion(save_path: str | None = None):
    """
    Visualize the forward diffusion process on a real image.

    Args:
        save_path: Path to save the visualization
    """
    ensure_dirs()

    # Get a random image from MNIST
    dataset = get_mnist_dataset(train=True)
    idx = np.random.randint(0, len(dataset))
    img, label = dataset[idx]
    img = img.to(DEVICE)

    # Normalize to [-1, 1]
    img = img * 2 - 1

    # Run forward diffusion
    xs = forward_diffusion(img)

    # Select timesteps to visualize
    n_plots = 10
    timesteps = np.linspace(1, DIFFU_STEPS, n_plots, dtype=int)

    fig, axes = plt.subplots(2, n_plots + 1, figsize=(2 * n_plots, 5))

    # Row labels
    axes[0, 0].text(0.5, 0.5, 'Step-by-step', ha='center', va='center', fontsize=10)
    axes[0, 0].axis('off')
    axes[1, 0].text(0.5, 0.5, 'Direct', ha='center', va='center', fontsize=10)
    axes[1, 0].axis('off')

    # Plot step-by-step diffusion
    for i, t in enumerate(timesteps):
        axes[0, i + 1].imshow(tensor_to_image(xs[t]), cmap='gray')
        axes[0, i + 1].set_title(f't={t}')
        axes[0, i + 1].axis('off')

        # Plot direct diffusion for comparison
        xt, _ = q_xt_x0(img, t)
        axes[1, i + 1].imshow(tensor_to_image(xt), cmap='gray')
        axes[1, i + 1].axis('off')

    fig.suptitle(f'Forward Diffusion Process (Label: {label})')
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, 'forward_diffusion.png')
    fig.savefig(save_path)
    print(f"Saved forward diffusion visualization to {save_path}")
    plt.close(fig)


def visualize_backward_diffusion(model: torch.nn.Module, save_path: str | None = None):
    """
    Visualize the backward (reverse) diffusion process.

    Args:
        model: Trained UNet model
        save_path: Path to save the visualization
    """
    ensure_dirs()
    model.eval()

    n_classes = NB_LABEL
    n_timesteps = 10
    timesteps = np.linspace(1, DIFFU_STEPS, n_timesteps, dtype=int)[::-1]

    fig, axes = plt.subplots(n_classes, n_timesteps, figsize=(2 * n_timesteps, 2 * n_classes))

    with torch.no_grad():
        # Start from noise
        x = torch.randn(n_classes, NB_CHANNEL, IMG_SIZE, IMG_SIZE, device=DEVICE)

        # One sample per class
        vec = torch.arange(n_classes, device=DEVICE)
        vec = torch.nn.functional.one_hot(vec, num_classes=NB_LABEL).to(dtype=torch.float32)

        for t in track(range(DIFFU_STEPS, 0, -1), description="Visualizing backward diffusion"):
            t_tensor = torch.tensor([[t]] * n_classes, device=DEVICE, dtype=torch.float32)
            x = p_xt_1_xt_x0_pred(model, x, t_tensor, vec)

            if t in timesteps:
                t_idx = timesteps.tolist().index(t)
                for class_idx in range(n_classes):
                    axes[class_idx, t_idx].imshow(tensor_to_image(x[class_idx]), cmap='gray')
                    if class_idx == 0:
                        axes[class_idx, t_idx].set_title(f't={t}')
                    if t_idx == 0:
                        axes[class_idx, t_idx].set_ylabel(f'Class {class_idx}')
                    axes[class_idx, t_idx].set_xticks([])
                    axes[class_idx, t_idx].set_yticks([])

    fig.suptitle('Backward Diffusion Process')
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, 'backward_diffusion.png')
    fig.savefig(save_path)
    print(f"Saved backward diffusion visualization to {save_path}")
    plt.close(fig)


def generate_grid(model: torch.nn.Module, n_per_class: int = 10, save_path: str | None = None):
    """
    Generate a grid of samples, organized by class.

    Args:
        model: Trained UNet model
        n_per_class: Number of samples per class
        save_path: Path to save the grid
    """
    ensure_dirs()

    # Generate samples
    labels = []
    for class_idx in range(NB_LABEL):
        labels.extend([class_idx] * n_per_class)

    samples = generate_samples(model, labels=labels)
    samples = samples.cpu().numpy()

    # Create grid
    fig, axes = plt.subplots(NB_LABEL, n_per_class, figsize=(n_per_class, NB_LABEL))

    for class_idx in range(NB_LABEL):
        for sample_idx in range(n_per_class):
            idx = class_idx * n_per_class + sample_idx
            axes[class_idx, sample_idx].imshow(samples[idx].transpose(1, 2, 0).squeeze(), cmap='gray')
            axes[class_idx, sample_idx].axis('off')

            if sample_idx == 0:
                axes[class_idx, sample_idx].set_ylabel(f'{class_idx}')

    fig.suptitle('Generated MNIST Digits')
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, 'generated_grid.png')
    fig.savefig(save_path)
    print(f"Saved generated grid to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sample from trained diffusion model")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(CHECKPOINT_DIR, "diffusion_latest.pt"),
                        help="Path to model checkpoint")
    parser.add_argument("--n-samples", type=int, default=10, help="Samples per class")
    parser.add_argument("--attention", action="store_true", help="Use attention in model")
    parser.add_argument("--forward", action="store_true", help="Visualize forward diffusion")
    parser.add_argument("--backward", action="store_true", help="Visualize backward diffusion")
    parser.add_argument("--grid", action="store_true", help="Generate sample grid")
    parser.add_argument("--all", action="store_true", help="Run all visualizations")

    args = parser.parse_args()

    # Load model
    model = UNetMNIST(use_attention=args.attention).to(DEVICE)

    if os.path.exists(args.checkpoint):
        model = load_checkpoint(model, os.path.basename(args.checkpoint))
    else:
        print(f"Warning: Checkpoint {args.checkpoint} not found. Using untrained model.")

    # Run visualizations
    if args.forward or args.all:
        visualize_forward_diffusion()

    if args.backward or args.all:
        visualize_backward_diffusion(model)

    if args.grid or args.all:
        generate_grid(model, n_per_class=args.n_samples)

    # Default: generate grid if no specific option selected
    if not (args.forward or args.backward or args.grid or args.all):
        generate_grid(model, n_per_class=args.n_samples)
