"""
Sampling and evaluation script for trained diffusion model.
Generate samples from a trained model and visualize results.
"""

from models import UNetMNIST
from src.utils import ensure_dirs, save_plot, tensor_to_image, load_checkpoint
from src.dataloader import get_mnist_dataset
from src.diffusion import p_xt_1_xt_x0_pred, forward_diffusion, q_xt_x0
from src.config import (
    DEVICE, DIFFU_STEPS, NB_CHANNEL, IMG_SIZE, NB_LABEL,
    CHECKPOINT_DIR,
)
import os
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


def visualize_forward_diffusion(save_path: str | None = None) -> go.Figure:
    """
    Visualize the forward diffusion process on a real image.

    Args:
        save_path: Path to save the visualization
    Returns:
        Plotly figure object
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

    # Create plotly figure
    fig = make_subplots(
        rows=2, cols=n_plots + 1,
        horizontal_spacing=0.01,
        vertical_spacing=0.1
    )

    # Add row labels
    fig.add_annotation(
        text='Step-by-step', xref="x domain", yref="y domain",
        x=0.5, y=0.5, showarrow=False, font=dict(size=10),
        row=1, col=1
    )
    fig.add_annotation(
        text='Direct', xref="x domain", yref="y domain",
        x=0.5, y=0.5, showarrow=False, font=dict(size=10),
        row=2, col=1
    )

    # Plot step-by-step diffusion
    for i, t in enumerate(timesteps):
        step_img = tensor_to_image(xs[t]).squeeze()[::-1]
        fig.add_trace(
            go.Heatmap(z=step_img, colorscale='gray', showscale=False),
            row=1, col=i + 2
        )
        # Add title annotation for each column
        fig.add_annotation(
            text=f't={t}',
            xref=f'x{i + 2} domain', yref=f'y{i + 2} domain',
            x=0.5, y=1.15, showarrow=False, font=dict(size=10)
        )

        # Plot direct diffusion for comparison
        xt, _ = q_xt_x0(img, t)
        direct_img = tensor_to_image(xt).squeeze()[::-1]
        fig.add_trace(
            go.Heatmap(z=direct_img, colorscale='gray', showscale=False),
            row=2, col=i + 2
        )

    fig.update_layout(
        title_text=f'Forward Diffusion Process (Label: {label})',
        width=200 * n_plots,
        height=500,
        showlegend=False
    )

    # Hide axes for all subplots
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    if save_path:
        save_plot(fig, save_path)

    return fig


def visualize_backward_diffusion(model: torch.nn.Module, save_path: str | None = None) -> go.Figure:
    """
    Visualize the backward (reverse) diffusion process.

    Args:
        model: Trained UNet model
        save_path: Path to save the visualization
    Returns:
        Plotly figure object
    """
    ensure_dirs()
    model.eval()

    n_classes = NB_LABEL
    n_timesteps = 10
    timesteps = np.linspace(1, DIFFU_STEPS, n_timesteps, dtype=int)[::-1]

    # Store images to plot later
    images_to_plot = {}

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
                    images_to_plot[(class_idx, t_idx)] = {
                        'img': tensor_to_image(x[class_idx]).squeeze()[::-1],
                        't': t
                    }

    # Create plotly figure
    fig = make_subplots(
        rows=n_classes, cols=n_timesteps,
        horizontal_spacing=0.01,
        vertical_spacing=0.02
    )

    for class_idx in range(n_classes):
        for t_idx in range(n_timesteps):
            data = images_to_plot[(class_idx, t_idx)]
            fig.add_trace(
                go.Heatmap(z=data['img'], colorscale='gray', showscale=False),
                row=class_idx + 1, col=t_idx + 1
            )
            # Add column titles for first row
            if class_idx == 0:
                fig.add_annotation(
                    text=f"t={data['t']}",
                    xref=f'x{t_idx + 1} domain', yref=f'y{t_idx + 1} domain',
                    x=0.5, y=1.15, showarrow=False, font=dict(size=10)
                )
            # Add row labels for first column
            if t_idx == 0:
                fig.add_annotation(
                    text=f'Class {class_idx}',
                    xref=f'x{class_idx * n_timesteps + 1} domain',
                    yref=f'y{class_idx * n_timesteps + 1} domain',
                    x=-0.2, y=0.5, showarrow=False, font=dict(size=10),
                    textangle=-90
                )

    fig.update_layout(
        title_text='Backward Diffusion Process',
        width=200 * n_timesteps,
        height=200 * n_classes,
        showlegend=False
    )

    # Hide axes for all subplots
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    if save_path is not None:
        save_plot(fig, save_path)

    return fig


def generate_grid(model: torch.nn.Module, n_per_class: int = 10, save_path: str | None = None) -> go.Figure:
    """
    Generate a grid of samples, organized by class.

    Args:
        model: Trained UNet model
        n_per_class: Number of samples per class
        save_path: Path to save the grid
    Returns:
        Plotly figure object
    """
    ensure_dirs()

    # Generate samples
    labels = []
    for class_idx in range(NB_LABEL):
        labels.extend([class_idx] * n_per_class)

    samples = generate_samples(model, labels=labels)
    samples = samples.cpu().numpy()

    # Create plotly grid
    fig = make_subplots(
        rows=NB_LABEL, cols=n_per_class,
        horizontal_spacing=0.01,
        vertical_spacing=0.02
    )

    for class_idx in range(NB_LABEL):
        for sample_idx in range(n_per_class):
            idx = class_idx * n_per_class + sample_idx
            img = samples[idx].transpose(1, 2, 0).squeeze()[::-1]
            fig.add_trace(
                go.Heatmap(z=img, colorscale='gray', showscale=False),
                row=class_idx + 1, col=sample_idx + 1
            )
            # Add row labels for first column
            if sample_idx == 0:
                fig.add_annotation(
                    text=f'{class_idx}',
                    xref=f'x{class_idx * n_per_class + 1} domain',
                    yref=f'y{class_idx * n_per_class + 1} domain',
                    x=-0.2, y=0.5, showarrow=False, font=dict(size=10)
                )

    fig.update_layout(
        title_text='Generated MNIST Digits',
        width=100 * n_per_class,
        height=100 * NB_LABEL,
        showlegend=False
    )

    # Hide axes for all subplots
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    if save_path:
        save_plot(fig, save_path)

    return fig


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
