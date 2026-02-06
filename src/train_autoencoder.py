"""
Training script for MNIST autoencoder.
"""

from models import Autoencoder, AEModule
from src.utils import ensure_dirs, save_checkpoint
from src.dataloader import get_autoencoder_dataloaders
from src.config import DEVICE, BATCH_SIZE, NUM_WORKERS
import os
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rich.progress import track
import mlflow

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train_autoencoder(
    epochs: int | None = None,
    learning_rate: float = 1e-3,
    batch_size: int = BATCH_SIZE,
    latent_channels: int = 1,
    val_split: float = 0.1,
    test_split: float = 0.1,
    patience: int | None = 5,
    checkpoint_path: str | None = None,
    run_name: str | None = None,
):
    """
    Train the autoencoder model.

    Args:
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Training batch size
        latent_channels: Number of channels in latent space
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        patience: Number of epochs to wait for validation loss improvement before stopping
        checkpoint_path: Path to checkpoint to resume training from
        run_name: Name for this training run (for logging)
    """
    if epochs is None and patience is None:
        raise ValueError("Must specify either epochs or patience for training")
    if patience is not None and patience <= 0:
        raise ValueError("Patience must be a positive integer")

    ensure_dirs()

    # Initialize model
    model = Autoencoder(input_channels=1, latent_channels=latent_channels).to(DEVICE)

    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    if run_name is None:
        from datetime import datetime
        run_name = f"autoencoder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    mlflow.set_experiment("MNIST Autoencoder")
    mlflow.start_run(run_name=run_name)

    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.functional.binary_cross_entropy

    # Get dataloaders
    train_loader, val_loader, test_loader = get_autoencoder_dataloaders(
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        val_split=val_split,
        test_split=test_split,
    )

    # Early stopping tracking
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # Training loop
    epoch = 0
    while epochs is None or epoch < epochs:
        epoch += 1

        model.train()
        epoch_loss = 0.0

        if epochs:
            description = f"Epoch {epoch}/{epochs}"
        else:
            description = f"Epoch {epoch}"
        if patience:
            description += f" (Patience: {patience-epochs_without_improvement})"

        for batch_idx, (x, target) in enumerate(track(train_loader, description=description)):
            optimizer.zero_grad()

            # Forward pass
            x_recon = model(x)
            loss = criterion(x_recon, target)

            # Backward pass
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        mlflow.log_metric("train/loss", avg_loss, step=epoch)

        val_loss = evaluate_autoencoder(model, val_loader, criterion)
        test_loss = evaluate_autoencoder(model, test_loader, criterion)

        mlflow.log_metric("val/loss", val_loss, step=epoch)
        mlflow.log_metric("test/loss", test_loss, step=epoch)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save best checkpoint
            save_checkpoint(model, "autoencoder_best.pt")
        else:
            epochs_without_improvement += 1

        # Save regular checkpoints
        save_checkpoint(model, f"autoencoder_epoch_{epoch:03d}.pt")
        save_checkpoint(model, "autoencoder_latest.pt")

        # Stop if no improvement
        if patience and epochs_without_improvement >= patience:
            print(f"Early stopping: No improvement for {patience} epochs")
            break

        # Visualize results
        train_fig = visualize_reconstructions(model, train_loader)
        val_fig = visualize_reconstructions(model, val_loader)
        test_fig = visualize_reconstructions(model, test_loader)

        mlflow.log_figure(train_fig, f"train/epoch_{epoch:03d}.png")
        mlflow.log_figure(val_fig, f"val/epoch_{epoch:03d}.png")
        mlflow.log_figure(test_fig, f"test/epoch_{epoch:03d}.png")

    mlflow.end_run()
    return model


def evaluate_autoencoder(
    model: AEModule,
    dataloader,
    criterion,
) -> float:
    """Evaluate autoencoder and return average loss."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, target in dataloader:
            x_recon = model(x)
            loss = criterion(x_recon, target)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def visualize_reconstructions(model: AEModule, dataloader, n_samples: int = 10) -> go.Figure:
    """Visualize original, latent, and reconstructed images."""
    model.eval()
    ensure_dirs()

    # Get a batch of samples
    x_batch, _ = next(iter(dataloader))
    x_batch = x_batch[:n_samples]

    with torch.no_grad():
        latent = model.encode(x_batch)
        x_recon = model.decode(latent)

    # Create visualization with plotly
    fig = make_subplots(
        rows=3, cols=n_samples + 1,
        subplot_titles=[''] * (3 * (n_samples + 1)),
        horizontal_spacing=0.01,
        vertical_spacing=0.05
    )

    # Add row labels as annotations
    row_labels = ['Original', 'Latent', 'Reconstructed']
    for row_idx, label in enumerate(row_labels):
        fig.add_annotation(
            text=label,
            xref="x domain", yref="y domain",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=12),
            row=row_idx + 1, col=1
        )

    # Plot images
    for i in range(n_samples):
        # Original
        fig.add_trace(
            go.Heatmap(z=x_batch[i].cpu().squeeze().numpy()[::-1], colorscale='gray', showscale=False),
            row=1, col=i + 2
        )
        # Latent
        fig.add_trace(
            go.Heatmap(z=latent[i].cpu().squeeze().numpy()[::-1], colorscale='gray', showscale=False),
            row=2, col=i + 2
        )
        # Reconstructed
        fig.add_trace(
            go.Heatmap(z=x_recon[i].cpu().squeeze().numpy()[::-1], colorscale='gray', showscale=False),
            row=3, col=i + 2
        )

    fig.update_layout(
        title_text='Autoencoder Results',
        width=200 * n_samples,
        height=600,
        showlegend=False
    )

    # Hide axes for all subplots
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MNIST autoencoder")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--latent-channels", type=int, default=1, help="Latent channels")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--test-split", type=float, default=0.1, help="Test split fraction")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")

    args = parser.parse_args()

    torch.multiprocessing.set_start_method("spawn", force=True)

    train_autoencoder(
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        latent_channels=args.latent_channels,
        val_split=args.val_split,
        test_split=args.test_split,
        patience=args.patience,
        checkpoint_path=args.checkpoint,
    )
