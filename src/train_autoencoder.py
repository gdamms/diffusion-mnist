"""
Training script for MNIST autoencoder.
"""

from models import Autoencoder, AEModule
from src.utils import ensure_dirs, save_checkpoint
from src.dataloader import get_autoencoder_dataloader
from src.config import DEVICE, BATCH_SIZE, NUM_WORKERS, CHECKPOINT_DIR, PLOTS_DIR
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from rich.progress import track
import mlflow

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train_autoencoder(
    epochs: int = 10,
    learning_rate: float = 1e-3,
    batch_size: int = BATCH_SIZE,
    latent_channels: int = 1,
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
        checkpoint_path: Path to checkpoint to resume training from
        run_name: Name for this training run (for logging)
    """
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
    # Use MSE loss instead of BCE for better reconstruction of continuous values
    criterion = nn.MSELoss()

    # Get dataloader
    dataloader = get_autoencoder_dataloader(
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
    )

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (x, target) in enumerate(track(dataloader, description=f"Epoch {epoch}/{epochs}")):
            optimizer.zero_grad()

            # Forward pass
            x_recon = model(x)
            loss = criterion(x_recon, target)

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        mlflow.log_metric("epoch_loss", avg_loss, step=epoch)

        # Save checkpoint
        save_checkpoint(model, f"autoencoder_epoch_{epoch:03d}.pt")
        save_checkpoint(model, "autoencoder_latest.pt")

    # Visualize results
    visualize_reconstructions(model, dataloader)

    mlflow.end_run()
    return model


def visualize_reconstructions(model: AEModule, dataloader, n_samples: int = 10):
    """Visualize original, latent, and reconstructed images."""
    model.eval()
    ensure_dirs()

    # Get a batch of samples
    x_batch, _ = next(iter(dataloader))
    x_batch = x_batch[:n_samples]

    with torch.no_grad():
        latent = model.encode(x_batch)
        x_recon = model.decode(latent)

    # Create visualization
    fig, axes = plt.subplots(3, n_samples + 1, figsize=(2 * n_samples, 6))

    # Labels
    axes[0, 0].text(0.5, 0.5, 'Original', ha='center', va='center', fontsize=12)
    axes[0, 0].axis('off')
    axes[1, 0].text(0.5, 0.5, 'Latent', ha='center', va='center', fontsize=12)
    axes[1, 0].axis('off')
    axes[2, 0].text(0.5, 0.5, 'Reconstructed', ha='center', va='center', fontsize=12)
    axes[2, 0].axis('off')

    # Plot images
    for i in range(n_samples):
        axes[0, i + 1].imshow(x_batch[i].cpu().squeeze().numpy(), cmap='gray')
        axes[0, i + 1].axis('off')

        axes[1, i + 1].imshow(latent[i].cpu().squeeze().numpy(), cmap='gray')
        axes[1, i + 1].axis('off')

        axes[2, i + 1].imshow(x_recon[i].cpu().squeeze().numpy(), cmap='gray')
        axes[2, i + 1].axis('off')

    fig.suptitle('Autoencoder Results')
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, 'autoencoder_results.png')
    fig.savefig(save_path)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MNIST autoencoder")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--latent-channels", type=int, default=1, help="Latent channels")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")

    args = parser.parse_args()

    torch.multiprocessing.set_start_method("spawn", force=True)

    train_autoencoder(
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        latent_channels=args.latent_channels,
        checkpoint_path=args.checkpoint,
    )
