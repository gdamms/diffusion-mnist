"""
Training script for MNIST diffusion model.
"""

from models import UNetMNIST
from src.utils import (
    ensure_dirs, save_checkpoint, tensor_to_image,
    figure_to_image,
)
from src.metrics import fid, kl_divergence, jsd
from src.diffusion import p_xt_1_xt_x0_pred
from src.dataloader import get_diffusion_dataloader, get_mnist_dataset
from src.config import (
    DEVICE, EPOCHS, LEARNING_RATE, BATCH_SIZE, NUM_WORKERS,
    DIFFU_STEPS, NB_CHANNEL, IMG_SIZE, NB_LABEL, CHECKPOINT_DIR, PLOTS_DIR
)
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from rich.progress import track
import mlflow

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train_diffusion(
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    predict_x0: bool = True,
    use_attention: bool = False,
    checkpoint_path: str | None = None,
    run_name: str | None = None,
):
    """
    Train the diffusion model.

    Args:
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Training batch size
        predict_x0: If True, model predicts x0. Otherwise predicts noise.
        use_attention: If True, use self-attention in UNet
        checkpoint_path: Path to checkpoint to resume training from
        run_name: Name for this training run (for logging)
    """
    ensure_dirs()

    # Setup run name and logging
    if run_name is None:
        from datetime import datetime
        run_name = f"diffusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    mlflow.set_experiment("MNIST Diffusion")
    mlflow.start_run(run_name=run_name)

    # Initialize model
    model = UNetMNIST(use_attention=use_attention).to(DEVICE)

    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        print(f"Loaded checkpoint from {checkpoint_path}")

    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Get dataloader
    dataloader = get_diffusion_dataloader(
        predict_x0=predict_x0,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
    )

    # Training loop
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (xt, t, vec, target) in enumerate(track(dataloader, description=f"Epoch {epoch}/{epochs}")):
            optimizer.zero_grad()

            # Forward pass
            pred = model(xt, t, vec)
            loss = criterion(pred, target)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            # Log training loss every 100 steps
            if global_step % 100 == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                mlflow.log_metric("train_loss", avg_loss, step=global_step)

        avg_loss = epoch_loss / len(dataloader)
        mlflow.log_metric("epoch_loss", avg_loss, step=epoch)

        # Save checkpoint every epoch
        save_checkpoint(model, f"diffusion_epoch_{epoch:03d}.pt")
        save_checkpoint(model, "diffusion_latest.pt")

        # Generate and log sample images
        if epoch % 1 == 0:
            evaluate_and_log(model, epoch, predict_x0)

    mlflow.end_run()
    return model


def evaluate_and_log(model: nn.Module, epoch: int, predict_x0: bool = True):
    """Generate samples and log metrics."""
    model.eval()

    with torch.no_grad():
        # Generate samples
        batch_size = 64
        n_batches = 4

        fakes = []
        for _ in range(n_batches):
            x = torch.randn(batch_size, NB_CHANNEL, IMG_SIZE, IMG_SIZE, device=DEVICE)
            vec = torch.randint(0, NB_LABEL, (batch_size,), device=DEVICE)
            vec = torch.nn.functional.one_hot(vec, num_classes=NB_LABEL).to(dtype=torch.float32)

            # Reverse diffusion
            for t in range(DIFFU_STEPS, 0, -1):
                t_tensor = torch.tensor([[t]] * batch_size, device=DEVICE, dtype=torch.float32)
                x = p_xt_1_xt_x0_pred(model, x, t_tensor, vec)

            x = x.cpu().numpy()
            x = (x - x.min()) / (x.max() - x.min() + 1e-8)
            fakes.append(x)

        fakes = np.concatenate(fakes)

        # Get real samples for comparison
        dataset = get_mnist_dataset(train=True)
        n_samples = len(fakes)
        reals = torch.stack([dataset[i][0] for i in range(n_samples)]).numpy()
        reals = reals * 2 - 1

        # Log metrics
        fid_score = fid(reals, fakes)
        kl_score = kl_divergence(reals, fakes)
        jsd_score = jsd(reals, fakes)

        mlflow.log_metric("FID", fid_score, step=epoch)
        mlflow.log_metric("KL Divergence", kl_score, step=epoch)
        mlflow.log_metric("JSD", jsd_score, step=epoch)

        # Log sample images
        fig, axes = plt.subplots(4, 8, figsize=(16, 8))
        for i, ax in enumerate(axes.flat):
            if i < len(fakes):
                ax.imshow(fakes[i].transpose(1, 2, 0).squeeze(), cmap='gray')
            ax.axis('off')
        fig.suptitle(f"Generated Samples - Epoch {epoch}")
        plt.tight_layout()

        mlflow.log_figure(fig, f"samples_epoch_{epoch:03d}.png")

        # Save to plots folder
        fig.savefig(os.path.join(PLOTS_DIR, f"samples_epoch_{epoch:03d}.png"))
        plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MNIST diffusion model")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--attention", action="store_true", help="Use self-attention")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--name", type=str, default=None, help="Run name")

    args = parser.parse_args()

    torch.multiprocessing.set_start_method("spawn", force=True)

    train_diffusion(
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        use_attention=args.attention,
        checkpoint_path=args.checkpoint,
        run_name=args.name,
    )
