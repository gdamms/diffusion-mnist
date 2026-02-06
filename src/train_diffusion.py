"""
Training script for MNIST diffusion model.
"""

from models import UNetMNIST
from src.utils import ensure_dirs, save_checkpoint
from src.metrics import fid, kl_divergence, jsd
from src.diffusion import p_xt_1_xt_x0_pred
from src.dataloader import get_diffusion_dataloaders, get_mnist_dataset
from src.config import (
    DEVICE, EPOCHS, LEARNING_RATE, BATCH_SIZE, NUM_WORKERS,
    DIFFU_STEPS, NB_CHANNEL, IMG_SIZE, NB_LABEL
)
import os
import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rich.progress import track
import mlflow

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train_diffusion(
    epochs: int | None = None,
    learning_rate: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    predict_x0: bool = True,
    use_attention: bool = False,
    val_split: float = 0.1,
    test_split: float = 0.1,
    patience: int | None = 5,
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

    # Get dataloaders
    train_loader, val_loader, test_loader = get_diffusion_dataloaders(
        predict_x0=predict_x0,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        val_split=val_split,
        test_split=test_split,
    )

    # Early stopping tracking
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # Training loop
    global_step = 0
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

        for batch_idx, (xt, t, vec, target) in enumerate(track(train_loader, description=description)):
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
                mlflow.log_metric("train/loss_step", avg_loss, step=global_step)

        avg_loss = epoch_loss / len(train_loader)
        mlflow.log_metric("train/loss", avg_loss, step=epoch)

        val_loss = evaluate_diffusion_loss(model, val_loader, criterion)
        test_loss = evaluate_diffusion_loss(model, test_loader, criterion)

        mlflow.log_metric("val/loss", val_loss, step=epoch)
        mlflow.log_metric("test/loss", test_loss, step=epoch)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save best checkpoint
            save_checkpoint(model, "diffusion_best.pt")
        else:
            epochs_without_improvement += 1

        # Save regular checkpoints
        save_checkpoint(model, f"diffusion_epoch_{epoch:03d}.pt")
        save_checkpoint(model, "diffusion_latest.pt")

        # Stop if no improvement
        if patience and epochs_without_improvement >= patience:
            print(f"Early stopping: No improvement for {patience} epochs")
            break

        # Generate and log sample images and metrics on test split
        if epoch % 1 == 0:
            evaluate_and_log(model, epoch, test_loader, predict_x0)

    mlflow.end_run()
    return model


def evaluate_diffusion_loss(
    model: nn.Module,
    dataloader,
    criterion,
) -> float:
    """Evaluate diffusion model and return average loss."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for xt, t, vec, target in dataloader:
            pred = model(xt, t, vec)
            loss = criterion(pred, target)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate_and_log(
    model: nn.Module,
    epoch: int,
    test_loader,
    predict_x0: bool = True,
):
    """Generate samples and log metrics evaluated on test split."""
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

        # Get real samples from test split
        n_samples = len(fakes)
        base_dataset = getattr(test_loader.dataset, "dataset", None)
        if base_dataset is None:
            base_dataset = get_mnist_dataset(train=True)
        n_samples = min(n_samples, len(base_dataset))
        reals = torch.stack([base_dataset[i][0] for i in range(n_samples)]).cpu().numpy()
        reals = reals * 2 - 1

        # Log metrics
        fid_score = fid(reals, fakes)
        kl_score = kl_divergence(reals, fakes)
        jsd_score = jsd(reals, fakes)

        mlflow.log_metric("test/FID", fid_score, step=epoch)
        mlflow.log_metric("test/KL Divergence", kl_score, step=epoch)
        mlflow.log_metric("test/JSD", jsd_score, step=epoch)

        # Log sample images using plotly
        fig = make_subplots(rows=4, cols=8, horizontal_spacing=0.01, vertical_spacing=0.02)
        for i in range(min(32, len(fakes))):
            row = i // 8 + 1
            col = i % 8 + 1
            img = fakes[i].transpose(1, 2, 0).squeeze()[::-1]
            fig.add_trace(
                go.Heatmap(z=img, colorscale='gray', showscale=False),
                row=row, col=col
            )
        fig.update_layout(
            title_text=f"Generated Samples - Epoch {epoch}",
            width=800,
            height=400,
            showlegend=False
        )
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

        mlflow.log_figure(fig, f"epoch_{epoch:03d}.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MNIST diffusion model")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--attention", action="store_true", help="Use self-attention")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--test-split", type=float, default=0.1, help="Test split fraction")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--name", type=str, default=None, help="Run name")

    args = parser.parse_args()

    torch.multiprocessing.set_start_method("spawn", force=True)

    train_diffusion(
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        use_attention=args.attention,
        val_split=args.val_split,
        test_split=args.test_split,
        patience=args.patience,
        checkpoint_path=args.checkpoint,
        run_name=args.name,
    )
