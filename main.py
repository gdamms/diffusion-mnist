"""
MNIST Diffusion Model

A diffusion-based generative model for MNIST digits.

Usage:
    Train diffusion model:
        python main.py train --epochs 10

    Train autoencoder:
        python main.py train-ae --epochs 10

    Generate samples:
        python main.py sample --checkpoint checkpoints/diffusion_latest.pt

    Visualize diffusion process:
        python main.py visualize --all
"""

import argparse
import torch


def main():
    parser = argparse.ArgumentParser(
        description="MNIST Diffusion Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train diffusion model
    train_parser = subparsers.add_parser("train", help="Train diffusion model")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    train_parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    train_parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--attention", action="store_true", help="Use self-attention")
    train_parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    train_parser.add_argument("--name", type=str, default=None, help="Run name")

    # Train autoencoder
    ae_parser = subparsers.add_parser("train-ae", help="Train autoencoder")
    ae_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    ae_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    ae_parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    ae_parser.add_argument("--latent-channels", type=int, default=1, help="Latent channels")
    ae_parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")

    # Sample from model
    sample_parser = subparsers.add_parser("sample", help="Generate samples")
    sample_parser.add_argument("--checkpoint", type=str, default="checkpoints/diffusion_latest.pt",
                               help="Path to model checkpoint")
    sample_parser.add_argument("--n-samples", type=int, default=10, help="Samples per class")
    sample_parser.add_argument("--attention", action="store_true", help="Use attention in model")

    # Visualize diffusion
    viz_parser = subparsers.add_parser("visualize", help="Visualize diffusion process")
    viz_parser.add_argument("--checkpoint", type=str, default="checkpoints/diffusion_latest.pt",
                            help="Path to model checkpoint")
    viz_parser.add_argument("--attention", action="store_true", help="Use attention in model")
    viz_parser.add_argument("--forward", action="store_true", help="Visualize forward diffusion")
    viz_parser.add_argument("--backward", action="store_true", help="Visualize backward diffusion")
    viz_parser.add_argument("--all", action="store_true", help="Run all visualizations")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Set multiprocessing start method
    torch.multiprocessing.set_start_method("spawn", force=True)

    if args.command == "train":
        from src.train_diffusion import train_diffusion
        train_diffusion(
            epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            use_attention=args.attention,
            checkpoint_path=args.checkpoint,
            run_name=args.name,
        )

    elif args.command == "train-ae":
        from src.train_autoencoder import train_autoencoder
        train_autoencoder(
            epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            latent_channels=args.latent_channels,
            checkpoint_path=args.checkpoint,
        )

    elif args.command == "sample":
        import os
        from src.config import DEVICE
        from src.sample import generate_grid
        from src.utils import load_checkpoint
        from models import UNetMNIST

        model = UNetMNIST(use_attention=args.attention).to(DEVICE)
        if os.path.exists(args.checkpoint):
            model = load_checkpoint(model, os.path.basename(args.checkpoint))
        else:
            print(f"Warning: Checkpoint {args.checkpoint} not found.")

        generate_grid(model, n_per_class=args.n_samples)

    elif args.command == "visualize":
        import os
        from src.config import DEVICE
        from src.sample import (
            visualize_forward_diffusion,
            visualize_backward_diffusion,
            generate_grid,
        )
        from src.utils import load_checkpoint
        from models import UNetMNIST

        model = UNetMNIST(use_attention=args.attention).to(DEVICE)
        if os.path.exists(args.checkpoint):
            model = load_checkpoint(model, os.path.basename(args.checkpoint))

        if args.forward or args.all:
            visualize_forward_diffusion()

        if args.backward or args.all:
            visualize_backward_diffusion(model)

        if args.all:
            generate_grid(model)


if __name__ == "__main__":
    main()
