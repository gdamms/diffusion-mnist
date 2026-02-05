# MNIST Diffusion Model

A diffusion-based generative model for MNIST digits implemented in PyTorch.

## Project Structure

```
diffusion-mnist/
├── main.py                 # Main entry point with CLI
├── models/                 # Neural network architectures
│   ├── __init__.py
│   ├── unet.py            # UNet for diffusion model
│   └── autoencoder.py     # Autoencoder for latent diffusion
├── src/                   # Source code modules
│   ├── __init__.py
│   ├── config.py          # Configuration and hyperparameters
│   ├── diffusion.py       # Diffusion process utilities
│   ├── dataloader.py      # Dataset and dataloader classes
│   ├── utils.py           # Helper functions and metrics
│   ├── train_diffusion.py # Diffusion training script
│   ├── train_autoencoder.py # Autoencoder training script
│   └── sample.py          # Sampling and visualization
├── checkpoints/           # Model checkpoints
├── plots/                 # Generated visualizations
├── data/                  # Dataset directory
└── runs/                  # TensorBoard logs
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train Diffusion Model
```bash
python main.py train --epochs 10 --lr 2e-4 --batch-size 64
```

### Train with Self-Attention
```bash
python main.py train --epochs 10 --attention
```

### Train Autoencoder (for latent diffusion)
```bash
python main.py train-ae --epochs 10
```

### Generate Samples
```bash
python main.py sample --checkpoint checkpoints/diffusion_latest.pt
```

### Visualize Diffusion Process
```bash
python main.py visualize --all
```

## Configuration

All hyperparameters can be found in `src/config.py`:
- `DIFFU_STEPS`: Number of diffusion steps (default: 1000)
- `EPOCHS`: Training epochs (default: 10)
- `BATCH_SIZE`: Batch size (default: 64)
- `LEARNING_RATE`: Learning rate (default: 2e-4)

## Model Architecture

The diffusion model uses a UNet architecture with:
- Timestep embedding
- Label conditioning (for class-conditional generation)
- Optional self-attention layers

## License

See [LICENSE](LICENSE) for details.
