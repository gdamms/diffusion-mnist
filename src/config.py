"""
Configuration file for MNIST Diffusion model.
Contains all hyperparameters and constants.
"""

import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image parameters (MNIST)
IMG_SIZE = 28
NB_CHANNEL = 1
NB_LABEL = 10

# Diffusion parameters
DIFFU_STEPS = 1000

# Noise schedule (linear beta schedule)
BETA = torch.linspace(1e-4, 2e-2, DIFFU_STEPS, device=DEVICE)
BETA = torch.cat((torch.tensor([0.0], device=DEVICE), BETA))
ALPHA = 1 - BETA
ALPHA_BAR = torch.cumprod(ALPHA, dim=0)

# Training parameters
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 2e-4
NUM_WORKERS = 4

# Paths
DATA_DIR = "data"
CHECKPOINT_DIR = "checkpoints"
PLOTS_DIR = "plots"
