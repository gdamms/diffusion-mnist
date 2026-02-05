"""
Data loading utilities for MNIST diffusion training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from .config import DEVICE, DIFFU_STEPS, NB_LABEL, DATA_DIR
from .diffusion import q_xt_x0


def get_mnist_dataset(train: bool = True) -> datasets.MNIST:
    """
    Load MNIST dataset.

    Args:
        train: If True, load training set. Otherwise load test set.

    Returns:
        MNIST dataset
    """
    return datasets.MNIST(
        root=DATA_DIR,
        train=train,
        download=True,
        transform=transforms.ToTensor(),
    )


class DiffusionDataset(Dataset):
    """
    Dataset wrapper for diffusion training.
    Returns noisy image, timestep, label, and target noise.

    Args:
        dataset: Base image dataset (e.g., MNIST)
        autoencoder: Optional autoencoder for latent diffusion
    """

    def __init__(self, dataset: Dataset, autoencoder: torch.nn.Module | None = None):
        super().__init__()
        self.dataset = dataset
        self.autoencoder = autoencoder

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        # Get image and label
        img, label = self.dataset[index]
        img = img.to(DEVICE)

        # Optionally encode to latent space
        if self.autoencoder is not None:
            with torch.no_grad():
                img = self.autoencoder.encode(img.unsqueeze(0)).squeeze(0)

        # Normalize to [-1, 1]
        img = img * 2 - 1

        # Sample random timestep and add noise
        t = torch.randint(1, DIFFU_STEPS + 1, (1,), device=DEVICE)
        xt, eps = q_xt_x0(img, t)

        # Convert label to one-hot vector
        vec = torch.nn.functional.one_hot(
            torch.tensor(min(label, NB_LABEL - 1)),
            num_classes=NB_LABEL,
        )

        return (
            xt.clone().detach().to(dtype=torch.float32, device=DEVICE),
            t.clone().detach().to(dtype=torch.float32, device=DEVICE),
            vec.clone().detach().to(dtype=torch.float32, device=DEVICE),
            eps,  # Target: the noise that was added
        )

    def __len__(self) -> int:
        return len(self.dataset)


class DiffusionDatasetX0(Dataset):
    """
    Dataset wrapper for diffusion training where model predicts x0 instead of noise.
    Returns noisy image, timestep, label, and target clean image.

    Args:
        dataset: Base image dataset (e.g., MNIST)
        autoencoder: Optional autoencoder for latent diffusion
    """

    def __init__(self, dataset: Dataset, autoencoder: torch.nn.Module | None = None):
        super().__init__()
        self.dataset = dataset
        self.autoencoder = autoencoder

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        # Get image and label
        img, label = self.dataset[index]
        img = img.to(DEVICE)

        # Optionally encode to latent space
        if self.autoencoder is not None:
            with torch.no_grad():
                img = self.autoencoder.encode(img.unsqueeze(0)).squeeze(0)

        # Normalize to [-1, 1]
        img = img * 2 - 1

        # Sample random timestep and add noise
        t = torch.randint(1, DIFFU_STEPS + 1, (1,), device=DEVICE)
        xt, _ = q_xt_x0(img, t)

        # Convert label to one-hot vector
        vec = torch.nn.functional.one_hot(
            torch.tensor(min(label, NB_LABEL - 1)),
            num_classes=NB_LABEL,
        )

        return (
            xt.clone().detach().to(dtype=torch.float32, device=DEVICE),
            t.clone().detach().to(dtype=torch.float32, device=DEVICE),
            vec.clone().detach().to(dtype=torch.float32, device=DEVICE),
            img.clone().detach().to(dtype=torch.float32, device=DEVICE),  # Target: clean image
        )

    def __len__(self) -> int:
        return len(self.dataset)


class AutoencoderDataset(Dataset):
    """
    Dataset wrapper for autoencoder training.
    Returns image as both input and target.

    Args:
        dataset: Base image dataset
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.dataset[idx][0].to(DEVICE)
        # data = data * 2 - 1  # Normalize to [-1, 1]
        return data, data


def get_diffusion_dataloader(
    predict_x0: bool = True,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    autoencoder: torch.nn.Module | None = None,
) -> DataLoader:
    """
    Create a DataLoader for diffusion training.

    Args:
        predict_x0: If True, model predicts x0. Otherwise predicts noise.
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        autoencoder: Optional autoencoder for latent diffusion

    Returns:
        DataLoader for training
    """
    mnist = get_mnist_dataset(train=True)

    if predict_x0:
        dataset = DiffusionDatasetX0(mnist, autoencoder)
    else:
        dataset = DiffusionDataset(mnist, autoencoder)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
    )


def get_autoencoder_dataloader(
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create a DataLoader for autoencoder training.

    Args:
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers

    Returns:
        DataLoader for training
    """
    mnist = get_mnist_dataset(train=True)
    dataset = AutoencoderDataset(mnist)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
    )
