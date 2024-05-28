import torch
from torch.utils.data import DataLoader, Dataset

from torchvision import datasets, transforms

import matplotlib.pyplot as plt

from trainer import Trainer


class PrintLayer(torch.nn.Module):
    def forward(self, x):
        print(x.shape)
        return x


class Autoencoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.latent_size = torch.prod(torch.tensor(latent_dim))
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim[0], 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 7 * 7, self.latent_size),
            torch.nn.ReLU(),
            torch.nn.Unflatten(1, latent_dim),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(self.latent_size, 64 * 7 * 7),
            torch.nn.ReLU(),
            torch.nn.Unflatten(1, (64, 7, 7)),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, input_dim, kernel_size=3, padding=1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


class AutoencoderDataset(Dataset):
    def __init__(self, dataset, device='cpu'):
        self.dataset = dataset
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx][0].to(self.device)
        return data, data

def main():
    torch.multiprocessing.set_start_method("spawn")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    mnist = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    dataset = AutoencoderDataset(mnist, device=device)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True,
                            num_workers=4, persistent_workers=True)

    # Initialize model
    model = Autoencoder(input_dim=(1, 28, 28), latent_dim=(1, 8, 8))
    # model.load_state_dict(torch.load('autoencoder.pth'))
    model.to(device)

    # Train model
    trainer = Trainer()
    lr = 1e-3
    epochs = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.functional.mse_loss
    trainer.train(model, dataloader, epochs, optimizer, criterion)

    # Save model
    torch.save(model.state_dict(), 'autoencoder.pth')

    # Visualize results
    n = 10
    with torch.no_grad():
        plt.figure(figsize=(2*n, 4))
        for i, j in enumerate(torch.randint(0, len(dataset), (n,))):
            x, _ = dataset[j]
            x = x.unsqueeze(0)
            x_hat = model(x)
            plt.subplot(2, n, i + 1)
            plt.imshow(x.cpu().squeeze().numpy(), cmap='gray')
            plt.axis('off')
            plt.subplot(2, n, i + n + 1)
            plt.imshow(x_hat.cpu().squeeze().numpy(), cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig('autoencoder.tmp.png')


if __name__ == '__main__':
    main()
