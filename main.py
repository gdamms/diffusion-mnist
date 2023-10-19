import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from torchvision import datasets, transforms

import matplotlib.pyplot as plt

import numpy as np

from trainer import Trainer


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Input
        # The input to the model is a 10 vector which represents the input image.
        # The Input is passed through layers to generate a 1x28x28, 1x14x14, 1x7x7 tensor.
        # -------
        # input: 1x10
        self.inconv1 = nn.Linear(10, 28 * 28)
        self.inconv2 = nn.Linear(10, 14 * 14)
        self.inconv3 = nn.Linear(10, 7 * 7)

        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image.
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer,
        # with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 28x28x1
        self.e11 = nn.Conv2d(1, 64, kernel_size=3, padding=1)  # output: 28x28x64
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # output: 28x28x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 14x14x64

        # input: 14x14x64
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # output: 14x14x128
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # output: 14x14x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 7x7x128

        # input: 7x7x128
        self.e31 = nn.Conv2d(129, 256, kernel_size=3, padding=1)  # output: 7x7x256
        self.e32 = nn.Conv2d(257, 256, kernel_size=3, padding=1)  # output: 7x7x256

        # Decoder
        # In the decoder, the output of the encoder is upsampled using the ConvTranspose2d function.
        # Each block in the decoder consists of two convolutional layers followed by an upsampling layer,
        # with the exception of the last block which does not include an upsampling layer.
        # -------
        # input: 7x7x256
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # output: 14x14x128
        self.d11 = nn.Conv2d(256, 128, kernel_size=3, padding=1)  # output: 14x14x(128x2)
        self.d12 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # output: 14x14x128

        # input: 14x14x128
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # output: 28x28x64
        self.d21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  # output: 28x28x(64x2)
        self.d22 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # output: 28x28x64

        # Output
        # The output of the decoder is passed through a convolutional layer with the Conv2d function to obtain the final output.
        # -------
        # input: 28x28x64
        self.outconv = nn.Conv2d(64, 1, kernel_size=1)  # output: 28x28x1

    def forward(self, x, y):
        # Input
        y = self.inconv3(y)
        y = y.view(-1, 1, 7, 7)

        # Encoder
        x = F.relu(self.e11(x))
        x1 = F.relu(self.e12(x))
        x = self.pool1(x1)

        x = F.relu(self.e21(x))
        x2 = F.relu(self.e22(x))
        x = self.pool2(x2)

        x = torch.cat([x, y], dim=1)
        x = F.relu(self.e31(x))
        x = torch.cat([x, y], dim=1)
        x = F.relu(self.e32(x))

        # Decoder
        x = self.upconv1(x)
        x = torch.cat([x, x], dim=1)
        x = F.relu(self.d11(x))
        x = F.relu(self.d12(x))

        x = self.upconv2(x)
        x = torch.cat([x, x1], dim=1)
        x = F.relu(self.d21(x))
        x = F.relu(self.d22(x))

        # Output
        x = self.outconv(x)

        return x


DIFFU_STEPS = 10


class MNISTDiffusionDataset(Dataset):
    def __init__(self, train=True):
        super().__init__()
        self.mnist_data = datasets.MNIST(root='./data', train=train, download=True, transform=transforms.ToTensor())

    def __getitem__(self, index):
        # Get the image and the label.
        img, label = self.mnist_data[index]

        # Add noise to the image.
        noise = np.random.normal(0, 1, (28, 28))
        alpha = np.random.uniform(1 / DIFFU_STEPS, 1.0)

        # The target is the image with the noise.
        target = img * alpha + noise * (1 - alpha)

        # The input is the image with more noise.
        input = img * (alpha - 1 / DIFFU_STEPS) + noise * (1 - alpha + 1 / DIFFU_STEPS)

        # Convert the label to a one-hot vector.
        vector = torch.nn.functional.one_hot(torch.tensor(label), num_classes=10)

        return (input.clone().detach().to(dtype=torch.float32),
                vector.clone().detach().to(dtype=torch.float32),
                target.clone().detach().to(dtype=torch.float32))

    def __len__(self):
        return len(self.mnist_data)


############
# Training #
############

# Define the device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model.
model = UNet().to(device)
# model.load_state_dict(torch.load('model.pth'))

# Define the optimizer.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Define the training dataset.
train_dataset = MNISTDiffusionDataset(train=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
trainer = Trainer()

# Train the model.
trainer.train(model, train_loader, 2, optimizer, F.mse_loss)

# Save the model.
torch.save(model.state_dict(), 'model.pth')


##############
# Evaluation #
##############

# Load the model.
model = UNet().to(device)
model.load_state_dict(torch.load('model.pth'))

# Set the model to evaluation mode.
model.eval()

n = 10
fig = plt.figure(figsize=(2 * 2 * n, 3 * 2))
gs = plt.GridSpec(nrows=3, ncols=2*2*n)
for i in range(n):
    # Get the i-th input and its label.
    input, vector, label = train_dataset[i]

    # Plot the input.
    ax = fig.add_subplot(gs[0:1, 1 + 4 * i:3 + 4 * i])
    ax.imshow(input[0], cmap='gray')
    ax.axis('off')

    # Plot the label.
    ax = fig.add_subplot(gs[1:2, 4 * i:2 + 4 * i])
    ax.imshow(label[0], cmap='gray')
    ax.axis('off')

    # Get the model output.
    output = model(input.unsqueeze(0).to(device), vector.unsqueeze(0).to(device))

    # Plot the model output.
    ax = fig.add_subplot(gs[1:2, 2 + 4 * i:4 + 4 * i])
    ax.imshow(output[0, 0].cpu().detach(), cmap='gray')
    ax.axis('off')

    # Plot the difference between the label and the model output.
    ax = fig.add_subplot(gs[2:3, 1 + 4 * i:3 + 4 * i])
    ax.imshow((label - output[0, 0].cpu().detach())[0], cmap='coolwarm')
    ax.axis('off')

fig.tight_layout()
fig.savefig('diff.tmp.png')


# Plot the evolution of the noise.
fig = plt.figure(figsize=(n, DIFFU_STEPS))
noises = np.random.normal(0, 1, (n, 1, 28, 28))
noises = torch.Tensor(noises).to(device)
vector = torch.nn.functional.one_hot(torch.tensor(range(n)), num_classes=10).to(device)
vector = vector.clone().detach().to(dtype=torch.float32)

# Apply the model multiple times.
for i in range(DIFFU_STEPS):
    noises = model(noises, vector)

    for j in range(n):
        ax = fig.add_subplot(DIFFU_STEPS, n, i * n + j + 1)
        ax.imshow(noises[j, 0].cpu().detach(), cmap='gray')
        ax.axis('off')

fig.tight_layout()
fig.savefig('diffu.tmp.png')

# Plot bench of generated images.
fig = plt.figure(figsize=(n, n))
noises = np.random.normal(0, 1, (n * n, 1, 28, 28))
noises = torch.Tensor(noises).to(device)
vector = torch.nn.functional.one_hot(torch.tensor([range(n)] * n), num_classes=10).to(device)
vector = vector.clone().detach().to(dtype=torch.float32)

# Apply the model multiple times.
for i in range(DIFFU_STEPS):
    noises = model(noises, vector)

for i in range(n * n):
    ax = fig.add_subplot(n, n, i + 1)
    ax.imshow(noises[i, 0].cpu().detach(), cmap='gray')
    ax.axis('off')

fig.tight_layout()
fig.savefig('bench.tmp.png')

plt.close('all')
