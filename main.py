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
        # The input to the model is a 11 vector which represents the desired label with the contextual information.
        # The Input is passed through layers to generate a 1x28x28, 1x14x14, 1x7x7 tensor.
        # -------
        # input: 1x11
        self.inconv1 = nn.Linear(11, 28 * 28)
        self.inconv2 = nn.Linear(11, 14 * 14)
        self.inconv3 = nn.Linear(11, 7 * 7)

        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image.
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer,
        # with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 28x28x1
        self.e11 = nn.Conv2d(1, 64, kernel_size=3,
                             padding=1)  # output: 28x28x64
        self.e12 = nn.Conv2d(64, 64, kernel_size=3,
                             padding=1)  # output: 28x28x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 14x14x64

        # input: 14x14x64
        self.e21 = nn.Conv2d(64, 128, kernel_size=3,
                             padding=1)  # output: 14x14x128
        self.e22 = nn.Conv2d(128, 128, kernel_size=3,
                             padding=1)  # output: 14x14x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 7x7x128

        # input: 7x7x128
        self.e31 = nn.Conv2d(129, 256, kernel_size=3,
                             padding=1)  # output: 7x7x256
        self.e32 = nn.Conv2d(257, 256, kernel_size=3,
                             padding=1)  # output: 7x7x256

        # Decoder
        # In the decoder, the output of the encoder is upsampled using the ConvTranspose2d function.
        # Each block in the decoder consists of two convolutional layers followed by an upsampling layer,
        # with the exception of the last block which does not include an upsampling layer.
        # -------
        # input: 7x7x256
        self.upconv1 = nn.ConvTranspose2d(
            256, 128, kernel_size=2, stride=2)  # output: 14x14x128
        self.d11 = nn.Conv2d(256, 128, kernel_size=3,
                             padding=1)  # output: 14x14x(128x2)
        self.d12 = nn.Conv2d(128, 128, kernel_size=3,
                             padding=1)  # output: 14x14x128

        # input: 14x14x128
        self.upconv2 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2)  # output: 28x28x64
        self.d21 = nn.Conv2d(128, 64, kernel_size=3,
                             padding=1)  # output: 28x28x(64x2)
        self.d22 = nn.Conv2d(64, 64, kernel_size=3,
                             padding=1)  # output: 28x28x64

        # Output
        # The output of the decoder is passed through a convolutional layer with the Conv2d function to obtain the final output.
        # -------
        # input: 28x28x64
        self.outconv = nn.Conv2d(64, 1, kernel_size=1)  # output: 28x28x2

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


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DIFFU_STEPS = 10
BETA = torch.linspace(0.0001, 0.2, DIFFU_STEPS+1, device=DEVICE)
ALPHA = 1 - BETA
ALPHA_BAR = torch.cumprod(ALPHA, dim=0)
SIGMA2 = BETA


def q_xt_x0(x0, t):
    alpha_bar = ALPHA_BAR[t]
    mean = x0 * torch.sqrt(alpha_bar)
    std = 1 - alpha_bar
    return torch.distributions.Normal(mean, std)


def q_xt_xt_1(xt_1, t):
    beta = BETA[t]
    mean = torch.sqrt(1 - beta) * xt_1
    std = beta
    return torch.distributions.Normal(mean, std)


def p_xt_1_xt(model, xt, vec):
    t = vec[..., -1].to(dtype=torch.long)
    eps_theta = model(xt, vec)
    alpha_bar = ALPHA_BAR[t].unsqueeze(
        -1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 28, 28)
    alpha = ALPHA[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 28, 28)
    beta = BETA[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 28, 28)
    eps_coef = (1 - alpha) / torch.sqrt(1 - alpha_bar)
    # mean = 1 / torch.sqrt(alpha) * (xt - eps_coef * eps_theta)
    mean = (xt - beta * eps_theta)
    # std = torch.sqrt(
    #     SIGMA2[t]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 28, 28)
    std = beta
    return torch.distributions.Normal(mean, std)


class MNISTDiffusionDataset(Dataset):
    def __init__(self, train=True):
        super().__init__()
        self.mnist_data = datasets.MNIST(
            root='./data',
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )

    def __getitem__(self, index):
        # Get the image and the label.
        img, label = self.mnist_data[index]
        img = img.to(DEVICE)

        # Add noise to the image.
        t_1 = torch.randint(0, DIFFU_STEPS, (1,), device=DEVICE)
        t = t_1 + 1
        xt_1 = q_xt_x0(img, t_1).sample()
        eps = torch.distributions.Normal(0, 1).sample(img.shape).to(DEVICE)
        xt = xt_1 * torch.sqrt(ALPHA[t]) + BETA[t] * eps

        # Convert the label to a one-hot vector.
        vector = torch.nn.functional.one_hot(
            torch.tensor(label),
            num_classes=10,
        )

        # Add contextual information (t) to the label.
        vector = torch.cat([vector, torch.tensor([t_1+1])])

        return (
            xt.clone().detach().to(dtype=torch.float32, device=DEVICE),
            vector.clone().detach().to(dtype=torch.float32, device=DEVICE),
            (
                img.clone().detach().to(dtype=torch.float32, device=DEVICE),
                eps,
                t,
            ),
        )

    def __len__(self):
        return len(self.mnist_data)


def loss_fn(y_pred, y_true):
    x0, eps, t = y_true
    return nn.MSELoss()(y_pred, eps)


mnist_data = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)
img, label = mnist_data[0]
img = img.to(DEVICE)
fig = plt.figure(figsize=(DIFFU_STEPS, 2))
for t_1 in range(0, DIFFU_STEPS):
    xt_1 = q_xt_x0(img, t_1).sample()
    x_t = q_xt_xt_1(xt_1, t_1+1).sample()
    ax = fig.add_subplot(2, DIFFU_STEPS, t_1 + 1)
    ax.imshow(xt_1[0].cpu(), cmap='gray')
    ax.axis('off')
    ax = fig.add_subplot(2, DIFFU_STEPS, DIFFU_STEPS + t_1 + 1)
    ax.imshow(x_t[0].cpu(), cmap='gray')
    ax.axis('off')
fig.tight_layout()
fig.savefig('img.tmp.png')


############
# Training #
############

torch.multiprocessing.set_start_method('spawn')

# Load the model.
model = UNet().to(DEVICE)
# model.load_state_dict(torch.load('model.pth'))

# Define the optimizer.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Define the training dataset.
train_dataset = MNISTDiffusionDataset(train=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
trainer = Trainer()
criterion = loss_fn
epochs = 1

# Train the model.
trainer.train(model, train_loader, epochs, optimizer, criterion)

# Save the model.
torch.save(model.state_dict(), 'model.pth')


##############
# Evaluation #
##############

# for data in train_loader:
#     input, vector, (x0, eps, t) = data
#     eps_theta = model(input, vector)

#     fig = plt.figure(figsize=(2, 2))
#     ax = fig.add_subplot(2, 2, 1)
#     ax.imshow(eps[0].cpu().transpose(0, 2).transpose(0, 1), cmap='gray')
#     ax.axis('off')
#     ax = fig.add_subplot(2, 2, 2)
#     ax.imshow(eps_theta[0].cpu().detach().transpose(
#         0, 2).transpose(0, 1), cmap='gray')
#     ax.axis('off')
#     ax = fig.add_subplot(2, 2, 3)
#     ax.imshow((eps[0] - eps_theta[0]).cpu().detach().transpose(
#         0, 2).transpose(0, 1), cmap='gray')
#     ax.axis('off')
#     ax = fig.add_subplot(2, 2, 4)
#     ax.imshow(x0[0].cpu().transpose(0, 2).transpose(0, 1), cmap='gray')
#     ax.axis('off')
#     fig.tight_layout()
#     fig.savefig('eps.tmp.png')
#     exit()

# Load the model.
model = UNet().to(DEVICE)
model.load_state_dict(torch.load('model.pth'))

# Set the model to evaluation mode.
model.eval()

n = 10
fig = plt.figure(figsize=(2 * 2 * n, 3 * 2))
gs = plt.GridSpec(nrows=3, ncols=2*2*n)
for i in range(n):
    # Get the i-th input and its label.
    input, vector, (x0, eps, t) = train_dataset[i]

    xt_1 = q_xt_x0(input, t-1).sample()
    xt = q_xt_xt_1(xt_1, t).sample()
    xt_1_pred = p_xt_1_xt(
        model,
        xt.unsqueeze(0),
        vector.unsqueeze(0),
    ).sample()

    # Plot the input.
    ax = fig.add_subplot(gs[0:1, 1 + 4 * i:3 + 4 * i])
    ax.imshow(xt[0].cpu(), cmap='gray')
    ax.axis('off')

    # Plot the label.
    ax = fig.add_subplot(gs[1:2, 4 * i:2 + 4 * i])
    ax.imshow(xt_1[0].cpu(), cmap='gray')
    ax.axis('off')

    # Plot the model output.
    ax = fig.add_subplot(gs[1:2, 2 + 4 * i:4 + 4 * i])
    ax.imshow(xt_1_pred[0, 0].cpu().detach(), cmap='gray')
    ax.axis('off')

    # Plot the difference between the label and the model output.
    ax = fig.add_subplot(gs[2:3, 1 + 4 * i:3 + 4 * i])
    ax.imshow(
        (xt_1.cpu() - xt_1_pred[0, 0].cpu().detach())[0], cmap='coolwarm')
    ax.axis('off')

fig.tight_layout()
fig.savefig('diff.tmp.png')


# Plot the evolution of the noise.
fig = plt.figure(figsize=(n, DIFFU_STEPS))
noises = np.random.normal(0, 1, (n, 1, 28, 28))
noises = torch.Tensor(noises).to(DEVICE)
vector = torch.nn.functional.one_hot(
    torch.tensor(range(n)), num_classes=10).to(DEVICE)
vector = vector.clone().detach().to(dtype=torch.float32)

# Apply the model multiple times.
for i in range(DIFFU_STEPS):
    t = DIFFU_STEPS - i - 1
    noises = p_xt_1_xt(
        model,
        noises,
        torch.cat([
            vector,
            torch.tensor([t] * n)
            .unsqueeze(-1)
            .to(device=DEVICE)
            .to(dtype=torch.long),
        ], dim=-1),
    ).sample()

    for j in range(n):
        ax = fig.add_subplot(DIFFU_STEPS, n, i * n + j + 1)
        ax.imshow(noises[j, 0].cpu().detach(), cmap='gray')
        ax.axis('off')

fig.tight_layout()
fig.savefig('diffu.tmp.png')

# Plot bench of generated images.
fig = plt.figure(figsize=(n, n))
noises = np.random.normal(0, 1, (n * n, 1, 28, 28))
noises = torch.Tensor(noises).to(DEVICE)
vector = torch.nn.functional.one_hot(
    torch.tensor([range(n)] * n), num_classes=10).to(DEVICE)
vector = vector.clone().detach().to(dtype=torch.float32)

# Apply the model multiple times.
for i in range(DIFFU_STEPS):
    t = DIFFU_STEPS - i - 1
    noises = model(noises, torch.cat(
        [vector, torch.tensor([[t] * n] * n).unsqueeze(-1)
         .to(DEVICE)], dim=-1))

for i in range(n * n):
    ax = fig.add_subplot(n, n, i + 1)
    ax.imshow(noises[i, 0].cpu().detach(), cmap='gray')
    ax.axis('off')

fig.tight_layout()
fig.savefig('bench.tmp.png')

plt.close('all')
