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

        ## Inputs:
        # xt: image at step t (1x28x28)
        # t: step number (1)
        # vec: one-hot vector of the label (10)

        ## Encoder for t
        self.encodet = nn.Linear(1, 28*28)

        ## Encoder for vec
        self.encodevec = nn.Linear(10, 28*28)

        ## UNet
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.maxpool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.maxpool2 = nn.MaxPool2d(2, 2) # 14x14 -> 7x7
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.upconv1 = nn.ConvTranspose2d(256, 128, 2, stride=2) # 7x7 -> 14x14
        self.conv7 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv8 = nn.Conv2d(128, 128, 3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2) # 14x14 -> 28x28
        self.conv9 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv10 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv11 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, xt, t, vec):
        # Encode t and vec
        t = F.relu(self.encodet(t))
        t = t.view(-1, 1, 28, 28)
        vec = F.relu(self.encodevec(vec))
        vec = vec.view(-1, 1, 28, 28)

        # Concat all 3
        x = torch.cat((xt, t, vec), dim=1)

        # UNet
        x1 = F.relu(self.conv1(x))
        x1 = F.relu(self.conv2(x1))
        x2 = self.maxpool1(x1)
        x2 = F.relu(self.conv3(x2))
        x2 = F.relu(self.conv4(x2))
        x3 = self.maxpool2(x2)
        x3 = F.relu(self.conv5(x3))
        x3 = F.relu(self.conv6(x3))
        x4 = self.upconv1(x3)
        x4 = torch.cat((x4, x2), dim=1)
        x4 = F.relu(self.conv7(x4))
        x4 = F.relu(self.conv8(x4))
        x5 = self.upconv2(x4)
        x5 = torch.cat((x5, x1), dim=1)
        x5 = F.relu(self.conv9(x5))
        x5 = F.relu(self.conv10(x5))
        x5 = self.conv11(x5)

        return x5


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIFFU_STEPS = 20
BETA = torch.linspace(0.0001, 0.2, DIFFU_STEPS, device=DEVICE)
ALPHA = 1 - BETA
ALPHA_BAR = torch.cumprod(ALPHA, dim=0)
SIGMA2 = BETA


def q_xt_xt_1(xt_1, t):
    t_ind = t.to(dtype=torch.long) if isinstance(t, torch.Tensor) else t
    beta = BETA[t_ind]
    mean = torch.sqrt(1 - beta) * xt_1
    std = beta
    return torch.distributions.Normal(mean, std)


def q_xt_x0(x0, t):
    t_ind = t.to(dtype=torch.long) if isinstance(t, torch.Tensor) else t
    alpha_bar = ALPHA_BAR[t_ind]
    mean = torch.sqrt(alpha_bar) * x0
    std = 1 - alpha_bar
    return torch.distributions.Normal(mean, std)


def p_xt_1_xt(model, xt, t, vec):
    t_ind = t.to(dtype=torch.long) if isinstance(t, torch.Tensor) else t
    alpha_bar_t = ALPHA_BAR[t_ind]
    alpha_bar_t_1 = ALPHA_BAR[t_ind-1]
    alpha_t = ALPHA[t_ind]
    beta_t = BETA[t_ind]
    beta_tilde = (1 - alpha_bar_t_1) / (1 - alpha_bar_t) * beta_t

    epsilon_theta = model(xt, t, vec)

    # sigma_theta = torch.exp(nu * torch.log(beta_t) + (1 - nu) * torch.log(beta_tilde))
    sigma_theta = beta_tilde
    mu_theta = (xt - beta_t / torch.sqrt(1 - alpha_bar_t) * epsilon_theta) / torch.sqrt(alpha_t)

    return torch.distributions.Normal(mu_theta, sigma_theta)


class MNISTDiffusionDataset(Dataset):
    def __init__(self, train=True):
        super().__init__()
        self.mnist_data = datasets.MNIST(
            root="./data",
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )

    def __getitem__(self, index):
        # Get the image and the label.
        img, label = self.mnist_data[index]
        img = img.to(DEVICE)

        # Add noise to the image.
        t = torch.randint(1, DIFFU_STEPS, (1,), device=DEVICE)
        xt = q_xt_x0(img, t).sample()
        eps = xt - img

        # Convert the label to a one-hot vector.
        vec = torch.nn.functional.one_hot(
            torch.tensor(label),
            num_classes=10,
        )

        return (
            xt.clone().detach().to(dtype=torch.float32, device=DEVICE),
            t.clone().detach().to(dtype=torch.float32, device=DEVICE),
            vec.clone().detach().to(dtype=torch.float32, device=DEVICE),
            eps,
        )

    def __len__(self):
        return len(self.mnist_data)


def loss(y_pred, y_true):
    return nn.MSELoss()(y_pred, y_true)



mnist_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)
img, label = mnist_data[0]
img = img.to(DEVICE)
fig = plt.figure(figsize=(DIFFU_STEPS, 2))
for t in range(1, DIFFU_STEPS):
    xt_1 = q_xt_x0(img, t - 1).sample()
    x_t = q_xt_xt_1(xt_1, t).sample()
    ax = fig.add_subplot(2, DIFFU_STEPS, t)
    ax.imshow(xt_1[0].cpu(), cmap="gray")
    ax.axis("off")
    ax = fig.add_subplot(2, DIFFU_STEPS, DIFFU_STEPS + t)
    ax.imshow(x_t[0].cpu(), cmap="gray")
    ax.axis("off")
fig.tight_layout()
fig.savefig("img.tmp.png")


############
# Training #
############

torch.multiprocessing.set_start_method("spawn")

# Load the model.
model = UNet().to(DEVICE)
model.load_state_dict(torch.load('model.pth'))

# Define the optimizer.
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

# Define the training dataset.
train_dataset = MNISTDiffusionDataset(train=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
trainer = Trainer()
criterion = loss
epochs = 5

# # Train the model.
# trainer.train(model, train_loader, epochs, optimizer, criterion)

# # Save the model.
# torch.save(model.state_dict(), "model.pth")


##############
# Evaluation #
##############

x = torch.randn(1, 1, 28, 28).to(DEVICE)
vec = torch.nn.functional.one_hot(torch.randint(0, 10, (1, 1)), num_classes=10).to(device=DEVICE, dtype=torch.float32)
print(vec)

fig = plt.figure(figsize=(DIFFU_STEPS, 2))
for ti in range(1, DIFFU_STEPS):
    t = torch.tensor([[ti]], device=DEVICE, dtype=torch.float32)
    x = p_xt_1_xt(model, x, t, vec).sample()
    ax = fig.add_subplot(1, DIFFU_STEPS, ti)
    ax.imshow(x[0, 0].detach().cpu(), cmap="gray")
    ax.axis("off")
fig.tight_layout()
fig.savefig("diffused.tmp.png")