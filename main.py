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

        ## UNet (3 channels input because we concatenate xt, t and vec)
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

DIFFU_STEPS = 200
BETA = torch.linspace(1e-4, 2e-2, DIFFU_STEPS, device=DEVICE)
BETA = torch.cat((torch.tensor([0.], device=DEVICE), BETA))
ALPHA = 1 - BETA
ALPHA_BAR = torch.cumprod(ALPHA, dim=0)
ALPHA_BAR_BAR = torch.cumsum(torch.sqrt(ALPHA_BAR), dim=0)

plt.figure()
plt.plot(BETA.cpu().detach().numpy(), label="beta")
plt.plot(ALPHA.cpu().detach().numpy(), label="alpha")
plt.plot(ALPHA_BAR.cpu().detach().numpy(), label="alpha_bar")
# plt.plot(ALPHA_BAR_BAR.cpu().detach().numpy(), label="aplha_bar_bar")
plt.plot(ALPHA_BAR_BAR.cpu().detach().numpy() * (1 - ALPHA).cpu().detach().numpy(), label="sigma_theta")
plt.legend()
plt.xlim(0, DIFFU_STEPS)
# plt.ylim(0, 1)
plt.savefig("beta_alpha.tmp.png")


def q_xt_xt_1(xt_1, t):
    t_ind = t.to(dtype=torch.long) if isinstance(t, torch.Tensor) else t

    alpha = ALPHA[t_ind]
    mean = torch.sqrt(alpha) * xt_1
    std = 1 - alpha
    xt = torch.distributions.Normal(mean, std).sample()

    print('xt_1 xt', xt_1.min(), xt_1.max(), xt.min(), xt.max())

    return xt

"""
A = sqrt(alpha)
B = 1 - alpha

x1 = A * x0 + B * e
x2 = A * x1 + B * e
   = A * (A * x0 + B * e) + B * e
   = A^2 * x0 + A * B * e + B * e
   = A^2 * x0 + (A * B + B) * e
   = A^2 * x0 + (A + 1) * B * e
x3 = A * x2 + B * e
   = A * (A^2 * x0 + (A * B + B) * e) + B * e
   = A^3 * x0 + A * (A * B + B) * e + B * e
   = A^3 * x0 + (A^2 * B + A * B + B) * e
   = A^3 * x0 + (A^2 + A + 1) * B * e
x4 = A * x3 + B * e
   = A * (A^3 * x0 + (A^2 + A + 1) * B * e) + B * e
   = A^4 * x0 + A * (A^2 + A + 1) * B * e + B * e
   = A^4 * x0 + (A^3 * B + A^2 * B + A * B + B) * e
   = A^4 * x0 + (A^3 + A^2 + A + 1) * B * e
"""


def q_xt_x0(x0, t):
    t_ind = t.to(dtype=torch.long) if isinstance(t, torch.Tensor) else t

    alpha = ALPHA[t_ind]
    alpha_bar = ALPHA_BAR[t_ind]
    alpha_bar_bar = ALPHA_BAR_BAR[t_ind - 1]
    mean = torch.sqrt(alpha_bar) * x0
    std = alpha_bar_bar * (1 - alpha)
    return torch.distributions.Normal(mean, std).sample()


def p_xt_1_xt(model, xt, t, vec):
    t_ind = t.to(dtype=torch.long) if isinstance(t, torch.Tensor) else t

    alpha_bar_t = ALPHA_BAR[t_ind]
    alpha_bar_t_1 = ALPHA_BAR[t_ind-1]
    alpha_t = ALPHA[t_ind]
    beta_t = BETA[t_ind]

    beta_tilde = (1 - alpha_bar_t_1) / (1 - alpha_bar_t) * beta_t

    epsilon_theta = model(xt, t, vec)

    sigma_theta = beta_tilde
    mu_theta = (xt - beta_t / torch.sqrt(1 - alpha_bar_t) * epsilon_theta) / torch.sqrt(alpha_t)
    if sigma_theta.abs().max() <= 0:
        return mu_theta

    return torch.distributions.Normal(mu_theta, sigma_theta).sample()


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

        # Normalize the image between -0.5 and 0.5.
        img = img - 0.5

        # Add noise to the image.
        t = torch.randint(1, DIFFU_STEPS, (1,), device=DEVICE)
        xt = q_xt_x0(img, t)
        eps = xt - img

        # Convert the label to a one-hot vector.
        vec = torch.nn.functional.one_hot(
            torch.tensor(label),
            num_classes=10,
        )

        return (
            # x_true
            xt.clone().detach().to(dtype=torch.float32, device=DEVICE),
            t.clone().detach().to(dtype=torch.float32, device=DEVICE),
            vec.clone().detach().to(dtype=torch.float32, device=DEVICE),
            # y_true
            eps,
        )

    def __len__(self):
        return len(self.mnist_data)


def loss(y_pred, y_true):
    return nn.MSELoss()(y_pred, y_true)


def forward_diffusion(x0):
    x = x0.clone()[0]
    xs = [x.cpu().detach().numpy()]
    for t in range(1, DIFFU_STEPS):
        x = q_xt_xt_1(x, t)
        xs.append(x.cpu().detach().numpy())
    return xs


if __name__ == '__main__':
    mnist_data = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )


    img, label = mnist_data[1]
    img = img.to(DEVICE) - 0.5

    fig = plt.figure(figsize=(DIFFU_STEPS, 2))

    xs = forward_diffusion(img)
    for t, x in enumerate(xs):
        ax = fig.add_subplot(2, DIFFU_STEPS, t+1)
        ax.imshow(x, cmap="gray")
        ax.axis("off")

    for t in range(1, DIFFU_STEPS):
        x = q_xt_x0(img, t)[0].cpu()
        print('x0', x.min(), x.max())
        ax = fig.add_subplot(2, DIFFU_STEPS, DIFFU_STEPS + t + 1)
        ax.imshow(x, cmap="gray")
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Define the trainiself.encodevec = nn.Linear(10, 28*28)ng dataset.
    train_dataset = MNISTDiffusionDataset(train=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    trainer = Trainer()
    criterion = loss
    epochs = 30

    # # Train the model.
    # trainer.train(model, train_loader, epochs, optimizer, criterion)

    # # Save the model.
    # torch.save(model.state_dict(), "model.pth")

    ############## 
    # Evaluation #
    ##############

    nb_plots = 5
    ti_plots = np.linspace(1, DIFFU_STEPS, nb_plots, dtype=int)
    n_values = [i for i in range(10)]

    fig = plt.figure(figsize=(nb_plots, len(n_values)))

    for attempti, n in enumerate(n_values):
        n = torch.tensor([[n]], device=DEVICE, dtype=torch.int64)
        x = torch.randn(1, 1, 28, 28).to(DEVICE)
        vec = torch.nn.functional.one_hot(n, num_classes=10).to(device=DEVICE, dtype=torch.float32)
        img_vec = model.encodevec(vec)
        img_vec = F.relu(img_vec)
        img_vec = img_vec.view(-1, 1, 28, 28)
        img_vec = img_vec.cpu().detach().numpy()

        for ti in range(DIFFU_STEPS, 0, -1):
            t = torch.tensor([[ti]], device=DEVICE, dtype=torch.float32)
            x = p_xt_1_xt(model, x, t, vec)
            if ti in ti_plots:
                ti_plotind = nb_plots - np.where(ti_plots == ti)[0][0]
                ax = fig.add_subplot(len(n_values), nb_plots, ti_plotind + nb_plots * attempti) 
                ax.imshow(x[0, 0].detach().cpu(), cmap="gray")
                ax.axis("off")
                ax.set_title(f"{ti}")
    fig.tight_layout()
    fig.savefig("diffused.tmp.png")
