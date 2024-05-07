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

DIFFU_STEPS = 300
BETA = torch.linspace(1e-4, 2e-2, DIFFU_STEPS, device=DEVICE)
BETA = torch.cat((torch.tensor([0.], device=DEVICE), BETA))
ALPHA = 1 - BETA
ALPHA_BAR = torch.cumprod(ALPHA, dim=0)


def q_xt_xt_1(xt_1, t):
    t_ind = t.to(dtype=torch.long) if isinstance(t, torch.Tensor) else t

    alpha = ALPHA[t_ind]
    mean = torch.sqrt(alpha) * xt_1
    std = torch.sqrt(1 - alpha)
    xt = torch.distributions.Normal(mean, std).sample()

    return xt

def q_xt_x0(x0, t):
    t_ind = t.to(dtype=torch.long) if isinstance(t, torch.Tensor) else t

    alpha_bar = ALPHA_BAR[t_ind]
    mean = torch.sqrt(alpha_bar) * x0
    std = torch.sqrt(1 - alpha_bar)
    return torch.distributions.Normal(mean, std).sample()


def p_xt_1_xt(model, xt, t, vec):
    t_ind = t.to(dtype=torch.long) if isinstance(t, torch.Tensor) else t

    alpha_bar_t = ALPHA_BAR[t_ind].view(-1, 1, 1, 1)
    alpha_bar_t_1 = ALPHA_BAR[t_ind-1].view(-1, 1, 1, 1)
    alpha_t = ALPHA[t_ind].view(-1, 1, 1, 1)
    beta_t = BETA[t_ind].view(-1, 1, 1, 1)

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

        # Normalize the image.
        img = img * 2 - 1

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
    for t in range(1, DIFFU_STEPS+1):
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
    ############
    # Training #
    ############

    torch.multiprocessing.set_start_method("spawn")

    # Load the model.
    model = UNet().to(DEVICE)
    try:
        model.load_state_dict(torch.load('model.pth'))
    except FileNotFoundError:
        print("No model found, training a new one.")
        pass

    # Define the optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    # Define the training dataset.
    train_dataset = MNISTDiffusionDataset(train=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                              num_workers=4, persistent_workers=True)
    trainer = Trainer()
    criterion = loss
    epochs = 0

    # Train the model.
    trainer.train(model, train_loader, epochs, optimizer, criterion)

    # Save the model.
    torch.save(model.state_dict(), 'model.pth')

    ############## 
    # Evaluation #
    ##############

    # Forward diffusion
    img, label = mnist_data[np.random.randint(0, len(mnist_data))]
    img = img.to(DEVICE) * 2 - 1

    nb_plots = 6
    plots_id = [i for i in np.linspace(1, DIFFU_STEPS, nb_plots, dtype=int)]

    xs = forward_diffusion(img)

    plt.figure(figsize=(nb_plots, 2.5))
    for t, x in enumerate(xs):
        if t not in plots_id:
            continue
        plot_i = plots_id.index(t)
        plt.subplot(2, nb_plots + 1, plot_i + 2)
        plt.title(f"t={t}")
        plt.imshow(x, cmap="gray")
        plt.axis("off")

    for t in range(1, DIFFU_STEPS+1):
        x = q_xt_x0(img, t)[0].cpu()
        if t not in plots_id:
            continue
        plot_i = plots_id.index(t)
        plt.subplot(2, nb_plots + 1, nb_plots + plot_i + 3)
        plt.imshow(x, cmap="gray")
        plt.axis("off")

    plt.subplot(2, nb_plots + 1, 1)
    plt.text(0, 0.5, "Implicit", fontsize=12)
    plt.axis("off")
    plt.subplot(2, nb_plots + 1, nb_plots + 2)
    plt.text(0, 0.5, "Explicit", fontsize=12)
    plt.axis("off")

    plt.suptitle("Forward diffusion")
    plt.tight_layout()
    plt.savefig("forward_diffusion.tmp.png")


    # Backward diffusion
    nb_plots = 6
    t_plots = np.linspace(1, DIFFU_STEPS, nb_plots, dtype=int)

    n_classes = 10

    x = torch.randn(n_classes, 1, 28, 28,device=DEVICE)
    vec = torch.tensor([[i] for i in range(n_classes)], dtype=torch.int64)
    vec = torch.nn.functional.one_hot(vec, num_classes=10).to(device=DEVICE, dtype=torch.float32)

    plt.figure(figsize=(nb_plots, n_classes))
    plt.suptitle("Backward diffusion")
    for t in range(DIFFU_STEPS, 0, -1):
        t_tensor = torch.tensor([[t]] * n_classes, device=DEVICE, dtype=torch.float32)
        x = p_xt_1_xt(model, x, t_tensor, vec)
        if t in t_plots:
            t_plot_i = nb_plots - t_plots.tolist().index(t) - 1
            for class_i in range(n_classes):
                plt.subplot(n_classes, nb_plots, t_plot_i + nb_plots * class_i + 1)
                if class_i == 0:
                    plt.title(f"t={t}")
                plt.imshow(x[class_i, 0].detach().cpu(), cmap="gray")
                plt.axis("off")
    plt.tight_layout()
    plt.savefig("backward_diffusion.tmp.png")


    # Benchmark
    x = torch.randn(nb_plots * n_classes, 1, 28, 28).to(DEVICE)
    vec = sum([[[i]] * nb_plots for i in range(n_classes)], [])
    vec = torch.tensor(vec, device=DEVICE, dtype=torch.int64)
    vec = torch.nn.functional.one_hot(vec, num_classes=10).to(device=DEVICE, dtype=torch.float32)

    for ti in range(DIFFU_STEPS, 0, -1):
        t = torch.tensor([[ti]] * n_classes * nb_plots, device=DEVICE, dtype=torch.float32)
        x = p_xt_1_xt(model, x, t, vec)

    plt.figure(figsize=(nb_plots, n_classes))
    for i in range(nb_plots):
        for j in range(n_classes):
            id = i * n_classes + j
            plt.subplot(n_classes, nb_plots, id + 1)
            plt.imshow(x[id, 0].detach().cpu(), cmap="gray")
            plt.axis("off")
    plt.tight_layout()
    plt.savefig("benchmark.tmp.png")

    plt.show()