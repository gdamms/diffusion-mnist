import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from torchvision import datasets, transforms

import matplotlib.pyplot as plt

import numpy as np
import os
import cv2

from trainer import Trainer


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        ## Inputs:
        # xt: image at step t (NB_CHANNEL*IMG_SIZE*IMG_SIZE)
        # t: step number (1)
        # vec: one-hot vector of the label (NB_LABEL)

        ## Encoder for t
        self.encodet = nn.Linear(1, IMG_SIZE*IMG_SIZE)

        ## Encoder for vec
        self.encodevec = nn.Linear(NB_LABEL, IMG_SIZE*IMG_SIZE)

        ## UNet (2 more channels input because we concatenate xt with t and vec)
        self.conv1 = nn.Conv2d(NB_CHANNEL+2, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.maxpool3 = nn.MaxPool2d(2, 2)
        self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        self.upconv1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv9 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv10 = nn.Conv2d(256, 256, 3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv11 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv12 = nn.Conv2d(128, 128, 3, padding=1)
        self.upconv3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv13 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv14 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv15 = nn.Conv2d(64, NB_CHANNEL, 3, padding=1)

    def forward(self, xt, t, vec):
        # Encode t and vec
        t = F.relu(self.encodet(t))
        t = t.view(-1, 1, IMG_SIZE, IMG_SIZE)
        vec = F.relu(self.encodevec(vec))
        vec = vec.view(-1, 1, IMG_SIZE, IMG_SIZE)

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
        x4 = self.maxpool3(x3)
        x4 = F.relu(self.conv7(x4))
        x4 = F.relu(self.conv8(x4))
        x5 = self.upconv1(x4)
        x5 = torch.cat((x5, x3), dim=1)
        x5 = F.relu(self.conv9(x5))
        x5 = F.relu(self.conv10(x5))
        x6 = self.upconv2(x5)
        x6 = torch.cat((x6, x2), dim=1)
        x6 = F.relu(self.conv11(x6))
        x6 = F.relu(self.conv12(x6))
        x7 = self.upconv3(x6)
        x7 = torch.cat((x7, x1), dim=1)
        x7 = F.relu(self.conv13(x7))
        x7 = F.relu(self.conv14(x7))
        x7 = self.conv15(x7)

        return x7


class LFWcrop(Dataset):
    def __init__(self):
        super().__init__()
        self.path = './data/lfwcrop_color/faces'
        self.files = os.listdir(self.path)

    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.path, self.files[index]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1)) / 255
        return torch.tensor(img, dtype=torch.float32), 0

    def __len__(self):
        return len(self.files)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIFFU_STEPS = 1000
BETA = torch.linspace(1e-4, 2e-2, DIFFU_STEPS, device=DEVICE)
BETA = torch.cat((torch.tensor([0.], device=DEVICE), BETA))
ALPHA = 1 - BETA
ALPHA_BAR = torch.cumprod(ALPHA, dim=0)


# dataset = datasets.MNIST(
#     root="./data",
#     train=True,
#     download=True,
#     transform=transforms.ToTensor(),
# )
# dataset = datasets.LFWPeople(
#     root="./data",
#     download=True,
#     transform=transforms.Compose([
#         transforms.Resize((64, 64)),
#         transforms.ToTensor(),
#     ]),
# )
dataset = LFWcrop()

img = dataset[0][0]
NB_CHANNEL, IMG_SIZE, _ = img.shape
NB_LABEL = 1

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

    return torch.distributions.Normal(mu_theta, torch.sqrt(sigma_theta)).sample()


class DiffusionDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        # Get the image and the label.
        img, label = self.dataset[index]
        img = img.to(DEVICE)

        # Normalize the image.
        img = img * 2 - 1

        # Add noise to the image.
        t = torch.randint(1, DIFFU_STEPS, (1,), device=DEVICE)
        xt = q_xt_x0(img, t)
        eps = xt - img

        # Convert the label to a one-hot vector.
        vec = torch.nn.functional.one_hot(
            torch.tensor(min(label, NB_LABEL-1)),
            num_classes=NB_LABEL,
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
        return len(self.dataset)


def loss(y_pred, y_true):
    return nn.MSELoss()(y_pred, y_true)


def forward_diffusion(x0):
    x = x0.clone()
    xs = [x.cpu()]
    for t in range(1, DIFFU_STEPS+1):
        x = q_xt_xt_1(x, t)
        xs.append(x.cpu())
    return xs


def tensor_to_image(tensor):
    img = tensor.clone().detach().cpu().numpy().transpose(1, 2, 0)
    img = img / 2 + 0.5
    img -= img.min()
    img /= img.max()
    return img


if __name__ == '__main__':
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
    train_dataset = DiffusionDataset(dataset)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                              num_workers=4, persistent_workers=True)
    trainer = Trainer()
    criterion = loss
    epochs = 100

    # Train the model.
    trainer.train(model, train_loader, epochs, optimizer, criterion)

    # Save the model.
    torch.save(model.state_dict(), 'model.pth')

    ############## 
    # Evaluation #
    ##############

    # Forward diffusion
    img, label = dataset[np.random.randint(0, len(dataset))]
    img = img.to(DEVICE) * 2 - 1

    nb_plots = 10
    plots_id = [i for i in np.linspace(1, DIFFU_STEPS, nb_plots, dtype=int)]

    xs = forward_diffusion(img)

    plt.figure(figsize=(nb_plots, 2.5))
    for t, x in enumerate(xs):
        if t not in plots_id:
            continue
        plot_i = plots_id.index(t)
        plt.subplot(2, nb_plots + 1, plot_i + 2)
        plt.title(f"t={t}")
        plt.imshow(tensor_to_image(x))
        plt.axis("off")

    for t in range(1, DIFFU_STEPS+1):
        x = q_xt_x0(img, t).cpu()
        if t not in plots_id:
            continue
        plot_i = plots_id.index(t)
        plt.subplot(2, nb_plots + 1, nb_plots + plot_i + 3)
        plt.imshow(tensor_to_image(x))
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

    x = torch.randn(n_classes, NB_CHANNEL, IMG_SIZE, IMG_SIZE, device=DEVICE)
    vec = torch.tensor([[min(i, NB_LABEL-1)] for i in range(n_classes)], dtype=torch.int64)
    vec = torch.nn.functional.one_hot(vec, num_classes=NB_LABEL).to(device=DEVICE, dtype=torch.float32)

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
                plt.imshow(tensor_to_image(x[class_i]))
                plt.axis("off")
    plt.tight_layout()
    plt.savefig("backward_diffusion.tmp.png")


    # Benchmark
    x = torch.randn(nb_plots * n_classes, NB_CHANNEL, IMG_SIZE, IMG_SIZE).to(DEVICE)
    vec = sum([[[min(i, NB_LABEL-1)]] * nb_plots for i in range(n_classes)], [])
    vec = torch.tensor(vec, device=DEVICE, dtype=torch.int64)
    vec = torch.nn.functional.one_hot(vec, num_classes=NB_LABEL).to(device=DEVICE, dtype=torch.float32)

    for ti in range(DIFFU_STEPS, 0, -1):
        t = torch.tensor([[ti]] * n_classes * nb_plots, device=DEVICE, dtype=torch.float32)
        x = p_xt_1_xt(model, x, t, vec)

    plt.figure(figsize=(nb_plots, n_classes))
    for i in range(nb_plots):
        for j in range(n_classes):
            id = i * n_classes + j
            plt.subplot(n_classes, nb_plots, id + 1)
            plt.imshow(tensor_to_image(x[id]))
            plt.axis("off")
    plt.tight_layout()
    plt.savefig("benchmark.tmp.png")

    # plt.show()