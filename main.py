import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.attention as attention
from torch.utils.data import DataLoader, Dataset
from rich.progress import track

from torchvision import datasets, transforms

import matplotlib.pyplot as plt

import numpy as np
import os
import cv2

from trainer import Trainer
from autoencoder import Autoencoder
from utils import *


class SelfAttention(nn.Module):
    def __init__(self, nb_channels, nb_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(nb_channels, nb_heads)

    def forward(self, x):
        _, c, w, h = x.shape
        x = x.view(-1, c, w*h)
        x = x.permute(2, 0, 1)
        x, _ = self.attention(x, x, x)
        x = x.permute(1, 2, 0)
        x = x.view(-1, c, w, h)
        return x


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
        # self.att1 = SelfAttention(128, 8)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        # self.att2 = SelfAttention(256, 8)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.upconv1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = nn.Conv2d(256, 128, 3, padding=1)
        # self.att3 = SelfAttention(128, 8)
        self.conv8 = nn.Conv2d(128, 128, 3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv10 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv11 = nn.Conv2d(64, NB_CHANNEL, 3, padding=1)

    def forward(self, xt, t, vec):
        # Encode t and vec
        t = F.relu(self.encodet(t / DIFFU_STEPS))
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
        # x2 = self.att1(x2)
        x2 = F.relu(self.conv4(x2))
        x3 = self.maxpool2(x2)
        x3 = F.relu(self.conv5(x3))
        # x3 = self.att2(x3)
        x5 = F.relu(self.conv6(x3))
        x6 = self.upconv1(x5)
        x6 = torch.cat((x6, x2), dim=1)
        x6 = F.relu(self.conv7(x6))
        # x6 = self.att3(x6)
        x6 = F.relu(self.conv8(x6))
        x7 = self.upconv2(x6)
        x7 = torch.cat((x7, x1), dim=1)
        x7 = F.relu(self.conv9(x7))
        x7 = F.relu(self.conv10(x7))
        x7 = self.conv11(x7)

        return x7

class FolderDataset(Dataset):
    def __init__(self, path, size=(32, 32)):
        super().__init__()
        self.path = path
        self.size = size
        self.files = os.listdir(self.path)

    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.path, self.files[index]))
        img = cv2.resize(img, self.size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1)) / 255
        return torch.tensor(img, dtype=torch.float32), 0

    def __len__(self):
        return len(self.files)


def q_xt_xt_1(xt_1, t):
    t_ind = t.to(dtype=torch.long) if isinstance(t, torch.Tensor) else t

    alpha = ALPHA[t_ind]
    mean = torch.sqrt(alpha) * xt_1
    std = torch.sqrt(1 - alpha)

    eps = torch.randn(xt_1.shape, device=DEVICE)
    xt = mean + std * eps

    return xt, eps

def q_xt_x0(x0, t):
    t_ind = t.to(dtype=torch.long) if isinstance(t, torch.Tensor) else t

    alpha_bar = ALPHA_BAR[t_ind]
    mean = torch.sqrt(alpha_bar) * x0
    std = torch.sqrt(1 - alpha_bar)

    eps = torch.randn(x0.shape, device=DEVICE)
    xt = mean + std * eps

    return xt, eps

def p_xt_1_xt(model, xt, t, vec):
    t_ind = t.to(dtype=torch.long) if isinstance(t, torch.Tensor) else t

    alpha_bar_t = ALPHA_BAR[t_ind].view(-1, 1, 1, 1)
    alpha_bar_t_1 = ALPHA_BAR[t_ind-1].view(-1, 1, 1, 1)
    alpha_t = ALPHA[t_ind].view(-1, 1, 1, 1)
    beta_t = BETA[t_ind].view(-1, 1, 1, 1)

    beta_tilde = (1 - alpha_bar_t_1) / (1 - alpha_bar_t) * beta_t

    epsilon_theta = model(xt, t, vec)

    sigma_theta = torch.sqrt(beta_tilde)
    mu_theta = (xt - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * epsilon_theta) / torch.sqrt(alpha_t)

    mask_t0 = (t > 1).to(dtype=torch.float32).view(-1, 1, 1, 1)
    noise = torch.randn(xt.shape, device=DEVICE) * mask_t0

    return mu_theta + sigma_theta * noise


class DiffusionDataset(Dataset):
    def __init__(self, dataset, autoencoder=None):
        super().__init__()
        self.dataset = dataset
        self.autoencoder = autoencoder

    def __getitem__(self, index):
        # Get the image and the label.
        img, label = self.dataset[index]
        img = img.to(DEVICE)

        # Encode the image.
        if self.autoencoder is not None:
            img = self.autoencoder.encode(img.unsqueeze(0)).squeeze(0)

        # Normalize the image.
        img = img * 2 - 1

        # Add noise to the image.
        t = torch.randint(1, DIFFU_STEPS, (1,), device=DEVICE)
        xt, eps = q_xt_x0(img, t)

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
    xs = [x]
    for t in range(1, DIFFU_STEPS+1):
        x = q_xt_xt_1(x, t)[0]
        xs.append(x)
    return xs


def tensor_to_image(tensor):
    img = tensor.clone().detach().cpu().numpy().transpose(1, 2, 0)
    img -= img.min()
    img /= img.max()
    return img


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
# dataset = FolderDataset('data/lfwcrop_color/faces')
dataset = FolderDataset('data/edface')

autoencoder = None
# autoencoder = Autoencoder(1, 1).to(DEVICE)
# autoencoder.load_state_dict(torch.load('autoencoder.pth'))
# autoencoder.eval()

img = dataset[0][0].to(DEVICE)
if autoencoder is not None:
    img = autoencoder.encode(img.unsqueeze(0)).squeeze(0)
NB_CHANNEL, IMG_SIZE, _ = img.shape
NB_LABEL = 1

EPOCHS = 200
LEARNING_RATE = 2e-4


def epoch_callback(epoch_i, epochs, model, trainer):
    if epoch_i % 10 == 0 or epoch_i == epochs - 1:
        print("Calculating metrics...")
        with torch.no_grad():
            batch_size = 64
            n_batches = 16
            n_samples = batch_size * n_batches

            fakes = np.zeros((0, NB_CHANNEL, IMG_SIZE, IMG_SIZE))
            for _ in range(n_batches):
                x = torch.randn(batch_size, NB_CHANNEL, IMG_SIZE, IMG_SIZE).to(DEVICE)
                vec = torch.randint(0, NB_LABEL, (batch_size,)).to(DEVICE)
                vec = torch.nn.functional.one_hot(vec, num_classes=NB_LABEL).to(device=DEVICE, dtype=torch.float32)

                for t in range(DIFFU_STEPS, 0, -1):
                    t_tensor = torch.tensor([[t]] * batch_size, device=DEVICE, dtype=torch.float32)
                    x = p_xt_1_xt(model, x, t_tensor, vec)

                fakes = np.concatenate((fakes, x.cpu().numpy()))

            reals = torch.stack([dataset[i][0] for i in range(n_samples)]).cpu().numpy()
            reals = reals * 2 - 1

            trainer.writer.add_scalars('Metrics/FID', {trainer.date_time: fid(reals, fakes)}, epoch_i)
            trainer.writer.add_scalars('Metrics/KL', {trainer.date_time: kl(reals, fakes)}, epoch_i)
            trainer.writer.add_scalars('Metrics/RKL', {trainer.date_time: kl(fakes, reals)}, epoch_i)
            trainer.writer.add_scalars('Metrics/JSD', {trainer.date_time: jsd(reals, fakes)}, epoch_i)


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
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Define the training dataset.
    train_dataset = DiffusionDataset(dataset, autoencoder)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                              num_workers=4, persistent_workers=True)
    trainer = Trainer()
    criterion = loss
    epochs = EPOCHS

    # Train the model.
    trainer.train(model, train_loader, epochs, optimizer, criterion, epoch_callbacks=[epoch_callback])

    # Save the model.
    torch.save(model.state_dict(), 'model.pth')

    ############## 
    # Evaluation #
    ##############

    with torch.no_grad():
        # Forward diffusion
        img, label = dataset[np.random.randint(0, len(dataset))]
        img = img.to(DEVICE)
        if autoencoder is not None:
            img = autoencoder.encode(img.unsqueeze(0)).squeeze(0)
        img = img * 2 - 1

        nb_plots = 10
        plots_id = [i for i in np.linspace(1, DIFFU_STEPS, nb_plots, dtype=int)]

        xs = forward_diffusion(img)

        plt.figure(figsize=(nb_plots, 2.5))
        for plot_i, t in enumerate(plots_id):
            x = xs[t]
            plt.subplot(2, nb_plots + 1, plot_i + 2)
            plt.title(f"t={t}")
            plt.imshow(tensor_to_image(x), interpolation='none')
            plt.axis("off")

        for plot_i, t in enumerate(plots_id):
            x = q_xt_x0(img, t)[0]
            plt.subplot(2, nb_plots + 1, nb_plots + plot_i + 3)
            plt.imshow(tensor_to_image(x), interpolation='none')
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
        t_plots = np.linspace(1, DIFFU_STEPS, nb_plots, dtype=int)

        n_classes = 10

        x = torch.randn(n_classes, NB_CHANNEL, IMG_SIZE, IMG_SIZE, device=DEVICE)
        vec = torch.tensor([[min(i, NB_LABEL-1)] for i in range(n_classes)], dtype=torch.int64)
        vec = torch.nn.functional.one_hot(vec, num_classes=NB_LABEL).to(device=DEVICE, dtype=torch.float32)

        plt.figure(figsize=(nb_plots, n_classes))
        plt.suptitle("Backward diffusion")
        for t in track(range(DIFFU_STEPS, 0, -1), description='Diffusing...'):
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

        for ti in track(range(DIFFU_STEPS, 0, -1), description='Benchmarking...'):
            t = torch.tensor([[ti]] * n_classes * nb_plots, device=DEVICE, dtype=torch.float32)
            x = p_xt_1_xt(model, x, t, vec)

        x = x * 0.5 + 0.5
        x = x.clamp(0, 1)

        plt.figure(figsize=(nb_plots, n_classes))
        for i in range(nb_plots):
            for j in range(n_classes):
                id = i * n_classes + j
                img = x[id]
                if autoencoder is not None:
                    img = train_dataset.autoencoder.decode(img.unsqueeze(0)).squeeze(0)
                plt.subplot(n_classes, nb_plots, id + 1)
                plt.imshow(tensor_to_image(img))
                plt.axis("off")
        plt.tight_layout()
        plt.savefig("benchmark.tmp.png")


        # Metrics
        batch_size = 64
        n_batches = 16
        n_samples = batch_size * n_batches

        fakes = np.zeros((0, NB_CHANNEL, IMG_SIZE, IMG_SIZE))
        for _ in track(range(n_batches), description=f'Sampling {n_samples} images...'):
            x = torch.randn(batch_size, NB_CHANNEL, IMG_SIZE, IMG_SIZE).to(DEVICE)
            vec = torch.randint(0, NB_LABEL, (batch_size,)).to(DEVICE)
            vec = torch.nn.functional.one_hot(vec, num_classes=NB_LABEL).to(device=DEVICE, dtype=torch.float32)

            for t in range(DIFFU_STEPS, 0, -1):
                t_tensor = torch.tensor([[t]] * batch_size, device=DEVICE, dtype=torch.float32)
                x = p_xt_1_xt(model, x, t_tensor, vec)

            fakes = np.concatenate((fakes, x.cpu().numpy()))

        reals = torch.stack([dataset[i][0] for i in range(n_samples)]).cpu().numpy()
        reals = reals * 2 - 1


        fid_score = fid(reals, fakes)
        print(f"FID score: {fid_score}")

        kl_score = kl(reals, fakes)
        print(f"KL divergence: {kl_score}")

        rkl_score = kl(fakes, reals)
        print(f"Reverse KL divergence: {rkl_score}")

        jsd_score = jsd(reals, fakes)
        print(f"Jensen-Shannon divergence: {jsd_score}")


        # plt.show()
