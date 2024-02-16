from typing import Tuple, Optional

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from rich.progress import track

from trainer import Trainer


def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Input
        # The input to the model is a 11 vector which represents the desired label with the contextual information.
        # The Input is passed through layers to generate two feature maps of size 7x7.
        # -------
        # input: 1 (diffu step) and 10 (label)
        self.inconv1 = nn.Linear(1, 7 * 7)
        self.inconv2 = nn.Linear(10, 7 * 7)

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
        self.e31 = nn.Conv2d(130, 256, kernel_size=3,
                             padding=1)  # output: 7x7x256
        self.e32 = nn.Conv2d(258, 256, kernel_size=3,
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

    def forward(self, x, t, y):
        # Input (diffusion step)
        t = t.unsqueeze_(-1)
        t = t.to(torch.float32)
        t = self.inconv1(t)
        t = t.view(-1, 1, 7, 7)

        # Input (label)
        y = self.inconv2(y)
        y = y.view(-1, 1, 7, 7)

        # Encoder
        x = F.relu(self.e11(x))
        x1 = F.relu(self.e12(x))
        x = self.pool1(x1)

        x = F.relu(self.e21(x))
        x2 = F.relu(self.e22(x))
        x = self.pool2(x2)

        x = torch.cat([x, t, y], dim=1)
        x = F.relu(self.e31(x))
        x = torch.cat([x, t, y], dim=1)
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


class MNISTDiffusionDataset(Dataset):
    def __init__(self, train=True):
        super().__init__()
        self.mnist_data = datasets.MNIST(
            root='./data',
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )

    def __len__(self):
        return len(self.mnist_data)

    def __getitem__(self, index):
        # Get the image and the label.
        img, label = self.mnist_data[index]
        prompt = torch.nn.functional.one_hot(
            torch.tensor(label), 10).to(torch.float32)
        return img.to(device), prompt.to(device)


class DenoiseDiffusion:
    """
    ## Denoise Diffusion
    """

    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        """
        * `eps_model` is $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$ model
        * `n_steps` is $t$
        * `device` is the device to place constants on
        """
        super().__init__()
        self.eps_model = eps_model

        # Create $\beta_1, \dots, \beta_T$ linearly increasing variance schedule
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)

        # $\alpha_t = 1 - \beta_t$
        self.alpha = 1. - self.beta
        # $\bar\alpha_t = \prod_{s=1}^t \alpha_s$
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        # $T$
        self.n_steps = n_steps
        # $\sigma^2 = \beta$
        self.sigma2 = self.beta

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        #### Get $q(x_t|x_0)$ distribution

        \begin{align}
        q(x_t|x_0) &= \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)
        \end{align}
        """

        # [gather](utils.html) $\alpha_t$ and compute $\sqrt{\bar\alpha_t} x_0$
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        # $(1-\bar\alpha_t) \mathbf{I}$
        var = 1 - gather(self.alpha_bar, t)
        #
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        """
        #### Sample from $q(x_t|x_0)$

        \begin{align}
        q(x_t|x_0) &= \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)
        \end{align}
        """

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if eps is None:
            eps = torch.randn_like(x0)

        # get $q(x_t|x_0)$
        mean, var = self.q_xt_x0(x0, t)
        # Sample from $q(x_t|x_0)$
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, prompt: Optional[torch.Tensor] = None):
        """
        #### Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$

        \begin{align}
        \textcolor{lightgreen}{p_\theta}(x_{t-1} | x_t) &= \mathcal{N}\big(x_{t-1};
        \textcolor{lightgreen}{\mu_\theta}(x_t, t), \sigma_t^2 \mathbf{I} \big) \\
        \textcolor{lightgreen}{\mu_\theta}(x_t, t)
          &= \frac{1}{\sqrt{\alpha_t}} \Big(x_t -
            \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)
        \end{align}
        """

        # $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$
        eps_theta = self.eps_model(xt, t, prompt)
        # [gather](utils.html) $\bar\alpha_t$
        alpha_bar = gather(self.alpha_bar, t)
        # $\alpha_t$
        alpha = gather(self.alpha, t)
        # $\frac{\beta}{\sqrt{1-\bar\alpha_t}}$
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        # $$\frac{1}{\sqrt{\alpha_t}} \Big(x_t -
        #      \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        # $\sigma^2$
        var = gather(self.sigma2, t)

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        eps = torch.randn(xt.shape, device=xt.device)
        # Sample
        return mean + (var ** .5) * eps

    def loss(self, x0: torch.Tensor, prompt: Optional[torch.Tensor] = None, noise: Optional[torch.Tensor] = None):
        """
        #### Simplified Loss

        $$L_{\text{simple}}(\theta) = \mathbb{E}_{t,x_0, \epsilon} \Bigg[ \bigg\Vert
        \epsilon - \textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)
        \bigg\Vert^2 \Bigg]$$
        """
        # Get batch size
        batch_size = x0.shape[0]
        # Get random $t$ for each sample in the batch
        t = torch.randint(0, self.n_steps, (batch_size,),
                          device=x0.device, dtype=torch.long)

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if noise is None:
            noise = torch.randn_like(x0)

        # Sample $x_t$ for $q(x_t|x_0)$
        xt = self.q_sample(x0, t, eps=noise)
        # Get $\textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)$
        if prompt is None:
            eps_theta = self.eps_model(xt, t)
        else:
            eps_theta = self.eps_model(xt, t, prompt)

        # MSE loss
        return F.mse_loss(noise, eps_theta)


diffu_steps = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
model = torch.load('model.pt').to(device)
ddpm = DenoiseDiffusion(model, diffu_steps, device)

ds = MNISTDiffusionDataset()
dl = DataLoader(ds, batch_size=128, shuffle=True)
opti = torch.optim.Adam(model.parameters(), lr=1e-5)


for epoch in range(3):
    for x0, prompt in track(dl):
        loss = ddpm.loss(x0, prompt=prompt)
        opti.zero_grad()
        loss.backward()
        opti.step()

    print(f'Epoch {epoch}: {loss.item()}')

torch.save(model, 'model.pt')

n = 1
fig = plt.figure(figsize=(10, n))
x = torch.randn(10 * n, 1, 28, 28, device=device)
prompt = torch.nn.functional.one_hot(
    torch.tensor([range(10)] * n), num_classes=10).to(device, torch.float32)
prompt = prompt.reshape(-1, 10)
for i in track(range(diffu_steps)):
    for j in range(10 * n):
        t = diffu_steps - i - 1
        t = torch.tensor(t, device=device)
        x[j] = ddpm.p_sample(x[j:j+1], t, prompt[j])

for i in range(10 * n):
    plt.subplot(n, 10, i + 1)
    plt.axis('off')
    plt.imshow(x[i, 0].cpu().detach().numpy())
plt.tight_layout()
plt.savefig('test.png')
