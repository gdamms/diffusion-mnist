import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from torchvision import datasets
import torch
import os
from rich.progress import track

from main import UNet, q_xt_xt_1, p_xt_1_xt


os.makedirs('plots', exist_ok=True)
os.makedirs('plots/diffusion', exist_ok=True)
os.makedirs('plots/diffusion_inverse', exist_ok=True)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DIFFU_STEPS = 1000
BETA = np.linspace(1e-4, 2e-2, DIFFU_STEPS)
ALPHA = 1 - BETA
ALPHA_BAR = np.cumprod(ALPHA)

NB_BINS = 50
BIN_MIN = -4
BIN_MAX = 4


plt.figure()
plt.plot(BETA, label='beta')
plt.plot(ALPHA, label='alpha')
plt.plot(ALPHA_BAR, label='alpha_bar')
plt.legend()
plt.title('Alpha, Beta and Alpha_bar schedules')
plt.savefig('plots/alpha_beta.tmp.png')


mnist = datasets.MNIST('data', train=True, download=True)
img, label = mnist[np.random.randint(0, len(mnist))]
img = np.array(img) / 255 * 2 - 1

plt.figure()
plt.imshow(img, cmap='gray')
plt.title('Image')
plt.axis('off')
plt.savefig('plots/img.tmp.png')


plt.figure()
plt.hist(img.flatten(), bins=NB_BINS, range=(BIN_MIN, BIN_MAX))
plt.yscale('log')
plt.title('Image histogram')
plt.savefig('plots/img_hist.tmp.png')


def norm_dist(x, mean, std):
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))

x_norm = np.linspace(BIN_MIN, BIN_MAX, 100)
y_norm = norm_dist(x_norm, 0, 1) * 28**2 / NB_BINS * (BIN_MAX - BIN_MIN)

fig = plt.figure(figsize=(10, 5))
fig.suptitle('Diffusion naturelle')
gs = GridSpec(1, 3, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1:])

plots_to_save = np.linspace(1, DIFFU_STEPS, 100).astype(int)

xt = torch.tensor(img, device=DEVICE, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
for t in track(range(1, DIFFU_STEPS+1)):
    xt, eps = q_xt_xt_1(xt, t)

    if t not in plots_to_save:
        continue

    xt_numpy = xt.cpu().detach().numpy()[0, 0]

    ax1.clear()
    ax1.imshow(xt_numpy, cmap='gray')
    ax1.set_title(f'xt at t={t:04d}')
    ax1.axis('off')

    ax2.clear()
    ax2.hist(xt_numpy.flatten(), bins=NB_BINS, range=(BIN_MIN, BIN_MAX))
    ax2.plot(x_norm, y_norm, color='red', label='N(0, 1)')
    ax2.legend()
    ax2.set_yscale('log')
    ax2.set_ylim(y_norm.min(), 1e3)
    ax2.set_title(f'xt histogram')

    fig.savefig(f'plots/diffusion/{t:04d}.tmp.png')
os.system('convert -delay 20 -loop 0 plots/diffusion/*.png plots/diffusion.tmp.gif')



model = UNet().to(DEVICE)
model.load_state_dict(torch.load('model.pth'))

fig = plt.figure(figsize=(10, 5))
fig.suptitle('Diffusion inverse')
gs = GridSpec(1, 3, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1:])

xt = torch.randn(1, 1, 28, 28, device=DEVICE)
vec = torch.zeros(1, 10).to(DEVICE)
vec[0, label] = 1
for t in track(range(DIFFU_STEPS, 0, -1)):
    t_tensor = torch.tensor([[t]], device=DEVICE, dtype=torch.float32)
    xt = p_xt_1_xt(model, xt, t_tensor, vec)

    if t not in plots_to_save:
        continue

    xt_numpy = xt.cpu().detach().numpy()[0, 0]

    ax1.clear()
    ax1.imshow(xt_numpy, cmap='gray')
    ax1.set_title(f'xt at t={t:04d}')
    ax1.axis('off')

    ax2.clear()
    ax2.hist(xt_numpy.flatten(), bins=NB_BINS, range=(BIN_MIN, BIN_MAX))
    ax2.plot(x_norm, y_norm, color='red', label='N(0, 1)')
    ax2.legend()
    ax2.set_yscale('log')
    ax2.set_ylim(y_norm.min(), 1e3)
    ax2.set_title(f'xt histogram')

    fig.savefig(f'plots/diffusion_inverse/{t:04d}.tmp.png')
os.system('convert -delay 20 -loop 0 -reverse plots/diffusion_inverse/*.png plots/diffusion_inverse.tmp.gif')
