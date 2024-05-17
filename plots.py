import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from torchvision import datasets


def q_xt_x0(x0, t):
    alpha_bar = ALPHA_BAR[t]
    mean = np.sqrt(alpha_bar) * x0
    std = np.sqrt(1 - alpha_bar)

    eps = np.random.normal(0, 1, x0.shape)
    xt = mean + std * eps

    return xt, eps

def q_xt_xt_1(xt_1, t):
    alpha = ALPHA[t]
    mean = np.sqrt(alpha) * xt_1
    std = np.sqrt(1 - alpha)

    eps = np.random.normal(0, 1, xt_1.shape)
    xt = mean + std * eps

    return xt, eps


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
img = mnist.data[np.random.randint(0, len(mnist))].numpy() / 255

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
gs = GridSpec(1, 3, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1:])

xt = img
for t in range(DIFFU_STEPS):
    xt, eps = q_xt_xt_1(xt, t)

    ax1.clear()
    ax1.imshow(xt, cmap='gray')
    ax1.set_title(f'xt at t={t}')
    ax1.axis('off')

    ax2.clear()
    ax2.hist(xt.flatten(), bins=NB_BINS, range=(BIN_MIN, BIN_MAX))
    ax2.plot(x_norm, y_norm, color='red', label='N(0, 1)')
    ax2.set_yscale('log')
    ax2.set_ylim(y_norm.min(), 1e3)
    ax2.set_title(f'xt histogram')

    fig.savefig(f'plots/diffusion/{t:04d}.tmp.png')
