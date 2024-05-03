import numpy as np
import matplotlib.pyplot as plt

A = np.random.uniform(0.1, 0.9)
B = np.random.uniform(0.1, 0.9)

def q_xt_xt_1_simple(x, t):
    mean = A * x
    std = B
    return np.random.normal(mean, std)

def q_xt_x0_simple_damien(x, t):
    mean = A ** t * x
    std = np.sqrt(sum([A ** (2*i) for i in range(t)])) * B
    return np.random.normal(mean, std)


def q_xt_xt_1(x, t):
    alpha = ALPHA[t]
    mean = np.sqrt(alpha) * x
    std = 1 - alpha
    return np.random.normal(mean, std)

def q_xt_x0_paper(x, t):
    alpha_bar = ALPHA_BAR[t]
    mean = np.sqrt(alpha_bar) * x
    std = 1 - alpha_bar
    return np.random.normal(mean, std)

def q_xt_x0_damien(x, t):
    alpha_bar = ALPHA_BAR[t]
    cum_sq_sum = sum([np.prod(ALPHA[s+2:t+1]) * (1 - ALPHA[s+1])**2 for s in range(t)])
    mean = np.sqrt(alpha_bar) * x
    std = np.sqrt(cum_sq_sum)
    return np.random.normal(mean, std)

T = 100
BETA = np.concatenate(([0], np.linspace(1e-4, 2e-2, T)))
ALPHA = 1 - BETA
ALPHA_BAR = np.cumprod(ALPHA)

N = int(1e6)
x0 = 1

xs_implicit = np.array([x0] * N)
for t in range(1, T+1):
    xs_implicit = q_xt_xt_1(xs_implicit, t)

xs_explicit_paper = q_xt_x0_paper(np.array([x0] * N), T)
xs_explicit_damien = q_xt_x0_damien(np.array([x0] * N), T)

plt.figure()
bins = np.linspace(min(
                xs_implicit.min(),
                xs_explicit_paper.min(),
                xs_explicit_damien.min(),
            ), max(
                xs_implicit.max(),
                xs_explicit_paper.max(),
                xs_explicit_damien.max(),
            ), 100)
plt.hist(xs_implicit, bins=bins, alpha=0.5, label="q_xt_xt_1")
plt.hist(xs_explicit_paper, bins=bins, alpha=0.5, label="q_xt_x0_paper")
plt.hist(xs_explicit_damien, bins=bins, alpha=0.5, label="q_xt_x0_damien")
plt.legend()
plt.savefig("q_xt_xt_1_vs_q_xt_x0.tmp.png")
plt.show()

xs_implicit = np.array([x0] * N)
for t in range(1, T+1):
    xs_implicit = q_xt_xt_1_simple(xs_implicit, t)

xs_explicit_damien = q_xt_x0_simple_damien(np.array([x0] * N), T)

plt.figure()
bins = np.linspace(min(xs_implicit.min(), xs_explicit_damien.min()), max(xs_implicit.max(), xs_explicit_damien.max()), 100)
plt.hist(xs_implicit, bins=bins, alpha=0.5, label="q_xt_xt_1_simple")
plt.hist(xs_explicit_damien, bins=bins, alpha=0.5, label="q_xt_x0_simple_damien")
plt.legend()
plt.savefig("q_xt_xt_1_simple_vs_q_xt_x0_simple.tmp.png")
plt.show()

A1, B1, A2, B2, A3, B3 = np.random.uniform(0, 1, 6)
x1 = np.random.normal(A1, B1, N)
x2 = np.random.normal(A2 * x1, B2, N)
x3 = np.random.normal(A3 * x2, B3, N)
x3_ = np.random.normal(A1 * A2 * A3, np.sqrt(A3**2 * A2**2 * B1**2 + A3**2 * B2**2 + B3**2), N)

plt.figure()
bins = np.linspace(min(x3.min(), x3_.min()), max(x3.max(), x3_.max()), 100)
plt.hist(x3, bins=bins, alpha=0.5, label="normal")
plt.hist(x3_, bins=bins, alpha=0.5, label="product")
plt.legend()
plt.savefig("product_normal.tmp.png")
plt.show()
