import numpy as np


def fid(reals, fakes):
    """FID score calculation.

    Args:
        reals (numpy.array): Real images.
        fakes (numpy.array): Fake images.
    """
    print(reals.shape, fakes.shape)
    reals = reals.reshape(reals.shape[0], -1)
    fakes = fakes.reshape(fakes.shape[0], -1)

    mu_real = np.mean(reals, axis=0)
    mu_fake = np.mean(fakes, axis=0)
    sigma_real = np.cov(reals, rowvar=False)
    sigma_fake = np.cov(fakes, rowvar=False)

    diff = mu_real - mu_fake
    covmean = np.dot(sigma_real, sigma_fake.T)
    covmean = np.sqrt(covmean * (covmean > 0))
    print(np.trace(covmean))

    return diff @ diff + np.trace(sigma_real) + np.trace(sigma_fake) - 2 * np.trace(covmean)
