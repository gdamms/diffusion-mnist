import numpy as np
import scipy


def fid(reals: np.ndarray, fakes: np.ndarray) -> float:
    """
    Calculate Frechet Inception Distance (FID) score.

    Args:
        reals: Real images [N, C, H, W]
        fakes: Generated images [N, C, H, W]

    Returns:
        FID score (lower is better)
    """
    reals = reals.reshape(reals.shape[0], -1)
    fakes = fakes.reshape(fakes.shape[0], -1)

    mu_real = np.mean(reals, axis=0)
    mu_fake = np.mean(fakes, axis=0)
    sigma_real = np.cov(reals, rowvar=False)
    sigma_fake = np.cov(fakes, rowvar=False)

    diff = mu_real - mu_fake
    covmean, _ = scipy.linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)

    if not np.isfinite(covmean).all():
        eps = 1e-6
        offset = np.eye(sigma_real.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma_real + offset).dot(sigma_fake + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return diff @ diff + np.trace(sigma_real) + np.trace(sigma_fake) - 2 * np.trace(covmean)


def kl_divergence(reals: np.ndarray, fakes: np.ndarray) -> float:
    """
    Calculate KL divergence between real and fake image distributions.

    Args:
        reals: Real images [N, C, H, W]
        fakes: Generated images [N, C, H, W]

    Returns:
        KL divergence value
    """
    reals = reals.transpose(1, 0, 2, 3).reshape(reals.shape[1], -1)
    fakes = fakes.transpose(1, 0, 2, 3).reshape(fakes.shape[1], -1)

    hist_real = np.apply_along_axis(
        lambda a: np.histogram(a, bins=40, range=(-1, 1))[0], 1, reals
    )
    hist_fake = np.apply_along_axis(
        lambda a: np.histogram(a, bins=40, range=(-1, 1))[0], 1, fakes
    )

    # Add smoothing
    hist_real = hist_real + 1
    hist_fake = hist_fake + 1

    hist_real = hist_real / np.sum(hist_real)
    hist_fake = hist_fake / np.sum(hist_fake)

    return np.mean(np.log(hist_real / hist_fake))


def jsd(reals: np.ndarray, fakes: np.ndarray) -> float:
    """
    Calculate Jensen-Shannon divergence between real and fake image distributions.

    Args:
        reals: Real images [N, C, H, W]
        fakes: Generated images [N, C, H, W]

    Returns:
        JSD value
    """
    reals = reals.transpose(1, 0, 2, 3).reshape(reals.shape[1], -1)
    fakes = fakes.transpose(1, 0, 2, 3).reshape(fakes.shape[1], -1)

    hist_real = np.apply_along_axis(
        lambda a: np.histogram(a, bins=40, range=(-1, 1))[0], 1, reals
    )
    hist_fake = np.apply_along_axis(
        lambda a: np.histogram(a, bins=40, range=(-1, 1))[0], 1, fakes
    )

    hist_real = hist_real + 1
    hist_fake = hist_fake + 1

    hist_real = hist_real / np.sum(hist_real)
    hist_fake = hist_fake / np.sum(hist_fake)

    hist_avg = (hist_real + hist_fake) / 2

    return 0.5 * (np.mean(np.log(hist_real / hist_avg)) + np.mean(np.log(hist_fake / hist_avg)))
