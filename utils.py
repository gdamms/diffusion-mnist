import numpy as np
import scipy.linalg
import cv2


def fid(reals, fakes):
    """FID score calculation.

    Args:
        reals (numpy.array): Real images.
        fakes (numpy.array): Fake images.
    """
    reals = reals.reshape(reals.shape[0], -1)
    fakes = fakes.reshape(fakes.shape[0], -1)

    mu_real = np.mean(reals, axis=0)
    mu_fake = np.mean(fakes, axis=0)
    sigma_real = np.cov(reals, rowvar=False)
    sigma_fake = np.cov(fakes, rowvar=False)

    diff = mu_real - mu_fake
    covmean = np.dot(sigma_real, sigma_fake.T)
    covmean, _ = scipy.linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)

    if not np.isfinite(covmean).all():
        eps=1e-6
        offset = np.eye(sigma_real.shape[0]) * eps
        ncovmean = scipy.linalg.sqrtm((sigma_real + offset).dot(sigma_fake + offset))
        covmean = ncovmean

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return diff @ diff + np.trace(sigma_real) + np.trace(sigma_fake) - 2 * np.trace(covmean)


def kl(reals, fakes):
    """KL divergence calculation.

    Args:
        reals (numpy.array): Real images.
        fakes (numpy.array): Fake images.
    """
    reals = reals.transpose(1, 0, 2, 3).reshape(reals.shape[1], -1)
    fakes = fakes.transpose(1, 0, 2, 3).reshape(fakes.shape[1], -1)

    hist_real = np.apply_along_axis(lambda a: np.histogram(a, bins=40, range=(-1, 1))[0], 1, reals)
    hist_fake = np.apply_along_axis(lambda a: np.histogram(a, bins=40, range=(-1, 1))[0], 1, fakes)

    hist_real = hist_real + 1
    hist_fake = hist_fake + 1

    hist_real = hist_real / np.sum(hist_real)
    hist_fake = hist_fake / np.sum(hist_fake)

    return np.mean(np.log(hist_real / hist_fake))


def jsd(reals, fakes):
    """Jensen-Shannon divergence calculation.

    Args:
        reals (numpy.array): Real images.
        fakes (numpy.array): Fake images.
    """
    reals = reals.transpose(1, 0, 2, 3).reshape(reals.shape[1], -1)
    fakes = fakes.transpose(1, 0, 2, 3).reshape(fakes.shape[1], -1)

    hist_real = np.apply_along_axis(lambda a: np.histogram(a, bins=40, range=(-1, 1))[0], 1, reals)
    hist_fake = np.apply_along_axis(lambda a: np.histogram(a, bins=40, range=(-1, 1))[0], 1, fakes)

    hist_real = hist_real + 1
    hist_fake = hist_fake + 1

    hist_real = hist_real / np.sum(hist_real)
    hist_fake = hist_fake / np.sum(hist_fake)

    hist_avg = (hist_real + hist_fake) / 2

    return 0.5 * (np.mean(np.log(hist_real / hist_avg)) + np.mean(np.log(hist_fake / hist_avg)))


def haar(image):
    # Load the Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the result
    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Load the image
    image = cv2.imread('data/edface/500/03120500_000.png')

    # Perform face detection
    haar(image)