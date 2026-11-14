import numpy as np
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

def calculate_psnr(original, reconstructed):
    """
    Calculate PSNR for a batch of RGB images.
    Inputs have shape [B, C, H, W] and values in [0, 1].
    """

    original = original.detach().cpu().numpy()
    reconstructed = reconstructed.detach().cpu().numpy()

    scores = []

    for i in range(original.shape[0]):

        img_original = np.transpose(original[i], (1, 2, 0))
        img_reconstructed = np.transpose(
            reconstructed[i], (1, 2, 0)
        )

        score = peak_signal_noise_ratio(
            img_original,
            img_reconstructed,
            data_range=1.0
        )

        scores.append(score)

    return float(np.mean(scores))


def calculate_ssim(original, reconstructed):
    """
    Calculate SSIM for a batch of RGB images.
    """

    original = original.detach().cpu().numpy()
    reconstructed = reconstructed.detach().cpu().numpy()

    scores = []

    for i in range(original.shape[0]):

        img_original = np.transpose(original[i], (1, 2, 0))
        img_reconstructed = np.transpose(
            reconstructed[i], (1, 2, 0)
        )

        score = structural_similarity(
            img_original,
            img_reconstructed,
            channel_axis=2,
            data_range=1.0
        )

        scores.append(score)

    return float(np.mean(scores))
