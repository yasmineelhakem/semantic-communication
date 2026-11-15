
import numpy as np
from PIL import Image
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)


def calculer_metriques(image_originale, image_reconstruite):
    """Calcule PSNR et SSIM entre deux images RGB."""

    with Image.open(image_originale) as img:
        original = np.asarray(img.convert("RGB"))

    with Image.open(image_reconstruite) as img:
        reconstruite = np.asarray(img.convert("RGB"))

    if original.shape != reconstruite.shape:
        raise ValueError("Les images doivent avoir la même taille.")

    psnr = peak_signal_noise_ratio(
        original, reconstruite, data_range=255
    )

    ssim = structural_similarity(
        original,
        reconstruite,
        channel_axis=-1,
        data_range=255,
    )

    return float(psnr), float(ssim)
