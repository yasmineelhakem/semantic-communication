
"""
PROJET 07 — JSCC SYNTAXIQUE
ÉTAPE 6 : ÉVALUATION DU MODÈLE ENTRAÎNÉ

Objectif :
    Charger les poids du meilleur modèle JSCC.
    Transmettre une image de test à différents SNR.
    Sauvegarder les images reconstruites.
    Calculer PSNR et SSIM.

Entrée :
    data/test_images/test.jpg

Modèle :
    results/jscc/best_model.pth

Sorties :
    results/jscc/evaluation/reference.png
    results/jscc/evaluation/reconstruction_XdB.png
    results/jscc/evaluation/metrics.csv

Exécution :
    python -m scripts.evaluate_jscc
"""

import csv
import math
from pathlib import Path
from src.paths import project_path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)

from src.jscc.encoder import EncoderCNN
from src.jscc.decoder import DecoderCNN
from src.jscc.normalization import PowerNormalization
from src.channels.awgn import awgn


MODEL_PATH = project_path("results/jscc/best_model.pth")
IMAGE_PATH = project_path("data/test_images/test.png")
OUTPUT_DIR = project_path("results/jscc/evaluation")

SNR_VALUES = [0, 5, 10, 15, 20]
IMAGE_SIZE = 128
DEVICE = torch.device("cpu")


def charger_image(path):

    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

        array = (
            np.asarray(img, dtype=np.float32)
            / 255.0
        ).copy()

    tensor = torch.from_numpy(array).permute(
        2, 0, 1
    ).unsqueeze(0)

    return tensor, array


def sauver_image(tensor, path):

    array = tensor.squeeze(0).permute(
        1, 2, 0
    ).cpu().numpy()

    array = np.clip(array, 0, 1)

    image = Image.fromarray(
        (array * 255).round().astype(np.uint8)
    )

    image.save(path)

    return np.asarray(image)


def main():

    torch.manual_seed(42)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}"
        )

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image introuvable : {IMAGE_PATH}"
        )

    # Charger les modèles.
    encoder = EncoderCNN().to(DEVICE)
    decoder = DecoderCNN().to(DEVICE)
    normalizer = PowerNormalization()

    # Charger les poids entraînés.
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    encoder.eval()
    decoder.eval()

    print(
        f"Modèle chargé : epoch {checkpoint['epoch']}"
    )

    # Charger l'image originale.
    image, original = charger_image(IMAGE_PATH)
    image = image.to(DEVICE)

    Image.fromarray(
        (original * 255).round().astype(np.uint8)
    ).save(OUTPUT_DIR / "reference.png")

    resultats = []

    with torch.no_grad():

        # Encoder une seule fois.
        latent = encoder(image)
        symbols = normalizer(latent)

        for snr_db in SNR_VALUES:

            # Transmission AWGN.
            received = awgn(symbols, snr_db)

            # Décodage.
            reconstructed = decoder(received)

            # Sauvegarder l'image.
            output_path = (
                OUTPUT_DIR / f"reconstruction_{snr_db}dB.png"
            )

            reconstructed_array = sauver_image(
                reconstructed,
                output_path,
            )

            original_uint8 = (
                (original * 255).round().astype(np.uint8)
            )

            # Métriques calculées sur les images PNG.
            psnr = peak_signal_noise_ratio(
                original_uint8,
                reconstructed_array,
                data_range=255,
            )

            ssim = structural_similarity(
                original_uint8,
                reconstructed_array,
                channel_axis=-1,
                data_range=255,
            )

            print(
                f"SNR={snr_db:2d} dB | "
                f"PSNR={psnr:.2f} dB | "
                f"SSIM={ssim:.4f}"
            )

            resultats.append({
                "snr_db": snr_db,
                "psnr_db": float(psnr),
                "ssim": float(ssim),
            })

    # Export CSV.
    csv_path = OUTPUT_DIR / "metrics.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "snr_db",
                "psnr_db",
                "ssim",
            ],
        )

        writer.writeheader()
        writer.writerows(resultats)

    print("\nÉvaluation terminée.")
    print(f"Résultats : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
