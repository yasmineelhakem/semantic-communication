
"""
PROJET 07 — Communication Sémantique Images et Vidéos
INSAT — 2026/2027

ÉTAPE 7 : ÉVALUATION STATISTIQUE DU JSCC SYNTAXIQUE

Objectifs :
    - Tester le modèle sur plusieurs images inédites.
    - Évaluer plusieurs valeurs de SNR.
    - Répéter les transmissions avec différents bruits AWGN.
    - Calculer PSNR et SSIM.
    - Calculer les moyennes et écarts-types.
    - Générer automatiquement les graphiques.

Architecture :
    Image RGB
       -> Encodeur CNN
       -> Symboles complexes normalisés
       -> Canal AWGN
       -> Décodeur CNN
       -> Image reconstruite

Modèle :
    results/jscc/best_model.pth

Dataset :
    data/test_images/

Données de sortie :
    results/jscc/evaluation_multi/details.csv
    results/jscc/evaluation_multi/summary.csv

Graphiques :
    results/jscc/graphs/graph2/psnr_multi.png
    results/jscc/graphs/graph2/ssim_multi.png

Exécution :
    python -m scripts.evaluate_jscc_multi

Remarque :
    Les barres d'erreur représentent l'écart-type
    entre les images, après moyenne sur les
    répétitions du bruit pour chaque image.
"""

import csv
import random
from collections import defaultdict
from pathlib import Path
from src.paths import project_path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
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


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_PATH = project_path("results/jscc/best_model.pth")
DATA_DIR = project_path("data/test_images")

OUTPUT_DIR = project_path(
    "results/jscc/evaluation_multi"
)

GRAPHS_DIR = project_path(
    "results/jscc/graphs/graph2"
)

IMAGE_SIZE = 128

SNR_VALUES = [0, 5, 10, 15, 20]

REPETITIONS = 5
SEED = 42

DEVICE = torch.device("cpu")


# ============================================================
# 2. CHARGEMENT DES IMAGES
# ============================================================

def load_image(path):
    """
    Charge une image RGB et la convertit
    en tenseur PyTorch [1, 3, 128, 128].
    """

    with Image.open(path) as img:

        img = img.convert("RGB")

        img = img.resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        array = (
            np.asarray(img, dtype=np.float32)
            / 255.0
        ).copy()

    tensor = torch.from_numpy(array).permute(
        2, 0, 1
    ).unsqueeze(0)

    return tensor


# ============================================================
# 3. CHARGEMENT DU MODÈLE
# ============================================================

def load_model():

    encoder = EncoderCNN().to(DEVICE)
    decoder = DecoderCNN().to(DEVICE)
    normalizer = PowerNormalization()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    encoder.load_state_dict(
        checkpoint["encoder"]
    )

    decoder.load_state_dict(
        checkpoint["decoder"]
    )

    encoder.eval()
    decoder.eval()

    print(
        f"Checkpoint chargé : epoch {checkpoint['epoch']}"
    )

    return encoder, normalizer, decoder


# ============================================================
# 4. CALCUL DES MÉTRIQUES
# ============================================================

def calculate_metrics(
    original,
    reconstructed
):
    """
    Calcule PSNR et SSIM sur les pixels
    normalisés dans l'intervalle [0, 1].
    """

    psnr = peak_signal_noise_ratio(
        original,
        reconstructed,
        data_range=1.0,
    )

    ssim = structural_similarity(
        original,
        reconstructed,
        channel_axis=-1,
        data_range=1.0,
    )

    return float(psnr), float(ssim)


# ============================================================
# 5. SAUVEGARDE CSV
# ============================================================

def save_csv(path, rows, columns):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=columns
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 6. GÉNÉRATION DES GRAPHIQUES
# ============================================================

def plot_metric(
    summary,
    metric,
    ylabel,
    filename
):

    snrs = [
        row["snr_db"]
        for row in summary
    ]

    means = [
        row[f"{metric}_mean"]
        for row in summary
    ]

    stds = [
        row[f"{metric}_std"]
        for row in summary
    ]

    plt.figure(figsize=(9, 5))

    plt.errorbar(
        snrs,
        means,
        yerr=stds,
        marker="o",
        capsize=4,
        linewidth=2,
        label="JSCC syntaxique"
    )

    plt.xlabel("SNR du canal (dB)")
    plt.ylabel(ylabel)

    plt.title(
        f"JSCC Syntaxique - {ylabel} moyen vs SNR"
    )

    if metric == "ssim":
        plt.ylim(-0.05, 1.05)

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = GRAPHS_DIR / filename

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Graphique sauvegardé : {output_path}"
    )


# ============================================================
# 7. ÉVALUATION SUR PLUSIEURS IMAGES
# ============================================================

@torch.no_grad()
def evaluate(
    encoder,
    normalizer,
    decoder,
    paths
):

    details = []

    for index, path in enumerate(
        paths,
        start=1
    ):

        image = load_image(path).to(DEVICE)

        # Encoder l'image une seule fois.
        latent = encoder(image)

        symbols = normalizer(latent)

        original = (
            image.squeeze(0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        for snr_db in SNR_VALUES:

            for repeat in range(REPETITIONS):

                # Nouveau bruit AWGN à chaque répétition.
                received = awgn(
                    symbols,
                    snr_db
                )

                reconstructed = decoder(
                    received
                )

                output = (
                    reconstructed.squeeze(0)
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                )

                output = np.clip(
                    output,
                    0.0,
                    1.0
                )

                psnr, ssim = calculate_metrics(
                    original,
                    output
                )

                details.append({
                    "image_id": index,
                    "image": str(path),
                    "snr_db": snr_db,
                    "repetition": repeat + 1,
                    "psnr_db": psnr,
                    "ssim": ssim,
                })

        print(
            f"Image {index}/{len(paths)} "
            f"évaluée : {path.name}"
        )

    return details


# ============================================================
# 8. CALCUL DES STATISTIQUES
# ============================================================

def summarize(details):
    """
    1. Moyenne des répétitions de bruit par image.
    2. Moyenne et écart-type entre les images.

    Les répétitions sur une même image ne sont
    pas traitées comme des images indépendantes.
    """

    by_image = defaultdict(list)

    for row in details:

        key = (
            row["snr_db"],
            row["image_id"]
        )

        by_image[key].append(row)

    by_snr = defaultdict(list)

    for (snr, image_id), rows in by_image.items():

        psnr_mean = np.mean([
            r["psnr_db"]
            for r in rows
        ])

        ssim_mean = np.mean([
            r["ssim"]
            for r in rows
        ])

        by_snr[snr].append((
            psnr_mean,
            ssim_mean
        ))

    summary = []

    for snr in SNR_VALUES:

        values = np.asarray(
            by_snr[snr],
            dtype=np.float64
        )

        psnrs = values[:, 0]
        ssims = values[:, 1]

        summary.append({
            "snr_db": snr,
            "n_images": len(values),
            "repetitions": REPETITIONS,

            "psnr_mean": float(
                np.mean(psnrs)
            ),

            "psnr_std": float(
                np.std(psnrs, ddof=1)
            ) if len(psnrs) > 1 else 0.0,

            "ssim_mean": float(
                np.mean(ssims)
            ),

            "ssim_std": float(
                np.std(ssims, ddof=1)
            ) if len(ssims) > 1 else 0.0,
        })

    return summary


# ============================================================
# 9. PROGRAMME PRINCIPAL
# ============================================================

def main():

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Création automatique des dossiers.
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    GRAPHS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}"
        )

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dossier introuvable : {DATA_DIR}"
        )

    paths = sorted(
        p for p in DATA_DIR.rglob("*")
        if p.suffix.lower() in {
            ".jpg", ".jpeg", ".png"
        }
    )

    if not paths:
        raise ValueError(
            "Aucune image dans data/test_images/"
        )

    print("=" * 60)
    print("PROJET 07 — ÉVALUATION MULTI-IMAGES JSCC")
    print("=" * 60)

    print(f"Images : {len(paths)}")
    print(f"SNR : {SNR_VALUES}")
    print(f"Répétitions : {REPETITIONS}")
    print(f"Device : {DEVICE}")

    encoder, normalizer, decoder = load_model()

    # 1. Évaluation.
    details = evaluate(
        encoder,
        normalizer,
        decoder,
        paths
    )

    # 2. Statistiques.
    summary = summarize(details)

    # 3. Mesures individuelles.
    save_csv(
        OUTPUT_DIR / "details.csv",
        details,
        [
            "image_id",
            "image",
            "snr_db",
            "repetition",
            "psnr_db",
            "ssim"
        ]
    )

    # 4. Résumé statistique.
    save_csv(
        OUTPUT_DIR / "summary.csv",
        summary,
        [
            "snr_db",
            "n_images",
            "repetitions",
            "psnr_mean",
            "psnr_std",
            "ssim_mean",
            "ssim_std"
        ]
    )

    # 5. Affichage.
    print("\nRÉSULTATS MOYENS")
    print("-" * 60)

    for row in summary:

        print(
            f"SNR={row['snr_db']:2d} dB | "
            f"PSNR={row['psnr_mean']:.2f} "
            f"+/- {row['psnr_std']:.2f} dB | "
            f"SSIM={row['ssim_mean']:.4f} "
            f"+/- {row['ssim_std']:.4f}"
        )

    # 6. Graphiques dans GRAPH2.
    plot_metric(
        summary,
        metric="psnr",
        ylabel="PSNR (dB)",
        filename="psnr_multi.png"
    )

    plot_metric(
        summary,
        metric="ssim",
        ylabel="SSIM",
        filename="ssim_multi.png"
    )

    print("\n" + "=" * 60)
    print("ÉVALUATION TERMINÉE")
    print("=" * 60)

    print(
        f"Données : {OUTPUT_DIR}"
    )

    print(
        f"Graphiques : {GRAPHS_DIR}"
    )


if __name__ == "__main__":
    main()
