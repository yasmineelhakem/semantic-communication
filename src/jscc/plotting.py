
"""
PROJET 07 — Communication Sémantique Images et Vidéos
INSAT — 2026/2027

GRAPHIQUES JSCC SYNTAXIQUE — EXPÉRIENCE 1

Objectifs :
    Visualiser l'apprentissage du réseau et les performances
    de reconstruction sur une seule image de test.

Graphiques :
    1. MSE entraînement et validation selon les epochs.
    2. PSNR en fonction du SNR.
    3. SSIM en fonction du SNR.
    4. Comparaison visuelle des images reconstruites.

Sources :
    results/jscc/history.csv
    results/jscc/evaluation/metrics.csv
    results/jscc/evaluation/*.png

Sorties :
    results/jscc/graphs/graph1/

Exécution :
    python -m scripts.plot_jscc

Remarque :
    Cette expérience repose sur une seule image de test.
    Les résultats ne constituent pas encore des moyennes
    statistiques sur plusieurs images.
"""

import csv
from pathlib import Path
from src.paths import project_path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# 1. CONFIGURATION
# ============================================================

HISTORY = project_path("results/jscc/history.csv")
METRICS = project_path("results/jscc/evaluation/metrics.csv")
EVALUATION = project_path("results/jscc/evaluation")

OUTPUT = project_path("results/jscc/graphs/graph1")

SNR_VALUES = [0, 5, 10, 15, 20]


# ============================================================
# 2. FONCTIONS UTILITAIRES
# ============================================================

def read_csv(path):
    """Charge les lignes d'un fichier CSV."""

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        return list(csv.DictReader(f))


def save_plot(filename):
    """Enregistre le graphique actuel en PNG."""

    path = OUTPUT / filename

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Graphique enregistré : {path}")


# ============================================================
# 3. GRAPHIQUE D'ENTRAÎNEMENT
# ============================================================

def plot_training():

    if not HISTORY.exists():
        print("Historique d'entraînement introuvable.")
        return

    data = read_csv(HISTORY)

    if not data:
        print("Historique vide.")
        return

    epochs = [int(row["epoch"]) for row in data]
    train = [float(row["train_mse"]) for row in data]
    val = [float(row["val_mse"]) for row in data]

    best_index = val.index(min(val))

    plt.figure(figsize=(9, 5))

    plt.plot(
        epochs,
        train,
        marker="o",
        linewidth=2,
        label="Training MSE"
    )

    plt.plot(
        epochs,
        val,
        marker="s",
        linewidth=2,
        label="Validation MSE"
    )

    plt.scatter(
        epochs[best_index],
        val[best_index],
        marker="*",
        s=180,
        zorder=5,
        label=f"Best epoch: {epochs[best_index]}"
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("JSCC Syntaxique - Training vs Validation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_plot("training_loss.png")


# ============================================================
# 4. PSNR EN FONCTION DU SNR
# ============================================================

def plot_psnr(data):

    snrs = [float(row["snr_db"]) for row in data]
    psnrs = [float(row["psnr_db"]) for row in data]

    plt.figure(figsize=(9, 5))

    plt.plot(
        snrs,
        psnrs,
        marker="o",
        linewidth=2,
        label="JSCC syntaxique"
    )

    plt.xlabel("SNR du canal (dB)")
    plt.ylabel("PSNR (dB)")

    plt.title(
        "JSCC Syntaxique - PSNR vs SNR (1 image)"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_plot("psnr_vs_snr.png")


# ============================================================
# 5. SSIM EN FONCTION DU SNR
# ============================================================

def plot_ssim(data):

    snrs = [float(row["snr_db"]) for row in data]
    ssims = [float(row["ssim"]) for row in data]

    plt.figure(figsize=(9, 5))

    plt.plot(
        snrs,
        ssims,
        marker="s",
        linewidth=2,
        label="JSCC syntaxique"
    )

    plt.xlabel("SNR du canal (dB)")
    plt.ylabel("SSIM")
    plt.ylim(0, 1)

    plt.title(
        "JSCC Syntaxique - SSIM vs SNR (1 image)"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_plot("ssim_vs_snr.png")


# ============================================================
# 6. COMPARAISON VISUELLE
# ============================================================

def plot_reconstructions():

    paths = [
        ("Original", EVALUATION / "reference.png")
    ]

    for snr in SNR_VALUES:
        paths.append((
            f"SNR = {snr} dB",
            EVALUATION / f"reconstruction_{snr}dB.png"
        ))

    paths = [
        (label, path)
        for label, path in paths
        if path.exists()
    ]

    if not paths:
        print("Aucune image à comparer.")
        return

    fig, axes = plt.subplots(
        1,
        len(paths),
        figsize=(3.5 * len(paths), 4),
        squeeze=False
    )

    for ax, (label, path) in zip(axes[0], paths):

        with Image.open(path) as img:
            image = img.convert("RGB").copy()

        ax.imshow(image)
        ax.set_title(label)
        ax.axis("off")

    fig.suptitle(
        "JSCC Syntaxique - Reconstructions selon SNR"
    )

    fig.tight_layout()

    save_plot("reconstruction_comparison.png")


# ============================================================
# 7. PROGRAMME PRINCIPAL
# ============================================================

def main():

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("PROJET 07 — GRAPHIQUES JSCC / GRAPH1")
    print("=" * 60)

    plot_training()

    if METRICS.exists():

        data = sorted(
            read_csv(METRICS),
            key=lambda row: float(row["snr_db"])
        )

        if data:
            plot_psnr(data)
            plot_ssim(data)
        else:
            print("Fichier metrics.csv vide.")

    else:
        print("Fichier metrics.csv introuvable.")

    plot_reconstructions()

    print("\nGraphiques GRAPH1 générés !")


if __name__ == "__main__":
    main()
