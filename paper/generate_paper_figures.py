"""Generate French publication plots from archived CSVs; no experiments run."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent / "results"
OUTPUT = ROOT / "figures"
OUTPUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def finish(fig, name):
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def curves():
    data = rows(RESULTS / "classical/fec/metriques.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    for method, label, color, marker in [
        ("sans_fec", "Sans correction", "#27628f", "o"),
        ("avec_fec", "Hamming (7,4)", "#be5727", "s"),
    ]:
        values = [row for row in data if row["methode"] == method]
        positive = [row for row in values if float(row["ber"]) > 0]
        ax.semilogy([float(r["snr_db"]) for r in positive],
                    [float(r["ber"]) for r in positive], marker=marker,
                    color=color, label=label)
        zero = [r for r in values if float(r["ber"]) == 0]
        if zero:
            ax.scatter([float(r["snr_db"]) for r in zero], [1e-5] * len(zero),
                       marker="v", facecolors="none", edgecolors=color, s=45,
                       label="Aucune erreur observée (repère)" if method == "avec_fec" else None)
    ax.set(xlabel="SNR du canal (dB)", ylabel="Taux d'erreur binaire (BER)",
           xticks=[5, 8, 10, 12], ylim=(7e-6, .1))
    ax.grid(alpha=.22, which="both")
    ax.legend(fontsize=8, loc="upper right")
    finish(fig, "classique_ber")

    single = rows(RESULTS / "jscc/evaluation/metrics.csv")
    repeated = rows(RESULTS / "jscc/evaluation_multi/summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for ax, key, mean_key, ylabel in [
        (axes[0], "psnr_db", "psnr_mean", "PSNR (dB)"),
        (axes[1], "ssim", "ssim_mean", "SSIM"),
    ]:
        ax.plot([float(r["snr_db"]) for r in single],
                [float(r[key]) for r in single], "o-", color="#27628f",
                label="Une transmission (pixels 8 bits)")
        ax.plot([float(r["snr_db"]) for r in repeated],
                [float(r[mean_key]) for r in repeated], "s--", color="#be5727",
                label="Moyenne de 5 bruits (pixels réels)")
        ax.set(xlabel="SNR du canal (dB)", ylabel=ylabel, xticks=[0,5,10,15,20])
        ax.grid(alpha=.22)
    axes[0].legend(fontsize=6.7)
    finish(fig, "jscc_qualite")

    semantic = rows(RESULTS / "semantic_evaluation.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.45))
    for ax, key, ylabel in zip(axes, ["psnr", "ssim", "perceptual_loss"],
                               ["PSNR (dB)", "SSIM", "Perte perceptuelle"]):
        for model, label, style, color in [
            ("MSE", "MSE seule", "o-", "#27628f"),
            ("MSE + Perceptual", "MSE + perceptuelle", "s-", "#be5727"),
        ]:
            values = [r for r in semantic if r["model"] == model]
            ax.plot([float(r["snr_db"]) for r in values],
                    [float(r[key]) for r in values], style, color=color,
                    markersize=3, label=label)
        ax.set(xlabel="SNR du canal (dB)", ylabel=ylabel, xticks=[-5,0,10,20])
        ax.grid(alpha=.22)
    axes[0].legend(fontsize=6.7)
    finish(fig, "perceptuel_qualite")


def reconstructions():
    base = RESULTS / "classical/picture_tests/test_20261114_151430_101219"
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.05))
    for row, (method, label) in enumerate([("sans_fec", "Sans FEC"),
                                           ("avec_fec", "Hamming (7,4)")]):
        for col, snr in enumerate([None, 5, 8, 10, 12]):
            ax = axes[row, col]
            path = base / ("reference.png" if snr is None else f"{method}_{snr}dB.png")
            if path.exists():
                ax.imshow(plt.imread(path))
            else:
                ax.set_facecolor("#eeeeee")
                ax.text(.5, .5, "JPEG illisible", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)
            ax.set_title("Référence" if snr is None else f"{snr} dB")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col == 0:
                ax.set_ylabel(label)
    finish(fig, "classique_reconstructions")

    base = RESULTS / "jscc/evaluation"
    fig, axes = plt.subplots(1, 6, figsize=(10, 1.95))
    for ax, snr in zip(axes, [None, 0, 5, 10, 15, 20]):
        path = base / ("reference.png" if snr is None else f"reconstruction_{snr}dB.png")
        ax.imshow(plt.imread(path))
        ax.set_title("Référence" if snr is None else f"{snr} dB")
        ax.axis("off")
    finish(fig, "jscc_reconstructions")


if __name__ == "__main__":
    curves()
    reconstructions()
    print(f"Generated five publication figures in {OUTPUT}")
