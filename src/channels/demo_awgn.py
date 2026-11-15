"""Lancer depuis la racine avec: python -m scripts.demo_awgn"""
from pathlib import Path
from src.paths import project_path

import matplotlib
matplotlib.use("Agg")  # Fonctionne aussi sans écran graphique
import matplotlib.pyplot as plt
import torch

from src.channels.awgn import awgn, measure_snr_db, normalize_power


def main():
    torch.manual_seed(42)
    output = project_path("results/classical/constellations")
    output.mkdir(parents=True, exist_ok=True)

    # QPSK : 4 symboles complexes, chacun représentant deux bits.
    qpsk = torch.tensor([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=torch.complex64)
    indices = torch.randint(low=0, high=4, size=(20000,))
    transmitted = normalize_power(qpsk[indices])
    print(f"Puissance moyenne du signal transmis : {transmitted.abs().square().mean().item():.4f}")

    for snr_db in [0, 5, 10, 20]:
        received = awgn(transmitted, snr_db)
        snr_measured = measure_snr_db(transmitted, received)
        print(f"SNR demandé : {snr_db:>2} dB | SNR mesuré : {snr_measured:.2f} dB")

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(received.real[:700].numpy(), received.imag[:700].numpy(), s=6, alpha=0.35)
        ax.set(xlabel="Partie réelle (I)", ylabel="Partie imaginaire (Q)",
               title=f"Constellation QPSK sur AWGN — SNR {snr_db} dB")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output / f"constellation_{snr_db}dB.png", dpi=140)
        plt.close(fig)

    print("Graphiques enregistrés dans results/")


if __name__ == "__main__":
    main()
