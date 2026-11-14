"""Démonstration QPSK -> AWGN -> démodulation -> BER.

Exécution depuis la racine : python -m scripts.demo_ber
"""
import math
import torch

from src.channels.awgn import awgn


def main():
    torch.manual_seed(42)

    # Chaque symbole représente deux bits (mapping de Gray).
    bits_par_symbole = torch.tensor([[0, 0], [0, 1], [1, 1], [1, 0]])
    constellation = torch.tensor(
        [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=torch.complex64
    ) / math.sqrt(2)

    # Choisir au hasard 50 000 symboles = 100 000 bits.
    indices_envoyes = torch.randint(0, 4, (50_000,))
    symboles_envoyes = constellation[indices_envoyes]
    bits_envoyes = bits_par_symbole[indices_envoyes]

    for snr_db in [0, 5, 10, 20]:
        # Le signal reçoit du bruit dans le canal.
        symboles_recus = awgn(symboles_envoyes, snr_db)

        # Pour chaque symbole reçu, chercher le point QPSK le plus proche.
        distances = torch.abs(
            symboles_recus[:, None] - constellation[None, :]
        ).square()
        indices_decodes = distances.argmin(dim=1)
        bits_recus = bits_par_symbole[indices_decodes]

        nombre_erreurs = (bits_recus != bits_envoyes).sum().item()
        ber = nombre_erreurs / bits_envoyes.numel()
        print(f"SNR = {snr_db:2d} dB | erreurs = {nombre_erreurs:5d} / {bits_envoyes.numel()} | BER = {ber:.6f}")


if __name__ == "__main__":
    main()
