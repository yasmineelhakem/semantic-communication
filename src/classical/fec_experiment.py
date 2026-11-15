"""Étape 4 : JPEG + Hamming(7,4) + QPSK + AWGN + décodage.

Exécution depuis la racine du projet : python -m scripts.demo_fec
Ajoute le FEC à la chaîne classique; montre le BER et l'overhead.
"""

from io import BytesIO
from pathlib import Path
from src.paths import project_path
import math

import numpy as np
import torch
from PIL import Image

from src.channels.awgn import awgn
from src.classical.hamming74 import encode_hamming74, decode_hamming74


CONSTELLATION = torch.tensor(
    [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=torch.complex64
) / math.sqrt(2)
TABLE_BITS = torch.tensor([[0, 0], [0, 1], [1, 1], [1, 0]])
VERS_INDICES_QPSK = torch.tensor([0, 1, 3, 2])


def transmettre_qpsk(bits: np.ndarray, snr_db: float) -> np.ndarray:
    """Modulation QPSK, canal AWGN, décision au point le plus proche."""
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    padding = (-bits.size) % 2
    if padding:
        bits = np.pad(bits, (0, padding))
    paires = torch.from_numpy(bits.reshape(-1, 2).copy()).long()
    indices_binaires = 2 * paires[:, 0] + paires[:, 1]
    indices_qpsk = VERS_INDICES_QPSK[indices_binaires]
    symboles = CONSTELLATION[indices_qpsk]
    recus = awgn(symboles, snr_db)
    distances = (recus[:, None] - CONSTELLATION[None, :]).abs().square()
    bits_decodes = TABLE_BITS[distances.argmin(dim=1)].reshape(-1).numpy().astype(np.uint8)
    return bits_decodes[:-padding] if padding else bits_decodes


def sauver_jpeg(bits: np.ndarray, fichier: Path, image_sortie: Path) -> bool:
    """Sauvegarde le JPEG reçu, puis tente de le décoder en image PNG."""
    raw = np.packbits(bits).tobytes()
    fichier.write_bytes(raw)
    try:
        with Image.open(BytesIO(raw)) as image:
            image.load()
            image.convert("RGB").save(image_sortie)
        return True
    except (OSError, ValueError):
        return False


def main() -> None:
    torch.manual_seed(42)
    entree = project_path("data/input.jpg")
    sortie = project_path("results/classical/fec")
    sortie.mkdir(parents=True, exist_ok=True)
    if not entree.exists():
        raise FileNotFoundError("Place une image dans data/input.jpg")

    with Image.open(entree) as image:
        image = image.convert("RGB").resize((128, 128))
        image.save(sortie / "reference.png")
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=75)
    jpeg = buf.getvalue()
    bits = np.unpackbits(np.frombuffer(jpeg, dtype=np.uint8)).copy()
    bits_codes, padding = encode_hamming74(bits)
    taux = bits.size / bits_codes.size
    print(f"JPEG : {len(jpeg)} octets, {bits.size} bits utiles")
    print(f"Hamming(7,4) : {bits_codes.size} bits transmis | taux R={taux:.3f} (≈4/7)")
    print("Attention : le FEC consomme 75 % de symboles en plus à modulation QPSK identique.")
    print("Note : le BER non codé et le BER après FEC sont des mesures, pas encore une comparaison à débit égal.\n")

    for snr_db in (5, 8, 10, 12):
        # Sans FEC : transmettre les bits du JPEG directement.
        recu_sans = transmettre_qpsk(bits, snr_db)
        erreurs_sans = int(np.count_nonzero(bits != recu_sans))
        # Avec FEC : encoder avant le canal, décoder après la démodulation.
        recu_codes = transmettre_qpsk(bits_codes, snr_db)
        bits_avec, syndromes = decode_hamming74(recu_codes, padding)
        erreurs_avec = int(np.count_nonzero(bits != bits_avec))
        ok_sans = sauver_jpeg(
            recu_sans, sortie / f"sans_fec_{snr_db}dB.jpg", sortie / f"sans_fec_{snr_db}dB.png"
        )
        ok_avec = sauver_jpeg(
            bits_avec, sortie / f"avec_fec_{snr_db}dB.jpg", sortie / f"avec_fec_{snr_db}dB.png"
        )
        print(
            f"SNR={snr_db:2d} dB | BER sans FEC={erreurs_sans/bits.size:.6f} "
            f"({erreurs_sans} erreurs) | BER avec FEC={erreurs_avec/bits.size:.6f} "
            f"({erreurs_avec} erreurs) | syndromes non nuls={syndromes} "
            f"| JPEG lisible sans/avec={ok_sans}/{ok_avec}"
        )


if __name__ == "__main__":
    main()
