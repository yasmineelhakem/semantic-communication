"""Image JPEG -> bits -> QPSK -> AWGN -> QPSK -> image JPEG.

Exécuter depuis la racine du projet : python -m scripts.demo_image
"""
from io import BytesIO
from pathlib import Path
from src.paths import project_path
import math

import numpy as np
import torch
from PIL import Image

from src.channels.awgn import awgn


def main():
    torch.manual_seed(42)
    entree = project_path("data/input.jpg")
    sortie = project_path("results/classical/image")
    sortie.mkdir(parents=True, exist_ok=True)

    if not entree.exists():
        raise FileNotFoundError("Place une image dans data/input.jpg")

    # 1) Préparer une petite image et l'encoder en JPEG.
    with Image.open(entree) as image:
        image = image.convert("RGB").resize((128, 128))
        image.save(sortie / "reference.png")
        tampon = BytesIO()
        image.save(tampon, format="JPEG", quality=75)
    jpeg_envoye = tampon.getvalue()
    (sortie / "jpeg_compresse.jpg").write_bytes(jpeg_envoye)

    # 2) Convertir les octets JPEG en bits (8 bits par octet).
    bits_envoyes = np.unpackbits(
        np.frombuffer(jpeg_envoye, dtype=np.uint8)
    ).copy()
    paires = torch.from_numpy(bits_envoyes.reshape(-1, 2)).long()

    # 3) Mapper 2 bits vers 1 symbole QPSK (codage de Gray).
    constellation = torch.tensor(
        [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=torch.complex64
    ) / math.sqrt(2)
    table_bits = torch.tensor([[0, 0], [0, 1], [1, 1], [1, 0]])
    indices_binaires = 2 * paires[:, 0] + paires[:, 1]
    vers_indices_qpsk = torch.tensor([0, 1, 3, 2])
    symboles_envoyes = constellation[vers_indices_qpsk[indices_binaires]]

    # 4) Transmettre dans le canal AWGN (SNR élevé pour un premier essai).
    snr_db = 20
    symboles_recus = awgn(symboles_envoyes, snr_db)

    # 5) Démoduler : choisir le point QPSK le plus proche.
    distances = (symboles_recus[:, None] - constellation[None, :]).abs().square()
    bits_recus = table_bits[distances.argmin(dim=1)].reshape(-1).numpy().astype(np.uint8)

    # 6) Regrouper les bits en octets pour reconstruire le fichier JPEG.
    erreurs = int(np.count_nonzero(bits_envoyes != bits_recus))
    jpeg_recu = np.packbits(bits_recus).tobytes()
    (sortie / "jpeg_recu.jpg").write_bytes(jpeg_recu)

    print(f"Taille JPEG : {len(jpeg_envoye)} octets")
    print(f"Bits envoyés : {bits_envoyes.size}")
    print(f"SNR : {snr_db} dB | Erreurs : {erreurs} | BER : {erreurs / bits_envoyes.size:.6f}")
    print(f"Fichier JPEG identique : {jpeg_envoye == jpeg_recu}")

    try:
        with Image.open(BytesIO(jpeg_recu)) as image_recue:
            image_recue.load()
            image_recue.convert("RGB").save(sortie / "reconstruction.png")
        print("Image reconstruite : results/classical/image/reconstruction.png")
    except (OSError, ValueError) as exc:
        print(f"Décodage JPEG impossible : {exc}")
        print("C'est possible lorsqu'un JPEG contient des erreurs binaires.")


if __name__ == "__main__":
    main()
