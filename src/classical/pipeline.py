
"""
PROJET 07 — Communication Sémantique Apprise pour Images et Vidéos
INSAT | 2026–2027

ÉTAPE 4 : CHAÎNE CLASSIQUE DE TRANSMISSION AVEC ET SANS FEC

Objectif :
    Simuler la transmission d'une image JPEG à travers un canal
    sans fil AWGN et étudier l'impact du codage correcteur
    Hamming (7,4) sur les erreurs et la qualité d'image.

Chaîne de transmission :

    Image originale (128 x 128)
              |
        Compression JPEG
              |
        Conversion en bits
              |
       +------+------+
       |             |
     Sans FEC      Avec FEC
       |          Hamming(7,4)
       |             |
       +------+------+
              |
        Modulation QPSK
              |
          Canal AWGN
              |
       Démodulation QPSK
              |
       Décodage Hamming
       (si FEC activé)
              |
        Décodage JPEG
              |
        Image reconstruite

Expériences :
    - SNR : 5, 8, 10 et 12 dB
    - Comparaison avec et sans Hamming (7,4)
    - Calcul du BER (Bit Error Rate)
    - Calcul du PSNR (Peak Signal-to-Noise Ratio)
    - Calcul du SSIM (Structural Similarity Index)
    - Vérification de la lisibilité du JPEG
    - Calcul du surcoût de transmission du FEC

Remarques :
    - PSNR et SSIM sont calculés uniquement si le JPEG
      reçu est décodable.
    - La référence est l'image RGB avant compression JPEG.
    - Les résultats sont sauvegardés dans un fichier CSV.
    - La comparaison avec/sans FEC n'est pas à débit égal.
    - Le code Hamming (7,4) corrige au plus une erreur
      par bloc de 7 bits.

Entrée :
    data/input.jpg

Sorties :
    results/classical/fec/reference.png
    results/classical/fec/sans_fec_XdB.jpg
    results/classical/fec/sans_fec_XdB.png (si décodable)
    results/classical/fec/avec_fec_XdB.jpg
    results/classical/fec/avec_fec_XdB.png (si décodable)
    results/classical/fec/metriques.csv

Exécution depuis la racine du projet :
    python -m scripts.demo_fec
"""

from io import BytesIO
from pathlib import Path
from src.paths import project_path
import math
import csv

import numpy as np
import torch
from PIL import Image

from src.channels.awgn import awgn
from src.classical.hamming74 import (
    encode_hamming74,
    decode_hamming74,
)
from src.metrics.image_quality import calculer_metriques


# ============================================================
# 1. CONFIGURATION QPSK
# ============================================================

CONSTELLATION = torch.tensor(
    [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j],
    dtype=torch.complex64,
) / math.sqrt(2)

TABLE_BITS = torch.tensor([
    [0, 0],
    [0, 1],
    [1, 1],
    [1, 0],
])

VERS_INDICES_QPSK = torch.tensor([0, 1, 3, 2])


# ============================================================
# 2. TRANSMISSION QPSK + CANAL AWGN
# ============================================================

def transmettre_qpsk(
    bits: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """
    Transmet une séquence binaire sur un canal AWGN.

    Étapes :
        1. Regroupement des bits par paires.
        2. Modulation QPSK.
        3. Ajout du bruit AWGN.
        4. Démodulation par distance minimale.
        5. Reconstruction des bits reçus.

    Retour :
        Tableau NumPy contenant les bits reçus.
    """

    bits = np.asarray(
        bits, dtype=np.uint8
    ).reshape(-1)

    # QPSK nécessite un nombre pair de bits.
    padding = (-bits.size) % 2

    if padding:
        bits = np.pad(bits, (0, padding))

    # Regrouper les bits par paires.
    paires = torch.from_numpy(
        bits.reshape(-1, 2).copy()
    ).long()

    indices_binaires = (
        2 * paires[:, 0] + paires[:, 1]
    )

    indices_qpsk = VERS_INDICES_QPSK[
        indices_binaires
    ]

    # Modulation.
    symboles = CONSTELLATION[indices_qpsk]

    # Transmission sur le canal AWGN.
    recus = awgn(symboles, snr_db)

    # Démodulation : symbole le plus proche.
    distances = (
        recus[:, None] - CONSTELLATION[None, :]
    ).abs().square()

    bits_decodes = (
        TABLE_BITS[distances.argmin(dim=1)]
        .reshape(-1)
        .numpy()
        .astype(np.uint8)
    )

    # Retirer les bits de remplissage QPSK.
    if padding:
        return bits_decodes[:-padding]

    return bits_decodes


# ============================================================
# 3. RECONSTRUCTION JPEG
# ============================================================

def sauver_jpeg(
    bits: np.ndarray,
    fichier: Path,
    image_sortie: Path,
) -> bool:
    """
    Reconstruit un fichier JPEG à partir des bits reçus.

    Retour :
        True  : image JPEG décodable.
        False : image JPEG illisible.

    Si le décodage réussit, une version PNG est enregistrée.
    """

    raw = np.packbits(bits).tobytes()

    # Sauvegarder les données JPEG reçues.
    fichier.write_bytes(raw)

    # Éviter de conserver un ancien PNG si
    # le nouveau JPEG est illisible.
    if image_sortie.exists():
        image_sortie.unlink()

    try:
        with Image.open(BytesIO(raw)) as image:
            image.load()

            image.convert("RGB").save(
                image_sortie
            )

        return True

    except (OSError, ValueError):
        return False


# ============================================================
# 4. CALCUL DES MÉTRIQUES D'IMAGE
# ============================================================

def evaluer_image(
    reference: Path,
    reconstruite: Path,
    lisible: bool,
) -> tuple:
    """
    Calcule PSNR et SSIM si le JPEG est décodable.

    Si le JPEG est illisible :
        PSNR = None
        SSIM = None
    """

    if not lisible:
        return None, None

    psnr, ssim = calculer_metriques(
        reference,
        reconstruite,
    )

    return psnr, ssim


# ============================================================
# 5. EXPÉRIENCE PRINCIPALE
# ============================================================

def main() -> None:
    """
    Compare la transmission JPEG avec et sans Hamming.

    Pour chaque SNR :
        - Transmission sans FEC.
        - Transmission avec FEC.
        - Calcul du BER.
        - Reconstruction JPEG.
        - Calcul du PSNR et du SSIM.
        - Sauvegarde des résultats.
    """

    # Reproductibilité du bruit simulé.
    torch.manual_seed(42)

    # Chemins.
    entree = project_path("data/input.jpg")
    sortie = project_path("results/classical/fec")

    sortie.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not entree.exists():
        raise FileNotFoundError(
            "Place une image dans data/input.jpg"
        )

    # --------------------------------------------------------
    # A. Préparation de l'image
    # --------------------------------------------------------

    with Image.open(entree) as image:

        image = (
            image.convert("RGB")
            .resize((128, 128))
        )

        reference = sortie / "reference.png"
        image.save(reference)

        buffer = BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=75,
        )

    jpeg = buffer.getvalue()

    # Convertir les octets JPEG en bits.
    bits = np.unpackbits(
        np.frombuffer(
            jpeg,
            dtype=np.uint8,
        )
    ).copy()

    # --------------------------------------------------------
    # B. Codage correcteur Hamming (7,4)
    # --------------------------------------------------------

    bits_codes, padding = encode_hamming74(
        bits
    )

    taux = bits.size / bits_codes.size

    overhead = (
        bits_codes.size / bits.size - 1
    ) * 100

    print("\n" + "=" * 65)
    print("PROJET 07 — TRANSMISSION CLASSIQUE JPEG + QPSK + AWGN")
    print("=" * 65)

    print(
        f"\nJPEG : {len(jpeg)} octets"
    )

    print(
        f"Bits utiles : {bits.size}"
    )

    print(
        f"Bits avec Hamming : {bits_codes.size}"
    )

    print(
        f"Taux de codage : {taux:.3f}"
    )

    print(
        f"Surcoût FEC : {overhead:.1f} %"
    )

    print(
        "\nAttention : comparaison à SNR identique, "
        "mais pas à débit identique."
    )

    # --------------------------------------------------------
    # C. Simulation pour différents SNR
    # --------------------------------------------------------

    snr_values = [5, 8, 10, 12]

    resultats = []

    for snr_db in snr_values:

        print("\n" + "-" * 65)
        print(f"SNR = {snr_db} dB")
        print("-" * 65)

        # ====================================================
        # TRANSMISSION SANS FEC
        # ====================================================

        recu_sans = transmettre_qpsk(
            bits,
            snr_db,
        )

        erreurs_sans = int(
            np.count_nonzero(
                bits != recu_sans
            )
        )

        ber_sans = erreurs_sans / bits.size

        fichier_sans = (
            sortie / f"sans_fec_{snr_db}dB.jpg"
        )

        image_sans = (
            sortie / f"sans_fec_{snr_db}dB.png"
        )

        ok_sans = sauver_jpeg(
            recu_sans,
            fichier_sans,
            image_sans,
        )

        psnr_sans, ssim_sans = evaluer_image(
            reference,
            image_sans,
            ok_sans,
        )

        # ====================================================
        # TRANSMISSION AVEC FEC
        # ====================================================

        recu_codes = transmettre_qpsk(
            bits_codes,
            snr_db,
        )

        bits_avec, syndromes = decode_hamming74(
            recu_codes,
            padding,
        )

        erreurs_avec = int(
            np.count_nonzero(
                bits != bits_avec
            )
        )

        ber_avec = erreurs_avec / bits.size

        fichier_avec = (
            sortie / f"avec_fec_{snr_db}dB.jpg"
        )

        image_avec = (
            sortie / f"avec_fec_{snr_db}dB.png"
        )

        ok_avec = sauver_jpeg(
            bits_avec,
            fichier_avec,
            image_avec,
        )

        psnr_avec, ssim_avec = evaluer_image(
            reference,
            image_avec,
            ok_avec,
        )

        # ====================================================
        # AFFICHAGE DES RÉSULTATS
        # ====================================================

        print("\nSANS FEC")
        print(f"  Erreurs : {erreurs_sans}")
        print(f"  BER : {ber_sans:.6f}")
        print(f"  JPEG lisible : {ok_sans}")

        if psnr_sans is not None:
            print(f"  PSNR : {psnr_sans:.2f} dB")
            print(f"  SSIM : {ssim_sans:.4f}")
        else:
            print("  PSNR : N/A")
            print("  SSIM : N/A")

        print("\nAVEC FEC")
        print(f"  Erreurs : {erreurs_avec}")
        print(f"  BER : {ber_avec:.6f}")
        print(f"  Syndromes non nuls : {syndromes}")
        print(f"  JPEG lisible : {ok_avec}")

        if psnr_avec is not None:
            print(f"  PSNR : {psnr_avec:.2f} dB")
            print(f"  SSIM : {ssim_avec:.4f}")
        else:
            print("  PSNR : N/A")
            print("  SSIM : N/A")

        # ====================================================
        # ENREGISTREMENT DES RÉSULTATS
        # ====================================================

        resultats.append({
            "snr_db": snr_db,
            "methode": "sans_fec",
            "erreurs": erreurs_sans,
            "ber": ber_sans,
            "jpeg_lisible": ok_sans,
            "psnr_db": psnr_sans,
            "ssim": ssim_sans,
        })

        resultats.append({
            "snr_db": snr_db,
            "methode": "avec_fec",
            "erreurs": erreurs_avec,
            "ber": ber_avec,
            "jpeg_lisible": ok_avec,
            "psnr_db": psnr_avec,
            "ssim": ssim_avec,
        })

    # --------------------------------------------------------
    # D. Export des résultats vers CSV
    # --------------------------------------------------------

    fichier_csv = sortie / "metriques.csv"

    with fichier_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "snr_db",
                "methode",
                "erreurs",
                "ber",
                "jpeg_lisible",
                "psnr_db",
                "ssim",
            ],
        )

        writer.writeheader()
        writer.writerows(resultats)

    print("\n" + "=" * 65)
    print("SIMULATION TERMINÉE")
    print("=" * 65)

    print(
        f"\nRésultats enregistrés dans : {sortie}"
    )

    print(
        f"Métriques CSV : {fichier_csv}"
    )


# ============================================================
# 6. POINT D'ENTRÉE DU PROGRAMME
# ============================================================

if __name__ == "__main__":
    main()
