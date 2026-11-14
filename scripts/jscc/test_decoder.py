
"""
PROJET 07 — JSCC SYNTAXIQUE
TEST DE LA CHAÎNE COMPLÈTE AVANT ENTRAÎNEMENT

Chaîne :
    Image -> Encodeur CNN -> Normalisation
          -> Canal AWGN -> Décodeur CNN
          -> Image reconstruite

Exécution :
    python -m scripts.test_decoder

Remarque :
    Les modèles ne sont pas encore entraînés.
    Ce test vérifie uniquement le fonctionnement
    et les dimensions des différents composants.
"""

import torch

from src.jscc.encoder import EncoderCNN
from src.jscc.normalization import PowerNormalization
from src.jscc.decoder import DecoderCNN
from src.channels.awgn import awgn, measure_snr_db


def main():

    torch.manual_seed(42)

    # Initialisation des composants.
    encoder = EncoderCNN()
    normalizer = PowerNormalization()
    decoder = DecoderCNN()

    encoder.eval()
    decoder.eval()

    # Image RGB fictive.
    image = torch.rand(1, 3, 128, 128)

    snr_db = 10.0

    with torch.no_grad():

        # Étape 1 : encodage.
        latent = encoder(image)

        # Étape 2 : normalisation.
        symbols = normalizer(latent)

        # Étape 3 : transmission AWGN.
        received = awgn(symbols, snr_db)

        # Étape 4 : décodage.
        reconstructed = decoder(received)

    # Vérifications.
    print("Image originale :", image.shape)
    print("Sortie encodeur :", latent.shape)
    print("Symboles transmis :", symbols.shape)
    print("Symboles reçus :", received.shape)
    print("Image reconstruite :", reconstructed.shape)

    print(
        "\nSNR mesuré :",
        f"{measure_snr_db(symbols, received):.2f} dB"
    )

    print(
        "Pixels minimum :",
        reconstructed.min().item()
    )

    print(
        "Pixels maximum :",
        reconstructed.max().item()
    )

    # Vérifier les dimensions et les pixels.
    assert reconstructed.shape == image.shape
    assert reconstructed.min().item() >= 0.0
    assert reconstructed.max().item() <= 1.0

    print("\nTest du décodeur réussi !")


if __name__ == "__main__":
    main()
