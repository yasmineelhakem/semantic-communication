
"""
PROJET 07 — JSCC SYNTAXIQUE
ÉTAPE 3 : TEST DU CANAL AWGN

Objectif :
    Tester la transmission des symboles complexes
    produits par l'encodeur CNN à travers le canal AWGN.

Chaîne :
    Image -> Encodeur CNN -> Normalisation
          -> AWGN -> Symboles bruités

Vérifications :
    - Dimensions des tenseurs
    - Puissance moyenne normalisée
    - SNR mesuré
    - Propagation des gradients

Exécution :
    python -m scripts.test_jscc_channel
"""

import torch

from src.jscc.encoder import EncoderCNN
from src.jscc.normalization import PowerNormalization
from src.channels.awgn import awgn, measure_snr_db


def main():

    torch.manual_seed(42)

    encoder = EncoderCNN()
    normalizer = PowerNormalization()

    snr_db = 10.0

    # Image fictive RGB 128x128.
    image = torch.rand(1, 3, 128, 128)

    # Étape 1 : encodage CNN.
    latent = encoder(image)

    # Étape 2 : symboles complexes normalisés.
    symbols = normalizer(latent)

    # Étape 3 : passage dans AWGN.
    received = awgn(symbols, snr_db)

    # Vérification de la puissance.
    power = symbols.abs().square().mean().item()

    # Vérification du SNR.
    measured_snr = measure_snr_db(
        symbols,
        received
    )

    print("Image originale :", image.shape)
    print("Sortie encodeur :", latent.shape)
    print("Symboles transmis :", symbols.shape)
    print("Symboles reçus :", received.shape)

    print(f"\nPuissance moyenne : {power:.6f}")
    print(f"SNR demandé : {snr_db:.2f} dB")
    print(f"SNR mesuré : {measured_snr:.2f} dB")

    print("\nPremier symbole transmis :")
    print(symbols[0, 0].item())

    print("\nPremier symbole reçu :")
    print(received[0, 0].item())

    # Vérifier que les gradients traversent AWGN.
    loss = received.abs().square().mean()
    loss.backward()

    gradient = encoder.network[0].weight.grad

    assert gradient is not None
    assert torch.isfinite(gradient).all()

    print("\nPropagation des gradients : OK")
    print("Test du canal JSCC réussi !")


if __name__ == "__main__":
    main()
