
"""
PROJET 07 — Test de l'étape 2

Objectif :
    Vérifier la conversion de la représentation latente
    en symboles complexes et la normalisation de puissance.

Exécution :
    python -m scripts.test_normalization
"""

import torch

from src.jscc.encoder import EncoderCNN
from src.jscc.normalization import PowerNormalization


def main():

    torch.manual_seed(42)

    # Encodeur déjà implémenté à l'étape 1.
    encoder = EncoderCNN()
    encoder.eval()

    # Nouveau module de normalisation.
    normalizer = PowerNormalization()

    # Image RGB fictive 128x128.
    image = torch.rand(1, 3, 128, 128)

    with torch.no_grad():

        # Étape 1 : encoder l'image.
        latent = encoder(image)

        # Étape 2 : former et normaliser les symboles.
        symbols = normalizer(latent)

    # Afficher les dimensions.
    print("Image originale :", image.shape)
    print("Sortie encodeur :", latent.shape)
    print("Symboles complexes :", symbols.shape)

    # Afficher quelques symboles.
    print("\nPremiers symboles complexes :")
    print(symbols[0, :5])

    # Vérifier la puissance moyenne.
    power = symbols.abs().square().mean()

    print(f"\nPuissance moyenne : {power.item():.6f}")

    # Calcul du Channel Bandwidth Ratio.
    nombre_symboles = symbols.shape[1]
    nombre_pixels = image.shape[2] * image.shape[3]

    cbr = nombre_symboles / nombre_pixels

    print("Nombre de symboles :", nombre_symboles)
    print(f"CBR : {cbr:.3f}")

    # Vérifications automatiques.
    assert symbols.shape == (1, 2048)
    assert torch.is_complex(symbols)
    assert torch.allclose(
        power,
        torch.tensor(1.0),
        atol=1e-5
    )

    print("\nTest réussi !")


if __name__ == "__main__":
    main()
