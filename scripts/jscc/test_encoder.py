
"""
PROJET 07 — Test de l'encodeur JSCC syntaxique

Exécution :
    python -m scripts.test_encoder
"""

import torch
from src.jscc.encoder import EncoderCNN


def main():

    torch.manual_seed(42)

    # Initialiser le modèle
    encoder = EncoderCNN()
    encoder.eval()

    # Générer une image RGB fictive
    image = torch.rand(1, 3, 128, 128)

    # Appliquer l'encodeur
    with torch.no_grad():
        latent = encoder(image)

    print("Image originale :", image.shape)
    print("Sortie encodeur :", latent.shape)

    print("Valeurs latentes :", latent.numel())
    print("Futurs symboles complexes :", latent.numel() // 2)

    # Compter les paramètres entraînables
    params = sum(
        p.numel()
        for p in encoder.parameters()
        if p.requires_grad
    )

    print("Paramètres entraînables :", params)


if __name__ == "__main__":
    main()
