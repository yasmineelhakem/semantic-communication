
"""
PROJET 07 — Communication Sémantique Images et Vidéos
INSAT — 2026/2027

ÉTAPE 2 : SYMBOLES COMPLEXES ET NORMALISATION

Objectif :
    Transformer la représentation latente produite par
    l'encodeur CNN en symboles complexes.

Chaîne :
    Encodeur CNN
         |
    4096 valeurs réelles
         |
    Regroupement par paires (I, Q)
         |
    2048 symboles complexes
         |
    Normalisation de puissance
         |
    Symboles complexes de puissance moyenne = 1

Entrée :
    [batch, 16, 16, 16]

Sortie :
    [batch, 2048] (complex64)

La normalisation est différentiable pour permettre
l'entraînement de bout en bout du modèle JSCC.
"""

import torch
from torch import nn


class PowerNormalization(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, z):

        # 1. Récupérer le nombre d'images.
        batch_size = z.shape[0]

        # 2. Aplatir les valeurs latentes.
        z_flat = z.reshape(batch_size, -1)

        # Vérifier que les valeurs peuvent être regroupées par 2.
        if z_flat.shape[1] % 2 != 0:
            raise ValueError(
                "Le nombre de valeurs latentes doit être pair."
            )

        # 3. Former les paires (réel, imaginaire).
        z_pairs = z_flat.reshape(batch_size, -1, 2)

        # 4. Convertir les paires en symboles complexes.
        symbols = torch.view_as_complex(
            z_pairs.contiguous()
        )

        # 5. Calculer la puissance moyenne par image.
        power = (
            symbols.abs().square()
            .mean(dim=1, keepdim=True)
        )

        # 6. Normaliser la puissance moyenne à environ 1.
        normalized_symbols = symbols / torch.sqrt(
            power + 1e-8
        )

        return normalized_symbols
