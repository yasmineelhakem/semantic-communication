from torch import nn
from src.jscc.encoder import EncoderCNN
from src.jscc.decoder import DecoderCNN
from src.jscc.normalization import PowerNormalization
from src.channels.awgn import awgn


class SyntaxicJSCC(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = EncoderCNN()
        self.normalizer = PowerNormalization()
        self.decoder = DecoderCNN()

    def forward(self, images, snr_db):

        # Encodage de l'image.
        latent = self.encoder(images)

        # Conversion en symboles complexes
        # et normalisation de puissance.
        symbols = self.normalizer(latent)

        # Transmission AWGN.
        received = awgn(symbols, snr_db)

        # Reconstruction de l'image.
        reconstructed = self.decoder(received)

        return reconstructed
