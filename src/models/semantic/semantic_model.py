import torch.nn as nn

from src.models.semantic.encoder import SemanticEncoder
from src.models.semantic.decoder import SemanticDecoder
from src.channels.awgn import awgn_channel


class SemanticCommunicationModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = SemanticEncoder()
        self.decoder = SemanticDecoder()

    def forward(self, x, snr_db):

        # Step 1: Encode the input image
        encoded = self.encoder(x)

        # Step 2: Transmit through AWGN channel
        received = awgn_channel(encoded, snr_db)

        # Step 3: Reconstruct the image
        reconstructed = self.decoder(received)

        return reconstructed
