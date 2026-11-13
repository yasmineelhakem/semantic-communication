import torch

from src.models.semantic.semantic_model import (
    SemanticCommunicationModel
)


def test_semantic_model_output_shape():

    model = SemanticCommunicationModel()

    # Batch of 4 RGB images, 128x128
    x = torch.rand(4, 3, 128, 128)

    reconstructed = model(x, snr_db=10)

    assert reconstructed.shape == x.shape


def test_semantic_model_output_range():

    model = SemanticCommunicationModel()

    x = torch.rand(2, 3, 128, 128)

    reconstructed = model(x, snr_db=10)

    assert torch.all(reconstructed >= 0)
    assert torch.all(reconstructed <= 1)


def test_semantic_model_gradients():

    model = SemanticCommunicationModel()

    x = torch.rand(2, 3, 128, 128)

    reconstructed = model(x, snr_db=10)

    loss = torch.mean((reconstructed - x) ** 2)

    loss.backward()

    assert model.encoder.encoder[0].weight.grad is not None
