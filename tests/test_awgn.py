import torch
from src.channels.awgn import awgn_channel


def test_awgn_shape():
    x = torch.randn(4, 2, 16, 16)

    y = awgn_channel(x, snr_db=10)

    assert y.shape == x.shape


def test_awgn_no_nan():
    x = torch.randn(4, 2, 16, 16)

    y = awgn_channel(x, snr_db=10)

    assert torch.isfinite(y).all()


def test_awgn_noise_increases_when_snr_decreases():
    x = torch.randn(4, 2, 16, 16)

    y_high = awgn_channel(x, snr_db=20)
    y_low = awgn_channel(x, snr_db=0)

    error_high = ((y_high - x / torch.sqrt(
        (x[:, 0].pow(2) + x[:, 1].pow(2)).mean(
            dim=(1, 2), keepdim=True
        ).clamp_min(1e-12)
    ).unsqueeze(1)) ** 2).mean()

    error_low = ((y_low - x / torch.sqrt(
        (x[:, 0].pow(2) + x[:, 1].pow(2)).mean(
            dim=(1, 2), keepdim=True
        ).clamp_min(1e-12)
    ).unsqueeze(1)) ** 2).mean()

    assert error_low > error_high
