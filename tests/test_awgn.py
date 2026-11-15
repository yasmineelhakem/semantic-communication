
import unittest

import torch

from src.channels.awgn import (
    awgn_channel,
    awgn,
    normalize_power,
    measure_snr_db,
)


# Tests de la version GitHub (I/Q réels)

def test_awgn_shape():
    x = torch.randn(4, 2, 16, 16)
    y = awgn_channel(x, snr_db=10)
    assert y.shape == x.shape


def test_awgn_no_nan():
    x = torch.randn(4, 2, 16, 16)
    y = awgn_channel(x, snr_db=10)
    assert torch.isfinite(y).all()


def test_awgn_noise_increases_when_snr_decreases():
    torch.manual_seed(0)
    x = torch.randn(4, 2, 16, 16)

    power = (x[:, 0].square() + x[:, 1].square()).mean(
        dim=(1, 2), keepdim=True
    )
    normalized_x = x / torch.sqrt(
        power.clamp_min(1e-12)
    ).unsqueeze(1)

    y_high = awgn_channel(x, snr_db=20)
    y_low = awgn_channel(x, snr_db=0)

    error_high = (y_high - normalized_x).square().mean()
    error_low = (y_low - normalized_x).square().mean()

    assert error_low > error_high


# Tests de la version locale (tenseurs complexes)

class AwgnTests(unittest.TestCase):

    def test_normalized_power(self):
        torch.manual_seed(0)
        z = torch.complex(
            torch.randn(3, 1000),
            torch.randn(3, 1000)
        )
        p = normalize_power(z).abs().square().mean(dim=-1)
        self.assertTrue(
            torch.allclose(p, torch.ones_like(p), atol=1e-5)
        )

    def test_snr_measurement(self):
        torch.manual_seed(0)
        z = normalize_power(
            torch.complex(
                torch.randn(100_000),
                torch.randn(100_000)
            )
        )
        for snr_db in (0, 10, 20):
            y = awgn(z, snr_db)
            self.assertAlmostEqual(
                measure_snr_db(z, y),
                snr_db,
                delta=0.15
            )

    def test_gradient_through_channel(self):
        torch.manual_seed(0)
        real_part = torch.randn(64, requires_grad=True)
        z = torch.complex(real_part, torch.randn(64))
        y = awgn(normalize_power(z), 10)
        loss = y.abs().square().mean()
        loss.backward()

        self.assertIsNotNone(real_part.grad)
        self.assertTrue(
            torch.isfinite(real_part.grad).all().item()
        )

    def test_rejects_real_input(self):
        with self.assertRaises(TypeError):
            awgn(torch.randn(20), 10)


if __name__ == "__main__":
    unittest.main()
