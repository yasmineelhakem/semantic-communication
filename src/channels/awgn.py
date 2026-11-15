
"""AWGN channel utilities for classical and neural communication pipelines."""

import math

import torch


# Original GitHub implementation: two real I/Q channels
def awgn_channel(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Apply AWGN to a tensor shaped [batch, 2, ...]."""
    if x.ndim < 2 or x.shape[1] != 2:
        raise ValueError("Input must have shape [batch, 2, ...]")

    power = x[:, 0].pow(2) + x[:, 1].pow(2)
    mean_power = power.mean(
        dim=tuple(range(1, power.ndim)),
        keepdim=True,
    )

    scale = torch.sqrt(mean_power.clamp_min(1e-12))
    x_normalized = x / scale.unsqueeze(1)

    snr_linear = 10 ** (snr_db / 10)
    noise_std = math.sqrt(1 / (2 * snr_linear))
    noise = torch.randn_like(x_normalized) * noise_std

    return x_normalized + noise


# Local implementation: native complex PyTorch tensors
def normalize_power(z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize complex symbol sequences to unit average power."""
    if not torch.is_complex(z):
        raise TypeError("z doit être un tenseur PyTorch complexe.")
    power = z.abs().square().mean(dim=-1, keepdim=True)
    return z / torch.sqrt(power.clamp_min(eps))


def awgn(z: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add circular complex Gaussian noise using input signal power."""
    if not torch.is_complex(z):
        raise TypeError("AWGN exige des symboles complexes (ex.: torch.complex64).")
    if z.ndim == 0 or z.shape[-1] == 0:
        raise ValueError("z doit contenir une séquence non vide de symboles.")
    if not math.isfinite(float(snr_db)):
        raise ValueError("snr_db doit être fini.")

    power = z.detach().abs().square().mean(dim=-1, keepdim=True)
    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    sigma = torch.sqrt(power / (2.0 * snr_linear))
    noise = sigma * (
        torch.randn_like(z.real) + 1j * torch.randn_like(z.real)
    )
    return z + noise


def measure_snr_db(sent: torch.Tensor, received: torch.Tensor) -> float:
    """Measure empirical SNR between transmitted and received signals."""
    if sent.shape != received.shape:
        raise ValueError("Les deux tenseurs doivent avoir la même forme.")
    p_signal = sent.detach().abs().square().mean().item()
    p_noise = (received.detach() - sent.detach()).abs().square().mean().item()
    if p_noise == 0.0:
        return float("inf")
    return 10.0 * math.log10(p_signal / p_noise)
