import torch
import math

def awgn_channel(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    Simulate a complex AWGN channel.

    Expected input:
        x: Tensor of shape [batch, 2, ...]
           Channel 1 represents the real part (I).
           Channel 2 represents the imaginary part (Q).

    Args:
        x: Transmitted complex symbols represented by two real channels.
        snr_db: Signal-to-noise ratio in dB.

    Returns:
        Noisy received symbols with the same shape as x.
    """

    if x.ndim < 2 or x.shape[1] != 2:
        raise ValueError("Input must have shape [batch, 2, ...]")

    # Calculate average complex power for each sample
    power = x[:, 0].pow(2) + x[:, 1].pow(2)
    mean_power = power.mean(
        dim=tuple(range(1, power.ndim)),
        keepdim=True
    )

    # Normalize average power to 1
    scale = torch.sqrt(mean_power.clamp_min(1e-12))
    x_normalized = x / scale.unsqueeze(1)

    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)

    # Each real component receives half the complex noise variance
    noise_std = math.sqrt(1 / (2 * snr_linear))

    noise = torch.randn_like(x_normalized) * noise_std

    # Received signal = transmitted signal + Gaussian noise
    y = x_normalized + noise

    return y
