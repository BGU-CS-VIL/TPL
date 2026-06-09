import math
import torch
import torch.nn as nn


def _channel_schedule(input_channels: int, min_ch: int = 32) -> list:
    """
    Halving schedule: input_channels → ... → min_ch.
    Returns [] if input_channels <= min_ch (go directly to bottleneck).
    """
    if input_channels <= min_ch:
        return []
    n = math.ceil(math.log2(input_channels / min_ch))
    schedule = []
    for i in range(1, n + 1):
        c = max(min_ch, input_channels >> i)
        schedule.append(c)
    if schedule[-1] != min_ch:
        schedule.append(min_ch)
    return schedule


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm1d(out_ch),
        nn.GELU(),
    )


class FeatureVAE(nn.Module):
    """
    Channel-agnostic 1-D convolutional VAE.

    Encoder and decoder depth/width are derived automatically from
    input_channels via a power-of-2 halving schedule — no architecture
    changes needed when switching between modalities (hl_features at 384,
    512; kp2d at J*2; raw; etc.).

    Example schedules:
        input_channels=384 → enc: [192, 96, 48, 32] → bottleneck
        input_channels=512 → enc: [256, 128, 64, 32] → bottleneck
        input_channels=34  → enc: [32]               → bottleneck
        input_channels=3   → enc: []                 → bottleneck (direct)

    Args:
        input_channels:      D (feature dimension of the input signal)
        bottleneck_channels: latent channels (default 1 → scalar trajectory)
        signal_len:          T (not used structurally; kept for API compat)
        min_ch:              smallest intermediate channel width (default 32)
    """

    def __init__(self, input_channels: int, bottleneck_channels: int = 1,
                 signal_len: int = 121, min_ch: int = 32):
        super().__init__()
        schedule = _channel_schedule(input_channels, min_ch)

        enc_layers, prev = [], input_channels
        for ch in schedule:
            enc_layers.append(_conv_block(prev, ch))
            prev = ch
        self.encoder_shared = nn.Sequential(*enc_layers)
        self.encoder_mu     = nn.Conv1d(prev, bottleneck_channels, kernel_size=3, padding=1)
        self.encoder_logvar = nn.Conv1d(prev, bottleneck_channels, kernel_size=3, padding=1)

        dec_layers, prev = [], bottleneck_channels
        for ch in reversed(schedule):
            dec_layers.append(_conv_block(prev, ch))
            prev = ch
        dec_layers.append(nn.Conv1d(prev, input_channels, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor):
        h = self.encoder_shared(x)
        return self.encoder_mu(h), self.encoder_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        """x: (B, D, T) — NaN positions must be zeroed before calling."""
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z
