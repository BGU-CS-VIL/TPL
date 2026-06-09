import torch
import torch.nn.functional as F
from difw import Cpab


class CpabAugment:
    """
    Applies a random diffeomorphic warp to a batch of sequences during training.

    Because the warp is diffeomorphic it preserves topology — peaks stay peaks,
    valleys stay valleys — so the label is unchanged. The aligner must then learn
    to undo arbitrary timing shifts, which prevents it memorising the exact
    timing offsets in the training set.

    Args:
        tess_size:     CPAB tessellation size (matches the aligner's tess_size).
        scale:         std of the random θ. Small (0.05) = subtle, large (0.3) = strong.
        zero_boundary: pin endpoints (recommended True to avoid edge drift).
        device:        'cpu' or 'cuda'.
    """

    def __init__(self, tess_size: int = 16, scale: float = 0.1,
                 zero_boundary: bool = True, device: str = "cuda"):
        cpab_device = "gpu" if "cuda" in device else "cpu"
        self.T     = Cpab(tess_size, backend="pytorch",
                          device=cpab_device, zero_boundary=zero_boundary)
        self.scale = scale
        self.d     = self.T.params.d

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)  — may contain NaNs (variable-length sequences).
        Returns warped x with the same shape and NaN pattern.
        """
        B, C, T = x.shape
        nan_mask = torch.isnan(x)

        theta = torch.randn(B, 1, self.d, device=x.device) * self.scale
        x_clean = torch.nan_to_num(x, nan=0.0)

        grid = self.T.uniform_meshgrid(n_points=T).repeat(B, 1).to(x.device)
        grid_t = self.T.transform_grid_ss(grid, theta.squeeze(1), method="closed_form")

        warped = torch.zeros_like(x_clean)
        for c in range(C):
            warped[:, c, :] = self.T.interpolate(
                x_clean[:, c, :].unsqueeze(-1), grid_t, outsize=T
            ).squeeze(-1)

        warped[nan_mask] = float("nan")
        return warped


class AffineAugment:
    """
    Applies random per-batch affine transforms along the time axis:
        - time stretch / compress  (scale ∈ [1-s, 1+s])
        - time shift               (translation ∈ [-shift_frac, +shift_frac] * T)
        - amplitude scale          (gain ∈ [1-a, 1+a])

    All ops are label-preserving. NaN pattern is shifted with the signal so
    padding remains at the ends.

    Args:
        scale_range:  max fractional stretch/compress (default 0.1 → ±10 %).
        shift_frac:   max fractional time shift       (default 0.1 → ±10 % of T).
        amplitude:    max fractional amplitude scale  (default 0.1 → ±10 %).
        p:            probability of applying augment to each sample (default 0.5).
    """

    def __init__(self, scale_range: float = 0.1, shift_frac: float = 0.1,
                 amplitude: float = 0.1, p: float = 0.5):
        self.scale_range = scale_range
        self.shift_frac  = shift_frac
        self.amplitude   = amplitude
        self.p           = p

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T) — may contain NaNs (variable-length sequences).
        Returns augmented x with the same shape and NaN pattern preserved.
        """
        B, C, T = x.shape
        out = x.clone()

        for b in range(B):
            if torch.rand(1).item() > self.p:
                continue

            nan_mask = torch.isnan(out[b])            # (C, T)
            x_b      = torch.nan_to_num(out[b], nan=0.0).unsqueeze(0)  # (1, C, T)

            # ── time warp (stretch/shift) via grid_sample ──────────────────
            scale = 1.0 + (torch.rand(1).item() * 2 - 1) * self.scale_range
            shift = (torch.rand(1).item() * 2 - 1) * self.shift_frac

            # grid_sample expects grid in [-1, 1]; shape (1, 1, T, 1) for 1-D
            t_grid = torch.linspace(-1, 1, T, device=x.device)
            t_grid = t_grid / scale + shift * 2       # stretch then shift
            t_grid = t_grid.clamp(-1, 1)

            # (1, 1, T, 2): H=1, W=T, (x, y) where y=0 (dummy height dim)
            grid_2d = torch.stack(
                [t_grid, torch.zeros_like(t_grid)], dim=-1
            ).unsqueeze(0).unsqueeze(0)               # (1, 1, T, 2)

            # grid_sample needs (B, C, H, W); use H=1
            x_4d    = x_b.unsqueeze(2)                # (1, C, 1, T)
            warped  = F.grid_sample(x_4d, grid_2d.expand(1, 1, 1, T),
                                    mode="bilinear", padding_mode="border",
                                    align_corners=True)
            x_b = warped.squeeze(2)                   # (1, C, T)

            # ── amplitude scale ────────────────────────────────────────────
            gain = 1.0 + (torch.rand(1).item() * 2 - 1) * self.amplitude
            x_b  = x_b * gain

            out[b] = x_b.squeeze(0)
            out[b][nan_mask] = float("nan")

        return out
