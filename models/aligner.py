import torch
import torch.nn as nn
from difw import Cpab
from tsai.all import InceptionTime, build_ts_model

_BACKBONES = {"InceptionTime": InceptionTime}


class TemporalAligner(nn.Module):
    """
    Learns a per-sample diffeomorphic warp that maps each sequence toward
    the class prototype.

    The warp lives in a low-dimensional smooth space (CPAB) — invertible
    by construction, so the prototype can always be recovered exactly.

    Args:
        signal_len:    length of the input sequence.
        channels:      number of input channels.
        tess_size:     tessellation resolution of the warp space (default 16).
        n_passes:      how many refinement passes to apply (default 2).
        zero_boundary: pin both endpoints — prevents global drift (default True).
        backbone:      encoder architecture name (default "InceptionTime").
        device:        "cuda" or "cpu".
    """

    def __init__(
        self,
        signal_len: int,
        channels: int,
        tess_size: int = 16,
        n_passes: int = 2,
        zero_boundary: bool = True,
        backbone: str = "InceptionTime",
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__()

        assert backbone in _BACKBONES, f"backbone must be one of {list(_BACKBONES)}"

        cpab_device  = "gpu" if "cuda" in device else "cpu"
        self.cpab    = Cpab(tess_size, backend="pytorch",
                            device=cpab_device, zero_boundary=zero_boundary)
        self.warp_dim  = self.cpab.params.d
        self.n_passes  = n_passes
        self.signal_len = signal_len
        self.device    = device
        self._n_ss     = 8   # squaring-and-scaling steps (CPAB internal)

        embed_dim = 64
        self.encoder = build_ts_model(
            arch=_BACKBONES[backbone], c_in=channels,
            c_out=embed_dim, seq_len=signal_len,
        )
        self.dropout = nn.Dropout(0.1)
        self.warp_head = nn.Linear(embed_dim, self.warp_dim)
        nn.init.normal_(self.warp_head.weight, std=1e-5)
        nn.init.normal_(self.warp_head.bias,   std=1e-5)

    # ── public API ─────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> dict:
        """
        Align a batch of sequences.

        Args:
            x: (B, C, T) — may contain NaNs for variable-length sequences.

        Returns dict with:
            "aligned"   (B, C, T)  — warped sequences
            "warp"      (B, n_passes, warp_dim)  — learned warp parameters
            "mask"      (B, C, T)  — bool, True where originally NaN
            "embedding" (B, embed_dim)  — last encoder output
        """
        B, C, T = x.shape
        has_nans = torch.any(torch.isnan(x))

        if has_nans:
            nan_mask = torch.isnan(x)
            x = torch.nan_to_num(x, nan=0.0)
        else:
            nan_mask = torch.zeros(B, C, T, dtype=torch.bool, device=x.device)

        warp = torch.zeros(B, self.n_passes, self.warp_dim, device=x.device)
        xt   = x.clone()
        for i in range(self.n_passes):
            emb      = self.encoder(xt)
            w        = self.warp_head(self.dropout(emb)).view(B, 1, self.warp_dim)
            xt       = self._warp(xt, w)
            warp[:, i, :] = w.squeeze(1)

        aligned = self._warp(x, warp)

        if has_nans:
            with torch.no_grad():
                nan_mask = self._warp(nan_mask.float(), warp).bool()

        return {"aligned": aligned, "warp": warp, "mask": nan_mask, "embedding": emb}

    def apply_warp(self, x: torch.Tensor, warp: torch.Tensor) -> torch.Tensor:
        """Apply a pre-computed warp to any sequence (e.g. frame indices, phases)."""
        return self._warp(x, warp)

    def invert(self, x: torch.Tensor, warp: torch.Tensor) -> torch.Tensor:
        """Invert a warp — recovers the original sequence from the aligned one."""
        if self.n_passes > 1:
            warp = torch.flip(warp, dims=[1])
        return self._warp(x, -warp)

    # ── internals ──────────────────────────────────────────────────────────────

    def _grid(self, n: int) -> torch.Tensor:
        return self.cpab.uniform_meshgrid(n_points=self.signal_len).repeat(n, 1)

    def _warp_grid(self, warp: torch.Tensor) -> torch.Tensor:
        B, n_passes, _ = warp.shape
        grid = self._grid(B).to(warp.device)
        out  = torch.zeros(B, self.signal_len, device=warp.device)
        for i in range(n_passes):
            w_i    = warp[:, i, :] / 2 ** self._n_ss
            grid_t = self.cpab.transform_grid_ss(grid, w_i,
                                                 method="closed_form",
                                                 N=self._n_ss, time=1)
            out   += grid_t - grid
        return out + grid

    def _warp(self, x: torch.Tensor, warp: torch.Tensor) -> torch.Tensor:
        if warp.dim() == 2:
            warp = warp.unsqueeze(1)
        grid = self._warp_grid(warp)
        B, C, _ = x.shape
        out = torch.zeros_like(x)
        for c in range(C):
            out[:, c, :] = self.cpab.interpolate(
                x[:, c, :].unsqueeze(-1), grid, outsize=self.signal_len
            ).squeeze(-1)
        return out
