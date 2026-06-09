import torch
import torch.nn as nn
import torch.nn.functional as F


class NaNMSE(nn.Module):
    """MSE that ignores NaN positions — safe for variable-length padded sequences."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(pred) & ~torch.isnan(target)
        diff = (pred - target) ** 2
        return diff[mask].mean()


class VAELoss(nn.Module):
    """
    Full VAE training objective:
        recon  — NaN-safe MSE between reconstruction and input
        kl     — KL divergence (β-weighted), masked to valid positions
        smooth — penalises jumps in the latent trajectory
        align  — pairwise MSE across subsampled latent trajectories
                 (pulls all sequences toward each other before ICAE)

    Args:
        beta:          KL weight (default 0.1).
        smooth_w:      smoothness weight (default 0.5).
        align_w:       pairwise alignment weight (default 1.0).
        n_align_pts:   number of evenly-spaced points used for pairwise alignment.
    """

    def __init__(self, beta: float = 0.1, smooth_w: float = 0.5,
                 align_w: float = 1.0, n_align_pts: int = 30):
        super().__init__()
        self.beta        = beta
        self.smooth_w    = smooth_w
        self.align_w     = align_w
        self.n_align_pts = n_align_pts
        self.recon       = NaNMSE()

    def forward(self, recon, target, mu, logvar, z, mask):
        """
        Args:
            recon:   (B, C, T) VAE reconstruction
            target:  (B, C, T) original input (may contain NaNs)
            mu:      (B, 1, T) latent mean
            logvar:  (B, 1, T) latent log-variance
            z:       (B, 1, T) sampled latent
            mask:    (B, C, T) bool — True where valid (non-NaN)
        """
        recon_loss = self.recon(recon, target)

        # A timestep is "valid" only when ALL channels are valid.
        # For feature modalities: NaN = padding (end of sequence only).
        # For pose/mesh: NaN = occluded joint — a frame with any occluded
        # joint is treated as partially valid; we use all() so the latent
        # z is only computed from fully-observed frames.
        mask_1d  = mask.all(dim=1, keepdim=True)                     # (B, 1, T)
        kl       = 1 + logvar - mu.pow(2) - logvar.exp()
        kl       = torch.where(mask_1d, kl, torch.zeros_like(kl))
        kl_loss  = -0.5 * kl.sum() / mask_1d.sum().clamp(min=1)

        z_diff      = z[:, :, 1:] - z[:, :, :-1]
        valid_pairs = mask_1d[:, :, 1:] & mask_1d[:, :, :-1]
        smooth_loss = torch.where(valid_pairs, z_diff ** 2,
                                  torch.zeros_like(z_diff))
        smooth_loss = smooth_loss.sum() / valid_pairs.sum().clamp(min=1)

        z_sub       = self._subsample(z, mask_1d)
        align_loss  = self._pairwise_mse(z_sub)

        total = (recon_loss
                 + self.beta     * kl_loss
                 + self.smooth_w * smooth_loss
                 + self.align_w  * align_loss)
        return total, recon_loss, kl_loss, smooth_loss, align_loss

    def _subsample(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Uniformly sample n_align_pts valid timesteps per sequence. (B, n_pts)"""
        B, _, T = z.shape
        z_sq = z.squeeze(1)
        out  = torch.zeros(B, self.n_align_pts, device=z.device)
        for b in range(B):
            valid_idx = mask[b, 0].nonzero(as_tuple=False).squeeze(1)
            if len(valid_idx) == 0:
                continue
            n    = min(self.n_align_pts, len(valid_idx))
            idx  = valid_idx[torch.linspace(0, len(valid_idx) - 1, n,
                                            dtype=torch.long, device=z.device)]
            out[b, :n] = z_sq[b, idx]
        return out

    @staticmethod
    def _pairwise_mse(z_sub: torch.Tensor) -> torch.Tensor:
        B = z_sub.size(0)
        if B < 2:
            return torch.tensor(0.0, device=z_sub.device)
        loss, count = 0.0, 0
        for i in range(B):
            for j in range(i + 1, B):
                loss  += F.mse_loss(z_sub[i], z_sub[j])
                count += 1
        return loss / count


class ICAE(nn.Module):
    """
    Inverse Consistency Averaging Error.

    Prototype = mean of the `n_ref` aligned sequences whose valid length is
    closest to the batch median. The prototype is inverse-warped back to the
    original domain and compared against each unaligned sequence.

    Args:
        n_ref: sequences averaged into the prototype (default 3).
               1 = strict median, higher = smoother reference.
    """

    def __init__(self, n_ref: int = 3):
        super().__init__()
        self.n_ref = n_ref
        self.loss  = NaNMSE()

    def forward(self, X, Xt, y_true, warp, model, mask, **kwargs):
        """
        Args:
            X:      (B, C, T) original (unaligned) sequences
            Xt:     (B, C, T) aligned sequences
            y_true: (B,) class labels
            warp:   (B, n_passes, warp_dim) learned warp parameters
            model:  TemporalAligner — used only for .invert()
            mask:   (B, C, T) bool — True where NaN
        """
        total = 0.0
        for cls in torch.unique(y_true):
            sel    = y_true == cls
            Xt_k   = Xt[sel]
            X_k    = X[sel]
            mask_k = mask[sel]

            ref       = self._prototype(Xt_k, mask_k)
            ref_stack = ref.unsqueeze(0).expand(Xt_k.shape[0], -1, -1)
            ref_inv   = model.invert(ref_stack, warp[sel])
            total    += self.loss(ref_inv, X_k)
        return total

    def _prototype(self, Xt_k: torch.Tensor, mask_k: torch.Tensor) -> torch.Tensor:
        # valid_len: number of fully-valid timesteps per sample
        # mask_k is True=NaN, so ~mask_k = valid; .all(dim=1) = all channels valid
        valid_len  = (~mask_k).all(dim=1).sum(dim=-1).float()    # (N,)
        dist       = torch.abs(valid_len - torch.median(valid_len))
        n          = min(self.n_ref, len(Xt_k))
        idx        = torch.topk(dist, k=n, largest=False).indices
        return Xt_k[idx].mean(dim=0)                             # (C, T)
