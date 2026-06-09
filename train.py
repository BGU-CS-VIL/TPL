"""
TPL-2026 — training entry point

Usage:
    conda activate rfdtan_new
    export CUDA_HOME=$HOME/miniconda3/envs/rfdtan_new

    python train.py                                      # defaults
    python train.py train.epochs=500 loss.n_ref=5        # overrides
    python train.py data.path=/path/to/features          # custom data
"""

import os
import torch
import torch.optim as optim
import hydra
from omegaconf import DictConfig

from models.vae     import FeatureVAE
from models.aligner import TemporalAligner
from losses.tpl     import VAELoss, ICAE
from data.loader    import get_loaders
from data.augment   import CpabAugment, AffineAugment


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.train.save_dir, exist_ok=True)
    print(f"device={device}")

    # ── data ──────────────────────────────────────────────────────────────────
    trn_loader, _, class_map = get_loaders(
        cfg.data.path,
        modality=cfg.data.modality,
        batch_size=cfg.data.batch_size,
        val_split=cfg.data.val_split,
        num_workers=cfg.data.num_workers,
    )
    sample_feat, _, _ = next(iter(trn_loader))
    input_channels = sample_feat.shape[1]
    signal_len     = sample_feat.shape[2]
    print(f"channels={input_channels}  signal_len={signal_len}  classes={class_map}")

    # ── models ────────────────────────────────────────────────────────────────
    vae = FeatureVAE(
        input_channels=input_channels,
        bottleneck_channels=cfg.model.bottleneck,
        signal_len=signal_len,
    ).to(device)

    aligner = TemporalAligner(
        signal_len=signal_len,
        channels=cfg.model.bottleneck,
        tess_size=cfg.model.tess_size,
        n_passes=cfg.model.n_recurrences,
        device=str(device),
    ).to(device)

    # ── augmentation ──────────────────────────────────────────────────────────
    cpab_aug = CpabAugment(
        tess_size=cfg.model.tess_size,
        scale=cfg.augment.cpab.scale,
        zero_boundary=cfg.augment.cpab.zero_boundary,
        device=str(device),
    ) if cfg.augment.cpab.enabled else None

    affine_aug = AffineAugment(
        scale_range=cfg.augment.affine.scale_range,
        shift_frac=cfg.augment.affine.shift_frac,
        amplitude=cfg.augment.affine.amplitude,
        p=cfg.augment.affine.p,
    ) if cfg.augment.affine.enabled else None

    # ── optimizers + losses ───────────────────────────────────────────────────
    opt_vae   = optim.Adam(vae.parameters(),
                           lr=cfg.train.lr,
                           weight_decay=cfg.train.weight_decay_vae)
    opt_align = optim.AdamW(aligner.parameters(),
                            lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay_aligner,
                            eps=1e-8, amsgrad=True)

    vae_loss  = VAELoss(
        beta=cfg.loss.beta,
        smooth_w=cfg.loss.smooth_w,
        align_w=cfg.loss.align_w,
    )
    icae_loss = ICAE(n_ref=cfg.loss.n_ref)

    # ── train loop ────────────────────────────────────────────────────────────
    for epoch in range(1, cfg.train.epochs + 1):
        vae.train(); aligner.train()
        total, v_sum, r_sum, a_sum = 0.0, 0.0, 0.0, 0.0

        for x, y, _ in trn_loader:
            x = x.to(device)
            y = y.to(device)
            if affine_aug is not None:
                x = affine_aug(x)
            if cpab_aug is not None:
                x = cpab_aug(x)

            mask = ~torch.isnan(x)
            x_in = torch.nan_to_num(x, nan=0.0)

            opt_vae.zero_grad(); opt_align.zero_grad()

            recon, mu, logvar, z = vae(x_in)

            v, recon_l, _kl, _smooth, _align = vae_loss(
                recon=recon, target=x, mu=mu, logvar=logvar, z=z, mask=mask
            )

            nan_mask_1d = mask.any(dim=1, keepdim=True)
            z_masked    = z.masked_fill(~nan_mask_1d, float("nan"))
            aligned     = aligner(z_masked)

            a   = icae_loss(
                X=z_masked, Xt=aligned["aligned"], y_true=y,
                warp=aligned["warp"], model=aligner, mask=aligned["mask"],
            )
            lam  = torch.sigmoid(torch.tensor(
                cfg.train.align_schedule_k * (epoch - cfg.train.align_schedule_t0),
                dtype=torch.float32,
            ))
            loss = v + lam * a

            loss.backward()
            opt_vae.step(); opt_align.step()

            total += loss.item(); v_sum += v.item()
            r_sum += recon_l.item(); a_sum += a.item()

        n = len(trn_loader)
        if epoch % cfg.train.log_every == 0 or epoch == 1:
            print(f"[{epoch:4d}/{cfg.train.epochs}] "
                  f"loss={total/n:.4f}  vae={v_sum/n:.4f}  "
                  f"recon={r_sum/n:.4f}  icae={a_sum/n:.4f}")

    # ── save ──────────────────────────────────────────────────────────────────
    out = os.path.join(cfg.train.save_dir, "checkpoint.pt")
    torch.save({"vae": vae.state_dict(), "aligner": aligner.state_dict()}, out)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
