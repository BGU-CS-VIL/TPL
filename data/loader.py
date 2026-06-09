"""
Minimal data loader — expects .npy files under:
  data_path/<class_name>/<video_name>.npy

Each .npy is either:
  - a dict  (dino, openclip modalities)
  - a plain (T, D) array (pose, mesh, raw modalities)

The `modality` argument selects which representation to load.
See data/modalities.py to register custom modalities.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from data.modalities import extract


class FeatureDataset(Dataset):
    def __init__(self, data_path, modality: str = "dino"):
        self.samples = []

        class_dirs = sorted(d for d in os.listdir(data_path)
                            if os.path.isdir(os.path.join(data_path, d)))
        self.class_to_idx = {c: i for i, c in enumerate(class_dirs)}

        for cls in class_dirs:
            cls_dir = os.path.join(data_path, cls)
            for fname in sorted(os.listdir(cls_dir)):
                if not fname.endswith(".npy"):
                    continue
                raw  = np.load(os.path.join(cls_dir, fname), allow_pickle=True)
                feat = extract(modality, raw)          # (T, D)
                self.samples.append((feat.T,           # (D, T)
                                     self.class_to_idx[cls],
                                     fname))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat, label, name = self.samples[idx]
        return torch.tensor(feat), torch.tensor(label, dtype=torch.long), name


def pad_collate(batch):
    feats, labels, names = zip(*batch)
    T_max = max(f.shape[-1] for f in feats)
    padded = torch.full((len(feats), feats[0].shape[0], T_max), float("nan"))
    for i, f in enumerate(feats):
        padded[i, :, :f.shape[-1]] = f
    return padded, torch.stack(labels), list(names)


def get_loaders(data_path, modality="dino", batch_size=64, val_split=0.1, num_workers=4):
    ds = FeatureDataset(data_path, modality=modality)
    n_val = max(1, int(len(ds) * val_split))
    n_trn = len(ds) - n_val
    trn_ds, val_ds = random_split(ds, [n_trn, n_val])

    trn = DataLoader(trn_ds, batch_size=batch_size, shuffle=True,
                     collate_fn=pad_collate, num_workers=num_workers)
    val = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                     collate_fn=pad_collate, num_workers=num_workers)
    return trn, val, ds.class_to_idx
