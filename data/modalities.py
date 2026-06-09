"""
Modality registry — maps a modality name to an extraction function.

Each extractor receives the loaded npy content (either a dict or a raw array)
and returns a float32 numpy array of shape (T, D).

The loader transposes to (D, T) for the model. NaN encodes invalid positions.

NaN semantics per modality type:
    feature-based (hl_features, raw):
        NaN = missing frames — padding at end only.
    spatial (pose, mesh):
        NaN = occluded / missing — scattered across (joint, frame).
    keypoints (kp2d, kp3d):
        Confidence / visibility stripped and converted to NaN mask on coords.
        kp2d: input (T, J, 2) or (T, J, 3) where dim-2 is confidence.
        kp3d: input (T, J, 3) or (T, J, 4) where dim-3 is confidence.
        Low-confidence joints → coords set to NaN.

Built-in modalities:
    hl_features — any high-level feature dict (auto-detects first *_features key)
    pose        — plain spatial array, auto-flat    (T, J*D)
    mesh        — plain spatial array, auto-flat    (T, V*3)
    raw         — plain array, used as-is           (T, D)
    kp2d        — 2-D keypoints + optional conf     (T, J*2)
    kp3d        — 3-D keypoints + optional conf     (T, J*3)

Adding a custom modality:
    from data.modalities import register
    register("smpl", lambda data: data["smpl_params"])         # dict key
    register("kp2d_custom", keypoints_2d(conf_threshold=0.3)) # custom conf
"""

import numpy as np
from typing import Callable

_REGISTRY: dict[str, Callable] = {}


def register(name: str, fn: Callable):
    _REGISTRY[name] = fn


def extract(name: str, raw) -> np.ndarray:
    """
    raw  : result of np.load(..., allow_pickle=True)
    Returns (T, D) float32 — spatial/joint dims flattened into D.
                             NaN marks invalid / occluded positions.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown modality '{name}'. Available: {sorted(_REGISTRY)}"
        )
    data = raw.item() if (isinstance(raw, np.ndarray) and raw.dtype == object) else raw
    feat = _REGISTRY[name](data).astype(np.float32)

    if feat.ndim == 3:
        feat = feat.reshape(feat.shape[0], -1)      # (T, J, C) → (T, J*C)
    elif feat.ndim != 2:
        raise ValueError(
            f"Modality '{name}' returned shape {feat.shape}; expected 2-D or 3-D."
        )
    return feat   # (T, D)


# ── keypoint helpers ───────────────────────────────────────────────────────────

def keypoints_2d(conf_threshold: float = 0.3) -> Callable:
    """
    Factory for 2-D keypoint extractor.

    Accepts arrays of shape:
        (T, J, 2)  — xy only, no confidence
        (T, J, 3)  — xy + confidence score in dim-2

    Low-confidence joints (conf < threshold) → xy set to NaN.
    Returns (T, J*2) float32.
    """
    def _extract(data: np.ndarray) -> np.ndarray:
        if isinstance(data, dict):
            raise TypeError("kp2d expects a plain array, not a dict.")
        arr = np.array(data, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]           # (T, J) → (T, J, 1) — unusual but safe
        T, J, C = arr.shape
        if C == 3:
            xy   = arr[:, :, :2].copy()           # (T, J, 2)
            conf = arr[:, :, 2]                   # (T, J)
            xy[conf < conf_threshold] = np.nan
        elif C == 2:
            xy = arr.copy()
        else:
            raise ValueError(f"kp2d: expected C=2 or 3, got {C}.")
        return xy.reshape(T, J * 2)               # (T, J*2)
    return _extract


def keypoints_3d(conf_threshold: float = 0.3) -> Callable:
    """
    Factory for 3-D keypoint extractor.

    Accepts arrays of shape:
        (T, J, 3)  — xyz only
        (T, J, 4)  — xyz + confidence in dim-3

    Low-confidence joints → xyz set to NaN.
    Returns (T, J*3) float32.
    """
    def _extract(data: np.ndarray) -> np.ndarray:
        if isinstance(data, dict):
            raise TypeError("kp3d expects a plain array, not a dict.")
        arr = np.array(data, dtype=np.float32)
        T, J, C = arr.shape
        if C == 4:
            xyz  = arr[:, :, :3].copy()
            conf = arr[:, :, 3]
            xyz[conf < conf_threshold] = np.nan
        elif C == 3:
            xyz = arr.copy()
        else:
            raise ValueError(f"kp3d: expected C=3 or 4, got {C}.")
        return xyz.reshape(T, J * 3)              # (T, J*3)
    return _extract


def _plain(data) -> np.ndarray:
    if isinstance(data, dict):
        raise TypeError(
            f"raw/pose/mesh expects a plain array .npy "
            f"(got dict with keys {list(data.keys())})."
        )
    return data


# ── hl_features: generic high-level feature extractor ─────────────────────────

def _hl_features(data) -> np.ndarray:
    """
    Reads from any dict-based .npy file that contains a *_features key.
    Picks the first key whose name ends with '_features'.
    Override by registering a custom 'hl_features' before loading data.
    """
    if not isinstance(data, dict):
        raise TypeError(
            "hl_features expects a dict .npy (produced by a feature extractor). "
            "For plain arrays use 'raw', 'pose', or 'mesh'."
        )
    feature_keys = [k for k in data if k.endswith("_features")]
    if not feature_keys:
        raise KeyError(
            f"No '*_features' key found in dict. Available keys: {list(data.keys())}"
        )
    return data[feature_keys[0]]


# ── built-in registrations ─────────────────────────────────────────────────────

register("hl_features", _hl_features)
register("pose",        _plain)
register("mesh",        _plain)
register("raw",         _plain)
register("kp2d",        keypoints_2d(conf_threshold=0.3))
register("kp3d",        keypoints_3d(conf_threshold=0.3))
