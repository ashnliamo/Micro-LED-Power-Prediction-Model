"""
LED Power Model
===============
The "model" half of the power-prediction pipeline.

It turns one cropped LED image into a small vector of physics-motivated
features, then maps those features to optical power (uW) with a simple,
interpretable regressor (Ridge by default; RandomForest optional).

Why features + a small regressor instead of a CNN?
  With ~200 labelled crops from a handful of arrays, a CNN would overfit and
  could "cheat" on position. A few hand-built features (chiefly the integrated
  intensity, which is physically proportional to radiant flux) give an
  interpretable model that trains in milliseconds and is easy to sanity-check.

Used by train_power_model.py (to fit/evaluate) and can be loaded later for
inference on new crops.

Requirements:
    pip install opencv-python numpy scikit-learn joblib
"""

import numpy as np
import cv2
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor


# Order matters: this is the column order of every feature vector.
FEATURE_NAMES = ["integrated", "peak", "area", "mean_raw", "p95", "sat_frac"]


def _blue_channel(crop_bgr):
    """Blue channel as float32. cv2 loads BGR, so blue is channel 0.

    The LEDs are blue and the brightness meter reads the blue channel, so we
    keep the model on the same physical quantity."""
    if crop_bgr.ndim == 2:
        return crop_bgr.astype(np.float32)
    return crop_bgr[:, :, 0].astype(np.float32)


def extract_features(crop_bgr):
    """One crop -> feature vector (len == len(FEATURE_NAMES)).

    Background is estimated from the crop's border frame (the LED sits in the
    centre), then subtracted so uneven illumination / neighbour glow doesn't
    leak in. The dominant feature is `integrated`: the background-subtracted
    intensity summed over the crop, which is the natural proxy for power.
    """
    blue = _blue_channel(crop_bgr)
    h, w = blue.shape
    b = max(4, min(h, w) // 8)                      # border frame width

    # background = median of the outer frame (robust to a few stray bright px)
    frame = np.concatenate([
        blue[:b, :].ravel(), blue[-b:, :].ravel(),
        blue[:, :b].ravel(), blue[:, -b:].ravel(),
    ])
    background = float(np.median(frame))

    sig = np.clip(blue - background, 0.0, None)      # background-subtracted signal
    peak = float(np.percentile(sig, 99.9))           # robust peak (ignore 1 hot px)
    thr = 0.15 * peak                                # "lit" threshold

    integrated = float(sig.sum())
    area = float((sig > thr).sum()) if peak > 0 else 0.0
    mean_raw = float(blue.mean())                    # analogue of the old meter
    p95 = float(np.percentile(sig, 95))
    sat_frac = float((blue >= 255).mean())           # saturation diagnostic

    return np.array([integrated, peak, area, mean_raw, p95, sat_frac], np.float32)


def features_from_file(path):
    """Read a crop PNG from disk (unicode-safe) and extract its features."""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return extract_features(img)


class PowerModel:
    """Thin wrapper around a scikit-learn pipeline: scale features -> regress power."""

    def __init__(self, kind="ridge", alpha=1.0, n_estimators=300, random_state=0):
        self.kind = kind
        self.feature_names = list(FEATURE_NAMES)
        if kind == "ridge":
            reg = Ridge(alpha=alpha)
        elif kind == "rf":
            reg = RandomForestRegressor(n_estimators=n_estimators,
                                        random_state=random_state, n_jobs=-1)
        else:
            raise ValueError(f"unknown model kind: {kind!r}")
        # scaling is a no-op for the tree but harmless; keeps one code path
        self.pipe = Pipeline([("scale", StandardScaler()), ("reg", reg)])

    def fit(self, X, y):
        self.pipe.fit(np.asarray(X, np.float64), np.asarray(y, np.float64))
        return self

    def predict(self, X):
        p = self.pipe.predict(np.asarray(X, np.float64))
        return np.clip(p, 0.0, None)                 # power can't be negative

    def predict_crop(self, crop_bgr):
        return float(self.predict(extract_features(crop_bgr).reshape(1, -1))[0])

    # ---- persistence -----------------------------------------------------
    def save(self, path):
        joblib.dump({"kind": self.kind,
                     "feature_names": self.feature_names,
                     "pipe": self.pipe}, path)

    @staticmethod
    def load(path):
        d = joblib.load(path)
        m = PowerModel(kind=d["kind"])
        m.feature_names = d["feature_names"]
        m.pipe = d["pipe"]
        return m
