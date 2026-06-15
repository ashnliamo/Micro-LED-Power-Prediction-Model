"""LED crop -> feature vector -> optical power (uW) via Ridge/RandomForest."""

import numpy as np
import cv2
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

FEATURE_NAMES = ["integrated", "peak", "area", "mean_raw", "p95", "sat_frac"]


def _blue_channel(crop_bgr):
    if crop_bgr.ndim == 2:
        return crop_bgr.astype(np.float32)
    return crop_bgr[:, :, 0].astype(np.float32)


def extract_features(crop_bgr):
    blue = _blue_channel(crop_bgr)
    h, w = blue.shape
    b = max(4, min(h, w) // 8)
    frame = np.concatenate([
        blue[:b, :].ravel(), blue[-b:, :].ravel(),
        blue[:, :b].ravel(), blue[:, -b:].ravel(),
    ])
    background = float(np.median(frame))                 # robust background estimate
    sig = np.clip(blue - background, 0.0, None)
    peak = float(np.percentile(sig, 99.9))
    thr = 0.15 * peak
    integrated = float(sig.sum())
    area = float((sig > thr).sum()) if peak > 0 else 0.0
    mean_raw = float(blue.mean())
    p95 = float(np.percentile(sig, 95))
    sat_frac = float((blue >= 255).mean())
    return np.array([integrated, peak, area, mean_raw, p95, sat_frac], np.float32)


def features_from_file(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return extract_features(img)


class PowerModel:
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
        self.pipe = Pipeline([("scale", StandardScaler()), ("reg", reg)])

    def fit(self, X, y):
        self.pipe.fit(np.asarray(X, np.float64), np.asarray(y, np.float64))
        return self

    def predict(self, X):
        return np.clip(self.pipe.predict(np.asarray(X, np.float64)), 0.0, None)

    def predict_crop(self, crop_bgr):
        return float(self.predict(extract_features(crop_bgr).reshape(1, -1))[0])

    def save(self, path):
        joblib.dump({"kind": self.kind, "feature_names": self.feature_names,
                     "pipe": self.pipe}, path)

    @staticmethod
    def load(path):
        d = joblib.load(path)
        m = PowerModel(kind=d["kind"])
        m.feature_names = d["feature_names"]
        m.pipe = d["pipe"]
        return m
