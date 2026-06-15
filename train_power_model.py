"""
Train the LED Power Model
=========================
Fits power_model.PowerModel on the crops you produced with
extract_training_crops.py, and evaluates it honestly.

Pipeline:
  1. Discover every array folder inside Training Crops/ and load its labelled
     crops (from each folder's labels.csv, or by parsing '<label>_<power>uW.png'
     filenames). Works with any number of array folders - just add or remove
     folders. Nested folders (e.g. a 'temp not using' container) are ignored.
  2. Extract features for every crop (power_model.extract_features).
  3. Evaluate with whatever CV schemes the data supports:
       * random k-fold            - how well it fits LEDs like those it saw
       * leave-one-ARRAY-out      - how well it generalises to a NEW array
         (only when >= 2 array folders are present; the honest test)
  4. Compare against a baseline:
       * integrated x scalar - single best physical feature, through origin
  5. Refit on ALL data and save the model + calibration / residual plots.

Outputs land in  Model/ :
    power_model.joblib              the trained model (load with PowerModel.load)
    metrics.csv                     MAE / RMSE / R2 for every method & scheme
    predictions_cv.csv              per-LED out-of-fold predictions
    calibration_cv.png              predicted vs measured (out-of-fold)
    residuals_by_power.png          residual vs measured power
    feature_importance.png          model coefficients / importances

This script also runs the crop-extraction step first: it rebuilds the labelled
crops in Training Crops/ from the raw arrays in Training data/ (image + LIV
spreadsheet), then trains on them. Pass --no-extract to skip that and train on
whatever crops already exist.

Usage:
    py train_power_model.py                 # extract crops + train RandomForest (default)
    py train_power_model.py ridge           # use Ridge instead of RandomForest
    py train_power_model.py --no-extract     # skip extraction, train on existing crops
    py train_power_model.py rf --no-extract  # combine flags in any order

Requirements:
    pip install opencv-python numpy scikit-learn matplotlib joblib openpyxl
"""

import os
import sys
import csv
import glob
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, LeaveOneGroupOut, cross_val_predict

from power_model import PowerModel, features_from_file, FEATURE_NAMES
import extract_training_crops as E
from extract_training_crops import pad_label

# CONFIG
BASE          = os.path.dirname(os.path.abspath(__file__))   # portable: this script's folder
CROPS_ROOT    = os.path.join(BASE, "Training Crops")
OUTPUT_FOLDER = os.path.join(BASE, "Model")

# Args (order-independent): a model kind ("ridge"/"rf") and an optional --no-extract flag
_args         = [a.lower() for a in sys.argv[1:]]
MODEL_KIND    = next((a for a in _args if a in ("ridge", "rf")), "rf")
SKIP_EXTRACT  = any(a in ("--no-extract", "--skip-extract") for a in _args)


# --------------------------------------------------------------------------- #
#  Data loading                                                                #
# --------------------------------------------------------------------------- #
def _power_from_filename(fname):
    """'C27_43.451uW.png' -> ('C27', 43.451). Returns None if it doesn't match."""
    stem = os.path.splitext(os.path.basename(fname))[0]
    if "_" not in stem or not stem.endswith("uW"):
        return None
    label, power = stem.rsplit("_", 1)
    try:
        return label, float(power[:-2])            # drop trailing 'uW'
    except ValueError:
        return None


def _read_array_folder(folder):
    """Yield (label, power_uW, crop_path) for one array folder.

    Prefers the folder's labels.csv; falls back to parsing crop filenames so
    any folder of '<label>_<power>uW.png' crops works on its own."""
    name = os.path.basename(folder.rstrip("\\/"))
    labels_csv = os.path.join(folder, "labels.csv")
    seen = False
    if os.path.exists(labels_csv):
        with open(labels_csv, newline="") as f:
            for r in csv.DictReader(f):
                crop = os.path.join(folder, os.path.basename(
                    r["crop_file"].replace("\\", "/")))
                if os.path.exists(crop):
                    seen = True
                    yield r["label"], float(r["Power_uW"]), crop
        if seen:
            return
    # no usable labels.csv -> parse filenames
    for crop in sorted(glob.glob(os.path.join(folder, "*.png"))):
        parsed = _power_from_filename(crop)
        if parsed:
            yield parsed[0], parsed[1], crop


def load_dataset():
    """Discover every array folder in CROPS_ROOT and load its labelled crops.

    An "array folder" is any immediate sub-folder that yields labelled crops
    (via labels.csv or '<label>_<power>uW.png' files). Folders without crops
    (e.g. a 'temp not using' container, whose array folders are one level
    deeper) are ignored, so moving folders in/out curates the training set.
    Returns X (NxF), y (N,), groups (array name per row), meta (list of dicts).
    """
    X, y, groups, meta = [], [], [], []
    found = []
    for folder in sorted(glob.glob(os.path.join(CROPS_ROOT, "*"))):
        if not os.path.isdir(folder):
            continue
        name = os.path.basename(folder)
        n0 = len(y)
        for label, power, crop in _read_array_folder(folder):
            X.append(features_from_file(crop))
            y.append(power)
            groups.append(name)
            meta.append({"array": name, "label": label})
        if len(y) > n0:
            found.append(f"{name} ({len(y) - n0})")
    if not found:
        raise FileNotFoundError(
            f"No array folders with crops found in {CROPS_ROOT}")
    print("Array folders found: " + ", ".join(found) + "\n")
    return np.array(X, np.float64), np.array(y, np.float64), np.array(groups), meta


# --------------------------------------------------------------------------- #
#  Metrics                                                                     #
# --------------------------------------------------------------------------- #
def metrics(y_true, y_pred):
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mae, rmse, r2


def scalar_through_origin(x, y):
    """Best k for y ~ k*x (least squares, no intercept)."""
    x = np.asarray(x, np.float64)
    denom = float(np.sum(x * x))
    return float(np.sum(x * y) / denom) if denom > 0 else 0.0


def cv_predict_scalar(x, y, splitter, groups=None):
    """Out-of-fold predictions for the single-feature 'k*x' baselines."""
    pred = np.zeros_like(y)
    for tr, te in splitter.split(x.reshape(-1, 1), y, groups):
        k = scalar_through_origin(x[tr], y[tr])
        pred[te] = k * x[te]
    return pred


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #
def _extract_crops():
    """Rebuild Training Crops/ from Training data/ (image + LIV sheet per array).

    Non-fatal: if there's no Training data/ to extract from, we just train on
    whatever crops already exist."""
    print("=" * 68)
    print(f"STEP 1/2  Extracting labelled crops from {E.TRAINING_DIR}")
    print("=" * 68)
    try:
        E.main()
    except FileNotFoundError as exc:
        print(f"  (no raw arrays to extract - {exc})")
        print("  -> training on existing Training Crops/ instead")
    print()


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    if SKIP_EXTRACT:
        print("(--no-extract: skipping crop extraction, using existing Training Crops/)\n")
    else:
        _extract_crops()

    print("=" * 68)
    print(f"STEP 2/2  Training power model ({MODEL_KIND})")
    print("=" * 68)
    X, y, groups, meta = load_dataset()
    n, f = X.shape
    arrays = sorted(set(groups))
    print(f"Loaded {n} labelled crops across {len(arrays)} arrays: {', '.join(arrays)}")
    print(f"Features: {FEATURE_NAMES}")
    print(f"Power range: {y.min():.3f} - {y.max():.3f} uW  (mean {y.mean():.2f})\n")
    print(f"Model: {MODEL_KIND}\n")

    # feature column index for the single-feature baseline
    fi_int = FEATURE_NAMES.index("integrated")

    # Build whatever CV schemes the data can actually support:
    #   * k-fold needs >= 2 samples (k capped at sample count, max 5)
    #   * leave-one-array-out needs >= 2 arrays
    schemes = {}
    if n >= 2:
        k = min(5, n)
        schemes[f"{k}-fold (random)"] = (KFold(n_splits=k, shuffle=True,
                                               random_state=0), None)
    if len(arrays) >= 2:
        schemes["leave-one-array-out"] = (LeaveOneGroupOut(), groups)
    if not schemes:
        raise ValueError(f"Need at least 2 labelled crops to evaluate (have {n}).")

    results = []          # rows for metrics.csv
    diag_pred = None      # out-of-fold model predictions used for plots/CSV
    diag_scheme = None

    print(f"{'method':<22}{'scheme':<22}{'MAE':>8}{'RMSE':>8}{'R2':>8}")
    print("-" * 68)
    for scheme_name, (splitter, grp) in schemes.items():
        # --- baseline (single best physical feature, through origin) ---
        for feat_name, fi in (("integrated x k", fi_int),):
            pred = cv_predict_scalar(X[:, fi], y, splitter, grp)
            mae, rmse, r2 = metrics(y, pred)
            results.append([feat_name, scheme_name, mae, rmse, r2])
            print(f"{feat_name:<22}{scheme_name:<22}{mae:>8.3f}{rmse:>8.3f}{r2:>8.3f}")

        # --- the model (all features) ---
        model = PowerModel(kind=MODEL_KIND)
        pred = cross_val_predict(model.pipe, X, y,
                                 cv=splitter.split(X, y, grp))
        pred = np.clip(pred, 0.0, None)
        mae, rmse, r2 = metrics(y, pred)
        results.append([f"model ({MODEL_KIND})", scheme_name, mae, rmse, r2])
        print(f"{'model (' + MODEL_KIND + ')':<22}{scheme_name:<22}{mae:>8.3f}{rmse:>8.3f}{r2:>8.3f}")
        # prefer leave-one-array-out for the diagnostics; else fall back to k-fold
        if diag_pred is None or scheme_name == "leave-one-array-out":
            diag_pred, diag_scheme = pred, scheme_name
        print("-" * 68)

    # ---- refit on ALL data and save ----
    final = PowerModel(kind=MODEL_KIND).fit(X, y)
    model_path = os.path.join(OUTPUT_FOLDER, "power_model.joblib")
    final.save(model_path)

    # ---- write metrics.csv ----
    with open(os.path.join(OUTPUT_FOLDER, "metrics.csv"), "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["method", "cv_scheme", "MAE_uW", "RMSE_uW", "R2"])
        for row in results:
            w.writerow([row[0], row[1], f"{row[2]:.4f}", f"{row[3]:.4f}", f"{row[4]:.4f}"])

    # ---- per-LED out-of-fold predictions (from the diagnostic CV scheme) ----
    with open(os.path.join(OUTPUT_FOLDER, "predictions_cv.csv"), "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["array", "label", "measured_uW", "predicted_uW", "abs_error_uW"])
        for m, yt, yp in zip(meta, y, diag_pred):
            w.writerow([m["array"], pad_label(m["label"]), f"{yt:.3f}",
                        f"{yp:.3f}", f"{abs(yp - yt):.3f}"])

    # ---- plots ----
    _plot_calibration(y, diag_pred, groups, diag_scheme,
                      os.path.join(OUTPUT_FOLDER, "calibration_cv.png"))
    _plot_residuals(y, diag_pred, diag_scheme,
                    os.path.join(OUTPUT_FOLDER, "residuals_by_power.png"))
    _plot_importance(final, X, y,
                     os.path.join(OUTPUT_FOLDER, "feature_importance.png"))

    print(f"\nSaved model -> {model_path}")
    print(f"Artifacts (metrics, predictions, plots) in: {OUTPUT_FOLDER}")
    print(f"\nDiagnostics use the '{diag_scheme}' scheme.")
    if "leave-one-array-out" in schemes:
        print("Read 'leave-one-array-out' as the realistic accuracy on a brand-new array.")
        print("If those R2/MAE numbers are weak, it usually means exposure differs")
        print("between arrays - capture a fixed reference or normalise per array.")
    else:
        print("Only one array present, so there's no cross-array (leave-one-array-out)")
        print("estimate - add another array folder to test generalisation.")


def _plot_calibration(y, pred, groups, scheme, path):
    plt.figure(figsize=(6, 6))
    for g in sorted(set(groups)):
        mask = groups == g
        plt.scatter(y[mask], pred[mask], s=28, alpha=0.8, label=g)
    lim = max(float(y.max()), float(pred.max())) * 1.05
    plt.plot([0, lim], [0, lim], "k--", lw=1, label="ideal")
    plt.xlabel("measured power (uW)")
    plt.ylabel("predicted power (uW)")
    plt.title(f"Calibration - {scheme} (out-of-fold)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def _plot_residuals(y, pred, scheme, path):
    plt.figure(figsize=(7, 4))
    plt.axhline(0, color="k", lw=1)
    plt.scatter(y, pred - y, s=24, alpha=0.8)
    plt.xlabel("measured power (uW)")
    plt.ylabel("residual: predicted - measured (uW)")
    plt.title(f"Residuals vs measured power ({scheme})")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def _plot_importance(model, X, y, path):
    reg = model.pipe.named_steps["reg"]
    plt.figure(figsize=(7, 4))
    if hasattr(reg, "coef_"):
        vals = reg.coef_                       # on standardised features
        title = "Ridge coefficients (standardised features)"
    else:
        vals = reg.feature_importances_
        title = "RandomForest feature importances"
    order = np.argsort(np.abs(vals))[::-1]
    names = [model.feature_names[i] for i in order]
    plt.bar(range(len(vals)), np.array(vals)[order])
    plt.xticks(range(len(vals)), names, rotation=30, ha="right")
    plt.ylabel("weight")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
