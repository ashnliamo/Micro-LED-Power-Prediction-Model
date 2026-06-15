"""Extract crops (from Training data/) then fit + evaluate the power model.

  py train_power_model.py                 # extract + train RandomForest (default)
  py train_power_model.py ridge           # use Ridge
  py train_power_model.py --no-extract     # train on existing Training Crops/
Flags combine in any order. Outputs (model, metrics, plots) land in Model/.
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

BASE          = os.path.dirname(os.path.abspath(__file__))
CROPS_ROOT    = os.path.join(BASE, "Training Crops")
OUTPUT_FOLDER = os.path.join(BASE, "Model")

_args        = [a.lower() for a in sys.argv[1:]]
MODEL_KIND   = next((a for a in _args if a in ("ridge", "rf")), "rf")
SKIP_EXTRACT = any(a in ("--no-extract", "--skip-extract") for a in _args)


def _power_from_filename(fname):
    """'C27_43.451uW.png' -> ('C27', 43.451), or None."""
    stem = os.path.splitext(os.path.basename(fname))[0]
    if "_" not in stem or not stem.endswith("uW"):
        return None
    label, power = stem.rsplit("_", 1)
    try:
        return label, float(power[:-2])
    except ValueError:
        return None


def _read_array_folder(folder):
    """Yield (label, power_uW, crop_path) from labels.csv, else from filenames."""
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
    for crop in sorted(glob.glob(os.path.join(folder, "*.png"))):
        parsed = _power_from_filename(crop)
        if parsed:
            yield parsed[0], parsed[1], crop


def load_dataset():
    """Load every array sub-folder of CROPS_ROOT -> X, y, groups, meta."""
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
        raise FileNotFoundError(f"No array folders with crops found in {CROPS_ROOT}")
    print("Array folders found: " + ", ".join(found) + "\n")
    return np.array(X, np.float64), np.array(y, np.float64), np.array(groups), meta


def metrics(y_true, y_pred):
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mae, rmse, r2


def scalar_through_origin(x, y):
    x = np.asarray(x, np.float64)
    denom = float(np.sum(x * x))
    return float(np.sum(x * y) / denom) if denom > 0 else 0.0


def cv_predict_scalar(x, y, splitter, groups=None):
    """Out-of-fold predictions for the 'k*x' baseline."""
    pred = np.zeros_like(y)
    for tr, te in splitter.split(x.reshape(-1, 1), y, groups):
        k = scalar_through_origin(x[tr], y[tr])
        pred[te] = k * x[te]
    return pred


def _extract_crops():
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
    n = len(y)
    arrays = sorted(set(groups))
    print(f"Loaded {n} labelled crops across {len(arrays)} arrays: {', '.join(arrays)}")
    print(f"Features: {FEATURE_NAMES}")
    print(f"Power range: {y.min():.3f} - {y.max():.3f} uW  (mean {y.mean():.2f})\n")
    print(f"Model: {MODEL_KIND}\n")

    fi_int = FEATURE_NAMES.index("integrated")

    # k-fold needs >= 2 samples; leave-one-array-out needs >= 2 arrays
    schemes = {}
    if n >= 2:
        k = min(5, n)
        schemes[f"{k}-fold (random)"] = (KFold(n_splits=k, shuffle=True,
                                               random_state=0), None)
    if len(arrays) >= 2:
        schemes["leave-one-array-out"] = (LeaveOneGroupOut(), groups)
    if not schemes:
        raise ValueError(f"Need at least 2 labelled crops to evaluate (have {n}).")

    results = []
    diag_pred = diag_scheme = None

    print(f"{'method':<22}{'scheme':<22}{'MAE':>8}{'RMSE':>8}{'R2':>8}")
    print("-" * 68)
    for scheme_name, (splitter, grp) in schemes.items():
        pred = cv_predict_scalar(X[:, fi_int], y, splitter, grp)        # baseline
        mae, rmse, r2 = metrics(y, pred)
        results.append(["integrated x k", scheme_name, mae, rmse, r2])
        print(f"{'integrated x k':<22}{scheme_name:<22}{mae:>8.3f}{rmse:>8.3f}{r2:>8.3f}")

        model = PowerModel(kind=MODEL_KIND)
        pred = np.clip(cross_val_predict(model.pipe, X, y, cv=splitter.split(X, y, grp)),
                       0.0, None)
        mae, rmse, r2 = metrics(y, pred)
        results.append([f"model ({MODEL_KIND})", scheme_name, mae, rmse, r2])
        print(f"{'model (' + MODEL_KIND + ')':<22}{scheme_name:<22}{mae:>8.3f}{rmse:>8.3f}{r2:>8.3f}")
        if diag_pred is None or scheme_name == "leave-one-array-out":
            diag_pred, diag_scheme = pred, scheme_name
        print("-" * 68)

    final = PowerModel(kind=MODEL_KIND).fit(X, y)
    model_path = os.path.join(OUTPUT_FOLDER, "power_model.joblib")
    final.save(model_path)

    with open(os.path.join(OUTPUT_FOLDER, "metrics.csv"), "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["method", "cv_scheme", "MAE_uW", "RMSE_uW", "R2"])
        for row in results:
            w.writerow([row[0], row[1], f"{row[2]:.4f}", f"{row[3]:.4f}", f"{row[4]:.4f}"])

    with open(os.path.join(OUTPUT_FOLDER, "predictions_cv.csv"), "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["array", "label", "measured_uW", "predicted_uW", "abs_error_uW"])
        for m, yt, yp in zip(meta, y, diag_pred):
            w.writerow([m["array"], pad_label(m["label"]), f"{yt:.3f}",
                        f"{yp:.3f}", f"{abs(yp - yt):.3f}"])

    _plot_calibration(y, diag_pred, groups, diag_scheme,
                      os.path.join(OUTPUT_FOLDER, "calibration_cv.png"))
    _plot_residuals(y, diag_pred, diag_scheme,
                    os.path.join(OUTPUT_FOLDER, "residuals_by_power.png"))
    _plot_importance(final, os.path.join(OUTPUT_FOLDER, "feature_importance.png"))

    print(f"\nSaved model -> {model_path}")
    print(f"Artifacts (metrics, predictions, plots) in: {OUTPUT_FOLDER}")
    print(f"\nDiagnostics use the '{diag_scheme}' scheme.")
    if "leave-one-array-out" in schemes:
        print("Read 'leave-one-array-out' as the realistic accuracy on a brand-new array.")
    else:
        print("Only one array present - add another to test cross-array generalisation.")


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


def _plot_importance(model, path):
    reg = model.pipe.named_steps["reg"]
    plt.figure(figsize=(7, 4))
    if hasattr(reg, "coef_"):
        vals, title = reg.coef_, "Ridge coefficients (standardised features)"
    else:
        vals, title = reg.feature_importances_, "RandomForest feature importances"
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
