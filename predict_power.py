"""Predict per-LED power for each array folder in Input/.

Each folder needs the array image + a measured CSV/xlsx; the measured LEDs anchor
a per-array linear calibration of the model's predictions. Output CSV + overlay
land in Output/. Train first with: py train_power_model.py
"""

import os
import csv
import glob
import numpy as np
import cv2

import extract_training_crops as E
from power_model import PowerModel, extract_features, FEATURE_NAMES

_INT_I = FEATURE_NAMES.index("integrated")
_PEAK_I = FEATURE_NAMES.index("peak")

BASE          = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER  = os.path.join(BASE, "Input")
OUTPUT_FOLDER = os.path.join(BASE, "Output")
MODEL_PATH    = os.path.join(BASE, "Model", "power_model.joblib")
SAVE_OVERLAY  = True
MIN_CALIB_PTS = 3

# An unlit LED has near-zero signal; force it to 0 uW instead of the model's floor.
DARK_INTEGRATED = 50000.0
DARK_PEAK       = 30.0

IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")


def _metrics(meas, pred):
    err = pred - meas
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((meas - meas.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mae, rmse, r2


def _fit_calibration(raw, meas):
    """meas ~ a*raw + b; identity (1, 0) if too few points or no spread."""
    raw = np.asarray(raw, np.float64)
    meas = np.asarray(meas, np.float64)
    if len(raw) < MIN_CALIB_PTS or raw.std() < 1e-6:
        return 1.0, 0.0
    a, b = np.polyfit(raw, meas, 1)
    return float(a), float(b)


def _read_measured_csv(path):
    """{(letter, number): power_uW} from a CSV with a label + power column."""
    out = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return out
        lc = next((c for c in reader.fieldnames
                   if c.strip().lower() in ("label", "channel")), None)
        pc = next((c for c in reader.fieldnames
                   if c.strip().lower() in ("power_uw", "power")), None)
        if lc is None or pc is None:
            return out
        for r in reader:
            key = E.normalize_label(r[lc])
            try:
                pw = float(r[pc])
            except (TypeError, ValueError):
                continue
            if key is not None:
                out[key] = pw
    return out


def _find_measured(folder):
    """(measured_dict, source) from a CSV (preferred) or xlsx LIV sheet; (None, None) if none."""
    for p in glob.glob(os.path.join(folder, "*.csv")):
        d = _read_measured_csv(p)
        if d:
            return d, os.path.basename(p)
    for p in glob.glob(os.path.join(folder, "*.xlsx")):
        try:
            powers, sheet = E.read_liv_powers(p)
        except Exception:
            continue
        if powers:
            return powers, f"{os.path.basename(p)} [{sheet}]"
    return None, None


def _find_image(folder):
    for ext in IMG_EXTS:
        hits = sorted(glob.glob(os.path.join(folder, ext)))
        if hits:
            return hits[0]
    return None


def predict_array(img_path, measured, template, labels, n_cols, model, out_folder, name):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"  ! could not read image for {name}")
        return

    blue, aligned, corners, shifts = E.align_image(img, template, n_cols)
    if aligned is None:
        print(f"  ! too few LEDs detected in {name} - skipped")
        return

    crop_size = E.crop_size_for(aligned)
    recs = []
    raw_anchor, meas_anchor = [], []
    n_off = 0
    for (row, col), (x, y) in zip(labels, aligned):
        crop = E.crop_led(img, x, y, crop_size, True)
        feat = extract_features(crop)
        off = bool(feat[_INT_I] < DARK_INTEGRATED and feat[_PEAK_I] < DARK_PEAK)
        raw = 0.0 if off else float(model.predict(feat.reshape(1, -1))[0])
        n_off += off
        label = E.led_label(row, col)
        m = measured.get(E.normalize_label(label))
        recs.append({"label": label, "row": row, "col": col,
                     "x": int(round(x)), "y": int(round(y)),
                     "raw": raw, "off": off, "measured": m})
        if m is not None and not off:        # only lit, measured LEDs anchor calibration
            raw_anchor.append(raw)
            meas_anchor.append(m)

    n_anchor = len(raw_anchor)
    a, b = 1.0, 0.0
    if n_anchor >= 1:
        ra = np.array(raw_anchor); ma = np.array(meas_anchor)
        bmae, brmse, br2 = _metrics(ma, ra)
        print(f"  {name}: {n_anchor} measured LED(s) found")
        print(f"      BLIND (raw model)   MAE {bmae:6.2f}  RMSE {brmse:6.2f}  R2 {br2:6.3f}")
        a, b = _fit_calibration(ra, ma)
        if (a, b) != (1.0, 0.0):
            cal_anchor = np.clip(a * ra + b, 0.0, None)
            cmae, crmse, cr2 = _metrics(ma, cal_anchor)
            print(f"      calibration: power = {a:.3f} * raw + {b:.2f}")
            print(f"      CALIBRATED (on anchors, in-sample)"
                  f"  MAE {cmae:6.2f}  RMSE {crmse:6.2f}  R2 {cr2:6.3f}")
        else:
            print(f"      (not enough spread to calibrate - need >= {MIN_CALIB_PTS} "
                  f"measured LEDs across a power range; left uncalibrated)")
    else:
        print(f"  ! {name}: none of the measured LEDs landed on the grid "
              f"- cannot calibrate (check the labels in your measured file)")

    for rec in recs:
        rec["calibrated"] = 0.0 if rec["off"] else float(np.clip(a * rec["raw"] + b, 0.0, None))
        rec["final"] = rec["measured"] if rec["measured"] is not None else rec["calibrated"]

    if n_off:
        print(f"      {n_off} LED(s) detected as off (no light) -> set to 0 uW")

    _write_csv(recs, out_folder, name)
    if SAVE_OVERLAY:
        _write_overlay(img, recs, out_folder, name, crop_size)

    note = ""
    if shifts and any(shifts):
        note = f"   (grid auto-shifted row {shifts[0]:+d}, col {shifts[1]:+d})"
    n_pred = sum(1 for r in recs if r["measured"] is None)
    print(f"      wrote {len(recs)} LEDs ({n_pred} predicted, {n_anchor} measured)"
          f" -> {name}_predicted.csv{note}")


def _free_path(path):
    """`path`, or a numbered fallback if it's locked (open in Excel)."""
    if not os.path.exists(path):
        return path
    try:
        with open(path, "a"):
            pass
        return path
    except PermissionError:
        root, ext = os.path.splitext(path)
        for i in range(1, 100):
            alt = f"{root} ({i}){ext}"
            if not os.path.exists(alt):
                print(f"      ! {os.path.basename(path)} is open/locked "
                      f"- writing {os.path.basename(alt)} instead")
                return alt
        raise


def _write_csv(recs, out_folder, name):
    out_csv = _free_path(os.path.join(out_folder, f"{name}_predicted.csv"))
    with open(out_csv, "w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["label", "row", "col", "x", "y",
                      "predicted_raw_uW", "predicted_calibrated_uW",
                      "measured_uW", "final_uW"])
        for r in recs:
            wri.writerow([
                E.pad_label(r["label"]), r["row"], r["col"], r["x"], r["y"],
                round(r["raw"], 3), round(r["calibrated"], 3),
                "" if r["measured"] is None else round(r["measured"], 3),
                round(r["final"], 3),
            ])


def _write_overlay(img, recs, out_folder, name, crop_size):
    overlay = img.copy()
    half = crop_size // 2
    for r in recs:
        x, y = r["x"], r["y"]
        measured = r["measured"] is not None
        box_col = (255, 255, 0) if measured else (0, 255, 0)   # measured=cyan, predicted=green
        cv2.rectangle(overlay, (x - half, y - half), (x + half, y + half), box_col, 2)
        cv2.putText(overlay, E.pad_label(r["label"]), (x - half, y - half - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        tag = f"{r['final']:.1f}uW" + ("*" if measured else "")
        cv2.putText(overlay, tag, (x - half, y - half + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    h, w = overlay.shape[:2]
    cv2.imwrite(_free_path(os.path.join(out_folder, f"{name}_overlay.png")),
                cv2.resize(overlay, (min(1600, w), int(min(1600, w) * h / w))))


def main():
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model at {MODEL_PATH}.\nRun  py train_power_model.py  first.")

    model = PowerModel.load(MODEL_PATH)
    template, n_cols, n_rows, labels = E.load_template(E.TEMPLATE_CSV)

    folders = [p for p in sorted(glob.glob(os.path.join(INPUT_FOLDER, "*")))
               if os.path.isdir(p)]
    if not folders:
        print(f"Nothing to do. Put a folder in {INPUT_FOLDER} containing the array")
        print(f"image and a measured CSV/xlsx, then run again.")
        return

    print(f"Model: {os.path.basename(MODEL_PATH)}")
    print(f"Jobs: {len(folders)} folder(s)\n")

    for folder in folders:
        name = os.path.basename(folder.rstrip("\\/"))
        img_path = _find_image(folder)
        if img_path is None:
            print(f"  ! {name}: no image found in folder - skipped")
            continue
        measured, src = _find_measured(folder)
        if not measured:
            print(f"  ! {name}: no measured CSV/xlsx found in folder - skipped "
                  f"(measurements are required for calibration)")
            continue
        print(f"  [{name}] image: {os.path.basename(img_path)} | measured: {src}")
        predict_array(img_path, measured, template, labels, n_cols, model,
                      OUTPUT_FOLDER, name)

    print(f"\nDone. Results in: {OUTPUT_FOLDER}")
    print("  CSV columns: predicted_raw_uW (blind), predicted_calibrated_uW,")
    print("  measured_uW (blank if not measured), final_uW (measured or calibrated).")


if __name__ == "__main__":
    main()
