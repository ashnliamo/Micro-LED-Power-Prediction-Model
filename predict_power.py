"""
Predict LED Power from an Array Image
=====================================
Two ways to use the Input folder:

  A) Drop a single array IMAGE in Input/         -> blind prediction.
     Every LED's power is predicted from the photo alone.

  B) Drop a FOLDER in Input/ (like the training-data folders) containing:
         <the array image>            e.g. Analysis_image-Stitched_-9.png
         a measured CSV or .xlsx      the 40 (or however many) LEDs you measured
     -> CALIBRATED prediction. The measured LEDs are used as anchors to remove
        that array's exposure offset, so the unmeasured LEDs are predicted on the
        correct absolute scale. You also get an honest "blind" accuracy report
        (how the raw model did on the measured LEDs before calibration).

     The measured file can be either:
        * a CSV with a label column (label / Channel) and a Power_uW column
          (the same columns as Training Crops/<array>/labels.csv works too), or
        * the array's .xlsx workbook (its LIV sheet is read at
          I_Low=0.75 mA / VDD_50LED=4.5 V, exactly like the training pipeline).

For every input it writes a CSV + labelled overlay to the Output folder.

Train the model first if it doesn't exist yet:
    py extract_training_crops.py      # build labelled crops from training data
    py train_power_model.py           # fit + save Model/power_model.joblib

Usage:
    py predict_power.py               # processes every image/folder in Input/

Requirements:
    pip install opencv-python numpy scikit-learn joblib openpyxl
"""

import os
import csv
import glob
import numpy as np
import cv2

import extract_training_crops as E          # reuse the alignment pipeline
from power_model import PowerModel

# CONFIG
BASE          = r"C:\Users\liam.deacon\Desktop\brightness test rotate"
INPUT_FOLDER  = os.path.join(BASE, "Input")
OUTPUT_FOLDER = os.path.join(BASE, "Output")
MODEL_PATH    = os.path.join(BASE, "Model", "power_model.joblib")
SAVE_OVERLAY  = True                         # also write a labelled verification image
MIN_CALIB_PTS = 3                            # need this many measured LEDs to calibrate

IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")


# --------------------------------------------------------------------------- #
#  Metrics + calibration helpers                                              #
# --------------------------------------------------------------------------- #
def _metrics(meas, pred):
    """MAE, RMSE, R2 between measured and predicted arrays."""
    err = pred - meas
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((meas - meas.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mae, rmse, r2


def _fit_calibration(raw, meas):
    """Fit  meas ~ a*raw + b  on the measured anchors. Returns (a, b).

    Falls back to identity (1, 0) if there aren't enough points or the raw
    predictions have no spread (can't fit a line)."""
    raw = np.asarray(raw, np.float64)
    meas = np.asarray(meas, np.float64)
    if len(raw) < MIN_CALIB_PTS or raw.std() < 1e-6:
        return 1.0, 0.0
    a, b = np.polyfit(raw, meas, 1)
    return float(a), float(b)


# --------------------------------------------------------------------------- #
#  Reading the measured-LED file (CSV or xlsx)                                 #
# --------------------------------------------------------------------------- #
def _read_measured_csv(path):
    """Return {(letter, number): power_uW} from a CSV with a label + power column.

    Tolerant about column names: label may be 'label'/'Label'/'Channel'/'channel';
    power may be 'Power_uW'/'power_uW'/'power'/'Power'."""
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
    """Look in a folder for a measured-LED file. Prefer CSV, fall back to xlsx.

    Returns (measured_dict, source_description) or (None, None) if none found."""
    csvs = [p for p in glob.glob(os.path.join(folder, "*.csv"))]
    for p in csvs:
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


# --------------------------------------------------------------------------- #
#  Core: predict one array (image + optional measured anchors)                 #
# --------------------------------------------------------------------------- #
def predict_array(img_path, measured, template, labels, n_cols, model, out_folder, name):
    """Align, predict every LED, optionally calibrate against measured anchors,
    and write the output CSV + overlay. `measured` is {(letter,num): power} or None."""
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"  ! could not read image for {name}")
        return

    blue, aligned, corners, shifts = E.align_image(img, template, n_cols)
    if aligned is None:
        print(f"  ! too few LEDs detected in {name} - skipped")
        return

    # 1) raw prediction for every LED on the grid
    recs = []                                # one dict per LED
    raw_anchor, meas_anchor = [], []         # measured LEDs, for calibration/scoring
    for (row, col), (x, y) in zip(labels, aligned):
        crop = E.crop_led(img, x, y, E.CROP_SIZE, True)
        raw = float(model.predict_crop(crop))
        label = E.led_label(row, col)
        key = E.normalize_label(label)
        m = measured.get(key) if measured else None
        recs.append({"label": label, "row": row, "col": col,
                     "x": int(round(x)), "y": int(round(y)),
                     "raw": raw, "measured": m})
        if m is not None:
            raw_anchor.append(raw)
            meas_anchor.append(m)

    # 2) blind accuracy on the measured anchors (before any calibration)
    n_anchor = len(raw_anchor)
    a, b = 1.0, 0.0
    if n_anchor >= 1:
        ra = np.array(raw_anchor); ma = np.array(meas_anchor)
        bmae, brmse, br2 = _metrics(ma, ra)
        print(f"  {name}: {n_anchor} measured LED(s) found")
        print(f"      BLIND (raw model)   MAE {bmae:6.2f}  RMSE {brmse:6.2f}  R2 {br2:6.3f}")
        # 3) fit per-array calibration on the anchors
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
        print(f"  {name}: no measured LEDs supplied - blind prediction only")

    # 4) apply calibration to all LEDs, build final values
    for rec in recs:
        rec["calibrated"] = float(np.clip(a * rec["raw"] + b, 0.0, None))
        # final value to report/draw: measured if we have it, else calibrated
        rec["final"] = rec["measured"] if rec["measured"] is not None else rec["calibrated"]

    _write_csv(recs, out_folder, name, calibrated=(a, b) != (1.0, 0.0))
    if SAVE_OVERLAY:
        _write_overlay(img, recs, out_folder, name)

    note = ""
    if shifts and any(shifts):
        note = f"   (grid auto-shifted row {shifts[0]:+d}, col {shifts[1]:+d})"
    n_pred = sum(1 for r in recs if r["measured"] is None)
    print(f"      wrote {len(recs)} LEDs ({n_pred} predicted, {n_anchor} measured)"
          f" -> {name}_predicted.csv{note}")


def _free_path(path):
    """Return `path` if writable, else a numbered fallback (file open in Excel etc.).

    Avoids crashing when the previous output CSV/overlay is still open."""
    if not os.path.exists(path):
        return path
    try:                                   # can we overwrite the existing file?
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


def _write_csv(recs, out_folder, name, calibrated):
    out_csv = _free_path(os.path.join(out_folder, f"{name}_predicted.csv"))
    with open(out_csv, "w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["label", "row", "col", "x", "y",
                      "predicted_raw_uW", "predicted_calibrated_uW",
                      "measured_uW", "final_uW"])
        for r in recs:
            wri.writerow([
                E.pad_label(r["label"]), r["row"], r["col"], r["x"], r["y"],
                round(r["raw"], 3),
                round(r["calibrated"], 3),
                "" if r["measured"] is None else round(r["measured"], 3),
                round(r["final"], 3),
            ])


def _write_overlay(img, recs, out_folder, name):
    overlay = img.copy()
    half = E.CROP_SIZE // 2
    for r in recs:
        x, y = r["x"], r["y"]
        measured = r["measured"] is not None
        # measured LEDs in cyan box, predicted LEDs in green box
        box_col = (255, 255, 0) if measured else (0, 255, 0)
        cv2.rectangle(overlay, (x - half, y - half), (x + half, y + half), box_col, 2)
        cv2.putText(overlay, E.pad_label(r["label"]), (x - half, y - half - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        tag = f"{r['final']:.1f}uW" + ("*" if measured else "")
        cv2.putText(overlay, tag, (x - half, y - half + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    h, w = overlay.shape[:2]
    cv2.imwrite(_free_path(os.path.join(out_folder, f"{name}_overlay.png")),
                cv2.resize(overlay, (min(1600, w), int(min(1600, w) * h / w))))


# --------------------------------------------------------------------------- #
#  Main: walk the Input folder (folders = calibrated, loose images = blind)    #
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model at {MODEL_PATH}.\n"
            "Run  py extract_training_crops.py  then  py train_power_model.py  first.")

    model = PowerModel.load(MODEL_PATH)
    template, n_cols, n_rows, labels = E.load_template(E.TEMPLATE_CSV)

    # folders in Input/ -> calibrated jobs
    folders = [p for p in sorted(glob.glob(os.path.join(INPUT_FOLDER, "*")))
               if os.path.isdir(p)]
    # loose images directly in Input/ -> blind jobs
    loose = []
    for ext in IMG_EXTS:
        loose.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))
    loose.sort()

    if not folders and not loose:
        print(f"Nothing to do. Put either:")
        print(f"  - a single array image in {INPUT_FOLDER}  (blind prediction), or")
        print(f"  - a folder (image + measured CSV/xlsx) in {INPUT_FOLDER}  (calibrated)")
        return

    print(f"Model: {os.path.basename(MODEL_PATH)}")
    print(f"Jobs: {len(folders)} folder(s) + {len(loose)} loose image(s)\n")

    for folder in folders:
        name = os.path.basename(folder.rstrip("\\/"))
        img_path = _find_image(folder)
        if img_path is None:
            print(f"  ! {name}: no image found in folder - skipped")
            continue
        measured, src = _find_measured(folder)
        if src:
            print(f"  [{name}] image: {os.path.basename(img_path)} | measured: {src}")
        predict_array(img_path, measured, template, labels, n_cols, model,
                      OUTPUT_FOLDER, name)

    for fp in loose:
        name = os.path.splitext(os.path.basename(fp))[0]
        predict_array(fp, None, template, labels, n_cols, model,
                      OUTPUT_FOLDER, name)

    print(f"\nDone. Results in: {OUTPUT_FOLDER}")
    print("  CSV columns: predicted_raw_uW (blind), predicted_calibrated_uW,")
    print("  measured_uW (blank if not measured), final_uW (measured or calibrated).")


if __name__ == "__main__":
    main()
