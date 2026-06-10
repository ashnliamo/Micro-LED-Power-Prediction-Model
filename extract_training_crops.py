"""
Training Crop Extractor (image crops + measured power labels)
=============================================================
Walks a "training data" folder. Each sub-folder is one LED array and contains:
  * a stitched array image (.png/.jpg/...)
  * an .xlsx workbook with an LIV sheet (name contains "LIV")

For every array it:
  1. Reads the LIV sheet and keeps only the rows measured at
         I_Low_mA == 0.75   and   VDD_50LED_Volt == 4.5
     taking the corresponding Power_uW for each Channel (e.g. "X02").
  2. Aligns the coordinate template to the image (same detect/align/refine/snap
     pipeline as the brightness meter) and crops each LED.
  3. Saves ONLY the LEDs that have a measured power, named by their alphabetical
     label + power, into a per-array output folder. LEDs without a measured
     power are skipped.

A channel like "X02" maps to label "X2" (letter = column, number = row position
in the staggered scheme - see led_label).  Output crop files are named
    <label>_<power>uW.png        e.g.  X2_27.302uW.png
and a manifest CSV ties every crop to its array / label / power.

Requirements:
    pip install opencv-python numpy openpyxl
"""

import cv2
import numpy as np
import csv
import os
import glob
import re
import openpyxl

# CONFIG
TRAINING_DIR  = r"C:\Users\liam.deacon\Desktop\brightness test rotate\training data"
TEMPLATE_CSV  = r"C:\Users\liam.deacon\Desktop\brightness test rotate\array_coordinates_corrected.csv"
OUTPUT_FOLDER = r"C:\Users\liam.deacon\Desktop\brightness test rotate\Training Crops"

# Which LIV measurement to label crops with:
I_LOW_TARGET  = 0.75   # mA
VDD_TARGET    = 4.5    # V (VDD_50LED_Volt)
TOL           = 1e-6   # float compare tolerance

CROP_SIZE     = 160    # side length (px) of the SQUARE crop saved per LED.
SAVE_COLOR    = True   # True -> colour crop; False -> blue channel only
PAD_EDGE      = True   # zero-pad edge crops so every crop is exactly CROP_SIZE^2

REFINE        = True   # refine the 2-corner fit against all dots (recommended)
SNAP_RADIUS   = 10     # final per-dot nudge to local peak (px). 0 to disable.

# Dot detection tuning
THRESH_FRAC   = 0.35
MIN_AREA      = 40


# --------------------------------------------------------------------------- #
#  Labelling                                                                   #
# --------------------------------------------------------------------------- #
def led_label(row, col):
    """Map (row, col) -> array label like 'C33'.

    Column letter:  cols 1-12 -> A-L, cols 13-24 -> N-Y  (M is skipped).
    Row number depends on the column group:
      "odd"  columns {1,3,5,7,9,11,14,16,18,20,22,24} use
             1,3,5,7,9,11,13,15,19,21,23,25,27,29,31,33  (odds 1-33, skipping 17)
      "even" columns (the rest) use
             2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32  (clean evens)
    Only 17 is ever skipped (in the odd group); the even group has no gap.
    """
    letters = "ABCDEFGHIJKLNOPQRSTUVWXY"          # 24 letters, M omitted
    odd_cols = {1, 3, 5, 7, 9, 11, 14, 16, 18, 20, 22, 24}
    odd_rows  = [1, 3, 5, 7, 9, 11, 13, 15, 19, 21, 23, 25, 27, 29, 31, 33]
    even_rows = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    number = (odd_rows if col in odd_cols else even_rows)[row - 1]
    return f"{letters[col - 1]}{number}"


def normalize_label(text):
    """'X02' or 'X2' -> ('X', 2). Returns None if it doesn't look like a label."""
    m = re.fullmatch(r"\s*([A-Za-z]+)\s*0*(\d+)\s*", str(text))
    if not m:
        return None
    return (m.group(1).upper(), int(m.group(2)))


def build_label_index(n_rows, n_cols):
    """Reverse map  (letter, number) -> (row, col)  for the whole grid."""
    idx = {}
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            idx[normalize_label(led_label(r, c))] = (r, c)
    return idx


# --------------------------------------------------------------------------- #
#  Spreadsheet                                                                 #
# --------------------------------------------------------------------------- #
def read_liv_powers(xlsx_path):
    """Return {(letter, number): power_uW} for rows at I_LOW_TARGET / VDD_TARGET."""
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    liv = next((s for s in wb.sheetnames if "LIV" in s.upper()), None)
    if liv is None:
        wb.close()
        raise KeyError(f"no LIV sheet in {os.path.basename(xlsx_path)}")
    ws = wb[liv]

    rows = ws.iter_rows(values_only=True)
    header = [str(h).strip() if h is not None else "" for h in next(rows)]
    col = {name: i for i, name in enumerate(header)}
    ci, ii = col.get("Channel", 0), col.get("I_Low_mA", 2)
    vi, pi = col.get("VDD_50LED_Volt", 3), col.get("Power_uW", 4)

    powers = {}
    for r in rows:
        ch = r[ci]
        if ch is None:
            continue
        try:
            il, vd, pw = float(r[ii]), float(r[vi]), float(r[pi])
        except (TypeError, ValueError):
            continue
        if abs(il - I_LOW_TARGET) < TOL and abs(vd - VDD_TARGET) < TOL:
            key = normalize_label(ch)
            if key is not None:
                powers[key] = pw
    wb.close()
    return powers, liv


# --------------------------------------------------------------------------- #
#  Alignment / cropping pipeline (shared with the brightness meter)            #
# --------------------------------------------------------------------------- #
def load_template(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    n_cols = len(header) // 2
    pts, labels = [], []
    for r_idx, row in enumerate(rows):
        for c_idx in range(n_cols):
            pts.append([float(row[c_idx * 2]), float(row[c_idx * 2 + 1])])
            labels.append((r_idx + 1, c_idx + 1))
    return np.array(pts, np.float32), n_cols, len(rows), labels


def detect_dots(blue):
    mx = float(blue.max())
    _, mask = cv2.threshold(blue.astype(np.uint8), int(mx * THRESH_FRAC), 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_AREA:
            continue
        M = cv2.moments(cnt)
        if M["m00"]:
            centers.append([M["m10"] / M["m00"], M["m01"] / M["m00"]])
    return np.array(centers, np.float32)


def find_top_corners(detected):
    tl = detected[np.argmin(detected[:, 0] + detected[:, 1])]
    tr = detected[np.argmax(detected[:, 0] - detected[:, 1])]
    return tl, tr


def similarity_from_2pts(src, dst):
    (ax, ay), (bx, by) = src
    (Ax, Ay), (Bx, By) = dst
    vx, vy = bx - ax, by - ay
    Vx, Vy = Bx - Ax, By - Ay
    s = np.hypot(Vx, Vy) / np.hypot(vx, vy)
    ang = np.arctan2(Vy, Vx) - np.arctan2(vy, vx)
    c, sn = s * np.cos(ang), s * np.sin(ang)
    R = np.array([[c, -sn], [sn, c]])
    t = np.array([Ax, Ay]) - R @ np.array([ax, ay])
    M = np.zeros((2, 3), np.float32)
    M[:, :2] = R
    M[:, 2] = t
    return M


def refine_against_all(template, aligned_pts, detected, gate=40):
    src, dst = [], []
    for k, p in enumerate(aligned_pts):
        d2 = ((detected - p) ** 2).sum(axis=1)
        j = int(np.argmin(d2))
        if d2[j] <= gate * gate:
            src.append(template[k])
            dst.append(detected[j])
    if len(src) < 12:
        return None
    M, _ = cv2.estimateAffinePartial2D(
        np.array(src, np.float32), np.array(dst, np.float32),
        method=cv2.RANSAC, ransacReprojThreshold=6,
    )
    return M


def snap_to_peak(blue, x, y, radius):
    h, w = blue.shape
    x1, x2 = max(0, int(x - radius)), min(w, int(x + radius))
    y1, y2 = max(0, int(y - radius)), min(h, int(y + radius))
    region = blue[y1:y2, x1:x2]
    if region.size == 0:
        return x, y
    blurred = cv2.GaussianBlur(region, (7, 7), 0)
    _, _, _, mloc = cv2.minMaxLoc(blurred)
    return x1 + mloc[0], y1 + mloc[1]


def crop_led(img, cx, cy, size, pad):
    h, w = img.shape[:2]
    half = size // 2
    x1, y1 = int(round(cx - half)), int(round(cy - half))
    x2, y2 = x1 + size, y1 + size
    if not pad:
        return img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if img.ndim == 3:
        canvas = np.zeros((size, size, img.shape[2]), dtype=img.dtype)
    else:
        canvas = np.zeros((size, size), dtype=img.dtype)
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(w, x2), min(h, y2)
    if sx2 <= sx1 or sy2 <= sy1:
        return canvas
    canvas[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = img[sy1:sy2, sx1:sx2]
    return canvas


def align_image(img, template, n_cols):
    """Return aligned per-LED points (Nx2) in image space, plus the corners used."""
    blue = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[:, :, 2].astype(np.float32)
    detected = detect_dots(blue)
    if len(detected) < 4:
        return None, None, None
    t_tl, t_tr = template[0], template[n_cols - 1]
    img_tl, img_tr = find_top_corners(detected)
    M = similarity_from_2pts([t_tl, t_tr], [img_tl, img_tr])
    aligned = cv2.transform(template.reshape(-1, 1, 2), M).reshape(-1, 2)
    if REFINE:
        M2 = refine_against_all(template, aligned, detected)
        if M2 is not None:
            aligned = cv2.transform(template.reshape(-1, 1, 2), M2).reshape(-1, 2)
    return blue, aligned, (img_tl, img_tr)


# --------------------------------------------------------------------------- #
#  Per-array processing                                                        #
# --------------------------------------------------------------------------- #
IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")


def find_one(folder, patterns):
    for pat in patterns:
        hits = glob.glob(os.path.join(folder, pat))
        if hits:
            return sorted(hits)[0]
    return None


def process_array(array_dir, template, labels, n_cols, label_index, out_root):
    name = os.path.basename(array_dir.rstrip("\\/"))
    img_path = find_one(array_dir, IMG_EXTS)
    xlsx_path = find_one(array_dir, ("*.xlsx",))
    if img_path is None or xlsx_path is None:
        print(f"  ! {name}: missing image or xlsx - skipped")
        return []

    powers, liv_sheet = read_liv_powers(xlsx_path)
    if not powers:
        print(f"  ! {name}: no measurements at {I_LOW_TARGET}mA / {VDD_TARGET}V - skipped")
        return []

    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"  ! {name}: could not read image - skipped")
        return []
    blue, aligned, corners = align_image(img, template, n_cols)
    if aligned is None:
        print(f"  ! {name}: too few dots detected - skipped")
        return []
    blue_u8 = np.clip(blue, 0, 255).astype(np.uint8)

    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)

    overlay = img.copy()
    half = CROP_SIZE // 2
    rows_written = []
    matched = set()
    for (row, col), (x, y) in zip(labels, aligned):
        key = normalize_label(led_label(row, col))
        if key not in powers:
            continue                       # this LED has no measured power -> skip
        matched.add(key)
        power = powers[key]
        label = led_label(row, col)

        if SNAP_RADIUS > 0:
            x, y = snap_to_peak(blue, x, y, SNAP_RADIUS)
        source = img if SAVE_COLOR else blue_u8
        crop = crop_led(source, x, y, CROP_SIZE, PAD_EDGE)

        fname = f"{label}_{power}uW.png"
        cv2.imwrite(os.path.join(out_dir, fname), crop)
        rows_written.append({"array": name, "label": label, "row": row, "col": col,
                             "x": int(round(x)), "y": int(round(y)),
                             "Power_uW": power, "crop_file": os.path.join(name, fname)})

        cv2.rectangle(overlay, (int(x) - half, int(y) - half),
                      (int(x) + half, int(y) + half), (0, 255, 0), 2)
        cv2.putText(overlay, label, (int(x) - half, int(y) - half - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    for (x, y) in corners:
        cv2.circle(overlay, (int(x), int(y)), 34, (0, 0, 255), 4)

    # verification overlay (only the labelled LEDs are boxed)
    h, w = overlay.shape[:2]
    cv2.imwrite(os.path.join(out_root, f"{name}_verify.png"),
                cv2.resize(overlay, (min(1600, w), int(min(1600, w) * h / w))))

    # per-array manifest
    with open(os.path.join(out_dir, "labels.csv"), "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=["array", "label", "row", "col",
                                            "x", "y", "Power_uW", "crop_file"])
        wri.writeheader(); wri.writerows(rows_written)

    missing = sorted(set(powers) - matched)
    extra = ""
    if missing:
        extra = f"  ({len(missing)} measured channels not found on grid: " \
                f"{', '.join(a + str(b) for a, b in missing)})"
    print(f"  {name}: {len(rows_written)}/{len(powers)} measured LEDs cropped{extra}")
    return rows_written


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    template, n_cols, n_rows, labels = load_template(TEMPLATE_CSV)
    label_index = build_label_index(n_rows, n_cols)
    print(f"Template: {n_cols} x {n_rows} = {len(template)} dots")
    print(f"Labelling crops at I_Low={I_LOW_TARGET}mA, VDD={VDD_TARGET}V  |  "
          f"crop {CROP_SIZE}px  |  {'colour' if SAVE_COLOR else 'blue-channel'}\n")

    array_dirs = [d for d in sorted(glob.glob(os.path.join(TRAINING_DIR, "*")))
                  if os.path.isdir(d)]
    if not array_dirs:
        raise FileNotFoundError(f"No array sub-folders in {TRAINING_DIR}")
    print(f"Processing {len(array_dirs)} array(s):\n")

    all_rows = []
    for d in array_dirs:
        all_rows.extend(process_array(d, template, labels, n_cols, label_index, OUTPUT_FOLDER))

    manifest = os.path.join(OUTPUT_FOLDER, "training_manifest.csv")
    with open(manifest, "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=["array", "label", "row", "col",
                                            "x", "y", "Power_uW", "crop_file"])
        wri.writeheader(); wri.writerows(all_rows)

    print(f"\nDone. {len(all_rows)} labelled crops across {len(array_dirs)} arrays.")
    print(f"Output: {OUTPUT_FOLDER}")
    print(f"Combined manifest: {manifest}")


if __name__ == "__main__":
    main()
