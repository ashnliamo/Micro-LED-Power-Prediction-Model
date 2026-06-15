"""Crop measured LEDs from each array in Training data/ into Training Crops/.

Each array sub-folder has a stitched image + an .xlsx with an LIV sheet. For each
array: read the powers at I_Low=0.75mA / VDD=4.5V, align the rigid LED template to
the image, and save one crop per measured LED as '<label>_<power>uW.png'.
Also the shared alignment/crop library used by predict_power.py.
"""

import cv2
import numpy as np
import csv
import os
import glob
import re
import openpyxl

BASE          = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR  = os.path.join(BASE, "training data")
TEMPLATE_CSV  = os.path.join(BASE, "array_coordinates_corrected.csv")
OUTPUT_FOLDER = os.path.join(BASE, "Training Crops")

I_LOW_TARGET  = 0.75   # mA
VDD_TARGET    = 4.5    # V
TOL           = 1e-6

# Crop side = fraction of detected LED pitch (resolution-independent); CROP_SIZE is a fallback.
CROP_FRACTION = 0.60
CROP_SIZE     = 115
SAVE_COLOR    = True
PAD_EDGE      = True

# Rigid alignment (single similarity transform; the whole grid moves together).
GATE_FRAC          = 0.40   # match radius as a fraction of LED pitch (<0.5)
MAX_SCALE_DECREASE = 0.25   # grid may shrink at most this far below the pitch-implied scale
MAX_COL_SHIFT      = 6      # grid-offset search range, columns
MAX_ROW_SHIFT      = 4      # grid-offset search range, rows
MIN_SHIFT_GAIN     = 0.05   # accept a row shift only if it lands >=5% more points on LEDs

THRESH_FRAC   = 0.37
MIN_AREA      = 40


def led_label(row, col):
    """(row, col) -> label like 'C33'. Cols 1-12 -> A-L, 13-24 -> N-Y (no M);
    odd-group columns skip row 17, even group is clean evens."""
    letters = "ABCDEFGHIJKLNOPQRSTUVWXY"
    odd_cols = {1, 3, 5, 7, 9, 11, 14, 16, 18, 20, 22, 24}
    odd_rows  = [1, 3, 5, 7, 9, 11, 13, 15, 19, 21, 23, 25, 27, 29, 31, 33]
    even_rows = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    number = (odd_rows if col in odd_cols else even_rows)[row - 1]
    return f"{letters[col - 1]}{number}"


def pad_label(label):
    """'A1' -> 'A01' (zero-pad the number, for display/CSV)."""
    m = re.fullmatch(r"([A-Za-z]+)(\d+)", str(label))
    return f"{m.group(1)}{int(m.group(2)):02d}" if m else str(label)


def normalize_label(text):
    """'X02' or 'X2' -> ('X', 2), or None."""
    m = re.fullmatch(r"\s*([A-Za-z]+)\s*0*(\d+)\s*", str(text))
    if not m:
        return None
    return (m.group(1).upper(), int(m.group(2)))


def read_liv_powers(xlsx_path):
    """{(letter, number): power_uW} for LIV rows at I_LOW_TARGET / VDD_TARGET; also the sheet name."""
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


def similarity_from_2pts(src, dst):
    """Similarity (rotation + uniform scale + translation) mapping src pair -> dst pair."""
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


def _apply(M, pts):
    return cv2.transform(pts.reshape(-1, 1, 2), M).reshape(-1, 2)


def _nearest_dist(pts, detected):
    d2 = ((pts[:, None, :] - detected[None, :, :]) ** 2).sum(axis=2)
    return np.sqrt(d2.min(axis=1))


def _median_spacing(points):
    d2 = ((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    return float(np.median(np.sqrt(d2.min(axis=1))))


def _scale_of(M):
    return float(np.hypot(M[0, 0], M[1, 0]))


def _corner_candidates(detected):
    """Plausible top-left / top-right anchor dots (consensus picks the right pair)."""
    x, y = detected[:, 0], detected[:, 1]
    o_sum = np.argsort(x + y)
    o_diff = np.argsort(x - y)
    o_x, o_y = np.argsort(x), np.argsort(y)
    tl = list(o_sum[:3]) + list(o_x[:2]) + list(o_y[:2])
    tr = list(o_diff[-3:]) + list(o_x[-2:]) + list(o_y[:2])
    uniq = lambda idx: [detected[i] for i in dict.fromkeys(int(j) for j in idx)]
    return uniq(tl), uniq(tr)


def _best_slide(base, deltas, detected, gate):
    """Pick the translation k landing the most points on LEDs; keep k=0 unless the
    best clears MIN_SHIFT_GAIN. Returns (k, delta)."""
    inl = {k: int((_nearest_dist(base + d, detected) < gate).sum()) for k, d in deltas.items()}
    inl0 = inl[0]
    best = max(inl, key=lambda k: (inl[k], -abs(k)))
    if inl[best] - inl0 < max(4, MIN_SHIFT_GAIN * inl0):
        best = 0
    return best, deltas[best]


def _gap_column_offset(M, template, n_cols, n_rows, detected):
    """Column offset that keeps the three black columns empty: the centre 'N' gap
    (between cols 12 and 13) and the two edges beyond cols A and Y. Returns +k
    (move grid right k columns)."""
    tcolx = template.reshape(n_rows, n_cols, 2)[0, :, 0]
    gap_x = (tcolx[11] + tcolx[12]) / 2.0
    colA, colY = tcolx[0], tcolx[-1]
    colw = float(np.median(np.diff(tcolx[:12])))
    Minv = cv2.invertAffineTransform(M)
    xs = cv2.transform(np.asarray(detected, np.float32).reshape(-1, 1, 2),
                       Minv).reshape(-1, 2)[:, 0]
    best_k, best_viol = 0, None
    for k in range(-MAX_COL_SHIFT, MAX_COL_SHIFT + 1):
        c, a, y = gap_x + k * colw, colA + k * colw, colY + k * colw
        viol = int(((xs > c - colw / 2) & (xs < c + colw / 2)).sum())
        viol += int((xs < a - colw / 2).sum())
        viol += int((xs > y + colw / 2).sum())
        if best_viol is None or viol < best_viol or (viol == best_viol and abs(k) < abs(best_k)):
            best_viol, best_k = viol, k
    return best_k


def correct_grid_offset(M, template, n_cols, detected, gate):
    """Step 2: align columns via the black 'N' gap, then rows by inlier slides.
    Returns (M, row_shift, col_shift); +row_shift = up, +col_shift = left."""
    base = _apply(M, template)
    base0 = base[0]
    row_vec = _apply(M, template[n_cols:n_cols + 1])[0] - base0   # one row down
    n_rows = len(template) // n_cols

    def col_disp(k):
        if k == 0:
            return np.zeros(2, np.float32)
        d = _apply(M, template[abs(k):abs(k) + 1])[0] - base0     # |k| cols right (carries stagger)
        return d if k > 0 else -d

    kc = _gap_column_offset(M, template, n_cols, n_rows, detected)
    dcol = col_disp(kc)

    row_deltas = {k: (k * row_vec).astype(np.float32) for k in range(-MAX_ROW_SHIFT, MAX_ROW_SHIFT + 1)}
    kr, drow = _best_slide(base + dcol, row_deltas, detected, gate)

    M2 = M.copy()
    M2[:, 2] += (dcol + drow).astype(M.dtype)
    return M2, -kr, -kc


def fit_rigid_transform(template, n_cols, detected):
    """Best similarity transform placing the template on the detected LEDs:
    consensus over corner anchors (scale floored to the pitch), a rigid polish,
    then correct_grid_offset. Returns (M, (tl, tr), (row_shift, col_shift))."""
    t_tl, t_tr = template[0], template[n_cols - 1]
    spacing = _median_spacing(detected)
    gate = GATE_FRAC * spacing
    ref_scale = spacing / _median_spacing(template)
    min_scale = (1.0 - MAX_SCALE_DECREASE) * ref_scale

    tl_cands, tr_cands = _corner_candidates(detected)
    best_M, best_inliers, best_corners = None, -1, None
    fallback_M, fallback_corners, fallback_gap = None, None, np.inf
    for a in tl_cands:
        for b in tr_cands:
            if np.allclose(a, b):
                continue
            M = similarity_from_2pts([t_tl, t_tr], [a, b])
            gap = abs(_scale_of(M) - ref_scale)
            if gap < fallback_gap:
                fallback_M, fallback_corners, fallback_gap = M, (a, b), gap
            if _scale_of(M) < min_scale:
                continue
            inliers = int((_nearest_dist(_apply(M, template), detected) < gate).sum())
            if inliers > best_inliers:
                best_M, best_inliers, best_corners = M, inliers, (a, b)

    if best_M is None:
        best_M, best_corners = fallback_M, fallback_corners

    # rigid polish: re-fit a similarity to the consensus matches (scale floor still applies)
    M = best_M
    for _ in range(2):
        P = _apply(M, template)
        d = _nearest_dist(P, detected)
        keep = d < gate
        if keep.sum() < 12:
            break
        src = template[keep]
        dst = np.array([detected[np.argmin(((detected - p) ** 2).sum(1))]
                        for p in P[keep]], np.float32)
        M2, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                            ransacReprojThreshold=6)
        if M2 is None or _scale_of(M2) < min_scale:
            break
        M = M2.astype(np.float32)

    M, row_shift, col_shift = correct_grid_offset(M, template, n_cols, detected, gate)
    return M, best_corners, (row_shift, col_shift)


def led_pitch(points):
    """Median nearest-neighbour distance among aligned LED centres (px)."""
    pts = np.asarray(points, np.float64)
    if len(pts) < 2:
        return float(CROP_SIZE)
    nn = []
    for i in range(len(pts)):
        d = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
        d[i] = np.inf
        nn.append(d.min())
    return float(np.median(nn))


def crop_size_for(points):
    """Square crop side (px) = CROP_FRACTION * LED pitch; CROP_SIZE if unavailable."""
    pitch = led_pitch(points)
    if not np.isfinite(pitch) or pitch <= 0:
        return CROP_SIZE
    return max(16, int(round(CROP_FRACTION * pitch)))


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
    """Place the rigid template on the image -> (blue, aligned_pts, corners, shifts)."""
    blue = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[:, :, 2].astype(np.float32)
    detected = detect_dots(blue)
    if len(detected) < 4:
        return None, None, None, (0, 0)
    M, corners, shifts = fit_rigid_transform(template, n_cols, detected)
    aligned = _apply(M, template)
    return blue, aligned, corners, shifts


IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")


def find_one(folder, patterns):
    for pat in patterns:
        hits = glob.glob(os.path.join(folder, pat))
        if hits:
            return sorted(hits)[0]
    return None


def process_array(array_dir, template, labels, n_cols, out_root):
    name = os.path.basename(array_dir.rstrip("\\/"))
    img_path = find_one(array_dir, IMG_EXTS)
    xlsx_path = find_one(array_dir, ("*.xlsx",))
    if img_path is None or xlsx_path is None:
        print(f"  ! {name}: missing image or xlsx - skipped")
        return []

    powers, _ = read_liv_powers(xlsx_path)
    if not powers:
        print(f"  ! {name}: no measurements at {I_LOW_TARGET}mA / {VDD_TARGET}V - skipped")
        return []

    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"  ! {name}: could not read image - skipped")
        return []
    blue, aligned, corners, (row_shift, col_shift) = align_image(img, template, n_cols)
    if aligned is None:
        print(f"  ! {name}: too few dots detected - skipped")
        return []
    blue_u8 = np.clip(blue, 0, 255).astype(np.uint8)

    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)

    crop_size = crop_size_for(aligned)
    overlay = img.copy()
    half = crop_size // 2
    rows_written = []
    matched = set()
    for (row, col), (x, y) in zip(labels, aligned):
        key = normalize_label(led_label(row, col))
        if key not in powers:
            continue
        matched.add(key)
        power = powers[key]
        label = led_label(row, col)

        source = img if SAVE_COLOR else blue_u8
        crop = crop_led(source, x, y, crop_size, PAD_EDGE)

        fname = f"{label}_{power}uW.png"
        cv2.imwrite(os.path.join(out_dir, fname), crop)
        rows_written.append({"array": name, "label": pad_label(label), "row": row, "col": col,
                             "x": int(round(x)), "y": int(round(y)),
                             "Power_uW": power, "crop_file": os.path.join(name, fname)})

        cv2.rectangle(overlay, (int(x) - half, int(y) - half),
                      (int(x) + half, int(y) + half), (0, 255, 0), 2)
        cv2.putText(overlay, pad_label(label), (int(x) - half, int(y) - half - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    for (x, y) in corners:
        cv2.circle(overlay, (int(x), int(y)), 34, (0, 0, 255), 4)

    h, w = overlay.shape[:2]
    cv2.imwrite(os.path.join(out_root, f"{name}_verify.png"),
                cv2.resize(overlay, (min(1600, w), int(min(1600, w) * h / w))))

    with open(os.path.join(out_dir, "labels.csv"), "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=["array", "label", "row", "col",
                                            "x", "y", "Power_uW", "crop_file"])
        wri.writeheader(); wri.writerows(rows_written)

    missing = sorted(set(powers) - matched)
    extra = ""
    if missing:
        extra = f"  ({len(missing)} measured channels not found on grid: " \
                f"{', '.join(a + str(b) for a, b in missing)})"
    parts = []
    if row_shift > 0:   parts.append(f"up {row_shift} row(s)")
    elif row_shift < 0: parts.append(f"down {-row_shift} row(s)")
    if col_shift > 0:   parts.append(f"left {col_shift} col(s)")
    elif col_shift < 0: parts.append(f"right {-col_shift} col(s)")
    shift_note = f"  [shifted grid {', '.join(parts)}]" if parts else ""
    print(f"  {name}: {len(rows_written)}/{len(powers)} measured LEDs cropped "
          f"@ {crop_size}px (pitch {led_pitch(aligned):.0f}px){extra}{shift_note}")
    return rows_written


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    template, n_cols, n_rows, labels = load_template(TEMPLATE_CSV)
    print(f"Template: {n_cols} x {n_rows} = {len(template)} dots")
    print(f"Labelling crops at I_Low={I_LOW_TARGET}mA, VDD={VDD_TARGET}V  |  "
          f"crop {CROP_FRACTION:.0%} of LED pitch  |  "
          f"{'colour' if SAVE_COLOR else 'blue-channel'}\n")

    array_dirs = [d for d in sorted(glob.glob(os.path.join(TRAINING_DIR, "*")))
                  if os.path.isdir(d)]
    if not array_dirs:
        raise FileNotFoundError(f"No array sub-folders in {TRAINING_DIR}")
    print(f"Processing {len(array_dirs)} array(s):\n")

    all_rows = []
    for d in array_dirs:
        all_rows.extend(process_array(d, template, labels, n_cols, OUTPUT_FOLDER))

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
