"""
LED Crop Extractor (for gathering training images)
==================================================
Based on array_brightness_corner_aligned_box.py, but instead of measuring
brightness this program SAVES a cropped image of each individual LED.

For each image in a folder, it:
  1. Detects the blue LED dots.
  2. Finds the top-left and top-right corner dots automatically.
  3. Uses those two anchors to compute the rotation / scale / position that
     maps your (straightened) coordinate template onto the image.
  4. Optionally refines that fit against ALL dots (robust to a bad corner).
  5. Crops a fixed-size square around every aligned LED and writes it to disk.
  6. Writes a manifest CSV (one row per crop) and a verification overlay so you
     can confirm the alignment before trusting the crops.

The crops are named   <image>_r##_c##.png   so each file is traceable back to
its (row, col) position in the array. Pair these with your measured power
values (matched on image + row + col) to build a training set.

Requirements:
    pip install opencv-python numpy
"""

import cv2
import numpy as np
import csv
import os
import glob

# CONFIG
IMAGE_FOLDER  = r"C:\Users\liam.deacon\Desktop\brightness test rotate\Samples"
TEMPLATE_CSV  = r"C:\Users\liam.deacon\Desktop\brightness test rotate\array_coordinates_corrected.csv"
OUTPUT_FOLDER = r"C:\Users\liam.deacon\Desktop\brightness test rotate\Crops"

CROP_SIZE     = 160    # side length (px) of the SQUARE crop saved per LED.
                       # Match this to SAMPLE_SIZE in the brightness script so the
                       # training crops cover the same region you measure later.
                       # LED core ~80px, glow to ~150px, spacing ~170-195px.
                       # Keep <= ~190 to avoid overlapping neighbouring LEDs.
SAVE_COLOR    = True   # True  -> save the original BGR crop (full colour)
                       # False -> save just the blue channel (matches the meter)
PAD_EDGE      = True   # pad crops that fall off the image edge so every crop is
                       # exactly CROP_SIZE x CROP_SIZE (keeps training tensors uniform)

REFINE        = True   # refine the 2-corner fit against all dots (recommended)
SNAP_RADIUS   = 10     # final per-dot nudge to local peak (px). 0 to disable.

# Dot detection tuning
THRESH_FRAC   = 0.35   # brightness threshold as fraction of image max (lower = more sensitive)
MIN_AREA      = 40     # ignore blobs smaller than this (px) - filters noise


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


def load_template(path):
    """Read coordinate CSV -> (points Nx2, n_cols, n_rows, labels[(row,col)...])."""
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
    """Return centroids (Nx2) of bright blobs in the blue channel."""
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
    """Top-left = min(x+y); top-right = max(x-y). Works for moderate rotation."""
    tl = detected[np.argmin(detected[:, 0] + detected[:, 1])]
    tr = detected[np.argmax(detected[:, 0] - detected[:, 1])]
    return tl, tr


def similarity_from_2pts(src, dst):
    """Similarity transform (rotation+scale+translation) mapping src pair -> dst pair."""
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
    """One robust pass: match aligned template pts to nearest dots, re-fit with RANSAC."""
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
    """Return a size x size crop centred on (cx, cy).

    If pad is True, regions off the image edge are zero-padded so every crop is
    exactly size x size. If pad is False, the crop is clipped to the image (and
    may be smaller for edge LEDs)."""
    h, w = img.shape[:2]
    half = size // 2
    x1, y1 = int(round(cx - half)), int(round(cy - half))
    x2, y2 = x1 + size, y1 + size

    if not pad:
        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = min(w, x2), min(h, y2)
        return img[cy1:cy2, cx1:cx2]

    # zero-padded canvas, copy the in-bounds overlap into it
    if img.ndim == 3:
        canvas = np.zeros((size, size, img.shape[2]), dtype=img.dtype)
    else:
        canvas = np.zeros((size, size), dtype=img.dtype)
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(w, x2), min(h, y2)
    if sx2 <= sx1 or sy2 <= sy1:
        return canvas
    dx1, dy1 = sx1 - x1, sy1 - y1
    canvas[dy1:dy1 + (sy2 - sy1), dx1:dx1 + (sx2 - sx1)] = img[sy1:sy2, sx1:sx2]
    return canvas


def process_image(image_path, template, labels, n_cols, out_folder):
    name = os.path.splitext(os.path.basename(image_path))[0]
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"  ! could not read {name}")
        return []
    blue = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[:, :, 2].astype(np.float32)

    detected = detect_dots(blue)
    if len(detected) < 4:
        print(f"  ! too few dots detected in {name}")
        return []

    # --- corner-anchor alignment ---
    t_tl, t_tr = template[0], template[n_cols - 1]      # template top corners
    img_tl, img_tr = find_top_corners(detected)         # image top corners
    M = similarity_from_2pts([t_tl, t_tr], [img_tl, img_tr])
    aligned = cv2.transform(template.reshape(-1, 1, 2), M).reshape(-1, 2)

    # --- optional all-points refinement ---
    if REFINE:
        M2 = refine_against_all(template, aligned, detected)
        if M2 is not None:
            M = M2
            aligned = cv2.transform(template.reshape(-1, 1, 2), M).reshape(-1, 2)

    angle = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    scale = np.hypot(M[0, 0], M[1, 0])

    # per-image subfolder keeps thousands of crops tidy
    crop_dir = os.path.join(out_folder, name)
    os.makedirs(crop_dir, exist_ok=True)

    # blue channel as a saveable 8-bit image (clip just in case)
    blue_u8 = np.clip(blue, 0, 255).astype(np.uint8)

    # --- crop every LED ---
    overlay = img.copy()
    manifest = []
    half = CROP_SIZE // 2
    for (row, col), (x, y) in zip(labels, aligned):
        if SNAP_RADIUS > 0:
            x, y = snap_to_peak(blue, x, y, SNAP_RADIUS)

        source = img if SAVE_COLOR else blue_u8
        crop = crop_led(source, x, y, CROP_SIZE, PAD_EDGE)

        fname = f"{name}_r{row:02d}_c{col:02d}.png"
        cv2.imwrite(os.path.join(crop_dir, fname), crop)

        manifest.append({"image": name, "row": row, "col": col,
                         "label": led_label(row, col),
                         "x": int(round(x)), "y": int(round(y)),
                         "crop_file": os.path.join(name, fname)})

        cv2.rectangle(overlay, (int(x) - half, int(y) - half),
                      (int(x) + half, int(y) + half), (0, 255, 0), 2)

    # mark detected anchors
    for (x, y) in (img_tl, img_tr):
        cv2.circle(overlay, (int(x), int(y)), 34, (0, 0, 255), 4)

    # save downscaled verification overlay next to the crops
    h, w = overlay.shape[:2]
    cv2.imwrite(os.path.join(out_folder, f"{name}_verify.png"),
                cv2.resize(overlay, (min(1400, w), int(min(1400, w) * h / w))))

    print(f"  {name}: angle {angle:+.2f}deg, scale {scale:.3f}, "
          f"{len(detected)} dots, {len(manifest)} crops -> {crop_dir}")
    return manifest


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    template, n_cols, n_rows, labels = load_template(TEMPLATE_CSV)
    print(f"Template: {n_cols} x {n_rows} = {len(template)} dots")
    print(f"Crop size: {CROP_SIZE}px  |  {'colour' if SAVE_COLOR else 'blue-channel'}  |  "
          f"{'padded' if PAD_EDGE else 'clipped'} edges\n")

    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(IMAGE_FOLDER, e)))
    files.sort()
    if not files:
        raise FileNotFoundError(f"No images found in {IMAGE_FOLDER}")
    print(f"Processing {len(files)} image(s):\n")

    all_rows = []
    for fp in files:
        all_rows.extend(process_image(fp, template, labels, n_cols, OUTPUT_FOLDER))

    # one combined manifest across all images - the backbone of your training set
    manifest_path = os.path.join(OUTPUT_FOLDER, "crops_manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=["image", "row", "col", "label", "x", "y", "crop_file"])
        wri.writeheader(); wri.writerows(all_rows)

    print(f"\nDone. {len(all_rows)} crops written under:\n  {OUTPUT_FOLDER}")
    print(f"Manifest: {manifest_path}")
    print("\nNext: add a 'Power_uW' column to the manifest (matched on image+row+col)")
    print("to turn these crops into labelled training data.")


if __name__ == "__main__":
    main()
