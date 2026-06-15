# LED Power Prediction

Predicts each LED's optical power (uW) from a photo of an LED array, using a
trained RandomForest model on per-LED image features.

## What's in this bundle

```
power_model.py                  feature extraction + the model class
extract_training_crops.py       grid alignment / cropping pipeline (shared library)
predict_power.py                RUN the model on new array images   <- main entry point
train_power_model.py            (optional) re-extract crops + retrain the model
array_coordinates_corrected.csv LED grid template (required for alignment)
Model/power_model.joblib        the trained model (required to predict)
requirements.txt                Python dependencies
Input/                          drop array images / folders here
Output/                         results (CSV + labelled overlay) land here
```

All paths are relative to this folder, so you can unzip it anywhere.

## Setup (once)

Requires **Python 3.10+**. From inside this folder:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

> The model was saved with scikit-learn 1.9.0 (pinned in requirements.txt). Use
> that version or joblib may fail to load Model/power_model.joblib.

## Predict power for new arrays

Put a **folder** in `Input/` containing the array image **and** a file of measured
LEDs. The measured LEDs are used as anchors to correct that array's exposure, so
the unmeasured LEDs are predicted on the right absolute scale:
```
Input/
  my_array/
    my_array.png
    measured.csv          # columns: label,Power_uW   (e.g. A1,12.8)
```
The measured file can also be the array's `.xlsx` workbook (its LIV sheet is read
at I_Low=0.75 mA / VDD_50LED=4.5 V, the same condition used in training).

A folder with no measured file is skipped — measurements are required so the
prediction can be calibrated to that array's exposure.

Then run:
```bash
python predict_power.py
```

For each input you get in `Output/`:
- `<name>_predicted.csv` - per-LED label, position, and predicted power. Columns:
  `predicted_raw_uW` (model alone), `predicted_calibrated_uW`, `measured_uW`
  (blank if not measured), `final_uW` (your measured value if known, else the
  calibrated prediction).
- `<name>_overlay.png` - the image with each LED boxed and labelled.

## (Optional) Retrain the model

Only needed if you want to rebuild the model from raw data. You must supply a
`training data/` folder (one sub-folder per array, each containing the stitched
image + its LIV `.xlsx`):

```bash
python train_power_model.py            # extract crops from 'training data/' then train (RandomForest)
python train_power_model.py ridge      # use Ridge regression instead
python train_power_model.py --no-extract   # train on existing 'Training Crops/' without re-extracting
```

This overwrites `Model/power_model.joblib` and writes metrics/plots to `Model/`.
The leave-one-array-out numbers it prints are the realistic accuracy on a brand-new array.
