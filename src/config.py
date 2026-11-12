"""Central configuration for Phase 1 (data cleaning, EDA, preprocessing).

Everything that affects the shape of the processed dataset lives here, so the
output can be matched to whatever the already-trained model expects.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Raw data location. Override with:  export ASL_DATA_DIR=/path/to/dataset5
DATA_DIR = Path(os.environ["ASL_DATA_DIR"]) if os.environ.get("ASL_DATA_DIR") else ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
IMAGES_DIR = ROOT / "images"

KAGGLE_HANDLE = "mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out"

# --- Target format: must match what the trained model/backend already expects ---
IMG_SIZE = 100                 # 100x100 (as in the reference paper / existing model)
IMG_MODE = "RGB"               # 3 channels, color images only (depth is not used)
LETTERS = list("abcdefghiklmnopqrstuvwxy")   # 24 static ASL letters (no j, z: they involve motion)
SAMPLES_PER_CLASS = 470        # per-class cap used for a balanced set (reference paper uses 470)
TEST_SIZE = 0.2                # 80/20 split (the paper's 5-fold CV is a 80/20 split per fold)
SEED = 42

# --- Cleaning thresholds ---
MIN_SIDE_PX = 20               # smaller than this on either side -> unusable crop
MIN_MEAN_BRIGHTNESS = 10       # almost black frame
MAX_MEAN_BRIGHTNESS = 245      # almost white frame
MIN_CONTRAST = 3               # grey-level std below this -> blank / uniform frame
IQR_FACTOR = 3.0               # "extreme" outlier fence (Q1 - k*IQR, Q3 + k*IQR)
DROP_NEAR_DUPLICATES = True    # drop perceptual-hash duplicates within the same (user, letter)
