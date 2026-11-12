"""Helpers shared by the Phase 1 notebooks."""
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from . import config

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def locate_raw_data(download_if_missing=True):
    """Return the folder holding the raw images.

    Order: ASL_DATA_DIR / data/raw (if it already contains images) -> kagglehub download.
    """
    def has_images(p):
        return p.exists() and any(f.suffix.lower() in IMG_EXTS for f in p.rglob("*") if f.is_file())

    if has_images(config.DATA_DIR):
        return config.DATA_DIR
    if not download_if_missing:
        raise FileNotFoundError(f"No images found in {config.DATA_DIR}")
    import kagglehub  # needs Kaggle credentials (~/.kaggle/kaggle.json) or KAGGLE_USERNAME/KAGGLE_KEY

    path = Path(kagglehub.dataset_download(config.KAGGLE_HANDLE))
    print("Downloaded to:", path)
    return path


def parse_path(path, root):
    """Extract (kind, user, letter, frame) from a path like  <root>/dataset5/A/b/color_1_0002.png."""
    rel = path.relative_to(root)
    parts = rel.parts
    name = path.stem.lower()
    kind = "depth" if name.startswith("depth") else "color" if name.startswith("color") else "other"
    letter = path.parent.name.lower()
    if not (len(letter) == 1 and letter.isalpha()):
        letter = "unknown"
    user = next((p for p in parts[:-1] if len(p) == 1 and p.isupper()), "unknown")
    m = re.search(r"(\d+)_(\d+)$", name)
    frame = f"{m.group(1)}_{m.group(2)}" if m else name
    return kind, user, letter, frame


def scan_files(root):
    """Cheap inventory (no pixel reads): one row per image file."""
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            kind, user, letter, frame = parse_path(p, root)
            rows.append(dict(path=str(p), rel_path=str(p.relative_to(root)), ext=p.suffix.lower(),
                             size_bytes=p.stat().st_size, kind=kind, user=user, letter=letter, frame=frame))
    return pd.DataFrame(rows)


def _features(path):
    """Read one image and return quality features (or an error string)."""
    out = dict(path=path, readable=True, error=None)
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
        out["md5"] = hashlib.md5(raw).hexdigest()
        with Image.open(path) as im:
            im.load()
            out["width"], out["height"], out["mode"] = im.width, im.height, im.mode
            rgb = im.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float32)
        gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        out["mean_r"], out["mean_g"], out["mean_b"] = (float(arr[..., i].mean()) for i in range(3))
        out["brightness"] = float(gray.mean())
        out["contrast"] = float(gray.std())
        out["sharpness"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        out["phash"] = str(imagehash.phash(rgb))
    except Exception as e:  # corrupt / truncated / not an image
        out.update(readable=False, error=f"{type(e).__name__}: {e}"[:120])
    return out


def compute_features(paths, workers=8):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return pd.DataFrame(list(ex.map(_features, paths, chunksize=64)))


def iqr_bounds(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def load_for_model(path, size=None):
    """Exactly the transform used to build the processed dataset: RGB -> size x size (uint8)."""
    size = size or config.IMG_SIZE
    with Image.open(path) as im:
        return im.convert(config.IMG_MODE).resize((size, size), Image.BILINEAR)
