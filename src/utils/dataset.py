
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


#### ---------------------------------------------
#### Funkcie tykajuce sa datasetu a suborov
#### ---------------------------------------------
def get_bit_depth(img):
    mode = img.mode

    # Špeciálne módy s definovanou bit-depth
    if mode == "I;16":
        return 16
    if mode == "I":
        return 32
    if mode == "F":
        return 32  # float32
    if mode == "1":
        return 1   # 1-bit masky

    try:
        arr = np.array(img)
        return arr.dtype.itemsize * 8
    except:
        return None


def load_dataset_info(dataset_path):
    VALID_EXTS = {".jpg", ".jpeg", ".png"}

    filepaths = [p for p in Path(dataset_path).rglob("*") if p.suffix.lower() in VALID_EXTS]

    records = []

    for p in filepaths:
        try:
            img = Image.open(p)

            # rozmery
            w, h = img.size

            # farebný model (RGB, RGBA, L, P, I;16, ...)
            color_mode = img.mode

            # bitová hĺbka
            bit_depth = get_bit_depth(img)

            # brightness rátame vždy z RGB
            brightness = np.array(img.convert("RGB")).mean()

        except Exception as e:
            w, h = None, None
            color_mode = None
            bit_depth = None
            brightness = None

        records.append({
            "filepath": str(p),
            "label": p.parent.name,
            "format": p.suffix.lower(),
            "color_mode": color_mode,
            "bit_depth": bit_depth,
            "width": w,
            "height": h,
            "area": None if w is None else w * h,
            "aspect_ratio": None if w is None or h is None else w / h,
            "brightness": brightness
        })

    df = pd.DataFrame(records)
    return df

def find_image_extensions(root):
    files = glob.glob(os.path.join(root, "**", "*.*"), recursive=True)
    extensions = set()

    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext:
            extensions.add(ext)

    return extensions

def pick_random_custom_size(df, target_h, target_w, top_k=20):
    df = df.copy()
    df["size_diff"] = (df["height"] - target_h).abs() + (df["width"] - target_w).abs()
    closest = df.sort_values("size_diff").head(top_k)
    return closest.sample(1).filepath.values[0]

def pick_random(df, mask, k=1):
    subset = df[mask]
    if len(subset) == 0:
        return None
    return subset.sample(k).filepath.values[0]