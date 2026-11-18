"""E01 baseline A1: colour statistics of each scan's plain photo (color.jpg).

Mean and standard deviation of R, G, B and of H, S, V (PIL's HSV, 0-255), over the whole image and over its central box
(half the width and half the height): 24 features per photo.

    python photo_color_stats.py --images <commit dir>/dataset/images --out photo_color_stats.csv

Output columns: folder (<object>__<pose>__<scan8>), scan8, then the 24 features.
"""
import argparse
import csv
import glob
import os

import numpy as np
from PIL import Image


def stats(img):
    rgb = np.asarray(img.convert("RGB"), dtype=np.float64)
    hsv = np.asarray(img.convert("RGB").convert("HSV"), dtype=np.float64)
    out = {}
    h, w = rgb.shape[:2]
    regions = {"full": (slice(None), slice(None)), "center": (slice(h // 4, h - h // 4), slice(w // 4, w - w // 4))}
    for rname, (ys, xs) in regions.items():
        for arr, names in ((rgb, "RGB"), (hsv, "HSV")):
            part = arr[ys, xs].reshape(-1, 3)
            for i, ch in enumerate(names):
                out[f"{rname}_{ch}_mean"] = part[:, i].mean()
                out[f"{rname}_{ch}_std"] = part[:, i].std()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(a.images, "*", "*", "color.jpg")))
    rows = []
    for p in paths:
        folder = os.path.basename(os.path.dirname(p))
        with Image.open(p) as img:
            s = stats(img)
        rows.append({"folder": folder, "scan8": folder.rsplit("__", 1)[-1], **{k: f"{v:.4f}" for k, v in s.items()}})
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"{len(rows)} photos -> {a.out}")


if __name__ == "__main__":
    main()
