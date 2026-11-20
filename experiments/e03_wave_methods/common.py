"""E03 shared code: the scan table, linear (not gamma-encoded) laser frames, clipping, spot centre.

The 3DAI frames on GitHub are sRGB JPEG previews (longest side 800 px, 4:2:0 chroma). Light adds up in linear units,
so every E03 method first undoes the sRGB curve; pixels at or above SAT8 in the laser frame are treated as clipped
(their true value is only known to be at least the clip level), never as measurements.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

SAT8 = 245                       # 8-bit level from which a pixel counts as clipped (JPEG ringing pulls 255 down a bit)
CHANNELS = {"ch1": ("laser_ch1.jpg", 0, 635),   # name: file, camera channel that carries the laser, wavelength nm
            "ch2": ("laser_ch2.jpg", 0, 635),
            "ch4": ("laser_ch4.jpg", 1, 530)}   # ch3 (940 nm) is not seen by the colour camera
PRIMARY = ["cardboard", "foam", "metal", "paper", "plastic"]  # materials with at least two objects (E01's set)
POSSIBLY_MISLABELLED = ["taped black metal", "taped breadboard"]


def srgb_to_linear(v8):
    v = np.asarray(v8, dtype=float) / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


LIN_SAT = float(srgb_to_linear(SAT8))


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=float)


def scan_table(features_csv, images_root):
    """One row per scan that has laser features (E01's 80 scans): scan id, object label, material, pose, folder."""
    lf = pd.read_csv(features_csv)
    first = lf.groupby("scan_id").first().reset_index()
    folders = {os.path.basename(p).rsplit("__", 1)[-1]: p for p in glob.glob(os.path.join(images_root, "*", "*"))}
    rows = []
    for _, r in first.iterrows():
        d = folders.get(str(r["scan_id"])[:8])
        rows.append({"scan_id": r["scan_id"], "scan8": str(r["scan_id"])[:8], "object": r["label"],
                     "material": r["material_class"], "pose": r["angle_index"], "dir": d,
                     "surface": r.get("surface"), "transparency": r.get("transparency")})
    t = pd.DataFrame(rows)
    assert t["scan8"].is_unique
    return t


def box_blur(x, k):
    """mean over a (2k+1)^2 window, edges padded by replication"""
    if k <= 0:
        return x
    c = np.cumsum(np.pad(x, ((k + 1, k), (0, 0)), mode="edge"), 0)
    x = (c[2 * k + 1:] - c[:-2 * k - 1]) / (2 * k + 1)
    c = np.cumsum(np.pad(x, ((0, 0), (k + 1, k)), mode="edge"), 1)
    return (c[:, 2 * k + 1:] - c[:, :-2 * k - 1]) / (2 * k + 1)


def laser_only(scan_dir, ch):
    """Laser light alone in the laser's camera channel, linear units: D = lin(laser) - lin(dark).

    Returns (D, lower, valid, laser8) where lower[i] is the lower bound of D at clipped pixels (NaN elsewhere) and valid
    marks pixels where the laser can be measured at all (the dark frame itself not clipped). None if a file is missing.
    """
    fn, col, _ = CHANNELS[ch]
    pl, pd_ = os.path.join(scan_dir, fn), os.path.join(scan_dir, "laser_dark.jpg")
    if not (os.path.exists(pl) and os.path.exists(pd_)):
        return None
    a8, d8 = load_rgb(pl), load_rgb(pd_)
    if a8.shape != d8.shape:
        return None
    a, d = srgb_to_linear(a8[..., col]), srgb_to_linear(d8[..., col])
    D = a - d
    clipped = a8[..., col] >= SAT8
    lower = np.where(clipped, LIN_SAT - d, np.nan)
    valid = d8[..., col] < SAT8
    return D, lower, valid, a8


def spot_centre(D, valid, lower):
    """Brightest point of the smoothed laser-only image; if it is clipped, the centroid of the clipped blob around it."""
    s = box_blur(np.where(valid, D, 0.0), 1)
    y, x = np.unravel_index(np.argmax(s), s.shape)
    clip = ~np.isnan(lower)
    if clip[y, x]:
        H, W = D.shape
        y0, y1, x0, x1 = max(0, y - 25), min(H, y + 26), max(0, x - 25), min(W, x + 26)
        sub = clip[y0:y1, x0:x1]
        # keep the clipped pixels connected to (y, x) by a simple flood fill
        from scipy.ndimage import label
        lab, _ = label(sub)
        m = lab == lab[y - y0, x - x0]
        yy, xx = np.nonzero(m)
        return float(xx.mean() + x0), float(yy.mean() + y0), int(m.sum())
    return float(x), float(y), 0


def import_e01():
    """E01's code (feature table and model), so the baselines are built exactly as there."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "..", "e01_material_unseen_objects"))
    import run_e01  # noqa: E402
    return run_e01
