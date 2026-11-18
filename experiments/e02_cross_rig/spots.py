"""E02: red-laser spot crops and spot features from both rigs, computed the same way on each.

    python -P spots.py --tdai C:/Users/adain/wavecomputing_data/3dai/6556bec6898e/dataset/images \
        --arm C:/Users/adain/wavecomputing_data/arm_96b76af/dataset --boxes <boxes.csv> --out ../../data/e02

3DAI: the two 635 nm channels of every scan (laser_ch1.jpg, laser_ch2.jpg); the spot is found as the maximum of the
smoothed red excess of (laser frame - laser_dark), and nothing else from the dark frame is used. Arm: primary photos
only; the spot starts at the centre of its labeled box and is refined to the maximum of the smoothed red excess inside
the box. From there both rigs are processed identically: background = median red excess on a ring around the spot,
half-maximum radius r50 from the radial profile, features measured in units of r50 (so the cameras' different pixel
scales cancel). Writes features_3dai.csv, features_arm.csv and a contact sheet of crops for quality control.
"""
import argparse
import csv
import glob
import os

import numpy as np
from PIL import Image, ImageDraw

FEATS = ["prof_0.5", "prof_1", "prof_1.5", "prof_2", "prof_3", "prof_4", "core_r", "core_g", "ring_r", "ring_g",
         "core_sat", "speckle", "log_peak_bg", "ellip", "halo_frac", "w10_w50"]


def red_excess(a):
    return np.clip(a[..., 0] - (a[..., 1] + a[..., 2]) / 2.0, 0, None)


def smooth(x, k=2):
    """box blur of radius k (separable)"""
    if k <= 0:
        return x
    c = np.cumsum(np.pad(x, ((k + 1, k), (0, 0)), mode="edge"), 0)
    x = (c[2 * k + 1:] - c[:-2 * k - 1]) / (2 * k + 1)
    c = np.cumsum(np.pad(x, ((0, 0), (k + 1, k)), mode="edge"), 1)
    return (c[:, 2 * k + 1:] - c[:, :-2 * k - 1]) / (2 * k + 1)


def features(a, cx, cy, r_init):
    """a: HxWx3 float RGB; (cx, cy) spot centre; r_init: initial window radius in pixels. Returns dict or None."""
    H, W, _ = a.shape
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.hypot(xx - cx, yy - cy)
    ex = red_excess(a)
    ring = (rr >= 0.8 * r_init) & (rr <= r_init)
    if ring.sum() < 10:
        return None
    bg = float(np.median(ex[ring]))
    e = np.clip(ex - bg, 0, None)
    win = rr <= r_init
    edges = np.arange(0, r_init + 1, 1.0)
    prof = np.array([e[(rr >= lo) & (rr < lo + 1)].mean() if ((rr >= lo) & (rr < lo + 1)).any() else 0.0
                     for lo in edges[:-1]])
    peak = prof[:2].max()
    if peak <= 5:  # no visible spot above background
        return None

    def radius(frac):
        below = np.nonzero(prof <= frac * peak)[0]
        return float(below[0]) + 0.5 if len(below) else float(r_init)
    r50, r10 = max(radius(0.5), 1.0), radius(0.1)

    def at(k):
        m = (rr >= (k - 0.25) * r50) & (rr < (k + 0.25) * r50)
        return float(e[m].mean() / peak) if m.any() else 0.0
    core = rr <= 0.5 * r50 + 0.5
    ringm = (rr >= 1.5 * r50) & (rr <= 3 * r50)
    tot = a.sum(2) + 1e-6
    chrom = lambda m, ch: float((a[..., ch][m] / tot[m]).mean()) if m.any() else 0.0
    ann = (rr >= r50) & (rr <= 2 * r50)
    spk = float(e[ann].std() / (e[ann].mean() + 1e-6)) if ann.any() else 0.0
    m2 = rr <= 2 * r50
    w = e[m2] + 1e-9
    dx, dy = xx[m2] - cx, yy[m2] - cy
    cov = np.array([[np.sum(w * dx * dx), np.sum(w * dx * dy)], [np.sum(w * dx * dy), np.sum(w * dy * dy)]]) / w.sum()
    ev = np.sort(np.linalg.eigvalsh(cov))
    inner, outer = e[rr <= 2 * r50].sum(), e[rr <= min(4 * r50, r_init)].sum()
    return {"prof_0.5": at(0.5), "prof_1": at(1), "prof_1.5": at(1.5), "prof_2": at(2), "prof_3": at(3),
            "prof_4": at(4), "core_r": chrom(core, 0), "core_g": chrom(core, 1), "ring_r": chrom(ringm, 0),
            "ring_g": chrom(ringm, 1), "core_sat": float((a[core].max(1) >= 250).mean()) if core.any() else 0.0,
            "speckle": spk, "log_peak_bg": float(np.log((peak + bg + 1) / (bg + 1))),
            "ellip": float(np.sqrt(ev[0] / ev[1])) if ev[1] > 0 else 1.0,
            "halo_frac": float(1 - inner / outer) if outer > 0 else 0.0, "w10_w50": r10 / r50,
            "_r50": r50, "_cx": cx, "_cy": cy}


def refine(ex, x0, y0, x1, y1):
    H, W = ex.shape
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(W, int(x1) + 1), min(H, int(y1) + 1)
    sub = smooth(ex[y0:y1, x0:x1], 2)
    y, x = np.unravel_index(np.argmax(sub), sub.shape)
    return x0 + x, y0 + y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tdai", required=True, help="3DAI dataset/images folder")
    ap.add_argument("--arm", required=True, help="arm dataset folder (train/, valid/)")
    ap.add_argument("--boxes", required=True, help="per-box table with materials (boxes.csv)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    thumbs = []
    # ---- 3DAI
    rows3 = []
    for scan in sorted(glob.glob(os.path.join(a.tdai, "*", "*"))):
        material = os.path.basename(os.path.dirname(scan))
        name = os.path.basename(scan)
        obj = name.split("__angle")[0]
        dark_p = os.path.join(scan, "laser_dark.jpg")
        if not os.path.exists(dark_p):
            continue
        dark = np.asarray(Image.open(dark_p).convert("RGB"), dtype=float)
        for ch in ("laser_ch1", "laser_ch2"):
            p = os.path.join(scan, ch + ".jpg")
            if not os.path.exists(p):
                continue
            img = np.asarray(Image.open(p).convert("RGB"), dtype=float)
            if img.shape != dark.shape:
                continue
            d = smooth(red_excess(np.clip(img - dark, 0, None)), 2)
            y, x = np.unravel_index(np.argmax(d), d.shape)
            f = features(img, x, y, 40)
            row = {"rig": "3dai", "sample": f"{name}/{ch}", "object": obj, "material": material, "channel": ch,
                   "ok": f is not None}
            if f:
                row.update(f)
                if len(thumbs) < 24 and ch == "laser_ch1":
                    thumbs.append((img, f, f"3DAI {material}: {obj[:22]}"))
            rows3.append(row)
    # ---- arm
    rowsa = []
    for b in csv.DictReader(open(a.boxes, encoding="utf-8")):
        if b["variant"] != "primary":
            continue
        p = os.path.join(a.arm, b["split"], "images", b["image_file"])
        if not os.path.exists(p):
            continue
        img = np.asarray(Image.open(p).convert("RGB"), dtype=float)
        cx, cy, w, h = (float(b[k]) for k in ("cx_px", "cy_px", "w_px", "h_px"))
        x, y = refine(red_excess(img), cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        f = features(img, x, y, max(w, h) * 1.5)
        row = {"rig": "arm", "sample": f"{b['image_file']}#{b['box_index']}", "object": b["object_id"],
               "object_name": b["object_name"], "material": b["material_coarse"], "material_fine": b["material"],
               "capture_date": b["capture_date"], "ok": f is not None}
        if f:
            row.update(f)
            if len(thumbs) < 48 and len(thumbs) >= 24 and b["material_coarse"] in ("plastic", "metal"):
                thumbs.append((img, f, f"ARM {b['material_coarse']}: {b['object_name'][:22]}"))
        rowsa.append(row)
    for name, rows in (("features_3dai.csv", rows3), ("features_arm.csv", rowsa)):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(os.path.join(a.out, name), "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys)
            wr.writeheader()
            wr.writerows(rows)
        print(name, len(rows), "rows,", sum(r["ok"] for r in rows), "with a spot")
    # ---- contact sheet of crops (4 x r50 around the spot)
    T = 120
    sheet = Image.new("RGB", (8 * T, ((len(thumbs) + 7) // 8) * (T + 14)), "white")
    dr = ImageDraw.Draw(sheet)
    for i, (img, f, title) in enumerate(thumbs):
        R = max(4 * f["_r50"], 6)
        im = Image.fromarray(img.astype(np.uint8)).crop((int(f["_cx"] - R), int(f["_cy"] - R),
                                                          int(f["_cx"] + R), int(f["_cy"] + R))).resize((T, T))
        x, y = (i % 8) * T, (i // 8) * (T + 14)
        sheet.paste(im, (x, y + 14))
        dr.text((x + 2, y + 1), title[:24], fill=(0, 0, 0))
    sheet.save(os.path.join(a.out, "crops_sheet.jpg"), quality=85)


if __name__ == "__main__":
    main()
