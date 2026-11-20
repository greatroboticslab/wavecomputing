"""E03 method 1: speckle degrees of freedom, a polarizer-free reading of depolarization and path-length spread.

    python -P m1_speckle.py --simulate --out ../../results/e03            # estimator check with a known answer
    python -P m1_speckle.py --features <laser_features.csv> --images <3DAI images> --m2 ../../data/e03/features_m2_spots.csv --out ../../data/e03

Physics. Laser light returned from a rough surface forms speckle whose intensity follows an exponential law (contrast
K = 1) when all of it shares one polarization and its path lengths differ by less than the laser's coherence length.
Light that wanders inside a material loses both: its two polarizations make two independent speckle patterns, and
paths longer than the coherence length add more. The sum of M independent patterns has K^2 = 1/M. So M, read from
the speckle in a ring around the spot, says how much of the returned light was depolarized or delayed, with no
polarizer: M = 1 for light returned at the surface (bare metal), M = 2 and more for light returned from inside
(plastic, paper, foam). The camera's own blurring of speckle multiplies every M by the same factor, which a ratio
between rings or against a metal reference cancels.

Estimator. K_I^2 from the laser-only image normalized by its smooth radial trend. Camera noise is removed by
multiplying two frames taken with the same laser (their noise is independent, the speckle is not), and the surface's
own texture (print, grain) by the plain white-light frame: K_s^2 = (1 + K_I^2) / (1 + K_T^2) - 1, M = 1 / K_s^2.

The 3DAI files on GitHub hold one frame per laser, as an 800 px JPEG preview. --simulate renders spots with known M
and runs the estimator on raw linear frames (two per laser) and on frames passed through a preview pipeline (sRGB,
downscaling, 8-bit, JPEG quality 85 with 4:2:0 chroma, one frame per laser), to see whether the files can show it.
"""
import argparse
import io
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this folder (our code); python -P leaves it out
from common import box_blur, srgb_to_linear  # noqa: E402

N = 240


def speckle(n, grain, rng):
    """fully developed speckle, mean 1, grain about `grain` px (circular pupil)"""
    f = np.fft.fftfreq(n)
    FX, FY = np.meshgrid(f, f)
    pupil = np.hypot(FX, FY) <= 1.0 / (2.0 * grain)
    E = np.fft.ifft2(np.fft.fft2(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) * pupil)
    I = np.abs(E) ** 2
    return I / I.mean()


def linear_to_srgb8(x):
    x = np.clip(x, 0, 1)
    v = np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
    return np.clip(np.round(v * 255), 0, 255).astype(np.uint8)


def radial_mean(img, cx, cy, rmax):
    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(xx - cx, yy - cy)
    rb = np.floor(r).astype(int)
    sums = np.bincount(rb.ravel(), img.ravel(), minlength=rmax + 2)
    cnt = np.bincount(rb.ravel(), minlength=rmax + 2)
    prof = sums / np.maximum(cnt, 1)
    return np.interp(r, np.arange(len(prof)) + 0.5, prof), r


def k2_texture(dark_a, dark_b, mask):
    """texture contrast^2 of the white-light frame in the ring, noise removed by the cross-product of two frames"""
    sa, sb = box_blur(dark_a, 6), box_blur(dark_b, 6)
    ra, rb = dark_a[mask] / sa[mask], dark_b[mask] / sb[mask]
    return float(np.mean(ra * rb) - np.mean(ra) * np.mean(rb))


def m_estimate(D_a, D_b, dark_a, dark_b, cx, cy, r1, r2, clipped=None):
    """M from the ring r1..r2. D_b/dark_b may be None (one frame only): then camera noise stays in."""
    two = D_b is not None
    Dm = 0.5 * (D_a + D_b) if two else D_a
    trend, r = radial_mean(Dm, cx, cy, int(r2) + 4)
    mask = (r >= r1) & (r <= r2) & (trend > 0)
    if clipped is not None:
        mask &= ~clipped
    if mask.sum() < 30:
        return np.nan, np.nan
    # the surface's coarse shading modulates the laser light too: take it out with the white-light frame, so that
    # only fine texture (handled by K_T) and speckle are left
    sd = box_blur(dark_a, 6)
    tl = sd / np.mean(sd[mask])
    xa = D_a[mask] / (trend[mask] * tl[mask])
    xb = (D_b[mask] / (trend[mask] * tl[mask])) if two else xa
    m2 = np.mean(xa * xb) if two else np.mean(xa * xa)
    k2_i = m2 / (np.mean(0.5 * (xa + xb))) ** 2 - 1
    k2_t = k2_texture(dark_a, dark_b if dark_b is not None else dark_a, mask)
    k2_s = (1 + k2_i) / (1 + k2_t) - 1
    return float(1.0 / max(k2_s, 1e-3)), float(k2_s)


def render(M, grain, rng, full_well=10000.0, read=5.0):
    """laser frames (two) and white-light frames (two) of one spot, linear, 12-bit; laser in the red channel"""
    yy, xx = np.mgrid[0:N, 0:N]
    cx = cy = N / 2 + rng.uniform(-0.5, 0.5)
    r = np.hypot(xx - cx, yy - cy)
    env = 0.55 * np.exp(-0.5 * (r / 4.0) ** 2) + 0.25 * np.exp(-r / 10.0)
    S = sum(speckle(N, grain, rng) for _ in range(M)) / M
    T = np.exp(box_blur(rng.normal(0, 0.1, (N, N)), 2) * 3.0)
    T /= T.mean()
    amb = 0.15

    def shot(x):
        e = rng.poisson(np.clip(x, 0, None) * full_well) + rng.normal(0, read, x.shape)
        return np.clip(np.round(e / full_well * 4095), 0, 4095) / 4095

    laser = [shot(T * (amb + env * S)) for _ in range(2)]
    dark = [shot(T * amb) for _ in range(2)]
    return laser, dark, cx, cy


def preview(img_red, img_amb, factor, quality=85):
    """sRGB, downscale, 8-bit, JPEG with 4:2:0 chroma, decode, back to linear; returns the red channel"""
    rgb = np.stack([linear_to_srgb8(img_red), linear_to_srgb8(img_amb), linear_to_srgb8(img_amb)], -1)
    im = Image.fromarray(rgb)
    if factor != 1.0:
        im = im.resize((int(round(N / factor)), int(round(N / factor))), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, subsampling=2)
    out = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), float)
    return srgb_to_linear(out[..., 0])


def auc(a, b):
    a, b = np.asarray(a), np.asarray(b)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(np.mean([(x < y) + 0.5 * (x == y) for x in a for y in b]))


def simulate(out):
    rng = np.random.default_rng(0)
    reps = 30
    res = {}
    for grain in (1.5, 3.0):
        for pipe in ("raw_two_frames", "raw_one_frame", "preview_x1.0", "preview_x1.6", "preview_x2.4"):
            est = {}
            for M in (1, 2, 4):
                vals = []
                for _ in range(reps):
                    laser, dark, cx, cy = render(M, grain, rng)
                    if pipe == "raw_two_frames":
                        m, _ = m_estimate(laser[0] - dark[0], laser[1] - dark[1], dark[0], dark[1], cx, cy, 6, 18)
                    elif pipe == "raw_one_frame":
                        m, _ = m_estimate(laser[0] - dark[0], None, dark[0], None, cx, cy, 6, 18)
                    else:
                        f = float(pipe.split("x")[1])
                        L = preview(laser[0], dark[0], f)
                        Dk = preview(dark[0], dark[0], f)
                        m, _ = m_estimate(L - Dk, None, Dk, None, cx / f, cy / f, 6 / f, 18 / f)
                    vals.append(m)
                est[M] = vals
            res[f"grain{grain}_{pipe}"] = {
                "median_M_hat": {str(M): float(np.nanmedian(v)) for M, v in est.items()},
                "iqr_M_hat": {str(M): [float(np.nanpercentile(v, 25)), float(np.nanpercentile(v, 75))] for M, v in est.items()},
                "auc_1_vs_2": auc(est[1], est[2]), "auc_2_vs_4": auc(est[2], est[4])}
            r = res[f"grain{grain}_{pipe}"]
            print(f"grain {grain} {pipe:15s} M_hat medians {r['median_M_hat']}  AUC 1v2 {r['auc_1_vs_2']:.2f}  2v4 {r['auc_2_vs_4']:.2f}",
                  flush=True)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "m1_speckle_simulation.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"reps_per_cell": reps, "ring_px_full_res": [6, 18], "results": res,
                   "note": "M_hat = 1/K_s^2; the camera's pixel and blur effects scale all M alike, so compare M across "
                           "materials within a pipeline, not against the true M"}, f, indent=1)


def real(args):
    """the one-frame estimator on the 3DAI previews, in a ring just outside each clipped core (exploratory)"""
    import pandas as pd
    from common import CHANNELS, laser_only, load_rgb, scan_table
    t = scan_table(args.features, args.images)
    sp = pd.read_csv(args.m2).set_index(["scan_id", "ch"])
    rows = []
    for s in t.itertuples():
        if s.dir is None:
            continue
        dk = srgb_to_linear(load_rgb(os.path.join(s.dir, "laser_dark.jpg")))
        for ch in CHANNELS:
            key = (s.scan_id, ch)
            if key not in sp.index or not bool(sp.loc[key, "ok"]):
                continue
            q = sp.loc[key]
            got = laser_only(s.dir, ch)
            D, lower, valid, _ = got
            col = CHANNELS[ch][1]
            r1 = float(q["r_clip"]) + 2
            m, k2 = m_estimate(D, None, dk[..., col], None, float(q["x0"]), float(q["y0"]), r1, r1 + 10,
                               clipped=~np.isnan(lower) | ~valid)
            rows.append({"scan_id": s.scan_id, "ch": ch, "M_hat": m, "K2_s": k2})
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "features_m1_spots.csv"), index=False)
    print(pd.DataFrame(rows).groupby("ch")["M_hat"].describe().round(2).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--features")
    ap.add_argument("--images")
    ap.add_argument("--m2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    simulate(a.out) if a.simulate else real(a)
