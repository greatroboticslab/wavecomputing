"""E03 method 2: split each laser spot into light reflected at the surface and light that went under it.

    python -P m2_scatter.py --features <laser_features.csv> --images <3DAI dataset/images> --out ../../data/e03
    python -P m2_scatter.py --synthetic --out <dir>        # recovers known parameters from made-up spots

Model of the laser-only image D = lin(laser) - lin(dark) around the spot, in linear units:

    D(r) = A g(r) + B (g * R_d)(r) + c

g is the beam as this camera sees it (a Gaussian of width sigma_c), R_d the normalized diffusion profile of
Christensen and Burley (2015), a one-parameter fit to Monte Carlo subsurface transport that integrates to 1, and c a
background offset. A is the light returned at the surface, B the light that travelled under it, d how far it travels.

sigma_c is calibrated per laser without labels: the narrowest spots of that laser in the whole data set (10th
percentile of a single-Gaussian fit) are taken as the beam itself, since no spot can be narrower than the beam.
Clipped pixels are censored (the model is only asked to reach their clip level), so a saturated core costs no bias.

Features per laser, none of which uses brightness or colour: surface fraction A/(A+B); log(d / sigma_c), the
subsurface length in beam widths; the fraction of returned light outside 2.5 beam widths; and the extra elongation of
the subsurface halo over the beam's own footprint (fibres spread light along themselves; tilt stretches both alike).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import j0

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this folder (our code); python -P leaves it out
from common import CHANNELS, laser_only, scan_table, spot_centre  # noqa: E402

RW = 30          # fit window radius, px
R_BG = (32, 40)  # half-size of the cut-out window, px (window radius RW plus margin)
R_NOISE = 60     # noise scale: robust spread of D farther than this from the spot (no laser light there)
SNR_MIN = 5.0    # spots fainter than this are not fitted
BORDER = 35      # spots closer than this to the frame edge are not fitted
D_GRID = np.geomspace(0.15, 400.0, 90)
C_LO, C_HI = -0.1, 0.3  # flat offset in the window: frame-to-frame drift, or a laser's own wide faint glow
R_GRID = np.arange(0.0, 46.0, 0.05)


def burley_ft(k, d):
    """2D Fourier transform of the normalized diffusion profile (e^{-r/d} + e^{-r/3d}) / (8 pi d r); equals 1 at k=0"""
    return (1.0 / np.sqrt(k ** 2 + d ** -2) + 1.0 / np.sqrt(k ** 2 + (3 * d) ** -2)) / (4 * d)


def profile_table(sigma):
    """(g * R_d)(r) on R_GRID for every d in D_GRID, by the Hankel transform; and the encircled energy of each"""
    k = np.concatenate([np.geomspace(1e-6, 0.05, 500)[:-1], np.linspace(0.05, 14.0 / sigma, 4000)])
    G = np.exp(-0.5 * (sigma * k) ** 2)
    J = j0(np.outer(R_GRID, k))
    H = np.empty((len(D_GRID), len(R_GRID)))
    for i, d in enumerate(D_GRID):
        H[i] = np.trapezoid(J * (burley_ft(k, d) * G * k)[None, :], k, axis=1) / (2 * np.pi)
    E = np.concatenate([np.zeros((len(D_GRID), 1)),
                        np.cumsum(0.5 * (H[:, 1:] * R_GRID[1:] + H[:, :-1] * R_GRID[:-1]) * np.diff(R_GRID), 1)
                        * 2 * np.pi], 1)
    return H, E


def gauss(r, s):
    return np.exp(-0.5 * (r / s) ** 2) / (2 * np.pi * s ** 2)


class Table:
    def __init__(self, sigma):
        self.sigma = sigma
        self.H, self.E = profile_table(sigma)
        self.logd = np.log(D_GRID)

    def _row(self, M, logd):
        j = int(np.clip(np.searchsorted(self.logd, logd) - 1, 0, len(self.logd) - 2))
        w = (logd - self.logd[j]) / (self.logd[j + 1] - self.logd[j])
        return (1 - w) * M[j] + w * M[j + 1]

    def h(self, r, logd):
        return np.interp(r, R_GRID, self._row(self.H, logd))

    def enclosed(self, R, logd):
        return float(np.interp(R, R_GRID, self._row(self.E, logd)))


def noise_scale(D, lower, valid, x, y):
    """robust spread of the laser-only image far from the spot, where no laser light falls"""
    H, W = D.shape
    yy, xx = np.ogrid[0:H, 0:W]
    far = ((xx - x) ** 2 + (yy - y) ** 2 > R_NOISE ** 2) & valid & np.isnan(lower)
    v = D[far]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) + 1e-4


def window(D, lower, valid, x, y, s0):
    H, W = D.shape
    x0, x1, y0, y1 = int(x) - R_BG[1], int(x) + R_BG[1] + 1, int(y) - R_BG[1], int(y) + R_BG[1] + 1
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    sl = (slice(y0, y1), slice(x0, x1))
    r = np.hypot(xx - x, yy - y)
    m = (r <= RW) & valid[sl]
    return {"xx": xx[m].astype(float), "yy": yy[m].astype(float), "D": D[sl][m], "low": lower[sl][m], "c0": 0.0,
            "s0": s0}


def residuals(model, w):
    cen = ~np.isnan(w["low"])
    obs = np.where(cen, w["low"], w["D"])
    s = np.sqrt(w["s0"] ** 2 + (0.05 * np.clip(obs, 0, None)) ** 2)
    res = (w["D"] - model) / s
    res[cen] = np.clip(w["low"][cen] - model[cen], 0, None) / s[cen]
    return res


def fit_gauss(w, x, y):
    """single Gaussian A g_sigma + c, censored; for the label-free beam width"""
    def f(p):
        A, ls, x0, y0, c = p
        return residuals(A * gauss(np.hypot(w["xx"] - x0, w["yy"] - y0), np.exp(ls)) + c, w)
    A0 = max(float(np.nansum(np.where(np.isnan(w["low"]), w["D"], w["low"]))), 1e-3)
    best = None
    for s_init in (1.0, 3.0, 8.0):
        p0 = [A0, np.log(s_init), x, y, w["c0"]]
        try:
            r = least_squares(f, p0, bounds=([0, np.log(0.4), x - 4, y - 4, C_LO], [np.inf, np.log(20), x + 4, y + 4, C_HI]),
                              loss="soft_l1", f_scale=2.0, max_nfev=400)
        except ValueError:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best


def fit_two(w, x, y, tab):
    """A g + B (g * R_d) + c, censored"""
    sig = tab.sigma

    def model(p):
        A, B, logd, x0, y0, c = p
        r = np.hypot(w["xx"] - x0, w["yy"] - y0)
        return A * gauss(r, sig) + B * tab.h(r, logd) + c

    tot = max(float(np.nansum(np.where(np.isnan(w["low"]), w["D"] - w["c0"], w["low"]))), 1e-3)
    best = None
    for frac, d_init in ((0.7, 0.5 * sig), (0.5, 2 * sig), (0.3, 8 * sig)):
        p0 = [frac * tot, (1 - frac) * tot, np.log(d_init), x, y, w["c0"]]
        try:
            r = least_squares(lambda p: residuals(model(p), w), p0,
                              bounds=([0, 0, np.log(D_GRID[0]), x - 4, y - 4, C_LO],
                                      [np.inf, np.inf, np.log(D_GRID[-1]), x + 4, y + 4, C_HI]),
                              loss="soft_l1", f_scale=2.0, max_nfev=600)
        except ValueError:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best, model


def moments_axis_ratio(xx, yy, wgt, x0, y0):
    if wgt.sum() <= 0 or len(wgt) < 6:
        return np.nan
    dx, dy = xx - x0, yy - y0
    C = np.array([[np.sum(wgt * dx * dx), np.sum(wgt * dx * dy)], [np.sum(wgt * dx * dy), np.sum(wgt * dy * dy)]]) / wgt.sum()
    ev = np.sort(np.linalg.eigvalsh(C))
    return float(np.sqrt(ev[1] / ev[0])) if ev[0] > 0 else np.nan


def features_from_fit(w, p, tab, model):
    A, B, logd, x0, y0, c = p
    sig = tab.sigma
    R = 2.5 * sig
    tot = A + B
    e_out = (A * np.exp(-0.5 * (R / sig) ** 2) + B * (1 - tab.enclosed(R, logd))) / tot if tot > 0 else np.nan
    r = np.hypot(w["xx"] - x0, w["yy"] - y0)
    cen = ~np.isnan(w["low"])
    halo = (r >= 1.5 * sig) & ~cen
    Hres = w["D"] - A * gauss(r, sig) - c
    q_h = moments_axis_ratio(w["xx"][halo], w["yy"][halo], np.clip(Hres[halo], 0, None), x0, y0)
    if cen.sum() >= 6:
        q_c = moments_axis_ratio(w["xx"][cen], w["yy"][cen], np.ones(int(cen.sum())), x0, y0)
    else:
        core = (r <= 2 * sig) & ~cen
        q_c = moments_axis_ratio(w["xx"][core], w["yy"][core], np.clip(w["D"][core] - c, 0, None), x0, y0)
    # model-free wing slope: log-log slope of the unclipped radial profile from just outside the clipped core
    r_c = float(r[cen].max()) if cen.any() else 0.0
    edges = np.arange(np.floor(r_c) + 2, RW - 1)
    lr, lv = [], []
    for e0 in edges:
        sel = (r >= e0) & (r < e0 + 1) & ~cen
        if sel.sum() >= 4:
            v = float(np.mean(w["D"][sel] - c))
            if v > 0:
                lr.append(np.log(e0 + 0.5)); lv.append(np.log(v))
    slope = float(np.polyfit(lr, lv, 1)[0]) if len(lr) >= 5 else np.nan
    return {"f_surf": (A / tot if tot > 0 else np.nan) if not cen.any() else np.nan,
            "log_d_beam": logd - np.log(sig), "e_out": e_out,
            "aniso_excess": np.log(q_h) - np.log(q_c) if q_h > 0 and q_c > 0 else np.nan,
            "wing_slope": slope, "r_clip": r_c}


def crop(D, lower, valid, x, y, half=20):
    H, W = D.shape
    xi, yi = int(round(x)), int(round(y))
    if xi - half < 0 or yi - half < 0 or xi + half + 1 > W or yi + half + 1 > H:
        return None
    sl = (slice(yi - half, yi + half + 1), slice(xi - half, xi + half + 1))
    c = np.where(np.isnan(lower[sl]), D[sl], lower[sl])
    return np.where(valid[sl], np.clip(c, 0, None), 0.0).astype(np.float32)


def run(args):
    os.makedirs(args.out, exist_ok=True)
    scans = scan_table(args.features, args.images)
    lf = pd.read_csv(args.features)
    invalid = {(r.scan_id, f"ch{int(r.channel)}") for r in lf.itertuples() if str(r.valid).strip().lower() not in ("true", "1")}
    # pass 1: spot, window, single Gaussian
    spots = []
    for s in scans.itertuples():
        for ch in CHANNELS:
            row = {"scan_id": s.scan_id, "ch": ch, "found": False, "flag_invalid": (s.scan_id, ch) in invalid}
            spots.append(row)
            if s.dir is None:
                continue
            got = laser_only(s.dir, ch)
            if got is None:
                continue
            D, lower, valid, _ = got
            x, y, nclip = spot_centre(D, valid, lower)
            Hh, Ww = D.shape
            if x < BORDER or y < BORDER or x > Ww - BORDER or y > Hh - BORDER:
                row.update(reason="border")
                continue
            w = window(D, lower, valid, x, y, noise_scale(D, lower, valid, x, y))
            if w is None:
                row.update(reason="window")
                continue
            peak = float(np.nanmax(np.where(np.isnan(w["low"]), w["D"], w["low"])))
            row.update(found=True, x=x, y=y, n_clip=int((~np.isnan(w["low"])).sum()), s0=w["s0"],
                       snr=(peak - w["c0"]) / w["s0"])
            row["_w"], row["_frame"] = w, (D, lower, valid)
            if row["snr"] >= SNR_MIN:
                g = fit_gauss(w, x, y)
                if g is not None and g.success:
                    row["sigma_gauss"] = float(np.exp(g.x[1]))
    sp = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in spots])
    sigma_c = {ch: float(np.nanpercentile(sp.loc[sp["ch"] == ch, "sigma_gauss"], 10)) for ch in CHANNELS}
    print("label-free beam widths (px):", {k: round(v, 3) for k, v in sigma_c.items()}, flush=True)
    tabs = {ch: Table(s) for ch, s in sigma_c.items()}
    # pass 2: the two-component fit
    out, crops = [], {}
    for r in spots:
        rec = {k: v for k, v in r.items() if not k.startswith("_")}
        rec["ok"] = False
        if r.get("found") and r.get("snr", 0) >= SNR_MIN and not r["flag_invalid"]:
            w = r["_w"]
            fit, model = fit_two(w, r["x"], r["y"], tabs[r["ch"]])
            if fit is not None and fit.success:
                A, B, logd, x0, y0, c = fit.x
                rec.update(A=A, B=B, d_px=float(np.exp(logd)), x0=x0, y0=y0, c=c, cost=float(fit.cost),
                           dof=int(len(w["D"]) - 6), logd_at_bound=bool(logd <= np.log(D_GRID[0]) + 1e-3
                                                                       or logd >= np.log(D_GRID[-1]) - 1e-3))
                rec.update(features_from_fit(w, fit.x, tabs[r["ch"]], model))
                rec["ok"] = True
                cr = crop(*r["_frame"], x0, y0)
                if cr is not None:
                    crops[f"{r['scan_id']}|{r['ch']}"] = cr
        out.append(rec)
    res = pd.DataFrame(out)
    res.to_csv(os.path.join(args.out, "features_m2_spots.csv"), index=False)
    np.savez_compressed(os.path.join(args.out, "crops.npz"), **crops)
    with open(os.path.join(args.out, "beam_widths.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"sigma_c_px": sigma_c, "rule": "10th percentile of single-Gaussian widths, spots with SNR >= 5"}, f,
                  indent=1)
    q = res.groupby("ch").agg(spots=("scan_id", "size"), found=("found", "sum"), fitted=("ok", "sum"),
                              at_bound=("logd_at_bound", "sum"), median_snr=("snr", "median"))
    print(q.to_string())
    return res


def synthetic(args):
    """Made-up spots with known A, B, d on a known beam, with clipping and noise; checks that the fit recovers them."""
    rng = np.random.default_rng(0)
    sig = 1.6
    tab = Table(sig)
    rows = []
    for i in range(60):
        A_frac = rng.uniform(0.05, 0.95)
        d = float(np.exp(rng.uniform(np.log(0.6), np.log(15))))
        tot = rng.uniform(5, 60)
        A, B = A_frac * tot, (1 - A_frac) * tot
        H = W = 101
        yy, xx = np.mgrid[0:H, 0:W]
        x, y = 50 + rng.uniform(-0.5, 0.5), 50 + rng.uniform(-0.5, 0.5)
        r = np.hypot(xx - x, yy - y)
        true = A * gauss(r, sig) + B * tab.h(r, np.log(d))
        dark = np.full((H, W), 0.05)
        obs = true + rng.normal(0, 0.004, (H, W)) + rng.normal(0, 0.03, (H, W)) * true
        lin_sat = 0.9
        clipped = obs + dark >= lin_sat
        lower = np.where(clipped, lin_sat - dark, np.nan)
        valid = np.ones((H, W), bool)
        w = window(obs, lower, valid, round(x), round(y), 0.004)
        fit, model = fit_two(w, round(x), round(y), tab)
        f = features_from_fit(w, fit.x, tab, model)
        rows.append({"true_f_surf": A_frac, "fit_f_surf": f["f_surf"], "true_log_d_beam": np.log(d / sig),
                     "fit_log_d_beam": f["log_d_beam"], "n_clip": int(clipped.sum())})
    t = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    t.to_csv(os.path.join(args.out, "m2_synthetic.csv"), index=False)
    rep = {"spearman_f_surf": float(t["true_f_surf"].corr(t["fit_f_surf"], method="spearman")),
           "spearman_log_d": float(t["true_log_d_beam"].corr(t["fit_log_d_beam"], method="spearman")),
           "median_abs_err_f_surf": float((t["true_f_surf"] - t["fit_f_surf"]).abs().median()),
           "median_abs_err_log_d": float((t["true_log_d_beam"] - t["fit_log_d_beam"]).abs().median()),
           "clipped_spots": int((t["n_clip"] > 0).sum()), "n": len(t)}
    print(json.dumps(rep, indent=1))
    with open(os.path.join(args.out, "m2_synthetic.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features")
    ap.add_argument("--images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    synthetic(a) if a.synthetic else run(a)
