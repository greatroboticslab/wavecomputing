"""E03 evaluation: do the wave-method features name the material of unseen objects, and do they measure the material
rather than the laser, the pose or the object's look? Pre-registered in EXPERIMENT_LOG.md, entry (m).

    python -P run_e03.py --features <laser_features.csv> --images <3DAI images> --data ../../data/e03 \
        --e02 ../../data/e02/features_3dai.csv --out ../../results/e03 [--n-perm 1000]
    python -P run_e03.py --synthetic --out <dir>       # pipeline check with a known answer

Feature sets (one row per scan): S, method 2 (subsurface length in beam widths, wing slope, halo elongation beyond the
beam's footprint, per laser, and green-minus-red differences); R, method 4 (reservoir states; reservoir kernel); RAW,
a linear readout on the same spot pixels without the reservoir; SPK, method 1 on the previews (exploratory); F, the
projector's fringe contrast (exploratory); E02, E02's 16 spot features for both red lasers; L, E01's laser features.
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this folder (our code); python -P leaves it out
from common import (CHANNELS, POSSIBLY_MISLABELLED, PRIMARY, import_e01, load_rgb, scan_table,  # noqa: E402
                    srgb_to_linear)

warnings.filterwarnings("ignore")
e01 = import_e01()
LASERS = list(CHANNELS)
S_PER_LASER = ["log_d_beam", "wing_slope", "aniso_excess"]
E02_FEATS = ["prof_0.5", "prof_1", "prof_1.5", "prof_2", "prof_3", "prof_4", "core_r", "core_g", "ring_r", "ring_g",
             "core_sat", "speckle", "log_peak_bg", "ellip", "halo_frac", "w10_w50"]
L_PER_CH = ["core", "halo_r50", "halo_r10", "halo_energy_10_40", "speckle", "add_r", "add_g", "add_b", "annulus_mean",
            "annulus_r", "annulus_g", "annulus_b", "saturated_core"]
F_FEATS = ["fringe_contrast_object", "fringe_contrast_object_std", "fringe_contrast_ratio", "fringe_reliable_fraction"]


# ------------------------------------------------------------------ feature tables
def per_laser_wide(spots, feats, key="scan_id"):
    w = spots.pivot_table(index=key, columns="ch", values=feats, aggfunc="first")
    w.columns = [f"{f}_{c}" for f, c in w.columns]
    return w


def build(args):
    scans = scan_table(args.features, args.images).set_index("scan_id")
    lf = pd.read_csv(args.features)
    sp = pd.read_csv(os.path.join(args.data, "features_m2_spots.csv"))
    sp.loc[~sp["ok"].astype(bool), S_PER_LASER + ["r_clip"]] = np.nan
    S = per_laser_wide(sp, S_PER_LASER)
    for f, short in (("log_d_beam", "dlogd"), ("wing_slope", "dslope")):
        red = S[[f"{f}_ch1", f"{f}_ch2"]].mean(axis=1, skipna=True)
        S[f"{short}_green_minus_red"] = S[f"{f}_ch4"] - red
    sets = {"S": S}
    sets["S_no_ch2"] = S[[c for c in S.columns if not c.endswith("_ch2")]].drop(
        columns=["dlogd_green_minus_red", "dslope_green_minus_red"])
    rclip = per_laser_wide(sp, ["r_clip"])
    m1 = pd.read_csv(os.path.join(args.data, "features_m1_spots.csv"))
    m1["log_M"] = np.log(m1["M_hat"].clip(0.1, 1000))
    SPK = per_laser_wide(m1, ["log_M"])
    SPK["dlogM_green_minus_red"] = SPK["log_M_ch4"] - SPK[["log_M_ch1", "log_M_ch2"]].mean(axis=1, skipna=True)
    sets["SPK"] = SPK
    z = np.load(os.path.join(args.data, "features_m4.npz"))
    ids = list(z["scan_id"])
    sets["R_states"] = pd.DataFrame(z["states"], index=ids, columns=[f"node{i}" for i in range(z["states"].shape[1])])
    sets["R_kernel"] = pd.DataFrame(z["kernel"], index=ids, columns=["k_red1_red2", "k_red1_green", "k_red2_green"])
    sets["RAW"] = pd.DataFrame(z["raw"], index=ids, columns=[f"px{i}" for i in range(z["raw"].shape[1])])
    e2 = pd.read_csv(args.e02)
    e2 = e2[e2["ok"].astype(str).str.lower() == "true"].copy()
    e2["scan8"] = e2["sample"].str.split("/").str[0].str.rsplit("__", n=1).str[-1]
    e2["ch"] = e2["channel"].str.replace("laser_", "", regex=False)
    by8 = {v: k for k, v in scans["scan8"].items()}
    e2["scan_id"] = e2["scan8"].map(by8)
    E02 = per_laser_wide(e2.dropna(subset=["scan_id"]), E02_FEATS)
    sets["E02"] = E02
    ltab, lcols, _ = e01.laser_table(lf)
    sets["L"] = ltab.set_index("scan_id")[lcols]
    sets["F"] = lf.groupby("scan_id")[F_FEATS].first()
    sets["S+R_kernel"] = S.join(sets["R_kernel"], how="left")
    for k in sets:
        sets[k] = sets[k].reindex(scans.index)
    # appearance at the spot: the lasers-off frame, linear RGB, 5 px around the red spot (ch1, else ch2, else ch4)
    app = []
    spi = sp.set_index(["scan_id", "ch"])
    for sid, s in scans.iterrows():
        rec = {"scan_id": sid}
        for ch in LASERS:
            if (sid, ch) in spi.index and bool(spi.loc[(sid, ch), "ok"]):
                x, y = float(spi.loc[(sid, ch), "x0"]), float(spi.loc[(sid, ch), "y0"])
                dk = srgb_to_linear(load_rgb(os.path.join(s["dir"], "laser_dark.jpg")))
                H, W, _ = dk.shape
                yy, xx = np.ogrid[0:H, 0:W]
                m = (xx - x) ** 2 + (yy - y) ** 2 <= 25
                rgb = dk[m].mean(0)
                tot = rgb.sum() + 1e-6
                rec.update(log_brightness=float(np.log(tot / 3 + 1e-4)), chrom_r=float(rgb[0] / tot),
                           chrom_g=float(rgb[1] / tot))
                break
        app.append(rec)
    app = pd.DataFrame(app).set_index("scan_id").reindex(scans.index)
    # per-laser tables for the validity checks
    per_laser = {"S": (sp, S_PER_LASER), "E02": (e2.dropna(subset=["scan_id"]), E02_FEATS),
                 "SPK": (m1, ["log_M"]), "r_clip (brightness, a control)": (sp, ["r_clip"])}
    lfl = lf.copy()
    lfl.loc[~e01.as_bool(lfl["valid"]), L_PER_CH] = np.nan
    lfl["ch"] = "ch" + lfl["channel"].astype(int).astype(str)
    per_laser["L"] = (lfl, L_PER_CH)
    return scans, sets, app, per_laser, rclip


# ------------------------------------------------------------------ scoring
def object_ba(P, y, groups, classes):
    op, ot = e01.by_object(P, y, groups, classes)
    return float(balanced_accuracy_score(ot, op)), op, ot


def evaluate(X, y, groups, classes):
    P = e01.leak_free(X, y, groups, "logreg", classes)
    ba, op, ot = object_ba(P, y, groups, classes)
    scan_ba = float(balanced_accuracy_score(y, np.array(classes)[P.argmax(1)]))
    return ba, scan_ba, op, ot, P


def boot_objects(op, ot, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ot))
    o, t = op.to_numpy(), ot.to_numpy()
    vals = []
    for _ in range(n):
        s = rng.choice(idx, len(idx))
        if len(set(t[s])) < 2:
            continue
        vals.append(balanced_accuracy_score(t[s], o[s]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def permutation(X, y, groups, classes, n, seed=0):
    rng = np.random.default_rng(seed)
    obj = pd.Series(y, index=groups).groupby(level=0).first()
    out = []
    for _ in range(n):
        lab = pd.Series(rng.permutation(obj.to_numpy()), index=obj.index)
        yp = np.array([lab[g] for g in groups])
        out.append(evaluate(X, yp, groups, classes)[0])
    return np.array(out)


def mcnemar(op_a, op_b, ot):
    a, b = (op_a == ot).to_numpy(), (op_b.reindex(ot.index) == ot).to_numpy()
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
    return {"a_right_b_wrong": n01, "a_wrong_b_right": n10, "p_exact": float(p)}


# ------------------------------------------------------------------ validity
def icc_c(a, b):
    """ICC(C,1), two-way, consistency, two measurements per item"""
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    n = len(a)
    if n < 8:
        return np.nan, n
    X = np.stack([a, b], 1)
    gm = X.mean()
    msr = 2 * np.sum((X.mean(1) - gm) ** 2) / (n - 1)
    msc = n * np.sum((X.mean(0) - gm) ** 2)
    sse = np.sum((X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + gm) ** 2)
    mse = sse / (n - 1)
    return float((msr - mse) / (msr + mse)), n


def validity(per_laser, scans, app):
    rows = []
    prim = scans[scans["material"].isin(PRIMARY)]
    for name, (tab, feats) in per_laser.items():
        t = tab.copy()
        for f in feats:
            w = t.pivot_table(index="scan_id", columns="ch", values=f, aggfunc="first").reindex(scans.index)
            # (1) two red lasers, same scans
            icc_l, n_l = icc_c(w.get("ch1", pd.Series(np.nan, index=w.index)).to_numpy(float),
                               w.get("ch2", pd.Series(np.nan, index=w.index)).to_numpy(float))
            for ch in [c for c in LASERS if c in w.columns]:
                v = w[ch]
                # (2) two poses of the same object (first scan of each pose)
                d = scans.assign(v=v).dropna(subset=["v"])
                pv = d.groupby(["object", "pose"])["v"].first().unstack("pose")
                icc_p, n_p = (icc_c(pv.iloc[:, 0].to_numpy(float), pv.iloc[:, 1].to_numpy(float))
                              if pv.shape[1] >= 2 else (np.nan, 0))
                # (3) material signal and (4) appearance leakage, on object means of the primary materials
                o = prim.assign(v=v).join(app).dropna(subset=["v", "log_brightness"]).groupby("object").agg(
                    v=("v", "mean"), material=("material", "first"), lb=("log_brightness", "mean"),
                    cr=("chrom_r", "mean"), cg=("chrom_g", "mean"))
                eta2, leak = np.nan, np.nan
                if len(o) >= 10 and o["v"].std() > 0:
                    gm = o["v"].mean()
                    ss_t = np.sum((o["v"] - gm) ** 2)
                    ss_b = sum(len(g) * (g["v"].mean() - gm) ** 2 for _, g in o.groupby("material"))
                    eta2 = float(ss_b / ss_t)
                    Dm = pd.get_dummies(o["material"], drop_first=True).astype(float)
                    X0 = np.column_stack([np.ones(len(o)), Dm.to_numpy()])
                    X1 = np.column_stack([X0, o[["lb", "cr", "cg"]].to_numpy()])
                    r0 = o["v"] - X0 @ np.linalg.lstsq(X0, o["v"], rcond=None)[0]
                    r1 = o["v"] - X1 @ np.linalg.lstsq(X1, o["v"], rcond=None)[0]
                    leak = float(1 - np.sum(r1 ** 2) / np.sum(r0 ** 2)) if np.sum(r0 ** 2) > 0 else np.nan
                rows.append({"set": name, "feature": f, "laser": ch, "icc_red1_red2": icc_l, "n_red_pairs": n_l,
                             "icc_pose": icc_p, "n_pose_pairs": n_p, "eta2_material": eta2,
                             "appearance_partial_r2": leak})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main
def run(args):
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    scans, sets, app, per_laser, rclip = build(args)
    res = {"n_scans_all": int(len(scans))}
    prim = scans[scans["material"].isin(PRIMARY) & scans["dir"].notna()]
    classes = sorted(prim["material"].unique())
    y, groups = prim["material"].to_numpy(), prim["object"].to_numpy()
    res["primary"] = {"n_scans": int(len(prim)), "n_objects": int(prim["object"].nunique()),
                      "objects_per_material": prim.groupby("material")["object"].nunique().to_dict()}
    n_perm = {"S": args.n_perm}
    preds = {}
    order = ["S", "R_states", "RAW", "R_kernel", "E02", "L", "SPK", "F", "S_no_ch2", "S+R_kernel"]
    for name in order:
        X = sets[name].loc[prim.index].to_numpy(float)
        keep = np.mean(np.isnan(X), 0) <= 0.20  # E01's rule: a feature missing in more than 20% of rows is dropped
        dropped = [c for c, k in zip(sets[name].columns, keep) if not k]
        X = X[:, keep]
        ba, sba, op, ot, P = evaluate(X, y, groups, classes)
        npm = n_perm.get(name, 200 if X.shape[1] <= 100 else 50)
        null = permutation(X, y, groups, classes, npm)
        res[name] = {"object_balanced_accuracy": ba, "ci95_objects": boot_objects(op, ot), "scan_balanced_accuracy": sba,
                     "object_accuracy": float((op == ot).mean()), "n_features": int(X.shape[1]), "dropped": dropped,
                     "null_mean": float(null.mean()), "null_95th": float(np.percentile(null, 95)),
                     "p_one_sided": float((1 + (null >= ba).sum()) / (1 + len(null))), "n_perm": int(len(null)),
                     "recall": {c: float((op[ot == c] == c).mean()) for c in classes}}
        preds[name] = op
        print(f"{name:11s} obj BA {ba:.3f} CI {res[name]['ci95_objects']} null {null.mean():.3f} "
              f"P {res[name]['p_one_sided']:.3f} ({X.shape[1]} feats, {npm} perms, {time.time() - t0:.0f}s)", flush=True)
    ot = pd.Series(y, index=groups).groupby(level=0).first()
    res["mcnemar"] = {"S_vs_L": mcnemar(preds["S"], preds["L"], ot), "S_vs_E02": mcnemar(preds["S"], preds["E02"], ot),
                      "R_states_vs_RAW": mcnemar(preds["R_states"], preds["RAW"], ot),
                      "S_vs_R_states": mcnemar(preds["S"], preds["R_states"], ot)}
    # exploratory: all nine materials (S); without the two possibly mislabelled objects (S)
    alls = scans[scans["dir"].notna()]
    ba_all, _, _, _, _ = evaluate(sets["S"].loc[alls.index].to_numpy(float), alls["material"].to_numpy(),
                                  alls["object"].to_numpy(), sorted(alls["material"].unique()))
    res["exploratory_S_all_nine_materials"] = ba_all
    keep_obj = prim[~prim["object"].isin(POSSIBLY_MISLABELLED)]
    keep_obj = keep_obj[keep_obj["material"].map(keep_obj.groupby("material")["object"].nunique()) >= 2]
    cl2 = sorted(keep_obj["material"].unique())
    res["exploratory_S_without_possibly_mislabelled"] = {
        "object_balanced_accuracy": evaluate(sets["S"].loc[keep_obj.index].to_numpy(float), keep_obj["material"].to_numpy(),
                                             keep_obj["object"].to_numpy(), cl2)[0], "materials": cl2}
    # validity
    v = validity(per_laser, scans, app)
    v.to_csv(os.path.join(args.out, "validity.csv"), index=False)
    summ = v.groupby("set").agg(median_icc_red1_red2=("icc_red1_red2", "median"), median_icc_pose=("icc_pose", "median"),
                                median_eta2=("eta2_material", "median"),
                                median_appearance_r2=("appearance_partial_r2", "median"), n_rows=("feature", "size"))
    res["validity_medians"] = summ.round(4).to_dict(orient="index")
    print(summ.round(3).to_string())
    # physics check, exploratory: material medians of the red subsurface length and wing slope (object means)
    S = sets["S"].loc[prim.index].assign(material=prim["material"], object=prim["object"])
    S["log_d_red"] = S[["log_d_beam_ch1", "log_d_beam_ch2"]].mean(axis=1)
    S["slope_red"] = S[["wing_slope_ch1", "wing_slope_ch2"]].mean(axis=1)
    om = S.groupby("object").agg(material=("material", "first"), log_d_red=("log_d_red", "mean"),
                                 slope_red=("slope_red", "mean"), log_d_green=("log_d_beam_ch4", "mean"))
    res["exploratory_material_medians"] = om.groupby("material")[["log_d_red", "slope_red", "log_d_green"]].median().round(3).to_dict(orient="index")
    pd.DataFrame({k: v.reindex(ot.index) for k, v in preds.items()}).assign(truth=ot).to_csv(
        os.path.join(args.out, "predictions_objects.csv"))
    res["runtime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, default=float)
    return res


def synthetic(args):
    """Same pipeline on made-up scans: a feature carrying each object's material should score high; pure noise should
    sit at the shuffled-label null on average and pass P < 0.05 about 5% of the time (10 noise draws)."""
    code = {m: j for j, m in enumerate(PRIMARY)}
    objs = [(m, i) for m, k in (("cardboard", 2), ("foam", 2), ("metal", 8), ("paper", 3), ("plastic", 16)) for i in range(k)]

    def make(seed, planted):
        rng = np.random.default_rng(seed)
        rows = []
        for m, i in objs:
            off = rng.normal(0, 0.5)
            for p in (1, 2):
                r = {"object": f"{m}{i}", "material": m, **{f"noise{j}": rng.normal() for j in range(10)}}
                if planted:
                    r["signal"] = code[m] + off + rng.normal(0, 0.3)
                rows.append(r)
        return pd.DataFrame(rows)

    out = {}
    d = make(0, True)
    classes = sorted(d["material"].unique())
    y, g = d["material"].to_numpy(), d["object"].to_numpy()
    X = d[["signal"] + [f"noise{j}" for j in range(10)]].to_numpy(float)
    ba = evaluate(X, y, g, classes)[0]
    null = permutation(X, y, g, classes, 100)
    out["planted"] = {"object_ba": ba, "null_mean": float(null.mean()), "p": float((1 + (null >= ba).sum()) / 101)}
    noise = []
    for seed in range(1, 11):
        d = make(seed, False)
        X = d[[f"noise{j}" for j in range(10)]].to_numpy(float)
        ba = evaluate(X, y, g, classes)[0]
        null = permutation(X, y, g, classes, 50, seed=seed)
        noise.append({"object_ba": ba, "null_mean": float(null.mean()), "p": float((1 + (null >= ba).sum()) / 51)})
    out["noise_10_draws"] = {"mean_object_ba": float(np.mean([n["object_ba"] for n in noise])),
                             "mean_null": float(np.mean([n["null_mean"] for n in noise])),
                             "n_p_below_0.05": int(sum(n["p"] < 0.05 for n in noise)), "draws": noise}
    rng = np.random.default_rng(0)
    a = rng.normal(size=60)
    out["icc_check"] = {"same": icc_c(a, a + rng.normal(0, 0.1, 60))[0], "unrelated": icc_c(a, rng.normal(size=60))[0]}
    print(json.dumps({k: v for k, v in out.items() if k != "noise_10_draws"} | {
        "noise_10_draws": {k: v for k, v in out["noise_10_draws"].items() if k != "draws"}}, indent=1))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "run_e03_synthetic.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features")
    ap.add_argument("--images")
    ap.add_argument("--data")
    ap.add_argument("--e02")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    synthetic(a) if a.synthetic else run(a)
