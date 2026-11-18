"""E01: can laser readings name the material of an object the model never saw, and do they beat a plain photo?

Pre-registered in EXPERIMENT_LOG.md, entry (a), before any model was fit to the real data.

    python run_e01.py --features laser_features.csv --a1 photo_color_stats.csv --a2 siglip2_embeddings.npz --out results
    python run_e01.py --synthetic --out <dir>      # pipeline check on made-up data with a known answer
"""
import argparse
import json
import os
import platform
import warnings

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")  # tiny classes make StratifiedKFold and lbfgs chatty

# laser features per channel (main camera, then the Kinect infrared sensor), and per scan
SPOT = ["core", "halo_r50", "halo_r10", "halo_energy_10_40", "speckle", "add_r", "add_g", "add_b", "annulus_mean",
        "annulus_r", "annulus_g", "annulus_b", "saturated_core",
        "ir_core", "ir_halo_r50", "ir_halo_r10", "ir_halo_energy_10_40", "ir_speckle", "ir_saturated_core"]
SCAN_LEVEL = ["red_green_ratio", "red2_green_ratio"]
CHANNELS = [1, 2, 3, 4]          # 1, 2 red 635 nm; 3 near-infrared 940 nm; 4 green 530 nm
MAX_MISSING = 0.20               # a listed feature missing in more than 20% of rows is dropped
POSSIBLY_MISLABELLED = ["taped black metal", "taped breadboard"]  # 3DAI handoff, 2026-09-15


def as_bool(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def laser_table(df):
    """One row per scan: the per-channel features (channels flagged invalid set missing) and the per-scan ones."""
    df = df.copy()
    keep = [c for c in SPOT if df[c].isna().mean() <= MAX_MISSING]
    keep_scan = [c for c in SCAN_LEVEL if df[c].isna().mean() <= MAX_MISSING]
    dropped = [c for c in SPOT + SCAN_LEVEL if c not in keep + keep_scan]
    df.loc[~as_bool(df["valid"]), keep] = np.nan
    df["channel"] = df["channel"].astype(int)
    assert not df.duplicated(["scan_id", "channel"]).any(), "duplicate (scan, channel) rows"
    wide = df.set_index(["scan_id", "channel"])[keep].unstack("channel")
    wide = wide.reindex(columns=pd.MultiIndex.from_product([keep, CHANNELS]))
    wide.columns = [f"{f}_ch{c}" for f, c in wide.columns]
    first = df.groupby("scan_id").first()
    tab = first[["label", "material_class", "angle_index", "sample_id"]].join(first[keep_scan]).join(wide)
    tab["scan8"] = [str(s)[:8] for s in tab.index]
    assert tab["scan8"].is_unique, "8-character scan ids collide"
    return tab.reset_index(), keep_scan + list(wide.columns), dropped


def make_model(kind):
    if kind == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))
    if kind == "rf":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample",
                                                    random_state=0, n_jobs=-1))
    raise ValueError(kind)


def fit_predict(X, y, tr, te, kind, classes):
    m = make_model(kind).fit(X[tr], y[tr])
    P = np.zeros((len(te), len(classes)))
    P[:, [classes.index(c) for c in m.classes_]] = m.predict_proba(X[te])
    return P


def leak_free(X, y, groups, kind, classes):
    """Leave one object out: each object's scans are predicted by a model that never saw that object."""
    P = np.zeros((len(y), len(classes)))
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        P[te] = fit_predict(X, y, tr, te, kind, classes)
    return P


def leaky(X, y, kind, classes, repeats):
    """5-fold split by scan (the other pose of the same object can be in training), repeated."""
    vals = []
    for seed in range(repeats):
        P = np.zeros((len(y), len(classes)))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            P[te] = fit_predict(X, y, tr, te, kind, classes)
        vals.append(balanced_accuracy_score(y, np.array(classes)[P.argmax(1)]))
    return [float(np.mean(vals)), float(np.std(vals))]


def by_object(P, y, groups, classes):
    d = pd.DataFrame(P, columns=classes)
    d["g"], d["y"] = groups, y
    agg = d.groupby("g")
    return agg[classes].mean().idxmax(axis=1), agg["y"].first()


def metrics(P, y, groups, classes):
    pred = np.array(classes)[P.argmax(1)]
    op, ot = by_object(P, y, groups, classes)
    present = [c for c in classes if c in set(y)]
    return {"n_scans": int(len(y)), "n_objects": int(len(ot)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
            "object_accuracy": float((op == ot).mean()),
            "object_balanced_accuracy": float(balanced_accuracy_score(ot, op)),
            "recall": {c: float((pred[y == c] == c).mean()) for c in present},
            "labels": present, "confusion": confusion_matrix(y, pred, labels=present).tolist()}


def bal_stat(P, y, classes):
    cl = np.array(classes)

    def f(idx):
        if len(np.unique(y[idx])) < 2:
            return None
        return balanced_accuracy_score(y[idx], cl[P[idx].argmax(1)])  # materials absent from a resample drop out
    return f


def boot(stat, groups, n, seed=0):
    """95% percentile interval of stat over resamples of whole objects."""
    rng = np.random.default_rng(seed)
    objs = np.unique(groups)
    where = {o: np.flatnonzero(groups == o) for o in objs}
    vals = []
    for _ in range(n):
        idx = np.concatenate([where[o] for o in rng.choice(objs, len(objs), replace=True)])
        v = stat(idx)
        if v is not None:
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def mcnemar(ca, cb):
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return {"first_only": b, "second_only": c, "p": float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0}


def run(tab, feats, dropped, a1, a2, n_boot, repeats):
    res = {"dropped_features": dropped, "n_laser_features": len(feats)}
    a1 = a1.set_index("scan8")
    a1_cols = [c for c in a1.columns if c != "folder"]
    a2_idx = {str(n).rsplit("__", 1)[-1]: i for i, n in enumerate(a2["names"])}
    res["n_scans_laser"] = int(len(tab))
    tab = tab[tab["scan8"].isin(a1.index) & tab["scan8"].isin(a2_idx)].reset_index(drop=True)
    res["n_scans_analysed"] = int(len(tab))
    XL = tab[feats].to_numpy(float)
    sets = {"L": XL, "A1": a1.loc[tab["scan8"], a1_cols].to_numpy(float),
            "A2": np.asarray(a2["emb"], float)[[a2_idx[s] for s in tab["scan8"]]]}
    sets["L+A2"] = np.hstack([sets["L"], sets["A2"]])
    counts = tab.groupby("material_class")["label"].nunique()
    classes = sorted(counts[counts >= 2].index)
    res["objects_per_material"] = {k: int(v) for k, v in counts.items()}
    res["primary_materials"] = classes
    prim = tab["material_class"].isin(classes).to_numpy()
    y, g = tab["material_class"].to_numpy()[prim], tab["label"].to_numpy()[prim]

    res["leak_free"], preds = {}, {}
    for name, X in sets.items():
        for kind in ("logreg", "rf"):
            P = leak_free(X[prim], y, g, kind, classes)
            preds[name, kind] = P
            m = metrics(P, y, g, classes)
            m["balanced_accuracy_ci95"] = boot(bal_stat(P, y, classes), g, n_boot)
            res["leak_free"][f"{name}/{kind}"] = m
    res["leaky"] = {f"{n}/{k}": leaky(sets[n][prim], y, k, classes, repeats)
                    for n, k in (("L", "logreg"), ("L", "rf"), ("A2", "logreg"))}

    res["paired"] = {}
    for other in ("A2", "A1"):
        PL, PO = preds["L", "logreg"], preds[other, "logreg"]
        fL, fO = bal_stat(PL, y, classes), bal_stat(PO, y, classes)

        def diff(idx, fL=fL, fO=fO):
            a, b = fL(idx), fO(idx)
            return None if a is None or b is None else a - b
        cL = (lambda op_ot: (op_ot[0] == op_ot[1]).to_numpy())(by_object(PL, y, g, classes))
        cO = (lambda op_ot: (op_ot[0] == op_ot[1]).to_numpy())(by_object(PO, y, g, classes))
        res["paired"][f"L-{other}/logreg"] = {
            "difference": res["leak_free"]["L/logreg"]["balanced_accuracy"]
            - res["leak_free"][f"{other}/logreg"]["balanced_accuracy"],
            "ci95": boot(diff, g, n_boot), "mcnemar_objects": mcnemar(cL, cO)}

    # P4: metal or not, every object
    yb = np.where(tab["material_class"].to_numpy() == "metal", "metal", "not metal")
    gb = tab["label"].to_numpy()
    Pb = leak_free(XL, yb, gb, "logreg", ["metal", "not metal"])
    mb = metrics(Pb, yb, gb, ["metal", "not metal"])
    op, ot = by_object(Pb, yb, gb, ["metal", "not metal"])
    k, n = int((op == ot).sum()), int(len(ot))
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    mb["object_accuracy_ci95"] = [float(ci.low), float(ci.high)]
    res["metal_vs_not"] = mb

    # exploratory
    ex = {}
    all_classes = sorted(counts.index)
    yall = tab["material_class"].to_numpy()
    for name in ("L", "A2"):
        ex[f"all_materials/{name}/logreg"] = metrics(leak_free(sets[name], yall, gb, "logreg", all_classes), yall, gb,
                                                     all_classes)
    for c in CHANNELS:
        cols = [i for i, f in enumerate(feats) if f.endswith(f"_ch{c}")]
        P = leak_free(XL[prim][:, cols], y, g, "logreg", classes)
        ex[f"channel_{c}_only/logreg"] = metrics(P, y, g, classes)
    keep = prim & ~tab["label"].isin(POSSIBLY_MISLABELLED).to_numpy()
    ys, gs = tab["material_class"].to_numpy()[keep], tab["label"].to_numpy()[keep]
    ex["without_possibly_mislabelled/L/logreg"] = metrics(leak_free(XL[keep], ys, gs, "logreg", classes), ys, gs, classes)
    res["exploratory"] = ex

    lf = res["leak_free"]
    p2 = res["leaky"]["L/logreg"][0] - lf["L/logreg"]["balanced_accuracy"]
    p3 = res["paired"]["L-A2/logreg"]["difference"]
    res["predictions"] = {
        "P1": {"rule": "L/logreg leak-free balanced accuracy: lower 95% bound > 0.20 (chance)",
               "value": lf["L/logreg"]["balanced_accuracy_ci95"][0],
               "supported": lf["L/logreg"]["balanced_accuracy_ci95"][0] > 0.20},
        "P2": {"rule": "leaky mean minus leak-free balanced accuracy >= 0.10 (L/logreg)", "value": p2,
               "supported": p2 >= 0.10},
        "P3": {"rule": "L/logreg beats A2/logreg on leak-free balanced accuracy (point estimate)", "value": p3,
               "supported": p3 > 0},
        "P4": {"rule": "metal vs not metal, all objects, L/logreg leak-free object accuracy >= 0.85",
               "value": mb["object_accuracy"], "supported": mb["object_accuracy"] >= 0.85}}
    scans = tab.loc[prim, ["scan_id", "label", "material_class", "angle_index"]].reset_index(drop=True)
    for name in ("L", "A2"):
        scans[f"pred_{name}_logreg"] = np.array(classes)[preds[name, "logreg"].argmax(1)]
    return res, scans


def report(res):
    f = lambda x: f"{x:.3f}"
    lines = ["# E01 results", "", f"Scans analysed: {res['n_scans_analysed']} of {res['n_scans_laser']} "
             f"(scans with laser features and a photo). Primary materials: {', '.join(res['primary_materials'])}.",
             f"Laser features: {res['n_laser_features']}; dropped for missingness: {', '.join(res['dropped_features']) or 'none'}.",
             "", "## Leak-free (leave one object out), primary materials", "",
             "| features / model | balanced acc. | 95% CI | macro F1 | object acc. | object bal. acc. |",
             "|---|---|---|---|---|---|"]
    for k, m in res["leak_free"].items():
        lo, hi = m["balanced_accuracy_ci95"]
        lines.append(f"| {k} | {f(m['balanced_accuracy'])} | {f(lo)}-{f(hi)} | {f(m['macro_f1'])} | "
                     f"{f(m['object_accuracy'])} | {f(m['object_balanced_accuracy'])} |")
    lines += ["", "Chance balanced accuracy: " + f(1 / len(res["primary_materials"])) + ".", "",
              "## Leaky split (5-fold by scan, 20 repeats)", "", "| features / model | mean | sd |", "|---|---|---|"]
    lines += [f"| {k} | {f(v[0])} | {f(v[1])} |" for k, v in res["leaky"].items()]
    lines += ["", "## Laser against photo (logistic regression, leak-free)", "",
              "| comparison | difference in balanced acc. | 95% CI | objects right only with laser / only with photo | McNemar p |",
              "|---|---|---|---|---|"]
    for k, v in res["paired"].items():
        mc = v["mcnemar_objects"]
        lines.append(f"| {k} | {f(v['difference'])} | {f(v['ci95'][0])} to {f(v['ci95'][1])} | "
                     f"{mc['first_only']} / {mc['second_only']} | {mc['p']:.3g} |")
    mb = res["metal_vs_not"]
    lines += ["", f"Metal vs not metal (all objects, L/logreg): object accuracy {f(mb['object_accuracy'])} "
              f"(95% CI {f(mb['object_accuracy_ci95'][0])}-{f(mb['object_accuracy_ci95'][1])}), "
              f"balanced accuracy {f(mb['balanced_accuracy'])}.", "", "## Predictions", ""]
    for k, p in res["predictions"].items():
        lines.append(f"- {k}: {p['rule']}: value {f(p['value'])} -> {'supported' if p['supported'] else 'NOT supported'}")
    lines += ["", "## Exploratory", "", "| analysis | balanced acc. | object acc. | n objects |", "|---|---|---|---|"]
    lines += [f"| {k} | {f(m['balanced_accuracy'])} | {f(m['object_accuracy'])} | {m['n_objects']} |"
              for k, m in res["exploratory"].items()]
    rec = res["leak_free"]["L/logreg"]
    lines += ["", "Per-material recall, L/logreg leak-free: " +
              ", ".join(f"{c} {f(v)}" for c, v in rec["recall"].items()) + ".", "",
              "Confusion matrix, L/logreg leak-free (rows true, columns predicted; order " +
              ", ".join(rec["labels"]) + "): " + json.dumps(rec["confusion"]) + "."]
    return "\n".join(lines) + "\n"


def synthetic(seed=0):
    """Made-up data shaped like the real table. Materials differ in laser features; each object adds its own
    offset (so splitting by scan leaks); photos carry only a weak material signal. Expected: P1, P2, P3 hold."""
    rng = np.random.default_rng(seed)
    mats = {"plastic": 16, "metal": 8, "paper": 3, "cardboard": 2, "foam": 2, "fabric": 1, "stone": 1, "wood": 1}
    centre = {m: rng.normal(0, 0.35, len(SPOT) + 2) for m in mats}
    rows, a1, names, emb, k = [], [], [], [], 0
    for m, n in mats.items():
        for j in range(n):
            obj = f"{m} object {j}"
            off = rng.normal(0, 1.0, len(SPOT) + 2)
            for pose in (1, 2):
                sid = f"{k:08x}-0000-synthetic"
                k += 1
                pose_off = rng.normal(0, 0.3, len(SPOT) + 2)
                for ch in CHANNELS:
                    v = centre[m] + off + pose_off + rng.normal(0, 0.3, len(SPOT) + 2) + 0.2 * ch
                    rows.append({"label": obj, "material_class": m, "angle_index": pose, "channel": ch,
                                 "valid": "True", "scan_id": sid, "sample_id": obj, **dict(zip(SPOT + SCAN_LEVEL, v))})
                weak = 0.1 * np.resize(centre[m], 24) + rng.normal(0, 1.0, 24)
                a1.append({"folder": f"{obj}__p{pose}__{sid[:8]}", "scan8": sid[:8], **{f"c{i}": x for i, x in enumerate(weak)}})
                names.append(f"{obj}__p{pose}__{sid[:8]}")
                emb.append(np.concatenate([0.1 * centre[m] + 0.5 * off, rng.normal(0, 1.0, 200)]))
    return pd.DataFrame(rows), pd.DataFrame(a1), {"names": np.array(names), "emb": np.array(emb)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features")
    ap.add_argument("--a1")
    ap.add_argument("--a2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--repeats", type=int, default=20)
    a = ap.parse_args()
    if a.synthetic:
        df, a1, a2 = synthetic()
    else:
        df, a1, a2 = pd.read_csv(a.features), pd.read_csv(a.a1), dict(np.load(a.a2, allow_pickle=False))
    tab, feats, dropped = laser_table(df)
    res, scans = run(tab, feats, dropped, a1, a2, a.boot, a.repeats)
    res["synthetic"] = a.synthetic
    res["versions"] = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                       "scipy": scipy.__version__, "scikit-learn": sklearn.__version__}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(a.out, "report.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(report(res))
    scans.to_csv(os.path.join(a.out, "predictions_scans.csv"), index=False, lineterminator="\n")
    print(report(res))


if __name__ == "__main__":
    main()
