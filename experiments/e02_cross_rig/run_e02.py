"""E02: train on the 3DAI scanner, test on the robotic arm (pre-registered in EXPERIMENT_LOG.md, entry (f)).

    python -P run_e02.py --data ../../data/e02 --out ../../results/e02 [--n-perm 1000]

Inputs: features_3dai.csv and features_arm.csv from spots.py. Materials: the four both rigs have (plastic, metal,
paper, foam). Test unit on the arm: an object-part, i.e. one object with one material label (the arm labels each laser
spot, so scissors have a plastic and a metal part); its prediction is the mean class probability over its photos.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut

FEATS = ["prof_0.5", "prof_1", "prof_1.5", "prof_2", "prof_3", "prof_4", "core_r", "core_g", "ring_r", "ring_g",
         "core_sat", "speckle", "log_peak_bg", "ellip", "halo_frac", "w10_w50"]
MATS = ["foam", "metal", "paper", "plastic"]
FLAGGED = ["insulated water bottle", "precision screwdriver", "robotic gripper finger assembly", "slip joint pliers",
           "stapler", "cardboard shipping box", "scissors"]  # label check of entry (f), made before any scoring


def zscore(X, ref):
    mu, sd = ref.mean(0), ref.std(0)
    return (X - mu) / np.where(sd > 0, sd, 1)


def model():
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)


def unit_predict(clf, X, units, classes):
    P = pd.DataFrame(clf.predict_proba(X), columns=clf.classes_).reindex(columns=classes, fill_value=0.0)
    P["unit"] = units
    M = P.groupby("unit").mean()
    return M.idxmax(axis=1)


def cross_rig(tr, te, classes, norm, y_tr=None):
    Xtr, Xte = tr[FEATS].to_numpy(float), te[FEATS].to_numpy(float)
    if norm == "per_rig":
        Xtr, Xte = zscore(Xtr, Xtr), zscore(Xte, Xte)
    else:
        Xte, Xtr = zscore(Xte, Xtr), zscore(Xtr, Xtr)
    clf = model().fit(Xtr, tr["material"] if y_tr is None else y_tr)
    pred_u = unit_predict(clf, Xte, te["unit"].to_numpy(), classes)
    truth_u = te.groupby("unit")["material"].first().reindex(pred_u.index)
    pred_b = clf.predict(Xte)
    return (balanced_accuracy_score(truth_u, pred_u), balanced_accuracy_score(te["material"], pred_b),
            pd.DataFrame({"truth": truth_u, "pred": pred_u}))


def boot_ci(df, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = df.index.to_numpy()
    vals = []
    for _ in range(n):
        s = df.loc[rng.choice(idx, len(idx))]
        if s["truth"].nunique() < 2:
            continue
        vals.append(balanced_accuracy_score(s["truth"], s["pred"]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def perm_null(tr, te, classes, norm, n, seed=0):
    rng = np.random.default_rng(seed)
    objs = tr.groupby("object")["material"].first()
    out = []
    for _ in range(n):
        lab = pd.Series(rng.permutation(objs.to_numpy()), index=objs.index)
        out.append(cross_rig(tr, te, classes, norm, y_tr=tr["object"].map(lab).to_numpy())[0])
    return np.array(out)


def within(df, classes, unit_col):
    X = zscore(df[FEATS].to_numpy(float), df[FEATS].to_numpy(float))
    y, g = df["material"].to_numpy(), df["object"].to_numpy()
    P = np.zeros((len(df), len(classes)))
    for tri, tei in LeaveOneGroupOut().split(X, y, g):
        if len(set(y[tri])) < len(classes):
            continue
        clf = model().fit(X[tri], y[tri])
        P[tei] = pd.DataFrame(clf.predict_proba(X[tei]), columns=clf.classes_).reindex(columns=classes,
                                                                                         fill_value=0).to_numpy()
    d = pd.DataFrame(P, columns=classes)
    d["unit"] = df[unit_col].to_numpy()
    M = d.groupby("unit").mean()
    truth = df.groupby(unit_col)["material"].first().reindex(M.index)
    return balanced_accuracy_score(truth, M.idxmax(axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-perm", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t3 = pd.read_csv(os.path.join(a.data, "features_3dai.csv"))
    ar = pd.read_csv(os.path.join(a.data, "features_arm.csv"))
    t3 = t3[t3["ok"] & t3["material"].isin(MATS)].reset_index(drop=True)
    ar = ar[ar["ok"] & ar["material"].isin(MATS)].reset_index(drop=True)
    t3["unit"] = t3["object"]
    ar["unit"] = ar["object"].astype(str) + "|" + ar["material"]
    res = {"n": {"3dai_samples": len(t3), "3dai_objects": t3.groupby("material")["object"].nunique().to_dict(),
                 "arm_boxes": len(ar), "arm_units": ar.groupby("material")["unit"].nunique().to_dict()}}
    analyses = {
        "primary_4class_per_rig": (t3, ar, MATS, "per_rig"),
        "4class_3dai_norm": (t3, ar, MATS, "3dai"),
        "plastic_metal_per_rig": (t3[t3["material"].isin(["plastic", "metal"])],
                                  ar[ar["material"].isin(["plastic", "metal"])], ["metal", "plastic"], "per_rig"),
        "sensitivity_unflagged_per_rig": (t3, ar[~ar["object_name"].isin(FLAGGED)], MATS, "per_rig"),
    }
    for name, (tr, te, classes, norm) in analyses.items():
        bu, bb, units = cross_rig(tr, te, classes, norm)
        null = perm_null(tr, te, classes, norm, a.n_perm if name.startswith("primary") or name.startswith("plastic")
                         else 200)
        res[name] = {"unit_balanced_acc": float(bu), "box_balanced_acc": float(bb), "ci95_units": boot_ci(units),
                     "n_units": int(len(units)), "chance": 1 / len(classes),
                     "null_mean": float(null.mean()), "null_95th": float(np.percentile(null, 95)),
                     "p_one_sided": float((1 + (null >= bu).sum()) / (1 + len(null))), "n_perm": int(len(null)),
                     "confusion": pd.crosstab(units["truth"], units["pred"]).to_dict()}
        print(name, json.dumps({k: res[name][k] for k in ("unit_balanced_acc", "ci95_units", "p_one_sided")}))
    res["within_3dai_logo"] = float(within(t3, MATS, "unit"))
    res["within_arm_logo"] = float(within(ar, MATS, "unit"))
    print("within 3dai", res["within_3dai_logo"], "within arm", res["within_arm_logo"])
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
