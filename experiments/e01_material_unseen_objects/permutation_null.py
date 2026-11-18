"""E01, post hoc (not in the plan of entry (a)): what leave-one-object-out scores when the material labels carry no
information. Materials are shuffled between objects (each object keeps one material for all its scans), the primary
analysis is rerun, and the observed balanced accuracy is compared with that null distribution.

    python permutation_null.py --features ... --a1 ... --a2 ... --out ../../results/e01/permutation_null.json \
        --n-l 500 --n-a2 200
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_e01 as e  # noqa: E402


def score(X, y, g, classes):
    return balanced_accuracy_score(y, np.array(classes)[e.leak_free(X, y, g, "logreg", classes).argmax(1)])


def null(X, y, g, classes, n, seed):
    objs = pd.Series(y, index=g).groupby(level=0).first()
    def one(k):
        perm = pd.Series(np.random.default_rng([seed, k]).permutation(objs.to_numpy()), index=objs.index)
        return score(X, perm.loc[g].to_numpy(), g, classes)
    return np.array(Parallel(n_jobs=-1)(delayed(one)(k) for k in range(n)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--a1", required=True)
    ap.add_argument("--a2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-l", type=int, default=500)
    ap.add_argument("--n-a2", type=int, default=200)
    a = ap.parse_args()
    tab, feats, _ = e.laser_table(pd.read_csv(a.features))
    a1 = pd.read_csv(a.a1).set_index("scan8")
    a2 = dict(np.load(a.a2, allow_pickle=False))
    idx = {str(n).rsplit("__", 1)[-1]: i for i, n in enumerate(a2["names"])}
    tab = tab[tab["scan8"].isin(a1.index) & tab["scan8"].isin(idx)].reset_index(drop=True)
    counts = tab.groupby("material_class")["label"].nunique()
    classes = sorted(counts[counts >= 2].index)
    prim = tab["material_class"].isin(classes).to_numpy()
    y, g = tab["material_class"].to_numpy()[prim], tab["label"].to_numpy()[prim]
    sets = {"L": tab[feats].to_numpy(float)[prim],
            "A2": np.asarray(a2["emb"], float)[[idx[s] for s in tab["scan8"]]][prim]}
    out = {"note": "post hoc; materials shuffled between objects; leave-one-object-out, logistic regression"}
    for name, n in (("L", a.n_l), ("A2", a.n_a2)):
        obs = score(sets[name], y, g, classes)
        nl = null(sets[name], y, g, classes, n, seed=0)
        out[name] = {"observed": float(obs), "n_permutations": n, "null_mean": float(nl.mean()), "null_sd": float(nl.std()),
                     "null_95th": float(np.percentile(nl, 95)), "p_one_sided": float((1 + (nl >= obs).sum()) / (1 + n))}
        print(name, json.dumps(out[name]))
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
