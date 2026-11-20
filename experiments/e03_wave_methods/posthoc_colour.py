"""E03 post hoc (not in plan (m)): do the comparison sets E02 and L still beat shuffled labels without their colour
features? Run after the results of (m) were seen, because both scored above shuffled labels while following the
object's look (validity check iv).

    python -P posthoc_colour.py --features <laser_features.csv> --images <3DAI images> --data ../../data/e03 \
        --e02 ../../data/e02/features_3dai.csv --out ../../results/e03
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this folder (our code); python -P leaves it out
import run_e03 as R  # noqa: E402

COLOUR = {"E02": ["core_r", "core_g", "ring_r", "ring_g"],
          "L": ["add_r", "add_g", "add_b", "annulus_r", "annulus_g", "annulus_b", "red_green_ratio", "red2_green_ratio"]}


def main():
    ap = argparse.ArgumentParser()
    for k in ("features", "images", "data", "e02", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    scans, sets, app, per_laser, rclip = R.build(a)
    prim = scans[scans["material"].isin(R.PRIMARY) & scans["dir"].notna()]
    classes = sorted(prim["material"].unique())
    y, groups = prim["material"].to_numpy(), prim["object"].to_numpy()
    out = {}
    for name, stems in COLOUR.items():
        tab = sets[name].loc[prim.index]
        cols = [c for c in tab.columns if not any(c == s or c.startswith(s + "_") for s in stems)]
        for variant, use in (("without_colour", cols), ("colour_only", [c for c in tab.columns if c not in cols])):
            X = tab[use].to_numpy(float)
            keep = np.mean(np.isnan(X), 0) <= 0.20
            X = X[:, keep]
            ba, sba, op, ot, P = R.evaluate(X, y, groups, classes)
            null = R.permutation(X, y, groups, classes, 200)
            out[f"{name}_{variant}"] = {"n_features": int(X.shape[1]), "object_balanced_accuracy": ba,
                                        "ci95_objects": R.boot_objects(op, ot), "null_mean": float(null.mean()),
                                        "p_one_sided": float((1 + (null >= ba).sum()) / (1 + len(null))),
                                        "recall": {c: float((op[ot == c] == c).mean()) for c in classes}}
            r = out[f"{name}_{variant}"]
            print(f"{name:3s} {variant:14s} {X.shape[1]:3d} feats  obj BA {ba:.3f} CI {np.round(r['ci95_objects'], 3)} "
                  f"null {r['null_mean']:.3f} P {r['p_one_sided']:.3f}  recall {r['recall']}", flush=True)
    with open(os.path.join(a.out, "posthoc_colour.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
