"""Markdown tables for results/e03/report.md, read from summary.json and validity.csv (no number copied by hand).

    python -P report_tables.py --res ../../results/e03
"""
import argparse
import json
import os

import pandas as pd

LABEL = {"S": "S: method 2 (subsurface length, halo elongation, green minus red)",
         "R_states": "R states: method 4 stand-in, 1000 reservoir outputs",
         "RAW": "RAW: linear readout on the same spot pixels, no reservoir",
         "R_kernel": "R kernel: method 4 stand-in, 3 laser-to-laser correlations",
         "E02": "E02: E02's spot features, both red lasers",
         "L": "L: E01's laser features",
         "SPK": "SPK: method 1 on the previews (exploratory)",
         "F": "F: projector fringe contrast (exploratory)",
         "S_no_ch2": "S without channel 2 (exploratory)",
         "S+R_kernel": "S + R kernel (exploratory)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    a = ap.parse_args()
    s = json.load(open(os.path.join(a.res, "summary.json"), encoding="utf-8"))
    print("| Feature set | Features | Balanced accuracy over objects [95% CI] | Shuffled labels: mean, 95th pct | P | Scan-level |")
    print("|---|---|---|---|---|---|")
    for k, lab in LABEL.items():
        r = s[k]
        print(f"| {lab} | {r['n_features']} | {r['object_balanced_accuracy']:.3f} [{r['ci95_objects'][0]:.3f}, "
              f"{r['ci95_objects'][1]:.3f}] | {r['null_mean']:.3f}, {r['null_95th']:.3f} | {r['p_one_sided']:.3f} "
              f"({r['n_perm']}) | {r['scan_balanced_accuracy']:.3f} |")
    print()
    print("Recall per material (objects):")
    print("| Set | " + " | ".join(sorted(s["S"]["recall"])) + " |")
    print("|---|" + "---|" * len(s["S"]["recall"]))
    for k in LABEL:
        print(f"| {k} | " + " | ".join(f"{s[k]['recall'][c]:.2f}" for c in sorted(s[k]["recall"])) + " |")
    print()
    print("McNemar (objects):", json.dumps(s["mcnemar"]))
    print("All nine materials (S):", round(s["exploratory_S_all_nine_materials"], 3))
    print("Without possibly mislabelled (S):", json.dumps(s["exploratory_S_without_possibly_mislabelled"]))
    print("Material medians (object means):", json.dumps(s["exploratory_material_medians"]))
    print()
    v = pd.read_csv(os.path.join(a.res, "validity.csv"))
    g = v.groupby("set").agg(icc_l=("icc_red1_red2", "median"), icc_p=("icc_pose", "median"),
                             eta=("eta2_material", "median"), leak=("appearance_partial_r2", "median"),
                             n=("feature", "nunique"))
    print("| Per-laser features of | Two red lasers agree (median ICC) | Two poses agree (median ICC) | Material signal (median eta^2) | Follows the look (median partial R^2) | Features |")
    print("|---|---|---|---|---|---|")
    for k, r in g.iterrows():
        print(f"| {k} | {r.icc_l:.2f} | {r.icc_p:.2f} | {r.eta:.2f} | {r.leak:.2f} | {int(r.n)} |")
    print()
    sv = v[v["set"].isin(["S", "SPK", "r_clip (brightness, a control)"])]
    print("| Set | Feature | Laser | Two-laser ICC (n) | Pose ICC (n) | eta^2 | Look partial R^2 |")
    print("|---|---|---|---|---|---|---|")
    for _, r in sv.iterrows():
        print(f"| {r['set']} | {r['feature']} | {r['laser']} | {r['icc_red1_red2']:.2f} ({int(r['n_red_pairs'])}) | "
              f"{r['icc_pose']:.2f} ({int(r['n_pose_pairs'])}) | {r['eta2_material']:.2f} | {r['appearance_partial_r2']:.2f} |")


if __name__ == "__main__":
    main()
