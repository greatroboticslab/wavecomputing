"""E02, post hoc (not in the plan of entry (f)): permutation test of the within-arm leave-one-object-out reference.
Materials are shuffled among the arm's objects (each object keeps one material per part), then the reference is rerun.
    python -P within_arm_null.py --data ../../data/e02 --out ../../results/e02/within_arm_null.json --n 200
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_e02 as e  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=200); a = ap.parse_args()
ar = pd.read_csv(os.path.join(a.data, "features_arm.csv"))
ar = ar[ar["ok"] & ar["material"].isin(e.MATS)].reset_index(drop=True)
ar["unit"] = ar["object"].astype(str) + "|" + ar["material"]
obs = e.within(ar, e.MATS, "unit")
units = ar.groupby("unit")["material"].first()
rng = np.random.default_rng(0); null = []
for k in range(a.n):
    lab = pd.Series(rng.permutation(units.to_numpy()), index=units.index)
    sh = ar.copy(); sh["material"] = sh["unit"].map(lab)
    null.append(e.within(sh, e.MATS, "unit"))
null = np.array(null)
res = {"note": "post hoc; materials shuffled among arm object-parts; leave-one-object-out within the arm",
       "observed": float(obs), "n_permutations": a.n, "null_mean": float(null.mean()),
       "null_95th": float(np.percentile(null, 95)), "p_one_sided": float((1 + (null >= obs).sum()) / (1 + a.n))}
print(json.dumps(res)); json.dump(res, open(a.out, "w"), indent=1)
