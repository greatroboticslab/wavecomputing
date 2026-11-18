"""E02, post hoc quality check (not in the plan of entry (f)): how often is the strongest red spot near an arm box
outside the box? For each primary-photo box: the maximum smoothed red excess inside the box (M_in) and in the
surrounding area out to 3 box sizes (M_out). "Outside" if M_out > 1.5 x M_in and M_out > 30 (a visible spot).
    python -P red_outside_check.py --arm <dataset> --boxes ../../data/e02/arm_boxes.csv --out ../../results/e02/red_outside.csv
"""
import argparse, csv, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spots import red_excess, smooth  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True); ap.add_argument("--boxes", required=True)
ap.add_argument("--out", required=True); a = ap.parse_args()
rows = []
for b in csv.DictReader(open(a.boxes, encoding="utf-8")):
    if b["variant"] != "primary":
        continue
    p = os.path.join(a.arm, b["split"], "images", b["image_file"])
    if not os.path.exists(p):
        continue
    img = np.asarray(Image.open(p).convert("RGB"), dtype=float)
    H, W, _ = img.shape
    cx, cy, w, h = (float(b[k]) for k in ("cx_px", "cy_px", "w_px", "h_px"))
    X0, Y0, X1, Y1 = int(max(0, cx - 1.5 * w)), int(max(0, cy - 1.5 * h)), int(min(W, cx + 1.5 * w)), int(min(H, cy + 1.5 * h))
    e = smooth(red_excess(img[Y0:Y1, X0:X1]), 2)
    yy, xx = np.mgrid[Y0:Y1, X0:X1]
    inside = (abs(xx - cx) <= w / 2) & (abs(yy - cy) <= h / 2)
    m_in, m_out = float(e[inside].max()) if inside.any() else 0.0, float(e[~inside].max()) if (~inside).any() else 0.0
    rows.append({"image_file": b["image_file"], "box_index": b["box_index"], "object_name": b["object_name"],
                 "material": b["material_coarse"], "m_in": round(m_in, 1), "m_out": round(m_out, 1),
                 "red_outside": int(m_out > 1.5 * m_in and m_out > 30)})
with open(a.out, "w", newline="", encoding="utf-8") as f:
    w_ = csv.DictWriter(f, fieldnames=list(rows[0])); w_.writeheader(); w_.writerows(rows)
n = len(rows); k = sum(r["red_outside"] for r in rows)
print(f"{k} of {n} primary boxes ({100 * k / n:.1f}%) have a stronger red spot outside the box than inside")
mats = sorted({r["material"] for r in rows})
for m in mats:
    s = [r for r in rows if r["material"] == m]
    print(f"  {m}: {sum(r['red_outside'] for r in s)} of {len(s)}")
