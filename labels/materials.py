"""One material list for every laser dataset, and the normalization of each dataset's own labels to it.

The list is the union of the 3DAI scanner's material classes and the robotic arm's materials, so neither dataset has
to be relabeled: each raw name maps to one canonical material and an optional detail (e.g. "plastic tape" -> plastic,
detail "tape"). Spelling is normalized (case, spaces); every other mapping is written out below.

    python -P labels/materials.py \
        --tdai data/3dai/6556bec6898e/dataset/laser_features.csv \
        --arm C:/Users/adain/wavecomputing_data/arm_96b76af/dataset/data.yaml --out labels/out

Writes out/3dai_objects.csv (one row per scanner object) and out/arm_classes.csv (one row per arm class, i.e. one
object part as the arm labels each laser spot).
"""
import argparse
import csv
import os
import re

import pandas as pd
import yaml

CANON = ["plastic", "metal", "paper", "cardboard", "foam", "rubber", "silicone", "leather", "fabric", "wood",
         "stone", "fiberglass", "composite"]

# raw name (lower case, single spaces) -> (canonical material, detail)
SYNONYMS = {
    "plastic tape": ("plastic", "tape"),
    "composite plastic": ("composite", "plastic"),
    "fiberglass cloth press": ("fiberglass", "cloth press"),
}

# corrections of single labels, each with its reason; applied by this script and marked in the output's note column
OVERRIDES = {
    # (dataset, object id, raw material): (canonical material, reason)
    ("arm", "OHMlPZx8zUzd", "composite"): ("cardboard", "cardboard shipping box: the photos show corrugated cardboard "
                                                       "under these spots; the class says composite (to confirm)"),
}

# labels the photos cannot settle; kept as they are, flagged in the note column
REVIEW = {}

# labels confirmed by a person looking at the object: (dataset, object id) -> (detail, surface, note)
CONFIRMED = {
    ("arm", "Uop6evwTVgMl"): ("painted", "matte",
                             "insulated water bottle: painted metal, matte paint (confirmed by William by eye, 18 Nov 2026)"),
    ("arm", "fduiadew"): ("painted", "glossy",
                         "stapler: painted metal, glossy paint (confirmed by William by eye, 18 Nov 2026)"),
}


def clean(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def normalize(dataset, obj_id, raw):
    r = clean(raw)
    if (dataset, obj_id, r) in OVERRIDES:
        mat, reason = OVERRIDES[(dataset, obj_id, r)]
        return mat, "", f"override: {reason}"
    if r in SYNONYMS:
        mat, detail = SYNONYMS[r]
    elif r in CANON:
        mat, detail = r, ""
    else:
        raise ValueError(f"{dataset}: material {raw!r} is not in the shared list; add it to CANON or SYNONYMS")
    if (dataset, obj_id) in CONFIRMED:
        d, _surface, conf = CONFIRMED[(dataset, obj_id)]
        return mat, "; ".join(x for x in (detail, d) if x), f"confirmed: {conf}"
    note = REVIEW.get((dataset, obj_id), "")
    return mat, detail, (f"review: {note}" if note else "")


ARM_RE = re.compile(r"^\s*object\s*-?\s*(?P<obj>.*?)\s*-\s*material\s*(?P<mat>.*?)\s*"
                    r"(?:-\s*color\s*(?P<color>.*?)\s*)?-?\s*pdfname\s*(?P<pdf>\S+)\s*$", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tdai", required=True, help="3DAI laser_features.csv")
    ap.add_argument("--arm", required=True, help="robotic arm dataset data.yaml")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    t = pd.read_csv(a.tdai)
    rows3 = []
    for obj, g in t.groupby("label"):
        raw = g["material_class"].iloc[0]
        assert g["material_class"].nunique() == 1, f"3DAI object {obj} has several material classes"
        mat, detail, note = normalize("3dai", obj, raw)
        sub = sorted({clean(s) for s in g["material_subclass"].dropna() if clean(s) not in ("na", "nan", "unknown")})
        rows3.append({"dataset": "3dai", "object": obj, "material_raw": raw, "material": mat,
                      "material_detail": "; ".join(x for x in [detail] + sub if x),
                      "surface": "; ".join(sorted({clean(s) for s in g["surface"].dropna()} - {"nan"})),
                      "transparency": "; ".join(sorted({clean(s) for s in g["transparency"].dropna()} - {"nan"})),
                      "label_source": "3DAI scan metadata", "note": note})

    names = yaml.safe_load(open(a.arm, encoding="utf-8"))["names"]
    rowsa = []
    for cid, name in enumerate(names):
        m = ARM_RE.match(name)
        if not m:
            raise ValueError(f"arm class {cid} not parsed: {name!r}")
        obj_id = m.group("pdf").strip()
        mat, detail, note = normalize("arm", obj_id, m.group("mat"))
        rowsa.append({"dataset": "arm", "class_id": cid, "object_id": obj_id, "object": clean(m.group("obj")),
                      "material_raw": m.group("mat").strip(), "material": mat, "material_detail": detail,
                      "color": clean(m.group("color") or ""),
                      "surface": CONFIRMED.get(("arm", obj_id), ("", "", ""))[1], "class_name_raw": name,
                      "label_source": "class name; written with the help of AI answers (pdf_for_labels/<object id>.pdf)",
                      "note": note})

    for fn, rows in (("3dai_objects.csv", rows3), ("arm_classes.csv", rowsa)):
        with open(os.path.join(a.out, fn), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    both = pd.DataFrame(rows3 + rowsa)
    key = both["object_id"].fillna(both["object"]) if "object_id" in both else both["object"]
    print(pd.crosstab(both.assign(k=key).drop_duplicates(["dataset", "k", "material"])["material"],
                      both.assign(k=key).drop_duplicates(["dataset", "k", "material"])["dataset"]).to_string())


if __name__ == "__main__":
    main()
