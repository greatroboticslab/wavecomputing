"""Download the 3DAI files that E01 uses, pinned to one commit, and record their SHA-256.

    python fetch_3dai.py --out ../../data/3dai

Writes <out>/<commit>/dataset/... (laser_features.csv, metadata.xlsx, README.md and every scan's color.jpg)
and <out>/<commit>/MANIFEST.tsv. The repository is public, so no token is needed.
"""
import argparse
import hashlib
import json
import os
import time
import urllib.request

REPO = "greatroboticslab/3DAI"
COMMIT = "6556bec6898e13bfc47c0bec4dce59f6d0194c43"  # main on 2026-11-18 (last commit 2026-11-10)
FILES = ["dataset/laser_features.csv", "dataset/metadata.xlsx", "dataset/README.md"]


def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "wavecomputing-fetch"}),
                                        timeout=60) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = os.path.join(a.out, COMMIT[:12])
    tree = json.loads(get(f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1"))
    assert not tree.get("truncated"), "tree listing truncated"
    photos = sorted(t["path"] for t in tree["tree"]
                    if t["type"] == "blob" and t["path"].startswith("dataset/images/") and t["path"].endswith("/color.jpg"))
    rows = []
    for path in FILES + photos:
        dst = os.path.join(root, *path.split("/"))
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            data = get(f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{path}")
            with open(dst, "wb") as f:
                f.write(data)
        with open(dst, "rb") as f:
            b = f.read()
        rows.append((path, len(b), hashlib.sha256(b).hexdigest()))
    with open(os.path.join(root, "MANIFEST.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# {REPO} @ {COMMIT}\npath\tbytes\tsha256\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    print(f"{len(rows)} files ({len(photos)} photos) in {root}")


if __name__ == "__main__":
    main()
