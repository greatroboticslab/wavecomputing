"""E02: download the 3DAI laser frames of every scan at the pinned commit (E01's), with a SHA-256 manifest.

    python -P fetch_3dai_laser.py --out C:/Users/adain/wavecomputing_data/3dai

Writes <out>/<commit[:12]>/dataset/images/<material>/<scan>/{laser_ch1..4,laser_dark,laser_zoom,color}.jpg and
MANIFEST.tsv. The files are only read by the E02 scripts, never executed.
"""
import argparse
import hashlib
import json
import os
import time
import urllib.request

REPO = "greatroboticslab/3DAI"
COMMIT = "6556bec6898e13bfc47c0bec4dce59f6d0194c43"  # the commit E01 used (main on 2026-11-18)
NAMES = ("laser_ch1.jpg", "laser_ch2.jpg", "laser_ch3.jpg", "laser_ch4.jpg", "laser_dark.jpg", "laser_zoom.jpg",
         "color.jpg")


def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "wavecomputing-e02"}),
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
    paths = sorted(t["path"] for t in tree["tree"] if t["type"] == "blob" and t["path"].startswith("dataset/images/")
                   and t["path"].rsplit("/", 1)[-1] in NAMES)
    rows = []
    for i, path in enumerate(paths):
        dst = os.path.join(root, path)
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            data = get(f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{path}")
            with open(dst, "wb") as f:
                f.write(data)
        data = open(dst, "rb").read()
        rows.append((path, len(data), hashlib.sha256(data).hexdigest()))
        if i % 100 == 0:
            print(f"{i + 1}/{len(paths)}", flush=True)
    with open(os.path.join(root, "MANIFEST.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# {REPO} @ {COMMIT}\npath\tbytes\tsha256\n")
        for r in rows:
            f.write("%s\t%d\t%s\n" % r)
    print(f"{len(rows)} files under {root}")


if __name__ == "__main__":
    main()
