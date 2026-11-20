"""E03 method 4, digital twin: a random optical projection of the laser spot, read out by a linear classifier.

    python -P m4_reservoir.py --crops ../../data/e03/crops.npz --out ../../data/e03

What a scattering medium in front of a camera computes (Saade et al. 2016): a random complex projection of the light
field followed by intensity detection, y = |W x|^2; only a linear readout on y is trained. Here x is the laser-only
spot (linear units, 41 x 41 px, clipped pixels at their clip level) of the three visible lasers, each scaled to unit
sum so the readout never sees brightness. Two outputs:

- states: y for the three spots side by side (N_RES values per scan), the reservoir's full state;
- kernel: the three correlations between the states the reservoir gives for each laser's spot on its own (red 1 vs
  red 2, red 1 vs green, red 2 vs green). A random projection of this kind preserves the overlap of its inputs, so these
  measure how alike the spot looks under two lasers; a physical device would compute them in the optics.

This is a digital stand-in, computed on photos. The physical version (the object itself, or a diffuser, as the
reservoir, with inputs we control) is described in docs/WAVE_METHODS.md.
"""
import argparse
import os

import numpy as np

N_RES = 1000
LASERS = ["ch1", "ch2", "ch4"]
SEED = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = np.load(a.crops)
    keys = list(z.keys())
    scans = sorted({k.split("|")[0] for k in keys})
    side = z[keys[0]].shape[0]
    n_in = side * side
    rng = np.random.default_rng(SEED)
    W1 = (rng.normal(size=(N_RES, n_in)) + 1j * rng.normal(size=(N_RES, n_in))) / np.sqrt(2 * n_in)
    W3 = (rng.normal(size=(N_RES, 3 * n_in)) + 1j * rng.normal(size=(N_RES, 3 * n_in))) / np.sqrt(2 * 3 * n_in)
    states, kernel, raw, have = [], [], [], []
    for s in scans:
        xs = []
        for ch in LASERS:
            c = z[f"{s}|{ch}"].astype(float).ravel() if f"{s}|{ch}" in z else np.zeros(n_in)
            xs.append(c / c.sum() if c.sum() > 0 else c)
        x = np.concatenate(xs)
        states.append(np.abs(W3 @ x) ** 2)
        ys = [np.abs(W1 @ xi) ** 2 if xi.sum() > 0 else None for xi in xs]
        kv = []
        for i, j in ((0, 1), (0, 2), (1, 2)):
            kv.append(float(np.corrcoef(ys[i], ys[j])[0, 1]) if ys[i] is not None and ys[j] is not None else np.nan)
        kernel.append(kv)
        raw.append(x)
        have.append([f"{s}|{ch}" in z for ch in LASERS])
    np.savez_compressed(os.path.join(a.out, "features_m4.npz"), scan_id=np.array(scans), states=np.array(states),
                        kernel=np.array(kernel), raw=np.array(raw, dtype=np.float32), have=np.array(have),
                        lasers=np.array(LASERS))
    print(len(scans), "scans;", "states", np.array(states).shape, "kernel non-missing per pair",
          np.sum(~np.isnan(np.array(kernel)), 0))


if __name__ == "__main__":
    main()
