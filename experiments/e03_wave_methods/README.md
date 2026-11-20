# E03: wave methods 1, 2 and 4 on the 3DAI laser frames

Methods: [docs/WAVE_METHODS.md](../../docs/WAVE_METHODS.md). Plan: lab notebook entry (m); results: entry (n) and
[results/e03](../../results/e03).

| Script | What it does |
|---|---|
| [`common.py`](common.py) | scan table (E01's 80 scans), frames in linear units, laser-only image, clipped pixels, spot centre |
| [`m2_scatter.py`](m2_scatter.py) | method 2: label-free beam width per laser, then a censored fit of surface light plus subsurface light (Christensen-Burley profile) to every spot; features in beam widths; `--synthetic` recovers known parameters |
| [`m1_speckle.py`](m1_speckle.py) | method 1: speckle degrees of freedom M in a ring around the spot; `--simulate` checks the estimator on spots with known M, on raw frames and through a preview pipeline |
| [`m4_reservoir.py`](m4_reservoir.py) | method 4, digital stand-in: random complex projection of the spot with intensity detection (reservoir states) and the correlations between lasers (reservoir kernel) |
| [`run_e03.py`](run_e03.py) | leave-one-object-out scoring of every feature set, bootstrap and permutation tests, and the validity checks (two red lasers, two poses, material signal, appearance leakage); `--synthetic` checks the pipeline on made-up data |

Run from this folder (frames from `fetch_3dai_laser.py` of E02):

```bash
IM=C:/Users/adain/wavecomputing_data/3dai/6556bec6898e/dataset/images
LF=../../data/3dai/6556bec6898e/dataset/laser_features.csv
python -P m2_scatter.py --features $LF --images $IM --out ../../data/e03
python -P m1_speckle.py --features $LF --images $IM --m2 ../../data/e03/features_m2_spots.csv --out ../../data/e03
python -P m4_reservoir.py --crops ../../data/e03/crops.npz --out ../../data/e03
python -P m1_speckle.py --simulate --out ../../results/e03
python -P run_e03.py --features $LF --images $IM --data ../../data/e03 --e02 ../../data/e02/features_3dai.csv --out ../../results/e03
```

The scripts put their own folder on the import path, so `python -P` (which leaves the current folder off) still finds
`common.py`; the frames are read from their own folder and never executed.
