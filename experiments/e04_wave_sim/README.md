# E04: the math of methods 1 and 4, and a wave simulation to check it

Simulation only; no real data. Plan: lab notebook entry (r), committed before scoring. Results: entry (s),
[results/e04](../../results/e04) and the weekly note
[docs/weekly_notes/2026-11-22](../../docs/weekly_notes/2026-11-22). Methods: [docs/WAVE_METHODS.md](../../docs/WAVE_METHODS.md).

A camera records only brightness. But the grain of a laser spot is interference, and its statistics are squared
correlations of the light field. E04 derives what the speckle contrast of one frame (F1) and the speckle correlation
between frames (wavelength steps F3, beam steps F4) equal in terms of the light's paths inside the material, and
checks the formulas at three levels.

| Script | What it does |
|---|---|
| [`fullwave.py`](fullwave.py) | **Exact wave solution.** The scalar wave equation for 3,198 point scatterers (Foldy-Lax equations, every order of multiple scattering), 16 random samples, 25 wavenumbers, 9 beam tilts; far-field speckle. `--selftest` checks energy conservation; `--analyze` compares the speckle correlations with the sum over transport paths for the same sample. |
| [`transport_mc.py`](transport_mc.py) | **Path Monte Carlo.** Paths through a half-space of dipole (Rayleigh) scatterers, each carrying its polarization vector; scores the light leaving towards the camera by distance from the entry point, path length and polarization (local estimator). `--check` compares with Chandrasekhar's exact half-space results. `slab_farfield` is the path-sum prediction for `fullwave.py`. |
| [`theory.py`](theory.py) | Closed forms: diffusion with the extrapolated boundary and complex absorption (F2), pupil overlap, a multimode diode's spectrum and coherence. |
| [`synth.py`](synth.py) | **Frames.** Draws the field on the object as a Gaussian random field with exactly the covariance the path statistics give (over path length, polarization, laser lines), passes it through the lens pupil, detects intensity, integrates on pixels, adds shot and read noise, a 12-bit converter and three bracketed exposures. Also the estimators: the spot's brightness trend and ring-wise cross-frame products. |
| [`run_e04.py`](run_e04.py) | The checks and the three frame experiments: speckle contrast ring by ring (F1), wavelength steps (F3), beam steps with a flat and a curved wavefront (F4). Writes `results/e04/*.json`. |
| [`figures.py`](figures.py) | Figures from the result files. |
| [`make_note.py`](make_note.py) | Fills the weekly note's numbers and tables from the result files. |

Run from this folder:

```bash
python -P transport_mc.py --check
python -P transport_mc.py --kernel ../../data/e04/kernel_rayleigh_n1.npz --photons 3000000 --seed 10
python -P fullwave.py --selftest
python -P fullwave.py --run ../../data/e04/fullwave.npz --realizations 16
python -P fullwave.py --analyze ../../data/e04/fullwave.npz --out ../../results/e04 --photons 2500000
python -P run_e04.py --kernel ../../data/e04/kernel_rayleigh_n1.npz --out ../../results/e04 --parts checks,m,wavelength,beam --objects 8
python -P figures.py --results ../../results/e04 --data ../../data/e04
python -P make_note.py --results ../../results/e04 --template ../../docs/weekly_notes/2026-11-22/weekly_note_template.tex --out ../../docs/weekly_notes/2026-11-22/weekly_note.tex
```

About 20 minutes for the exact wave solution and 25 for the frames on a 6-core desktop; the kernels and frames go to
`data/e04` (not in git).

Units and choices:

- Monte Carlo lengths are in scattering mean free paths, so one kernel serves every scattering strength; absorption is
  applied afterwards as exp(-mu_a s), so it serves every absorption too. Frames are in millimetres.
- Materials are made-up parameter sets (transport length 0.05, 0.2, 0.8 mm; absorption 0.1, 0.05, 0.02 per mm; 0% or
  30% surface light), not measurements of real materials. Bare metal is surface light only.
- Lasers: single-frequency, and a multimode diode (1 nm wide, lines 0.15 nm apart). The rig's real spectra are unknown.
- Scored rings: 200 speckle grains or more, and mean brightness above 0.002 of the spot centre.
- The beam-step frames treat the two polarization channels as independent (the small cross term between them is left
  out); the contrast and wavelength-step frames keep it.

Limits are listed in entry (s) and in the weekly note: small (Rayleigh) scatterers, no fibres, polarization without
internal reflection at the surface (a kernel for index 1.4 is computed, with the reflection's polarization
approximated, and not used yet), ideal lens, scalar exact solution.
