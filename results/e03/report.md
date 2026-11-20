# E03 results: wave methods 1, 2 and 4 on the 3DAI laser previews

Plan: lab notebook entry (m), commit `7bc3f4b`, pushed before scoring. Results: entry (n). Methods:
[docs/WAVE_METHODS.md](../../docs/WAVE_METHODS.md). Code: [experiments/e03_wave_methods](../../experiments/e03_wave_methods).

**In short.** On the 800 px JPEG previews in the 3DAI repository the new features do not work, and the validity checks
show it before any classifier does:

- **Method 2 (S).** The subsurface length disagrees between the scanner's two red lasers (ICC −0.05) and between an
  object's two poses (0.13 to 0.61). It ranks foam shortest and metal longest, the reverse of the physics. With the core
  clipped in 227 of 229 spots, only the far halo is left, and these features cannot read it reliably.
- **Method 4.** The digital stand-in adds nothing over a linear readout on the same pixels, as predicted.
- **Method 1.** The speckle M read on the previews agrees between the two lasers and between poses, and does not follow
  the object's look, but it carries no material signal here.
- **The older features** of E01 and E02 score above shuffled labels when every object counts once (0.30 and 0.40),
  mostly on the two cardboard objects.

## Naming the material of unseen objects

Leave one object out, 31 objects in five materials (chance 0.20; the shuffled-label mean is lower, see the table).
Balanced accuracy over objects; P is one-sided against shuffled labels (number of shuffles in brackets).

| Feature set | Features | Balanced accuracy over objects [95% CI] | Shuffled labels: mean, 95th pct | P | Scan-level |
|---|---|---|---|---|---|
| S: method 2 (subsurface length, halo elongation, green minus red) | 7 | 0.142 [0.022, 0.311] | 0.181, 0.379 | 0.574 (1000) | 0.221 |
| R states: method 4 stand-in, 1000 reservoir outputs | 1000 | 0.188 [0.103, 0.318] | 0.177, 0.313 | 0.412 (50) | 0.160 |
| RAW: linear readout on the same spot pixels, no reservoir | 5043 | 0.113 [0.067, 0.190] | 0.185, 0.282 | 0.961 (50) | 0.119 |
| R kernel: method 4 stand-in, 3 laser-to-laser correlations | 3 | 0.208 [0.046, 0.380] | 0.182, 0.421 | 0.358 (200) | 0.134 |
| E02: E02's spot features, both red lasers | 32 | 0.400 [0.214, 0.548] | 0.157, 0.305 | 0.015 (200) | 0.311 |
| L: E01's laser features | 78 | 0.304 [0.181, 0.512] | 0.172, 0.287 | 0.040 (200) | 0.245 |
| SPK: method 1 on the previews (exploratory) | 4 | 0.113 [0.050, 0.214] | 0.192, 0.425 | 0.736 (200) | 0.180 |
| F: projector fringe contrast (exploratory) | 4 | 0.138 [0.058, 0.252] | 0.180, 0.404 | 0.582 (200) | 0.131 |
| S without channel 2 (exploratory) | 4 | 0.062 [0.012, 0.147] | 0.194, 0.388 | 0.930 (200) | 0.158 |
| S + R kernel (exploratory) | 10 | 0.167 [0.047, 0.349] | 0.176, 0.351 | 0.448 (200) | 0.200 |

Recall per material (share of objects named right):

| Set | cardboard (2) | foam (2) | metal (8) | paper (3) | plastic (16) |
|---|---|---|---|---|---|
| S | 0.00 | 0.00 | 0.25 | 0.33 | 0.12 |
| R states | 0.00 | 0.00 | 0.50 | 0.00 | 0.44 |
| RAW | 0.00 | 0.00 | 0.00 | 0.00 | 0.56 |
| E02 | 1.00 | 0.00 | 0.62 | 0.00 | 0.38 |
| L | 0.00 | 0.00 | 0.50 | 0.33 | 0.69 |
| SPK | 0.00 | 0.00 | 0.12 | 0.00 | 0.44 |

Exact McNemar tests on objects: L right where S was wrong 13 times, the reverse 2 (P = 0.007); E02 against S 9 to 1
(P = 0.021); R states against RAW 4 to 2 (P = 0.69); R states against S 8 to 2 (P = 0.11). Exploratory: S on all
nine materials 0.056; S without the two possibly mislabelled objects 0.375 over four materials (no permutation test).

## Do the features measure the material?

Medians over each set's per-laser features. Two red lasers: ICC(C,1) between the 635 nm lasers on the same scans
(different beams, different points). Poses: ICC(C,1) between an object's two poses. Material signal: eta² of object
means over the five materials. Follows the look: within-material partial R² of the brightness and colour of the
object at the spot (lasers-off frame).

| Per-laser features of | Two red lasers agree | Two poses agree | Material signal | Follows the look | Features |
|---|---|---|---|---|---|
| S (method 2) | 0.07 | 0.19 | 0.08 | 0.13 | 3 |
| SPK (method 1) | 0.53 | 0.55 | 0.10 | 0.07 | 1 |
| E02 | 0.16 | 0.66 | 0.16 | 0.35 | 16 |
| L (scanner, colour camera) | 0.68 | 0.82 | 0.11 | 0.53 | 13 |
| clipped radius (brightness, a control) | 0.72 | 0.77 | 0.17 | 0.33 | 1 |

Per feature, S and SPK:

| Feature | Two red lasers (pairs) | Poses ch1 / ch2 / ch4 | Material signal ch1 / ch2 / ch4 | Follows the look ch1 / ch2 / ch4 |
|---|---|---|---|---|
| subsurface length `log_d_beam` | −0.05 (79) | 0.13 / 0.24 / 0.61 | 0.01 / 0.09 / 0.04 | 0.26 / 0.11 / 0.13 |
| wing slope `wing_slope` (not in S: missing in 21-44%) | 0.63 (42) | 0.10 / −0.16 / 0.58 | 0.08 / 0.37 / 0.14 | 0.13 / 0.28 / 0.17 |
| halo elongation `aniso_excess` | 0.07 (79) | 0.20 / 0.19 / 0.12 | 0.07 / 0.07 / 0.12 | 0.13 / 0.22 / 0.06 |
| speckle `log_M` (SPK) | 0.53 (79) | 0.55 / 0.47 / 0.74 | 0.15 / 0.05 / 0.10 | 0.00 / 0.07 / 0.14 |

Post hoc (entry (q)): SPK's agreement between the two red lasers is not where its ring sits. After the log M of each
laser is regressed on that laser's own clipped radius, the residuals of the two lasers still correlate at 0.43 (79
scans).

The control behaves as it should: the clipped radius agrees between lasers and poses, since a bright object clips
wider under both, and it follows the look (partial R² up to 0.74). Agreement alone therefore does not make a feature a
material property; the look check is needed beside it.

Physics check (exploratory), material medians of object means of the subsurface length in beam widths, red lasers:
foam 2.83, cardboard 3.23, paper 3.52, plastic 3.58, metal 3.61. Expected was metal shortest and foam longest.

## Predictions of (m)

| | Prediction | Outcome |
|---|---|---|
| P1 (primary) | S above shuffled labels, P < 0.05 | Not supported: 0.142, P = 0.57 |
| P2 | S's two-laser agreement above E02's and L's | Not supported: 0.07 against 0.16 and 0.68 |
| P3 | S follows the look less than E02 | Met by the letter (0.13 against 0.35), but S carries almost no material signal (0.08) and does not agree between lasers, so it follows nothing |
| P4 | R states no different from RAW | Supported: 4 to 2, P = 0.69 (both at chance) |
| P5 | Method 1 unreadable on the previews: two-laser ICC below 0.5 and not above shuffled labels | Half met: no material signal (P = 0.74), but the two lasers agree at 0.53 |

## Post hoc, not planned: are the comparison sets reading colour?

Run after these results, since E02 and L beat shuffled labels while following the look
([`posthoc_colour.py`](../../experiments/e03_wave_methods/posthoc_colour.py), [`posthoc_colour.json`](posthoc_colour.json),
200 shuffles each):

| Set | Features | Balanced accuracy over objects [95% CI] | Shuffled mean | P |
|---|---|---|---|---|
| E02 without its 8 colour features | 24 | 0.362 [0.159, 0.511] | 0.161 | 0.015 |
| E02 colour features only | 8 | 0.342 [0.111, 0.508] | 0.175 | 0.104 |
| L without its 26 colour features | 52 | 0.200 [0.119, 0.338] | 0.168 | 0.323 |
| L colour features only | 26 | 0.217 [0.094, 0.407] | 0.163 | 0.229 |

E02's shape features keep its score without colour, and both versions name both cardboard objects; on the other four
materials E02's recall averages about chance. L's score needs both halves. With two cardboard objects and more than a
dozen sets tested, this is a hint that the spot's shape in E02's units separates cardboard on this scanner, not a
result.

## Method 1 in simulation

From the development check of (m) (`m1_speckle_simulation.json`; 30 simulated spots per cell, M = 1, 2, 4).
AUC for telling M = 1 from M = 2, and M = 2 from M = 4:

| Pipeline | Speckle grain 1.5 px | Speckle grain 3 px |
|---|---|---|
| Raw, two frames per laser | 1.00, 1.00 | 1.00, 1.00 |
| Raw, one frame | 1.00, 1.00 | 1.00, 1.00 |
| sRGB + JPEG 85 (4:2:0), no downscaling | 1.00, 1.00 | 0.99, 0.99 |
| the same, downscaled 1.6x | 0.91, 0.81 | 0.94, 0.89 |
| the same, downscaled 2.4x | 0.72, 0.54 | 0.77, 0.71 |

JPEG alone keeps the order of M but shrinks the speckle (M read 2 to 5 times too high); downscaling is what loses it.

## Reading

The GitHub previews cannot test methods 1 and 2. The quality numbers of (m) predicted that (cores clipped in 227 of
229 spots, channel 2's halo wider than the window), and the validity checks confirm it independently of any
classifier: S's subsurface length is not the same quantity under the two red lasers, and it orders the materials
against the physics. The checks did their job, which is the part of E03 that carries over: on new captures, a
feature has to agree between the two red lasers and between poses and stay clear of the look before its accuracy
means anything. Two leads for the full data: the model-free wing slope is the most laser-consistent quantity of
method 2 (0.63), and the speckle M agrees between lasers and poses without following the look. The digital reservoir
did what P4 said, nothing; whatever a physical reservoir adds has to come from what a camera image does not hold.
The captures these methods need are listed in [docs/WAVE_METHODS.md](../../docs/WAVE_METHODS.md); the most important
are unclipped spots (bracketed exposures), raw full-resolution frames and two frames per laser.

Files: `summary.json`, `validity.csv`, `predictions_objects.csv`, `posthoc_colour.json`, `m1_speckle_simulation.json`,
`m2_synthetic.json`, `run_e03_synthetic.json`.
