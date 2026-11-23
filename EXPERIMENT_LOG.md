# Lab notebook

Dated entries, newest last. A plan is written and committed here before its experiment is scored; results come in a
later entry. Mistakes are corrected in a new dated entry, never by rewriting an old one. (Same rules as the AGAI
notebook, `greatroboticslab/AGAI`, `cluster/EXPERIMENT_LOG.md`.)

## 2026-11

- **2026-11-18, 02:40 CST. (a) Plan for E01: can laser readings name the material of an object the model never saw,
  and do they do better than a plain photo?** Written before any model was fit to the real data. Only the structure
  of the table was looked at (columns, counts, missing values, how scans match photos).
  - *Question.* Dr. Zhang's point (Discord, 2026-11-18): lasers tell the material, and there are few materials, so a
    few lasers can recognise many objects regardless of their shape. E01 tests the first half on the 3DAI scans: name
    the material of objects held out of training, from laser readings alone, against the same objects' plain photos.
  - *Data.* `greatroboticslab/3DAI` at commit 6556bec6898e13bfc47c0bec4dce59f6d0194c43, `dataset/laser_features.csv`:
    318 rows (scan x laser channel), 80 scans, 35 object labels (38 sample ids), 2 poses per object. Objects per
    material: plastic 16, metal 8, paper 3, cardboard 2, foam 2, fabric 1, fiberglass cloth press 1, stone 1, wood 1.
    Plain photos: `dataset/images/<material>/<object>__<pose>__<scan8>/color.jpg` (previews, longest side 800 px),
    matched to scans by the first 8 characters of `scan_id`; 79 of the 80 scans have one. All analyses use the scans
    that have both laser features and a photo, so every feature set sees the same scans. Files are fetched by
    `fetch_3dai.py`, which records their SHA-256.
  - *Unit, label, split group.* One scan; label `material_class`; scans are grouped by `label` (the object's name), so
    an object registered under two sample ids still stays on one side of a split.
  - *Primary set.* The materials with at least two objects (expected: cardboard, foam, metal, paper, plastic; 31
    objects). A material with one object cannot be named on an unseen object, since holding that object out removes the
    material from training; all materials are reported as an exploratory analysis.
  - *Feature sets.* **L (laser):** for each channel (1 and 2 red 635 nm, 3 near-infrared 940 nm, 4 green 530 nm) the
    main camera's spot features `core, halo_r50, halo_r10, halo_energy_10_40, speckle, add_r, add_g, add_b,
    annulus_mean, annulus_r, annulus_g, annulus_b, saturated_core` and the Kinect infrared sensor's `ir_core,
    ir_halo_r50, ir_halo_r10, ir_halo_energy_10_40, ir_speckle, ir_saturated_core`, plus the per-scan `red_green_ratio`
    and `red2_green_ratio`. A listed feature missing in more than 20% of rows is dropped (none is expected to be).
    Channels flagged `valid = false` are set missing. Left out: `cam_*` (empty), `tail_slope, tail_efold_px,
    annulus_speckle` (missing in 42-58% of rows), `flood, flood_background` (scene level, meaning unclear),
    `gain_factor, bg_noise, clip_radius` (camera and processing settings), `fringe_*` (projector white light, not
    laser). **A1 (photo colour):** mean and standard deviation of R, G, B, H, S, V over the whole photo and over its
    central box (24 values). **A2 (photo encoder):** the pooled image embedding of pretrained SigLIP 2 SO400M/16-512,
    the backbone of AGAI v2's classifier as released (no strawberry fine-tuning), photo resized to 512x512 (1152
    values; computed on Hamilton). **L+A2** (exploratory).
  - *Models,* the same for every feature set, fit on the training part only: (i, primary) median imputation,
    standardisation, multinomial logistic regression, C = 1.0, balanced class weights; (ii) random forest, 500 trees,
    balanced class weights, seed 0, median imputation.
  - *Evaluation.* Leak-free (primary): leave one object out, so each object's scans are predicted by a model that
    never saw that object. Leaky comparison: 5-fold split by scan, stratified by material, 20 repeats (seeds 0-19),
    where the object's other pose can be in training. Metrics: balanced accuracy over scans (mean recall over the
    materials; chance 0.20 with 5 materials) is primary; macro F1; object accuracy and object balanced accuracy (scan
    probabilities averaged per object). Uncertainty: 95% percentile bootstrap over objects, 2000 resamples, seed 0
    (a material absent from a resample drops out of that resample's average). Laser against photo: difference in
    balanced accuracy with the same bootstrap, and an exact McNemar test on which objects each gets right.
  - *Predictions.* **P1:** L, logistic regression, leak-free: balanced accuracy above chance, lower 95% bound > 0.20.
    **P2:** the leaky split overstates it by at least 0.10 (L, logistic regression, leaky mean minus leak-free).
    **P3:** L beats the photo encoder A2 on unseen objects (logistic regression, leak-free, point estimate; with 31
    objects the comparison has little power, so the interval and McNemar test are reported as they come out).
    **P4** (exploratory): metal or not, all objects, L, logistic regression, leak-free: object accuracy at least 0.85.
  - *Exploratory, named now.* All nine materials (L and A2); each laser channel alone (which wavelengths carry the
    information); without the two objects the 3DAI handoff flags as possibly mislabelled ("taped black metal",
    labelled paper, and "taped breadboard", labelled foam); the random-forest and L+A2 rows.
  - *Known limits.* 31 objects, 16 of them plastic; one scanner, nearly all scans from one session (2026-09-15); the
    material labels are the collector's; colour and material are linked for some objects (A1 and A2 measure how much a
    photo already gives). This is a pilot to decide what to collect next, not a result for a paper.
  - *Code.* `experiments/e01_material_unseen_objects/` (`fetch_3dai.py`, `photo_color_stats.py`, `embed_siglip2.py`,
    `run_e01.py`). `run_e01.py --synthetic` was run on made-up data with a known answer before this entry: with a
    strong per-object effect and weak photos, leak-free L 0.39 against leaky 0.98 and L above A2 and A1, as built.
- **2026-11-18, 02:55 CST. (b) E01 results: on objects held out of training the laser readings do little better than
  chance, and no better than a photo.**
  - *Run.* 79 of 80 scans (one scan has no photo); 31 objects in the five primary materials; 78 laser features, none
    dropped. Inputs (SHA-256): `laser_features.csv` b412a77dc36b90f3..., photo colour table 08abb316f6d9831d...,
    SigLIP 2 embeddings 06d9092a4f588bcf... (Hamilton, one RTX 5000 Ada, torch 2.9.1, transformers 5.20.2). Analysis
    on a Windows PC (Python 3.12.1, scikit-learn 1.9.0, numpy 2.4.6, pandas 3.0.3, scipy 1.18.0), 5 minutes. One code
    change after (a): `embed_siglip2.py` loads the model with `.float()` instead of `torch_dtype`, an argument
    transformers 5 changed; the analysis code is as committed in (a).
  - *Leak-free* (leave one object out), balanced accuracy, chance 0.20: laser L 0.249 (95% CI 0.146-0.431; object
    accuracy 0.516); photo encoder A2 0.231 (0.152-0.386; 0.516); photo colour A1 0.033 with logistic regression,
    worse than chance, and 0.184 with the random forest; laser plus photo encoder 0.237. Random forests are no better
    (L 0.214, A2 0.220).
  - *Leaky split* (5-fold by scan, 20 repeats): L 0.466, A2 0.778. Both look much better when the object's other pose
    is in training, the photo encoder most, since it recognises the object itself.
  - *Laser against photo.* Against the encoder: +0.018 (-0.118 to 0.169); each gets 3 objects right that the other
    gets wrong (McNemar P = 1). Against colour: +0.216 (0.070 to 0.423), 16 objects to 0, P = 3e-5.
  - *Metal or not,* all 35 objects: object accuracy 0.629 (exact 95% CI 0.449-0.785); recall on metal scans 0.42.
  - *Predictions.* P1 not supported (lower bound 0.146). P2 supported (0.466 - 0.249 = 0.217). P3 met by the letter of
    the rule (+0.018), but that is a tie, not a win. P4 not supported (0.629).
  - *Exploratory, named in (a).* All nine materials: L 0.110, A2 0.144. Each laser channel alone: 0.12-0.18, so no
    single wavelength carries it. Without the two possibly mislabelled objects: 0.188.
  - *Post hoc, not planned; ideas to test, not results.* The laser misses most metals whose surface is painted,
    textured or sparkly (the metal pot, "green painted"; the electronic metal backing; the unfinished metal block) and
    4 of 5 scans of the polished aluminium, where a mirror-like surface may send the beam away from the camera. It gets
    "taped black metal", labelled paper because of its masking tape, right on both scans: the beam sees the surface
    layer. The photo encoder gets the transparent and translucent plastics right more often than the laser.
  - *Reading.* On 31 objects these laser features do not name the material of a new object reliably, so this data
    cannot yet decide Dr. Zhang's claim that lasers recognise materials regardless of shape. It is small (two objects
    each for cardboard and foam), and its labels give the bulk material while the beam reports the surface. With a
    random split the same data would have looked like a working method (0.47 for the laser, 0.78 for the photo), the
    AGAI lesson again. What would make the test decisive: at least 5-10 objects per material, with mixed colours and
    finishes; a label for the surface the beam hits (paint, tape, coating) next to the bulk material; several angles
    for shiny objects; and the full-resolution laser frames from the scanner PC instead of the summary features.
    Files: `results/e01/` (`summary.json`, `report.md`, `predictions_scans.csv`).
- **2026-11-18, 02:58 CST. (c) Two notes.** Entry (b) is stamped 02:55 but was committed at 02:52 (commit 0f4d8bc);
  the stamp was written ahead. And for the next experiment: the robotic-arm dataset has more objects per material
  than 3DAI (plastic 15, metal 10, rubber 7, paper 2, foam 2; counted from its class names), so the same
  unseen-object test is possible there with larger groups, on photos where the laser spot is visible. Not planned in
  detail yet; that waits for the arm project's full image set (more than 3,000 images, many not yet labelled).
- **2026-11-18, 02:54 CST. (d) Correction to (c).** Entry (c) is stamped 02:58 but was committed at 02:53 (commit a9c26a1), the same mistake it reports about (b). From this entry on, stamps are read from the clock when the entry is written.
- **2026-11-18, 03:23 CST. (e) Independent re-check of E01, and a post hoc permutation test.**
  - An independent re-analysis, given only the plan (a) and the code, re-ran `run_e01.py` and reproduced every
    committed point estimate and a byte-identical `predictions_scans.csv`. It then rebuilt the analysis in separate
    code (its own pivot and photo lookup, leave-one-object-out with a check that the held-out object never reaches
    training, metrics, bootstrap and McNemar test): all 70 values compared match. No bugs were found.
  - Deviations from (a) it found; none touches a prediction:
    1. The random forest uses `class_weight="balanced_subsample"` (as committed with (a)); the text of (a) says
       balanced class weights. With `"balanced"`: leaky L/rf 0.581 instead of 0.369; leak-free L+A2/rf 0.280 instead
       of 0.183, A2/rf 0.247 instead of 0.220, A1/rf 0.144 instead of 0.184 (the same with forest seeds 1 and 2). No
       prediction uses the forest; the committed numbers stay the record.
    2. "Without possibly mislabelled" kept foam with a single object, whose recall is then 0 by construction, against
       (a)'s own rule for one-object materials. With the materials recounted (four, chance 0.25) it is 0.228; the same
       four materials with nothing removed give 0.311.
    3. Three objects (cardboard piece 01, polished metal 01, white breadboard 01) have 5 analysed scans, not 2, so they
       weigh 2.5 times in the scan-level metrics. With every object weighted equally, L is 0.258 and A2 0.263, so the
       sign of P3 flips; (b) already read it as a tie.
    4. Minor: `report.md` prints "20 repeats" as fixed text, and does not mark the forest and L+A2 rows as exploratory.
  - The re-analysis also showed that with materials shuffled between objects, leave-one-object-out scores about 0.16
    on average, below the nominal 0.20 (small groups, balanced weights). So 0.20 is not the right baseline.
  - *Post hoc permutation test* (not in (a); `permutation_null.py`): materials shuffled between objects, the primary
    analysis rerun. L: observed 0.249, null mean 0.168 (sd 0.052, 95th percentile 0.273, 500 shuffles), one-sided
    P = 0.084. A2: 0.231 against 0.176 (sd 0.050, 95th percentile 0.266, 200 shuffles), P = 0.10.
  - *Reading.* Unchanged. Both sit about 1.5 standard deviations above shuffled labels, which is a hint at most; neither
    clears the 95th percentile, and the verdicts of (b) stand (P1 not supported). File:
    `results/e01/permutation_null.json`.
  - Added `docs/RELATED_WORK.md`, a literature scan. Methods that work on held-out objects use 10-18 objects per
    material and near-infrared spectra, scattering measured per wavelength or several views; none of the optical ones
    reads through paint or tape.

- **2026-11-18, 16:33 CST. (f) Pre-registered: E02, train on the 3DAI scanner, test on the robotic arm** (Dr. Zhang's suggestion
  of 18 November, 16:05).
  - *Question.* Does a material classifier trained on the scanner's red-laser spots name the material under the arm's
    laser spot, on other objects and another camera?
  - *Data.* 3DAI at `6556bec` (E01's commit): the two 635 nm channels (`laser_ch1`, `laser_ch2`) of every scan of the
    four materials both rigs have, plastic 16 objects, metal 8, paper 3, foam 2. Arm (`roboticsarmAI` at `96b76af`):
    primary photos only (the `split_r` and `split_g` variants come from another, monochrome camera and are saturated),
    every labeled laser spot whose material is one of the four. The arm's material labels are Gemini appearance guesses
    and are given per spot, so an object can have parts of two materials (scissors: plastic handle, metal blade); the
    test unit is an object-part.
  - *Spots* (`experiments/e02_cross_rig/spots.py`). Scanner spot: maximum of the smoothed red excess of (laser frame
    minus `laser_dark`); arm spot: the same maximum inside the labeled box. From there both rigs are processed the same
    way: local background, half-maximum radius r50, 16 features in units of r50 (radial profile at 0.5 to 4 r50, core
    and ring chromaticity, core saturation, speckle contrast, peak over background, ellipticity, halo fraction, width
    ratio). Quality control before any scoring (`data/e02/crops_sheet.jpg`): the scanner shows a small round dot, the
    arm a large, saturated, elongated streak.
  - *Analysis* (`experiments/e02_cross_rig/run_e02.py`). Logistic regression (C = 1, balanced) trained on all scanner
    samples and applied to every arm spot; a unit's prediction is the mean probability over its spots. Primary:
    unit-level balanced accuracy over the four materials (chance 0.25) with features standardized within each rig
    separately (uses no labels); 95% bootstrap interval over units; permutation test, 1000 shuffles of the scanner's
    materials among its objects. Secondary: the same with standardization on the scanner only (raw transfer); plastic
    against metal (chance 0.5); spot-level balanced accuracy; leave-one-object-out within each rig as references.
    Sensitivity: without the 7 arm objects flagged below.
  - *Label check* (one photo per object, before scoring), for William or Finn to confirm by hand: insulated water
    bottle (spot on a painted coating), precision screwdriver (another entry of the same object says rubber), robotic
    gripper finger assembly (metal and plastic parts), slip joint pliers and scissors (the spot seems to be on the
    table in the photo checked), stapler (black surface, plastic or painted), cardboard shipping box (labeled plastic;
    the spot is on blue plastic tape).
  - *Predictions.* P1: primary balanced accuracy with a lower 95% bound above 0.25 and permutation P < 0.05; not
    expected, since the spots look very different on the two rigs. P2: leave-one-object-out within the arm above the
    cross-rig primary. P3: plastic against metal above 0.5 with P < 0.05; no expectation.
  - *Limits.* One wavelength in common (635 nm); different cameras, distances and angles; the arm's labels are
    appearance guesses; photos of one arm object are near-copies, hence the unit-level score.

- **2026-11-18, 16:38 CST. (g) E02 results: the red spot in a photo does not carry the material, and nothing transfers from the
  scanner to the arm.** As planned in (f) (`results/e02/report.md`, `results/e02/summary.json`): 129 scanner spots of
  29 objects for training, 1084 arm spots in 30 object-parts for testing.
  1. Primary (four materials, features standardized per rig): balanced accuracy 0.267 over object-parts (95% interval
     0.170 to 0.431; chance 0.25), against 0.263 for shuffled scanner labels (1000 shuffles), P = 0.48. P1 not
     supported.
  2. Standardized on the scanner only (raw transfer): 0.250, every part called the same material, P = 0.54.
  3. Plastic against metal: 0.517 (0.306 to 0.717; chance 0.50), P = 0.35. P3 not supported.
  4. Without the 7 flagged arm objects: 0.239, P = 0.59.
  5. References, leave one object out within one rig with the same 16 features: scanner 0.224, arm 0.358. Post hoc
     (not in the plan): the arm's 0.358 against shuffled arm labels has null mean 0.234 and 95th percentile 0.475,
     P = 0.16 (200 shuffles; `within_arm_null.py`, `results/e02/within_arm_null.json`). P2 holds as a number, not as
     a result.
  - *Reading.* There is nothing to transfer yet: the spot's shape, halo and colour, as an ordinary RGB camera records
    them, do not separate the four materials even within one rig (scanner 0.22; E01's per-channel laser features
    0.25), and the two rigs record the same 635 nm laser very differently (small round dot against large saturated
    streak), so features standardized per rig still do not line up. For a cross-rig test to mean something, both rigs
    need the raw laser signal (unsaturated exposures, the laser frames without the scene, several wavelengths,
    speckle) and more objects per material, as `docs/RELATED_WORK.md` recommends. The arm's material labels (Gemini
    appearance guesses, given per spot) still need the hand check listed in (f).

- **2026-11-18, 16:59 CST. (h) Correction to the label check of (f): most flags were wrong.** (f) judged each arm object from
  one photo and one box. The arm labels every laser spot by the part it lands on, and the sheet of every labeled part
  (`data/e02/flagged_7_objects.jpg`, 4 photos per part) shows: precision screwdriver = rubber grip (38 spots) and
  metal collar (23), the boxes on the matching parts in 7 of 8 photos (one rubber-labeled photo of 5 October shows a
  red handle, not this black-and-orange screwdriver); robotic gripper finger assembly = blue plastic fingers (36) and
  metal brackets (24), the plastic boxes on the blue fingers, the metal boxes on silver brackets in two photos and on a
  white part that may be plastic in one; slip joint pliers = metal jaws (24, all on metal) and yellow handle (16);
  cardboard shipping box = cardboard (40, labeled "composite") and the blue tape (12, labeled plastic), all matching;
  scissors = red plastic handle (49) and metal blade (25), matching. Still open: insulated water bottle (3 spots on a
  painted coating, labeled metal), stapler (26 spots on a black body labeled metal; paint or plastic cannot be told from
  the photos). New observation: in some photos the box surrounds a green laser dot while the red spot falls outside
  it, on the table (pliers handle 2 of 4 photos, stapler 2 of 4, scissors blade 1 of 4); E02's features are measured at
  the red maximum inside the box, so such boxes give no red spot. The E02 results of (g) stand as run; its sensitivity
  analysis rests on the flags of (f).

- **2026-11-18, 17:15 CST. (i) The two open labels of (h) settled by eye.** William looked at the objects: the insulated
  water bottle is painted metal with matte paint, the stapler painted metal with glossy paint. Both stay metal; the
  shared label list (`labels/`, which maps both datasets' labels to one list of materials without relabeling either)
  records them with detail "painted" and the paint's surface, since the laser reads the surface.

- **2026-11-18, 17:23 CST. (j) Correction: the arm's labels are its author's own, not AI guesses.** Entries (f) and (g)
  call the arm's material labels "Gemini appearance guesses". That was inferred from the arm project's code (it has a
  Gemini material labeler) and is wrong for this data: asked on 18 November (17:18), the arm's author answered that
  all current labels are his own and that no AI-suggested label has been approved or uploaded yet (the AI answers in
  `dataset/pdf_for_labels/` are suggestions; the build report kept `data.yaml` as the authority). He also confirmed
  that each box marks one laser spot; whether red or green is not recorded per box. Nothing in the results of (g)
  changes. `labels/materials.py` now names the source correctly.

- **2026-11-18, 17:24 CST. (k) On the boxes of (h).** William, who built the scanner on the same laser board: each arm box
  marks the point where both lasers aim; red or green spots outside a box come from refraction, reflection and the
  like at the surface, not from a box marking the wrong laser. So the boxes are right as labeled; for E02 it means the
  red maximum inside a box can be a secondary spot rather than the main one, which the results of (g) do not depend on.

- **2026-11-18, 17:26 CST. (l) Post hoc check of (k): the red spot is inside the box in 98% of cases.** For each arm box in
  a primary photo, the strongest smoothed red excess inside the box against the strongest in the area around it (out
  to 3 box sizes); a box counts as "red spot outside" when the outside maximum is more than 1.5 times the inside one and
  visible (above 30). 33 of 1596 boxes (2.1%): metal 13 of 263, plastic 15 of 566, rubber 2 of 303, foam, leather and
  paper 1 each, none for composite, fabric, stone or wood. So E02 measured the main red spot almost everywhere, and the
  result of (g) does not come from stray spots (`red_outside_check.py`, `results/e02/red_outside.csv`).

- **2026-11-20, 12:24 CST. (m) Pre-registered: E03, wave methods 1, 2 and 4 on the 3DAI laser frames.** Dr. Zhang
  (Discord, 20 November): make the method generalize regardless of how the data look, by looking inside the laser
  reflection and extracting the nature of the wave information; start with 1 (speckle), 2 (scattering model fit) and 4
  (wave computing), very innovative methods; 3 (polarization) needs a device his team will build. The methods are in
  `docs/WAVE_METHODS.md`. Written before any feature set was scored on the real data; only the label-free quality
  numbers below were looked at.
  - *Question.* Do features built from the physics of the reflection name the material of objects never seen in
    training, and do they measure the material rather than the laser, the pose or the object's look?
  - *Data.* 3DAI at `6556bec`, the 80 scans with laser features (E01's), frames `laser_ch1` and `laser_ch2` (635 nm),
    `laser_ch4` (530 nm) and `laser_dark`: JPEG previews, 800 px, 4:2:0 chroma. Channel 3 (940 nm) is not seen by the
    colour camera and is left out. The `laser_zoom.jpg` panels are 2x enlargements of preview-scale crops and add no
    detail. Primary set as in E01: the five materials with at least two objects, 31 objects in 72 scans (cardboard 2,
    foam 2, metal 8, paper 3, plastic 16); labels `material_class`, objects grouped by `label`.
  - *Common processing* (`experiments/e03_wave_methods/common.py`). The sRGB curve is undone so that light adds; the
    laser-only image is D = lin(laser) - lin(dark) in the laser's camera channel; pixels at 245 or more in the laser
    frame are clipped and known only to be at least the clip level.
  - *Method 2* (`m2_scatter.py`). D around the spot = A beam(r) + B (beam * R_d)(r) + c, with R_d the normalized
    diffusion profile of Christensen and Burley and the beam a Gaussian whose width is calibrated per laser without
    labels (10th percentile of single-Gaussian widths over all spots with SNR >= 5: ch1 3.09 px, ch2 5.68 px, ch4 1.70
    px); censored robust least squares in a 30 px window, noise from the frame beyond 60 px. Features per laser:
    `log_d_beam` (subsurface length in beam widths; d is capped at 400 px, meaning a halo wider than the window),
    `wing_slope` (model-free log-log slope of the unclipped halo), `aniso_excess` (elongation of the halo beyond the
    core's, for fibres); per scan, green minus mean red of `log_d_beam` and of `wing_slope`. Quality before scoring:
    spots found for 80, 79 and 78 of 80 scans (ch1, ch2, ch4), fitted 80, 79 and 70 (ch4: 4 channel-invalid, 5 below
    SNR 5). The core is clipped in 227 of the 229 fitted spots (median clipped radius 8.3, 19.7 and 7.1 px), so the
    surface fraction A/(A+B) is unidentified and is not used; d reaches its 400 px cap in 30, 64 and 13 spots.
    Changes made during development, from these quality numbers alone: d's range raised from 80 to 400 px, the flat
    offset c allowed from -0.1 to 0.3 (frame drift, or a laser's own faint glow; it was +-0.05), noise taken away from
    the spot instead of from a ring the halo reaches, the surface fraction kept only for unclipped spots, the wing
    slope added. Synthetic check (60 made-up spots, 38 clipped): Spearman 0.98 for log d, 0.86 for the surface
    fraction on the 22 unclipped (`results/e03/m2_synthetic.json`).
  - *Set S.* The per-laser features and the two differences, with E01's rule that a feature missing in more than 20%
    of rows is dropped. Clipping leaves too little halo for the wing slopes (missing 21%, 44% and 35%, the difference
    36%), so S has 7 features: `log_d_beam` and `aniso_excess` for each laser and the green-minus-red `log_d_beam`.
    Variant S_no_ch2 (exploratory): without channel 2, whose halo reaches the window edge in 64 of 79 spots.
  - *Method 4, digital stand-in* (`m4_reservoir.py`). Spot crops of 41 x 41 px around each fitted centre (clipped
    pixels at their clip level), each laser scaled to unit sum so brightness is gone; y = |W x|^2 with W complex
    Gaussian (1000 x 5043, seed 0). R_states: the 1000 outputs; R_kernel: the correlations between the outputs for each
    pair of lasers (3 values); RAW: a linear readout on the 5043 pixels themselves, without the projection.
  - *Method 1* (`m1_speckle.py`). M = 1/K_s^2, with K_s^2 = (1 + K_laser^2)/(1 + K_texture^2) - 1 in a ring around the
    spot, the texture from the lasers-off frame; with two frames per laser the noise cancels in their product. Run
    during development (`results/e03/m1_speckle_simulation.json`, 30 spots per cell): on raw frames M = 1, 2 and 4 are
    told apart perfectly (AUC 1.00); through a preview pipeline (sRGB, downscaling, JPEG quality 85 with 4:2:0) the
    separation of M = 1 from M = 2 falls to AUC 0.91 to 0.94 at 1.6x downscaling and 0.72 to 0.77 at 2.4x. How much the
    scanner downscales is not known. On the previews, with one frame per laser, in a 10 px ring just outside each
    clipped core: set SPK (log M per laser, green minus red), exploratory.
  - *Comparison sets.* E02 (E02's 16 spot features for both red lasers, 32), L (E01's 78 laser features), F (the
    projector's fringe contrast: `fringe_contrast_object`, `_object_std`, `_ratio`, `fringe_reliable_fraction`;
    projector light, not laser; exploratory).
  - *Evaluation* (`run_e03.py`), as in E01: leave one object out; median imputation, standardization, logistic
    regression (C = 1, balanced class weights). Primary metric: balanced accuracy over objects (scan probabilities
    averaged per object, so every object counts once; E01's scan-level number is reported too). 95% bootstrap over
    objects (2000); permutation test with materials shuffled among objects (S: 1000 shuffles; sets of at most 100
    features: 200; R_states and RAW: 50); exact McNemar tests on objects. Pipeline check on made-up data
    (`results/e03/run_e03_synthetic.json`): a planted signal 0.40 (P = 0.03); pure noise 0.153 on average over 10
    draws against a shuffled-label mean of 0.163, none with P < 0.05. One single noise draw had scored 0.48, by luck on
    the two-object materials, which is how much one object moves this metric.
  - *Validity checks,* on the per-laser features of S, E02, L (its 13 colour-camera features per channel) and SPK:
    (i) the two red lasers, ICC(C,1) between ch1 and ch2 on the same scans (different beams, points about 11 px apart);
    (ii) the two poses, ICC(C,1) between an object's two poses; (iii) material signal, eta^2 of object means over the
    five materials; (iv) appearance leakage, the within-material partial R^2 of the object's look at the spot (log
    brightness and two chromaticities of the lasers-off frame within 5 px). Control: the clipped radius, which
    brightness drives, should leak.
  - *Predictions.* **P1 (primary):** S above shuffled labels, one-sided P < 0.05. No firm expectation: S is measured in
    linear units with clipping handled, but E01 and E02 sat near chance on these 31 objects. **P2:** the median
    two-laser ICC of S's per-laser features (`log_d_beam`, `wing_slope`, `aniso_excess`) above the medians of E02's
    and of L's features. **P3:** the median appearance leakage of S's per-laser features below E02's. **P4:** R_states
    does not differ from RAW (exact McNemar P >= 0.05): a digital random projection adds no information, so a physical
    reservoir has to bring what the camera image does not hold (phase, coherence). **P5:** method 1 cannot be read
    from the previews: SPK's two-laser ICC below 0.5 and SPK not above shuffled labels (P >= 0.05).
  - *Exploratory, named now.* S_no_ch2; S + R_kernel; R_kernel alone; SPK; F; S on all nine materials; S without the two
    possibly mislabelled objects; material medians of the red and green subsurface lengths (expected order: metal
    shortest, foam longest).
  - *Limits.* 31 objects; previews of one frame per laser; cores clipped; channel 2's halo wider than the window; one
    session. This tests what the GitHub files allow, not the methods at full strength; the captures they need are listed
    in `docs/WAVE_METHODS.md`.

- **2026-11-20, 13:03 CST. (n) E03 results: on the GitHub previews the new features fail their own validity checks
  and do not name materials; the older features beat shuffled labels on the two cardboard objects.** As planned in (m)
  (`results/e03/report.md`, `summary.json`, `validity.csv`, `predictions_objects.csv`); 72 scans of 31 objects; run
  time 29 minutes.
  1. **P1 not supported.** S: balanced accuracy over objects 0.142 (95% CI 0.022 to 0.311) against a shuffled-label
     mean of 0.181, P = 0.57 (1000 shuffles).
  2. **P2 not supported.** Median two-laser ICC of S's per-laser features 0.07, against 0.16 for E02 and 0.68 for L.
     S's subsurface length gives -0.05 between the two red lasers and 0.13 to 0.61 between poses.
  3. **P3 met by the letter only.** S follows the look less than E02 (median partial R^2 0.13 against 0.35), but it
     carries almost no material signal (eta^2 0.08) and does not agree between lasers, so there is little to follow.
  4. **P4 supported.** R states 0.188 and RAW 0.113 (both at chance); exact McNemar 4 to 2, P = 0.69.
  5. **P5 half met.** SPK carries no material signal (0.113, P = 0.74), but its log M agrees between the two red
     lasers at 0.53, above the 0.5 the prediction allowed, between poses at 0.55, and does not follow the look (0.07).
  6. **Comparison sets,** scored per object as (m) planned: E02 0.400 (P = 0.015), L 0.304 (P = 0.040); against S,
     exact McNemar 9 to 1 (P = 0.021) and 13 to 2 (P = 0.007). E01 reported L per scan (0.249; post hoc P = 0.084 in
     (e)); counting each object once it is 0.304. Both follow the look strongly (0.35 and 0.53). E02 names both cardboard
     objects; its recall on the other four materials averages about chance.
  7. **Exploratory, named in (m).** S without channel 2 0.062; S + R kernel 0.167; R kernel 0.208 (P = 0.36); F
     (fringe contrast) 0.138; S on all nine materials 0.056; S without the two possibly mislabelled objects 0.375 over
     four materials (no permutation test). Physics check reversed: red subsurface length in beam widths, material
     medians, foam 2.83, cardboard 3.23, paper 3.52, plastic 3.58, metal 3.61, where metal shortest and foam longest
     was expected. The brightness control (clipped radius) agrees between lasers (0.72) and poses (0.77) and follows the
     look (up to 0.74), as a control should: agreement alone does not make a feature a material property.
  - *Post hoc, not planned* (`posthoc_colour.py`, `results/e03/posthoc_colour.json`, 200 shuffles each), run because E02
    and L beat shuffled labels while following the look: E02 without its 8 colour features 0.362 (P = 0.015), colour
    features only 0.342 (P = 0.10); L without its 26 colour features 0.200 (P = 0.32), colour only 0.217 (P = 0.23).
    E02 names both cardboard objects in every version. With two cardboard objects and more than a dozen sets tested,
    this is a hint that the spot's shape in E02's units separates cardboard on this scanner, not a result.
  - *Reading.* The GitHub previews cannot test methods 1 and 2. Clipping (227 of 229 cores, (m)) leaves only the far
    halo, and S is not the same quantity under the two red lasers and orders the materials against the physics. The
    validity checks showed this without any classifier, and flagged the brightness control as following the look, so
    they carry over as the gate for new captures. Leads for the full data: the model-free wing slope agrees between the
    two red lasers at 0.63, and the speckle M agrees between lasers and poses without following the look. Next: the
    full-resolution bundle from the scanner PC, then captures with unclipped spots (bracketed exposures), two frames
    per laser and an aperture that makes speckle grains at least 2 px; the older features' cardboard hits to be
    rechecked there with more cardboard objects.

- **2026-11-20, 13:31 CST. (o) Correction: the arm's "more than 3,000 images" is an unchecked figure.** Entry (c) and
  `docs/DATA.md` give the arm project's full image set as more than 3,000 images, many unlabelled, as if counted. The
  number comes from Dr. Zhang's message of 17 November (23:43); the arm's author has not given a total, and none was
  checked. The counts checked are 1,705 images in the repository and 1,786 in the public Roboflow export
  (`mtsu-2h73y/laser-assisted-computer-vision`, version 4 of 22 October, the latest of four versions; looked at on
  20 November without signing in). `docs/DATA.md` and `README.md` now say so; the total is to be confirmed with the
  arm's author. Nothing in E01 to E03 used the figure.

- **2026-11-20, 13:59 CST. (p) The arm's total, from its author.** Asked after (o), the arm's author answered
  (Discord, 20 November, 13:55) that he has about 2,000 unlabelled images besides the roughly 1,700 labelled ones, and
  that those labels need to be updated. So the full set is about 3,700 images, in line with Dr. Zhang's "more than
  3,000". What the update changes is not known yet; `docs/DATA.md` and `README.md` now give these numbers.

- **2026-11-20, 14:41 CST. (q) Three checks.**
  1. *Post hoc, on E03's SPK.* The two red lasers' log M agree (ICC 0.53 in (n)) not because the ring sits just
     outside each clipped core: after the log M of each laser is regressed on that laser's own clipped radius, the
     residuals still correlate at 0.43 (79 scans). The rank correlation of log M with the clipped radius is -0.31,
     -0.41 and 0.03 (ch1, ch2, ch4). The lead named in (n) stands; what the previews' M measures stays open.
  2. *Sources of `docs/WAVE_METHODS.md` opened:* the arXiv preprint [17] and the five patents [2, 3, 6, 9, 20] exist and
     match, except that the trinamiX patent [3] determines a material property, such as skin, from the beam profile of
     the reflected spot (speckle included), which is closer to method 2 than the doc said. Its description is
     corrected and it is now named next to method 2's related work.
  3. *A claim corrected.* The doc and the paper draft said that lengths in beam widths mean the same on the scanner and
     the arm because both use the same laser board. The beam widens with distance, so that holds only at the same
     working distance, or after one measurement of the beam's width against distance, which turns the lengths into
     millimetres on either rig. Both texts now say so.

- **2026-11-22, 23:41 CST. (r) Plan for E04: the math of methods 1 and 4, and a wave simulation to check it.** Written before any
  comparison below was scored. Seen before writing: the machinery checks of item 1, the solver's energy check, and the
  speckle contrast of one sample (1.03).
  - *Why.* Dr. Zhang (Discord, 22 November): the methods are still an idea; he asked for the math and simulated results
    in a weekly note, and for a simulation built on the wave nature of light and its interaction with materials, later
    usable as synthetic training data. E03's simulation only drew speckle with a chosen M; it did not derive M from a
    material.
  - *Three levels* (`experiments/e04_wave_sim`).
    1. **Exact wave solution** (`fullwave.py`). The scalar wave equation solved for 3,198 point scatterers (Foldy-Lax
       equations, every order of multiple scattering): a cylinder of radius 9 wavelengths, 4 mean free paths thick, k x
       mean free path = 10, lit by a Gaussian beam (waist 3.5 wavelengths); 16 random samples, 25 wavenumbers over
       +-3%, 9 beam tilts. Far-field speckle 20 to 50 degrees from backscatter.
    2. **Transport Monte Carlo** (`transport_mc.py`). Paths through a half-space of electric-dipole (Rayleigh)
       scatterers, each carrying its polarization vector; scored is the light leaving towards a distant camera, by
       distance from the entry point, path length and polarization. Lengths in mean free paths, absorption applied
       afterwards.
    3. **Speckle frames** (`synth.py`, to be written against the formulas below). Fields drawn as Gaussian random
       fields with exactly the covariance the path statistics give (across path length, polarization, laser modes and
       beam positions), passed through the lens pupil, integrated on pixels with shot and read noise, two frames per
       laser; then E03's two-frame estimator.
  - *The math, fixed here.* W(x, s) is the 2 x 2 polarization coherency of the light leaving at x after path length s
    (surface light is a term at s = 0), n the refractive index, gamma the laser's degree of temporal coherence, h the
    lens's amplitude point-spread function, K the transport kernel (radiance leaving at a distance from where it
    entered).
    - F1, one exposure: K^2(x) = integral integral tr[W(x,s) W(x,s')] |gamma(n(s - s'))|^2 ds ds' / (integral tr W ds)^2,
      and M = 1/K^2. A single-frequency laser gives K^2 = (1 + P^2)/2 with P the degree of polarization; a share f of
      surface light over depolarized volume light gives K^2 = (1 + f^2)/2.
    - F2, diffusion: the Fourier transform over path length of the diffuse reflectance equals the steady-state
      reflectance with the absorption mu_a replaced by mu_a - i n dk.
    - F3, two exposures a wavenumber step dk apart: speckle correlation = ||Gamma(dk)||^2 / ||Gamma(0)||^2 (Frobenius
      norms), Gamma(dk) = integral W(x,s) exp(i dk n s) ds.
    - F4, two exposures with the beam at a and b: field correlation at image point x is
      Gamma_ab(x) = integral F_x(x') A(x'-a) A*(x'-b) d2x', with A the beam's field and F_x = (|h|^2 * K)(x - x') the
      distribution of entry points feeding x. (a) Flat wavefront, Gaussian beam of 1/e^2 radius w: the speckle
      correlation is at least exp(-d^2/w^2) for a step d, so at least 0.91 for d <= 0.3 w, for any material. (b)
      Wavefront of radius R_c: the correlation is the squared Fourier transform of F_x at q = k d / R_c; for a Gaussian
      F_x of rms width sigma it halves at d = 0.133 lambda R_c / sigma; surface light halves at d = 0.47 R_c NA (overlap
      of the shifted pupil).
    - F5, full wave against paths: spectral correlation = mean over directions of |sum_p w_p exp(i dk L_p)|^2 /
      |sum_p w_p|^2 with L_p the whole path; tilt correlation the same with the phase change of each path under the
      tilt.
  - *Predictions.*
    1. Machinery (seen): isotropic half-space, normal-exit radiance 0.3364 per steradian (Chandrasekhar's H(1)^2 / 8 pi)
       within 1% once truncated paths are counted; radiance over flux towards the Milne value 0.4008 with distance.
       Solver: extinction equals scattering.
    2. Full wave: speckle contrast 1 +- 0.05; the spectral and the tilt correlation curves within 0.10 of F5 at every
       step.
    3. Diffusion against Monte Carlo (F2): the wavenumber step at which the field correlation halves agrees within 15%
       at 10 and 20 transport lengths from the entry point.
    4. Polarization: beyond 5 mean free paths from a pencil beam the degree of polarization is below 0.1, so a
       single-frequency laser gives M above 1.98 there.
    5. Frames against F1: ring by ring, K^2 measured with the two-frame estimator and divided by the bare-metal
       reference is within 15% of F1 (rings of 200 grains or more); M(r) stays 1 on the metal and rises with r on volume
       scatterers under a multimode laser.
    6. Beam steps (F4): frames within 0.05 of the formula. Flat wavefront: surface-only and volume-only correlations
       differ by less than 0.1 for steps up to 0.3 w. Curved wavefront: there is a step at which volume light is below
       0.5 while surface light is above 0.9.
    7. Wavelength steps (F3): frames within 0.05 of the formula.
  - *Expected correction.* Prediction 6 says the stay-or-move test as worded in `docs/WAVE_METHODS.md` 4B (a fraction of
    a millimetre, any beam) does not hold for a collimated beam; it needs a diverging or converging beam, or steps
    about as large as the beam. If the simulation agrees, 4B and the capture list are corrected.
  - *Frame parameters.* 635 nm, n = 1.4, object grid 0.025 mm, camera pixel 0.05 mm, speckle grain 0.15 mm, beam 1/e^2
    radius 1.0 mm. Materials (made-up parameter sets, not measured): bare metal (surface only); three volume
    scatterers with transport length 0.05, 0.2 and 0.8 mm and absorption 0.1, 0.05 and 0.02 per mm, each with 0% and
    30% surface light. Lasers: single-frequency, and multimode (1.0 nm wide, modes 0.15 nm apart). The rig's real laser
    spectra are not known.
  - *Not covered.* Large (Mie) particles, fibres, the exact polarization at an index-mismatched surface, the camera's
    image pipeline, and any real material: this tests the math and the estimators, not accuracy on objects. A material
    classifier trained on such frames is the step after, and anything of that kind run now is exploratory.
