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
