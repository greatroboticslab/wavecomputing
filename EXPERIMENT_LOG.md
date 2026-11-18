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
