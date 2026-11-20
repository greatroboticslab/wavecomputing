# Laser data in the lab

Inventory as of 2026-11-18. Counts come from the files at the commits named; where a source's own summary differs, both
numbers are given.

## 3DAI scanner

`greatroboticslab/3DAI`, commit 6556bec (2026-11-10). A scanner built to recognise materials. Lasers switched by an
ESP32 (channels 1 and 2 red 635 nm, 3 near-infrared 940 nm, 4 green 530 nm), an RGB camera, a Kinect (infrared and
depth) and a projector for fringe-projection height maps.

Each scan (one object, one pose) stores a lasers-off frame, one frame per laser, the Kinect infrared frame for each,
the laser transient array, a plain photo, Kinect depth, the fringe stack with the reconstructed height map, and
features per laser channel: scatter halo radii, speckle contrast, reflectance, and their infrared equivalents,
normalised by camera gain.

- On GitHub (`dataset/`): JPEG previews (longest side 800 px), `metadata.xlsx` (one row per image) and
  `laser_features.csv` (one row per scan and laser channel).
- Full-resolution frames and arrays stay on the scanner PC (`scanner_data/scans/`, MongoDB in Docker). A training
  bundle is exported with `python -m scanner_system.export_bundle out.zip --include-arrays`.
- `laser_features.csv`: 318 rows, 80 scans, 35 object labels (38 sample ids), 2 poses per object. Objects per material:
  plastic 16, metal 8, paper 3, cardboard 2, foam 2, fabric 1, fiberglass cloth press 1, stone 1, wood 1. The export's
  own README (2026-09-19) counts 38 objects, 87 scans and 1,155 images.

Known problems:
- The first three objects (cardboard piece 01, polished metal 01, white breadboard 01) have channel 4 flagged invalid
  (a loose wire, since fixed).
- The red lasers saturate the spot core on light objects (`saturated_core = 1`). Halo and speckle are measured outside
  the core and stay valid.
- Two labels may be wrong: "taped black metal" (labelled paper) and "taped breadboard" (labelled foam).
- The second camera's features (`cam_*`) are empty. `tail_slope` and `tail_efold_px` are missing for 58% of rows,
  `annulus_speckle` for 42%.
- The 3D height calibration still uses the July curve, because no flat reference object has passed the check yet.

## Robotic arm

`greatroboticslab/roboticsarmAI`, commit 96b76af (2026-11-11). A Dobot arm with the same 4-laser controller. YOLO
and RF-DETR models detect the object the laser points at.

- `dataset/`: 1,705 images (train 1,365, validation 340), 1280x720 (1,581) and 752x480 (124). They were taken on 11
  days between 2026-09-11 and 2026-10-21, and each file name starts with its date and time. Roboflow export:
  `mtsu-2h73y/laser-assisted-computer-vision`, version 4 (1,786 images, CC BY 4.0, 2026-10-24).
- 47 classes. Each class is one material part of one physical object, 33 objects in all, for example "ballpoint pen,
  plastic, red". Materials: metal, plastic, rubber, paper, cardboard, composite, foam, fabric, leather, silicone,
  stone, wood.
- The project's own audit (`stats_after_split/dataset_stats_report.txt`) found that 188 of the 340 validation images
  (55%) have a near-duplicate in training, and 56 of them look identical. The split was made at random by image, so
  the reported result (RF-DETR, mAP50 0.944 on the validation split) is likely higher than it would be on new photos.
- Objects per material (counted from the class names): plastic 15, metal 10, rubber 7, paper 2, foam 2, and one
  object each for leather, composite, plastic tape, silicone, fabric, stone, cardboard, composite plastic and wood.
  Several objects have parts of two materials (a wrench handle and head, for example). Material names are not always
  spelled the same way ("Metal" and "metal", "Foam" and "foam").
- `dataset/dataset_build_report.txt` lists objects whose classes do not match their label PDF.
- Total collected: not known. Dr. Zhang said on 17 November that the arm's author had collected more than 3,000
  images; that number is his and has not been checked. The counts checked are 1,705 in the repository and 1,786 in
  the public Roboflow export, whose version 4 (22 October) is its latest (versions 1 to 3: 1,210, 1,443 and 1,609
  images). Images taken since may not be labelled yet. `laser_labeler/` in that repo is an assisted labelling tool.

## Other laser data (not used yet)

- SpectrometerLearner2 (2023-24): one CSV per measurement (wavelength sent, wavelength and intensity returned), 16
  subjects including strawberry leaf, labelled healthy or not.
- laserai, lasergui, laserLLM: laser displacement interferometry time series (nanometres) from a Moku:Go; the data is
  on the lab Google Drive.
- Irrigation-Laser-Yolo2, lasermoisturerecognition: camera images of a soil-moisture probe and soil, labelled by
  moisture level.
- The 4DAI data hub: 3DAI scans can be pushed to it (integration added 2026-11-06); its contents have not been checked.
