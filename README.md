# Wave computing

Lasers are waves. How a surface reflects and scatters laser light depends mostly on what it is made of, not on the
object's shape. If that holds, a few lasers can recognise the material of many different objects, including objects
never seen before. The long-term aim is learning driven by wave physics that needs few or no labels: self-supervised
and contrastive learning first, quantum-inspired learning later.

This repository starts from the AGAI pipeline (`greatroboticslab/AGAI`): its models, and above all its way of testing
them. Test objects are never seen in training, every plan is written down before it is scored, and every result is
reported.

## Data

| Source | What it records | Size | Labels |
|---|---|---|---|
| 3DAI scanner | 4 lasers (635, 635, 940, 530 nm), RGB camera, Kinect infrared and depth, projector height map, one plain photo per scan | 35 objects, 80 scans with laser features | material, surface, transparency |
| Robotic arm | photos of objects with the arm's laser dot on them | 1,705 labelled images in the repo, 1,786 in the latest Roboflow export; about 2,000 more unlabelled (the arm's author, 20 November) | box per object, labelled object + material + colour |
| Others | spectrometer spectra, laser interferometry, soil-moisture images | see the inventory | |

Versions, counts and known problems: [docs/DATA.md](docs/DATA.md).

## How the AGAI pipeline carries over

| AGAI (strawberry plants) | Here (laser) |
|---|---|
| Leak-free split: near-duplicate photos grouped, test images from plants never seen in training | Split by object and capture session, so a test object is never in training |
| Classifier (SigLIP 2, ResNet-50) | Material classifier on laser readings, and on the laser spot in photos |
| Part detector (RF-DETR) | Laser-spot and object detector (the arm project already trains RF-DETR and YOLO) |
| Report writer with claim checks | A short explanation of each material call that may cite only measured laser features |
| Calibration and abstention gate | Abstain when a reading is unreliable (saturated spot, a laser that did not fire) |
| Teacher labels and self-consistency checks | Self-supervised and contrastive pretraining on the unlabelled captures, a small labelled set for testing |
| Lab notebook, plans before scoring | The same: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) |

## Plan

1. **E01 (now).** From the 3DAI laser readings alone, name the material of objects the model never saw, and compare
   with a plain photo of the same objects. [experiments/e01_material_unseen_objects](experiments/e01_material_unseen_objects)
2. **Robotic-arm images.** The arm project's own audit found that 55% of its validation photos are near-copies of
   training photos. Rebuild the split by capture session, then name the material from the laser spot on unseen
   objects. Label the new captures (Dr. Zhang asked for help with this).
3. **Labels from the data itself.** Self-supervised and contrastive pretraining on every unlabelled laser capture,
   tested on the labelled objects.
4. **Wave physics.** Build what is known about scattering, speckle and wavelength into the model.

## Status

- 2026-11-18: repository started; E01 planned (lab notebook, entry (a)).
- 2026-11-18: E01 done (entry (b)). On 31 objects the 3DAI laser features name the material of an unseen object
  only a little better than chance (balanced accuracy 0.25, chance 0.20), the same as a photo through AGAI's image
  encoder (0.23). A random split would have shown 0.47 and 0.78. Next: more objects per material, labels for the
  surface the beam hits, and the full laser frames. [results/e01/report.md](results/e01/report.md)
- 2026-11-18: E01 re-checked independently (entry (e)): no bugs; against shuffled labels the laser's score has
  P = 0.08, a hint at most. Literature and what it means for the next collection: [docs/RELATED_WORK.md](docs/RELATED_WORK.md).
- 2026-11-18: E02 done (entries (f), (g)): trained on the scanner's red-laser spots and tested on the robotic arm's
  photos, as Dr. Zhang suggested. At chance (balanced accuracy 0.27 against 0.25; plastic against metal 0.52 against
  0.50). The spot in an ordinary photo does not separate the materials even within one rig, and the two rigs record
  the same laser very differently. [results/e02/report.md](results/e02/report.md)
- 2026-11-20: Methods for speckle, scattering and wave computing, asked for by Dr. Zhang, with what is new against
  the literature and what to capture: [docs/WAVE_METHODS.md](docs/WAVE_METHODS.md). E03 tried them on the scanner's
  GitHub previews (entries (m), (n)): there the new features fail their own checks (the two red lasers disagree, and
  the spot is clipped in 227 of 229 frames), so the test needs raw, unclipped captures. The checks themselves work and
  become the gate for new data. [results/e03/report.md](results/e03/report.md)
