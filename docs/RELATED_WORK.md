# Related work: recognising materials with lasers

A short literature scan (2026-11-18) to plan the next data collection and experiments. Items marked [unverified] were
not fully checked against the paper. Titles and key numbers of [1], [2] and [3] were checked against the papers.

## What others found

- On objects held out of training, published accuracies run from about 59% to 91%. They come from near-infrared
  spectra, scattering measured at several wavelengths, and several views of each object. These studies used 10 to 18
  objects per material.
- The highest numbers (98% from raw laser speckle, about 90% from time-of-flight depth errors) come from splits by image
  or by orientation of the same samples, so they do not show that a method works on new objects.
- None of the optical surface methods listed here sees through paint or tape: they read the outer layer, as E01 did. A
  method that heats the surface with a laser and watches it with a thermal camera did tell metals apart under tape.

## Sources

**Material recognition with active light**

1. Erickson et al., *Classification of Household Materials via Spectroscopy*, IEEE RA-L 2019,
   [arXiv:1805.04051](https://arxiv.org/abs/1805.04051). A near-infrared spectrometer (740-1070 nm) reached 79.1% with
   leave-one-object-out on metal, plastic, wood, paper and fabric, 10 objects each; a visible-range spectrometer 68.7%.
   Accuracy rose with more objects per class. Dataset SMM50.
2. Erickson et al., *Multimodal Material Classification for Robots using Spectroscopy and High Resolution Texture
   Imaging*, IROS 2020, [arXiv:2004.01160](https://arxiv.org/abs/2004.01160). 144 household objects, 8 classes; near
   infrared plus a lit close-up of the texture: 74.2% leave-one-object-out. Dataset SpectroVision.
3. Lee and Sankaranarayanan, *Spectral Subsurface Scattering for Material Classification*, ECCV 2024,
   [paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/725_ECCV_2024_paper.php). The closest to the 3DAI
   scanner: it measures the scattering halo around a spot at several wavelengths. 25 classes, each test fold a different
   set of objects scanned on a different day: spectral scattering 58.8%, broadband scattering 54.1%, spectral
   reflectance 41.3%. Painted objects were confused; it needs smooth surfaces. Data public.
4. Tanaka et al., *Material Classification Using Frequency- and Depth-Dependent Time-of-Flight Distortion*, CVPR 2017.
   How a Kinect v2's depth error changes with modulation frequency encodes subsurface scattering; 90.5% on 26
   materials, scored across orientations of the same samples [unverified].
5. Dogan et al., *SensiCut*, UIST 2021. Raw speckle from a 515 nm laser into ResNet-50: 98% on 30 sheet materials,
   split by image.
6. Xue et al., *Differential Angular Imaging for Material Recognition*, CVPR 2017,
   [arXiv:1612.02372](https://arxiv.org/abs/1612.02372). Samples physically held out: one view 74.3%, several views
   81.2%.
7. Gu and Liu, *Discriminative Illumination*, CVPR 2012. Learned multispectral lighting, so the reflection itself does
   part of the classifier's work; raw, unpainted materials only [accuracy unverified].

**Coatings and shiny metal**

8. Wolff, *Polarization-Based Material Classification from Specular Reflection*, IEEE TPAMI 1990. A polarisation ratio
   of the specular reflection separates metal from non-metal regardless of colour [unverified].
9. Liang et al., *Multimodal Material Segmentation*, CVPR 2022 (MCubeS dataset: RGB, polarisation and near infrared, 20
   materials). Polarisation helps most for metal against non-metal.
10. Dashpute et al., *Thermal Spread Functions*, CVPR 2023, [arXiv:2304.00696](https://arxiv.org/abs/2304.00696). A
    633 nm laser heats the surface and a thermal camera records how the heat spreads; 86% on 16 classes [split
    unverified]. Bare metal was hard, but taped metals were classified by how they conduct heat.

**Learning without labels**

11. Vecchio et al., *Φeat*, [arXiv:2511.11270](https://arxiv.org/abs/2511.11270), 2025. Contrastive pairs are the same
    material under controlled changes of lighting and geometry, rather than generic image augmentations.
12. Xu et al., *CREPE*, [arXiv:2204.01733](https://arxiv.org/abs/2204.01733), 2022. A contrastive embedding of raw
    speckle, learned without labels.
13. Eranti et al., *Raman Spectroscopy Pre-trained Encoder (RSPTE)*, IEEE Access 2026. Barlow Twins pretraining on
    spectra lifts accuracy with 20% of the labels from 9-57% to 50-86%.

**Computing with waves (the long-term direction)**

14. Saade et al., *Random Projections through Multiple Optical Scattering*, ICASSP 2016,
    [arXiv:1510.06664](https://arxiv.org/abs/1510.06664). Light through a scattering medium, recorded by a camera,
    computes random features that match a known kernel.
15. Rafayelyan et al., *Large-Scale Optical Reservoir Computing for Spatiotemporal Chaotic Systems Prediction*, Physical
    Review X 2020. A reservoir computer built on a scattering medium.
16. Pierangeli et al., *Photonic Extreme Learning Machine by Free-Space Optical Propagation*, Photonics Research 2021,
    [arXiv:2105.12123](https://arxiv.org/abs/2105.12123). Free-space propagation, camera nonlinearity and a linear
    readout match digital extreme learning machines.
17. Lin et al., *All-Optical Machine Learning Using Diffractive Deep Neural Networks*, Science 2018,
    [arXiv:1804.08711](https://arxiv.org/abs/1804.08711).
18. Wright et al., *Deep Physical Neural Networks Trained with Backpropagation*, Nature 2022,
    [arXiv:2104.13386](https://arxiv.org/abs/2104.13386).

## What this means for us

1. **More objects.** At least 10 per material, balanced, with different shapes within a material and similar shapes
   across materials; test on held-out objects, with folds by scan day [1, 3]. With two or three objects in some
   materials, E01's 0.25 cannot be told apart from chance.
2. **Take geometry out with the height map.** For each spot, record distance and incidence angle, halo radii in
   millimetres and ratios between wavelengths; keep the full radial profile and fit a scattering model per wavelength
   [3]; combine several views of an object [6].
3. **Polarisation.** Co- and cross-polarised analysers on each laser, and a camera at the mirror angle for polished
   metal [8, 9].
4. **Coatings.** Label the surface layer and the bulk material separately. A thermal channel could reach the bulk [10]
   (it needs a laser-safety review). Keep a test set of painted, taped, polished and anodised metals.
5. **More signals.** Near-infrared diodes around 780, 850 and 980 nm [1, 3]; raw speckle crops instead of a single
   contrast value [5]; if the Kinect is a v2, its depth at each modulation frequency [4].
6. **Next experiment without labels.** Pretrain a contrastive or Barlow Twins encoder on the unlabelled scans, with the
   same object across pose, distance and day as positive pairs [11-13]; train a linear classifier on top; test on
   held-out objects against E01.
7. **Physics track.** Treat the object's speckle as an optical random projection with a trained linear readout [14,
   16] and compare it with digital random features; later a diffuser reservoir [15] and trainable lighting or optics
   [7, 17, 18].
