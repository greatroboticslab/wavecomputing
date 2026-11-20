# Wave methods: what the laser reflection says about the material

Dr. Zhang's aim (Discord, 20 November 2026) is a method that generalizes regardless of how the data look, by looking
inside the laser reflection and extracting the nature of the wave information. He asked for very innovative methods
for **1** (speckle), **2** (a scattering model fit) and **4** (wave computing); **3** (polarization) needs a device,
which his team will build. This page proposes them, says which parts are new against the literature (search of 20
November 2026), and what the files we have already allow ([E03](../experiments/e03_wave_methods), lab notebook
entries (m) and (n)).

## What a photo of a laser spot mixes

1. **The material's physics**, which is what we want: how deep and how far light travels inside before it comes back,
   whether it comes back with its polarization and coherence, and how both change with wavelength.
2. **How the object looks:** colour, print, brightness, shape and pose.
3. **The rig:** the laser beam, the lens, exposure, clipping, JPEG.

Each method below measures 1 in a way meant to cancel 2 and 3, and each is checked the same way:

- the scanner's two red lasers (same wavelength, different beams, different points) must give the same value: the rig
  cancels;
- an object's two poses must give the same value: the shape cancels;
- within a material, the value must not follow the object's visible brightness and colour: the looks cancel;
- and it must name the material of objects never seen in training.

## 1. Speckle

### 1A. Speckle degrees of freedom, ring by ring (new as far as we found)

Laser light coming back from a rough surface is speckle. If the light keeps one polarization and its path lengths
differ by less than the laser's coherence length, the speckle intensity follows an exponential law, with contrast
K = 1. Light that wanders inside a material loses both. Its two polarizations make two independent speckle patterns,
and paths longer than the coherence length add more. The sum of M independent patterns has K² = 1/M, so M works as a
polarization and path-length meter that needs no polarizer. M is about 1 for light returned at the surface (bare
metal) and 2 or more for light returned from inside (plastic, paper, foam).

What is new is how M is read:

- **As a profile M(r)** outward from where the beam enters. Near the entry point the light is mostly scattered once;
  farther out it has travelled longer, so M rises. How fast it rises says how far light goes before it loses its
  polarization and coherence.
- **Camera noise removed** by multiplying two frames of the same laser: their noise is independent, their speckle is the
  same.
- **Surface texture removed** with the lasers-off frame. Print and grain are in both frames, speckle only in the laser
  frame: K_s² = (1 + K_laser²) / (1 + K_texture²) − 1.
- **The camera cancels** in the ratio M(r)/M(r₀), or against a bare-metal reference, because blurring of speckle by the
  camera multiplies every M by the same factor.

Speckle contrast as one number per sample has been used to recognise materials [1; patents 2, 3], and the
depolarization of speckle in turbid media is known physics [4]. We found no use of the radial profile.

M gives part of what the polarizer of method 3 would give, without the device. Once the polarizer exists, M measured
co-polarized and cross-polarized separates depolarization from path-length spread.

### 1B. Speckle size in a defocused frame, a ruler for the glow under the surface (known physics)

Defocus the camera, or record with a bare sensor. The speckle grain size is then about λz/D, where D is the diameter
of the patch the light leaves from: the beam alone on metal, the beam plus the glow under the surface on foam or
plastic. By the van Cittert-Zernike theorem the speckle's autocorrelation is the squared Fourier transform of the
glowing patch. So the glow is measured from the grain size, below the camera's resolution and without regard to
colour. The physics is known [5]. A patent uses grain-size criteria to tell skin from latex and silicone [6], and a 2025
study found that the simple exit-spot model underpredicts the change [7], so this needs calibration on known samples.

### 1C. Wavelength steps: the speckle computes a Fourier transform of path length (known physics)

Tune a diode laser's wavelength in small steps (a DFB laser moves about 0.1 nm per kelvin) and correlate the speckle
frames. The correlation against the wavelength step is the Fourier transform of the distribution of path lengths
inside the material. For path spreads of 1 µm, 1 mm and 1 cm (metal, plastic, white foam) the speckle decorrelates
after steps of about 60 nm, 0.06 nm and 0.006 nm. A small patch of pixels gives the transport length in absolute units,
with no imaging of the halo. This has been shown as a way to image the transport mean free path [8] and is listed in a
patent as a material feature [9]. It needs one temperature-controlled laser.

## 2. Scattering model fit

### 2A. Surface light and subsurface light, measured in beam widths (the calibration and censoring are new as far as we found)

Each spot is fitted as light returned at the surface plus light that travelled under it:

    D(r) = A · beam(r) + B · (beam ∗ R_d)(r) + c

R_d is the normalized diffusion profile of Christensen and Burley [10], a one-parameter fit to Monte Carlo light
transport. A is the surface light, B the subsurface light, d how far it travels; c takes up a flat offset. Three
choices make the fit independent of the rig and of the looks:

- **Label-free beam calibration.** Each laser's beam is measured from the rig's own data, as its narrowest spots (no
  spot can be narrower than the beam). No labels and no reference target are needed.
- **Lengths in beam widths.** The robotic arm uses the same laser board as the scanner, so a length in beam widths
  means the same thing on both rigs; a length in pixels does not.
- **Clipped pixels censored.** A clipped pixel only says "at least this bright", and the fit treats it that way instead
  of reading the clip level as a value.

The spot's brightness and colour are never features. Given enough resolution near the beam, the fit also separates
absorption (colour, i.e. the looks) from reduced scattering (the material's microstructure), and a classifier on the
scattering part alone would be colour-blind by construction.

Fitting the halo around a spot to name materials is the closest published work: 58.8% on 25 classes, tested on objects
scanned on another day [11], and laser dots from a depth camera [12]. The label-free calibration, the censoring and the
beam-width units are not in them.

### 2B. Halo elongation beyond the beam's footprint (known for wood only)

Fibres carry light along themselves. A laser spot on wood stretches along the grain (the tracheid effect, used in
sawmills to find knots and fibre direction [13]), and paper does the same along its fibres [14]. Tilt stretches a spot
too, but it stretches the beam's footprint and the halo alike, while fibres stretch only the halo. So the halo's
elongation in excess of the core's is a fibre detector for wood, paper, cardboard, fabric and fibre composites that does
not depend on the pose. We found no use of it to recognise materials.

### 2C. The scanner's projector as a scattering meter (known physics)

The 3DAI scanner already projects fringes for its height map. Light that enters at a bright fringe and leaves in a
dark one washes the fringes out, and finer fringes wash out more. Fringe contrast against fringe frequency is the
material's spatial frequency response (spatial frequency domain imaging), and two frequencies separate absorption
from scattering at every pixel of the object: thousands of measurements per object instead of one spot. In 3D scanning
this washout is a known nuisance [15, 16]; a 2026 preprint learns subsurface scattering from a 3D scanner's phase-shift
images and tests on five objects [17]. We found no use of it to recognise materials. It needs the fringe stacks the
scanner already saves, at two or more fringe frequencies.

## 4. Wave computing

### 4A. The object is the reservoir (nothing close found)

In optical reservoir computing a fixed scattering medium transforms the input light and only a linear readout is
trained [18, 19]. Here the medium is the object being recognised. The inputs are ours to control: the four lasers,
small steps of the beam, small steps of wavelength. The outputs are camera frames. What matters is what the readout is
trained on. The object's transmission matrix is different at every point of every object, but its statistics are set
by the material (random matrix theory). The correlation between the outputs for two inputs (two beam positions, two
wavelengths, two lasers) depends on those statistics and not on the particular point. So the readout is trained on
input-to-input correlations, the medium's kernel, which the optics computes. A digital readout on raw speckle exists
[1], and so does sending light from a sample into a separate photonic reservoir (patent [20]); we found none where the
sample itself is the reservoir.

### 4B. Speckle that stays, speckle that moves (new as far as we found)

Move the beam a fraction of a millimetre with the arm and compare speckle frames. Light reflected at the surface makes
speckle that belongs to the surface, so its grains stay where they are as the beam moves. Light that comes out from
inside travelled paths that start where the beam enters; when the beam moves, the paths change and the grains change.
The speckle correlation against beam step therefore separates surface from volume scattering and measures how far light
travels inside, with the arm we have. Speckle correlations under small tilts have been used to measure the scattering
phase function [21], and a patent lists correlations across illumination angles as a material feature [9]. We found no
use of the stay-or-move test.

### 4C. A diffuser in front of the camera (a cheap physical reservoir)

A piece of tape or ground glass in front of the lens turns the camera into an optical random projector of the light
coming back from the spot [18]. A plain camera records intensity only; the diffuser mixes the light field before the
camera squares it, so its frames carry coherence and phase information a plain photo loses. The readout stays linear.

## What the files we have allow (E03)

The 3DAI files on GitHub are 800 px JPEG previews of one frame per laser, and the spot core is clipped in nearly all of
them. E03 tried what they allow: 2A and 2B in full, 1A with one frame (exploratory), and 4A as a digital stand-in (a
random projection of the spot, read out linearly, which computes nothing the camera image does not already hold). The
plan is entry (m) of the lab notebook, the results entry (n) and [results/e03/report.md](../results/e03/report.md).

## What to capture next

- **No clipped spot.** Bracket three exposures per laser (for example 1, 1/8 and 1/64) and merge them. In the GitHub
  previews the beam and the light just under the surface are hidden by clipping.
- **Raw, full-resolution frames** (PNG or raw), not previews. For the existing scans, the scanner PC can export them
  (`python -m scanner_system.export_bundle out.zip --include-arrays`, see [DATA.md](DATA.md)).
- **Two frames per laser** and the lasers-off frame (noise and texture removal, 1A).
- **Aperture for speckle:** close the lens until a speckle grain covers 2 or more pixels. On the sensor a grain is about
  2.4 × wavelength × f-number, so at 635 nm f/4 gives about 6 µm.
- **A defocused frame per laser** (1B).
- **Beam steps on the arm:** five steps of 0.1 to 0.5 mm at each spot (4B).
- **The fringe stacks** at two or more fringe frequencies (2C).
- **Optional:** one temperature-controlled DFB laser (1C), a tape diffuser (4C), and the polarizers of method 3.
- **Objects:** at least 10 per material, mixed colours and finishes, including painted, taped and anodized metal; test
  on objects and capture days held out of training.

## Sources

Search of 20 November 2026. Marked [abstract] where only the abstract or metadata was checked.

1. Valent and Silberberg, *Scatterer recognition via analysis of speckle patterns*, Optica 2018,
   [doi:10.1364/OPTICA.5.000204](https://doi.org/10.1364/OPTICA.5.000204). (Which speckle statistics they used was not
   checked.)
2. US patent 8913251 (Canon): speckle contrast at several angles for roughness and a subsurface term,
   [link](https://patents.google.com/patent/US8913251).
3. US patent 11947013 (trinamiX): low speckle contrast from volume scattering as a material feature,
   [link](https://patents.google.com/patent/US11947013).
4. Li, Yao and Wang, *Degree of polarization in laser speckles from turbid media*, J. Biomed. Opt. 2002,
   [doi:10.1117/1.1483313](https://doi.org/10.1117/1.1483313). Also Thompson, Webb and Weiner, *Diffusive media
   characterization with laser speckle*, Appl. Opt. 1997, [doi:10.1364/AO.36.003726](https://doi.org/10.1364/AO.36.003726)
   [abstract].
5. Piederrière et al., *Scattering through fluids: speckle size measurement and Monte Carlo simulations*, Opt. Express
   2004, [doi:10.1364/OPEX.12.000176](https://doi.org/10.1364/OPEX.12.000176) [abstract].
6. US patent 9830520 (Safran): speckle autocorrelation and grain size to tell skin from latex, silicone and gelatin,
   [link](https://patents.google.com/patent/US9830520).
7. Kulcsar, Louie and Vitkin, *Laser speckle size and contrast investigation of volumetric scattering*, iScience 2025,
   [doi:10.1016/j.isci.2025.112433](https://doi.org/10.1016/j.isci.2025.112433).
8. Mikhailovskaya, Fade and Crassous, *Speckle decorrelation with wavelength shift as a simple way to image transport
   mean free path*, Eur. Phys. J. Appl. Phys. 2019, [doi:10.1051/epjap/2019180297](https://doi.org/10.1051/epjap/2019180297).
9. US patent 8760638 (Canon): two-wavelength speckle correlation with spectral reflectance, matched to a material
   database, [link](https://patents.google.com/patent/US8760638).
10. Christensen and Burley, *Approximate reflectance profiles for efficient subsurface scattering*, Pixar technical
    memo, 2015.
11. Lee and Sankaranarayanan, *Spectral subsurface scattering for material classification*, ECCV 2024,
    [paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00725.pdf).
12. Mao et al., *Surface material perception through multimodal learning*, IEEE JSTSP 2022,
    [doi:10.1109/JSTSP.2022.3171682](https://doi.org/10.1109/JSTSP.2022.3171682) [abstract].
13. Nyström, *Automatic measurement of fiber orientation in softwoods by using the tracheid effect*, Comput. Electron.
    Agric. 2003, [doi:10.1016/S0168-1699(03)00045-0](https://doi.org/10.1016/S0168-1699(03)00045-0) [abstract].
14. Linder et al., *Light scattering in fibrous media with different degrees of in-plane fiber alignment*, Opt. Express
    2014, [doi:10.1364/OE.22.016829](https://doi.org/10.1364/OE.22.016829) [abstract].
15. Holroyd and Lawrence, *An analysis of using high-frequency sinusoidal illumination to measure the 3D shape of
    translucent objects*, CVPR 2011, [doi:10.1109/CVPR.2011.5995536](https://doi.org/10.1109/CVPR.2011.5995536) [abstract].
16. Tariq et al., *Efficient estimation of spatially varying subsurface scattering parameters*, VMV 2006,
    [pdf](https://vgl.ict.usc.edu/Research/SVSSS/SVSSS_VMV06_bw.pdf).
17. Majumdar et al., *From phase to phenomenon: self-supervised learning of subsurface scattering with minimal
    phase-shift inputs*, arXiv 2026, [arXiv:2606.29461](https://arxiv.org/abs/2606.29461).
18. Saade et al., *Random projections through multiple optical scattering*, ICASSP 2016,
    [arXiv:1510.06664](https://arxiv.org/abs/1510.06664).
19. Rafayelyan et al., *Large-scale optical reservoir computing for spatiotemporal chaotic systems prediction*, Phys.
    Rev. X 2020.
20. US patent 10228280 (IBM): light from a sample sent into a photonic reservoir,
    [link](https://patents.google.com/patent/US10228280).
21. Alterman, Saiko and Levin, *Direct acquisition of volumetric scattering phase function using speckle correlations*,
    SIGGRAPH Asia 2022, [doi:10.1145/3550469.3555379](https://doi.org/10.1145/3550469.3555379).
