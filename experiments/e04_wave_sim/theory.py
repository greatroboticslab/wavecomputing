"""E04: closed-form pieces of the math (lab notebook (r), formulas F1 to F4).

Diffusion approximation for a half-space lit by a pencil beam, with the extrapolated boundary. The one fact that
carries the wave information: the Fourier transform over path length of the diffusely reflected light is the ordinary
steady-state reflectance with the absorption mu_a replaced by the complex number mu_a - i n dk (F2), because a path of
length s contributes exp(-mu_a s) exp(i n dk s).
"""
import numpy as np

H1 = 2.9078                                   # Chandrasekhar's H-function at mu = 1, conservative isotropic scattering
EXACT_NORMAL_RADIANCE = H1 ** 2 / (8 * np.pi)  # per steradian per unit power, half-space, normal in, normal out
MILNE_RADIANCE_OVER_FLUX = np.sqrt(3) * H1 / (4 * np.pi)


def boundary_A(n):
    """internal-reflection parameter A = (1 + R_eff) / (1 - R_eff) from the Fresnel integrals (Haskell et al. 1994)"""
    if n <= 1.0:
        return 1.0
    th = np.linspace(0, np.pi / 2, 20001)
    ci = np.cos(th)
    sint2 = n * n * (1 - ci * ci)
    R = np.ones_like(th)
    ok = sint2 < 1
    ct = np.sqrt(1 - sint2[ok])
    rs = (n * ci[ok] - ct) / (n * ci[ok] + ct)
    rp = (ci[ok] - n * ct) / (ci[ok] + n * ct)
    R[ok] = 0.5 * (rs ** 2 + rp ** 2)
    r_phi = np.trapezoid(2 * np.sin(th) * ci * R, th)
    r_j = np.trapezoid(3 * np.sin(th) * ci ** 2 * R, th)
    r_eff = (r_phi + r_j) / (2 - r_phi + r_j)
    return (1 + r_eff) / (1 - r_eff)


def diffusion(rho, mua, musp, A=1.0, n_dk=0.0):
    """(normal-exit radiance per steradian, flux) per unit area at distance rho from a pencil beam of unit power;
    n_dk = n x (wavenumber step) turns both into the field correlation across that step (complex numbers)."""
    rho = np.asarray(rho, float)
    D = 1.0 / (3.0 * musp)
    mueff = np.sqrt((mua - 1j * n_dk) / D + 0j)
    z0, zb = 1.0 / musp, 2.0 * A * D
    r1 = np.sqrt(z0 ** 2 + rho ** 2)
    r2 = np.sqrt((z0 + 2 * zb) ** 2 + rho ** 2)
    e1, e2 = np.exp(-mueff * r1), np.exp(-mueff * r2)
    phi = (e1 / r1 - e2 / r2) / (4 * np.pi * D)
    flux = (z0 * (mueff + 1 / r1) * e1 / r1 ** 2 + (z0 + 2 * zb) * (mueff + 1 / r2) * e2 / r2 ** 2) / (4 * np.pi)
    rad = (phi + 3 * flux) / (4 * np.pi)
    if np.all(np.asarray(n_dk) == 0):
        return rad.real, flux.real
    return rad, flux


def half_width(q, c, level=0.5):
    """first q at which |c| falls to `level` (linear interpolation); nan if it never does"""
    a = np.abs(c)
    below = np.nonzero(a < level)[0]
    if not len(below) or below[0] == 0:
        return np.nan
    i = below[0]
    return float(q[i - 1] + (a[i - 1] - level) / (a[i - 1] - a[i]) * (q[i] - q[i - 1]))


def pupil_overlap(u):
    """normalized overlap of a circular pupil with itself shifted by 2u radii (the amplitude correlation of surface
    speckle when the illumination is tilted by u x 2 NA)"""
    u = np.clip(np.abs(u), 0, 1)
    return (2 / np.pi) * (np.arccos(u) - u * np.sqrt(1 - u * u))


def multimode_spectrum(lam_mm, fwhm_nm=1.0, spacing_nm=0.15, span=1.6):
    """wavenumber offsets (rad/mm) and weights of the longitudinal modes of a Fabry-Perot diode with a Gaussian
    envelope; lam in mm"""
    m = np.arange(-int(span * fwhm_nm / spacing_nm), int(span * fwhm_nm / spacing_nm) + 1)
    dl = m * spacing_nm * 1e-6                       # mm
    w = np.exp(-4 * np.log(2) * (m * spacing_nm / fwhm_nm) ** 2)
    return -2 * np.pi * dl / lam_mm ** 2, w / w.sum()


def coherence2(dk, wts, path_diff_optical):
    """|gamma|^2 of a source with lines at wavenumber offsets dk and weights wts, at an optical path difference"""
    return np.abs(np.exp(1j * np.multiply.outer(path_diff_optical, dk)) @ wts) ** 2


def diffusion_path_density(rho, s, mua, musp, A=1.0):
    """normal-exit radiance per unit path length s (time-resolved diffusion, extrapolated boundary), unnormalized"""
    D = 1.0 / (3.0 * musp)
    z0, zb = 1.0 / musp, 2.0 * A * D
    r1sq, r2sq = z0 ** 2 + rho ** 2, (z0 + 2 * zb) ** 2 + rho ** 2
    e1, e2 = np.exp(-r1sq / (4 * D * s)), np.exp(-r2sq / (4 * D * s))
    phi = (4 * np.pi * D * s) ** -1.5 * (e1 - e2)
    flux = 0.5 * (4 * np.pi * D) ** -1.5 * s ** -2.5 * (z0 * e1 + (z0 + 2 * zb) * e2)
    return (phi + 3 * flux) * np.exp(-mua * s)
