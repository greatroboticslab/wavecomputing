"""E04: an exact solution of the wave equation, to check the path picture the larger simulation rests on.

The larger simulation (transport_mc.py, synth.py) treats the field at the camera as a sum over transport paths, each
with phase k x (path length). That is an approximation of wave physics. This module makes no such approximation: it
solves the scalar wave equation for N point scatterers exactly (every order of multiple scattering, all interference),

    psi_j = psi_in(r_j) + sum_{l != j} t G(r_j - r_l) psi_l,      G(r) = exp(ikr) / (4 pi r),   t = 4 pi i / k,

(Foldy-Lax equations; t is the strongest lossless point scatterer, cross-section 4 pi / k^2, with no index shift), and
reads the scattered far field F(khat) = t sum_j psi_j exp(-i k khat.r_j).

Sample: a cylinder of scatterers, optical thickness 4, k x (mean free path) = 10, lit from the top by a Gaussian beam.
Two experiments, each compared with the path-sum prediction of transport_mc.slab_farfield for the same geometry:

  spectral   the same sample at 25 wavenumbers; correlation of the far-field speckle against the wavenumber step.
             Path picture: |sum_p w_p exp(i dk L_p)|^2, the squared Fourier transform of the path-length distribution.
  tilt       the beam tilted in small steps; correlation of the speckle read at the correspondingly shifted direction
             (the memory effect). Path picture: the squared Fourier transform of the distribution of the sideways
             distance between where a path enters and where it leaves.

    python -P fullwave.py --selftest
    python -P fullwave.py --run ../../data/e04/fullwave.npz --realizations 12
    python -P fullwave.py --analyze ../../data/e04/fullwave.npz --out ../../results/e04
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transport_mc import slab_farfield  # noqa: E402

LAMBDA = 1.0
K0 = 2 * np.pi / LAMBDA
KELL = 10.0                      # k x mean free path
ELL = KELL / K0
SIGMA = 4 * np.pi / K0 ** 2
DENSITY = 1.0 / (ELL * SIGMA)
R_CYL = 9.0
L_CYL = 4.0 * ELL
W0 = 3.5
D_MIN = 0.2
EPS = np.linspace(-0.03, 0.03, 25)          # relative wavenumber steps
DKAPPA = 0.02 * K0
M_TILT = np.arange(0, 9)
SIN2 = (np.sin(np.radians(20.0)) ** 2, np.sin(np.radians(50.0)) ** 2)
N_TH, N_AZ = 4, 8


def positions(n, radius, thick, dmin, rng):
    pts = np.empty((0, 3))
    while len(pts) < n:
        m = int((n - len(pts)) * 1.3) + 32
        r = radius * np.sqrt(rng.uniform(size=m))
        ph = rng.uniform(0, 2 * np.pi, m)
        cand = np.stack([r * np.cos(ph), r * np.sin(ph), rng.uniform(0, thick, m)], 1)
        allp = np.vstack([pts, cand])
        pairs = cKDTree(allp).query_pairs(dmin, output_type="ndarray")
        bad = np.unique(pairs.max(1)) if len(pairs) else np.array([], int)
        keep = np.ones(len(allp), bool)
        keep[bad] = False
        pts = allp[keep]
    return pts[:n]


def beam(pos, k, w0, q=0.0):
    """Gaussian beam with its waist at the origin, travelling along (q/k, 0, sqrt(1 - q^2/k^2))"""
    u = np.array([q / k, 0.0, np.sqrt(1 - (q / k) ** 2)])
    zeta = pos @ u
    perp2 = (pos ** 2).sum(1) - zeta ** 2
    qf = 1 + 1j * zeta / (k * w0 * w0 / 2)
    return np.exp(1j * k * zeta - perp2 / (w0 * w0 * qf)) / qf


def solve(r, k, rhs):
    """exciting fields at the scatterers; r = pairwise distances with ones on the diagonal"""
    n = r.shape[0]
    t = 4j * np.pi / k
    A = np.exp(1j * k * r)
    A /= r
    A *= -t / (4 * np.pi)
    A[np.diag_indices(n)] = 1.0
    return np.linalg.solve(A, rhs), t


def farfield(pos, psi, t, kx, ky, kz):
    ph = np.exp(-1j * (np.outer(kx, pos[:, 0]) + np.outer(ky, pos[:, 1]) - np.outer(kz, pos[:, 2])))
    return t * (ph @ psi)


def direction_grid():
    m = int(np.ceil(np.sqrt(SIN2[1]) * K0 / DKAPPA)) + 1
    g = np.arange(-m, m + 1) * DKAPPA
    kx, ky = np.meshgrid(g, g)
    s2 = (kx ** 2 + ky ** 2) / K0 ** 2
    ok = (s2 >= SIN2[0]) & (s2 < SIN2[1])
    return kx[ok], ky[ok]


def selftest():
    """optical theorem for a random cluster under a plane wave: extinction = total scattering (energy conservation)"""
    rng = np.random.default_rng(0)
    pos = positions(400, 3.0, 3.0, D_MIN, rng)
    r = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    r[np.diag_indices(len(pos))] = 1.0
    psi, t = solve(r, K0, np.exp(1j * K0 * pos[:, 2]))
    ext = (1 / K0) * np.imag(np.sum(t * psi * np.exp(-1j * K0 * pos[:, 2])))
    n = 20000
    i = np.arange(n) + 0.5
    ct = 1 - 2 * i / n
    ph = np.pi * (1 + 5 ** 0.5) * i
    st = np.sqrt(1 - ct * ct)
    khat = np.stack([st * np.cos(ph), st * np.sin(ph), ct], 1)
    f = (t / (4 * np.pi)) * (np.exp(-1j * K0 * (khat @ pos.T)) @ psi)
    sca = 4 * np.pi * np.mean(np.abs(f) ** 2)
    out = {"extinction": float(ext), "scattering": float(sca), "ratio": float(sca / ext),
           "single_scatterer_sigma_x_N": float(SIGMA * len(pos))}
    print(json.dumps(out))
    return out


def run(out, n_real, seed0):
    n = int(round(DENSITY * np.pi * R_CYL ** 2 * L_CYL))
    kx0, ky0 = direction_grid()
    ks = K0 * (1 + EPS)
    I_spec = np.zeros((n_real, len(ks), len(kx0)), np.float32)
    I_tilt = np.zeros((n_real, len(M_TILT), len(kx0)), np.float32)
    print(f"N = {n} scatterers, density {DENSITY:.3f} per lambda^3, mean free path {ELL:.3f} lambda, "
          f"{len(kx0)} far-field directions", flush=True)
    t0 = time.time()
    for ir in range(n_real):
        rng = np.random.default_rng(seed0 + ir)
        pos = positions(n, R_CYL, L_CYL, D_MIN, rng)
        r = np.linalg.norm(pos[:, None] - pos[None], axis=2)
        r[np.diag_indices(n)] = 1.0
        for ik, k in enumerate(ks):
            if ik == len(ks) // 2:          # the centre wavenumber also serves the tilt experiment
                rhs = np.stack([beam(pos, k, W0, m * DKAPPA) for m in M_TILT], 1)
                psi, t = solve(r, k, rhs)
                for j, m in enumerate(M_TILT):
                    kx = kx0 + m * DKAPPA
                    kz = np.sqrt(k * k - kx ** 2 - ky0 ** 2)
                    I_tilt[ir, j] = np.abs(farfield(pos, psi[:, j], t, kx, ky0, kz)) ** 2
                psi0 = psi[:, 0]
            else:
                psi0, t = solve(r, k, beam(pos, k, W0))
            kx, ky = kx0 * k / K0, ky0 * k / K0
            kz = np.sqrt(k * k - kx ** 2 - ky ** 2)
            I_spec[ir, ik] = np.abs(farfield(pos, psi0, t, kx, ky, kz)) ** 2
        print(f"  realization {ir + 1}/{n_real} done, {time.time() - t0:.0f} s", flush=True)
        np.savez_compressed(out, I_spec=I_spec[:ir + 1], I_tilt=I_tilt[:ir + 1], kx0=kx0, ky0=ky0, eps=EPS,
                            m_tilt=M_TILT, n_scatterers=n)


def _normalize(I, sin2, edges):
    """divide by the mean over realizations and directions within bins of sin^2(theta)"""
    out = np.empty_like(I, dtype=float)
    ib = np.clip(np.searchsorted(edges, sin2) - 1, 0, len(edges) - 2)
    for b in range(len(edges) - 1):
        m = ib == b
        if m.any():
            out[..., m] = I[..., m] / I[..., m].mean()
    return out


def analyze(path, out, photons):
    z = np.load(path)
    I_spec, I_tilt, kx0, ky0 = z["I_spec"].astype(float), z["I_tilt"].astype(float), z["kx0"], z["ky0"]
    n_real = I_spec.shape[0]
    s2 = (kx0 ** 2 + ky0 ** 2) / K0 ** 2
    edges = np.linspace(SIN2[0], SIN2[1], 9)
    # --- spectral
    X = np.stack([_normalize(I_spec[:, a], s2, edges) for a in range(I_spec.shape[1])], 1) - 1.0   # (real, k, dir)
    nk = X.shape[1]
    cov = np.einsum("rad,rbd->ab", X, X) / (X.shape[0] * X.shape[2])
    var = np.diag(cov)
    corr = cov / np.sqrt(np.outer(var, var))
    spec = np.array([np.mean(np.diag(corr, d)) for d in range(nk)])
    spec_sd = np.array([np.std(np.diag(corr, d)) if nk - d > 1 else 0.0 for d in range(nk)])
    contrast = float(np.sqrt(var.mean()))
    # intensity distribution at the centre wavenumber
    xc = (X[:, nk // 2] + 1.0).ravel()
    hist, he = np.histogram(xc, bins=np.linspace(0, 6, 31), density=True)
    # --- tilt
    Y = []
    for j, m in enumerate(z["m_tilt"]):
        s2m = ((kx0 + m * DKAPPA) ** 2 + ky0 ** 2) / K0 ** 2
        Y.append(_normalize(I_tilt[:, j], s2m, np.linspace(s2m.min(), s2m.max() + 1e-9, 9)) - 1.0)
    Y = np.stack(Y, 1)
    tilt = np.array([np.mean(Y[:, 0] * Y[:, j]) / np.sqrt(np.mean(Y[:, 0] ** 2) * np.mean(Y[:, j] ** 2))
                     for j in range(Y.shape[1])])
    # --- path-sum prediction from transport Monte Carlo of the same sample
    dks = K0 * (EPS - EPS[0])
    qs = z["m_tilt"] * DKAPPA
    mc = slab_farfield(photons, ELL, R_CYL, L_CYL, W0, K0, np.linspace(SIN2[0], SIN2[1], N_TH + 1), N_AZ, dks, qs,
                       seed=11)
    A, B = mc["A"], mc["B"]
    spec_mc = np.mean(np.abs(A) ** 2 / np.abs(A[:, :1]) ** 2, 0)
    tilt_mc = np.mean(np.abs(B) ** 2 / np.abs(B[:, :1]) ** 2, 0)
    # grains: a rough count of independent speckle grains, for the sampling error of the correlations
    res = {"n_scatterers": int(z["n_scatterers"]), "realizations": int(n_real), "directions": int(len(kx0)),
           "k_ell": KELL, "optical_thickness": L_CYL / ELL, "contrast": contrast,
           "intensity_hist": {"edges": he.tolist(), "density": hist.tolist()},
           "spectral": {"dk_over_k": (EPS - EPS[0]).tolist(), "full_wave": spec.tolist(), "full_wave_sd_over_pairs": spec_sd.tolist(),
                        "path_sum": spec_mc.tolist(), "max_abs_diff": float(np.max(np.abs(spec - spec_mc)))},
           "tilt": {"q_over_k": (qs / K0).tolist(), "full_wave": tilt.tolist(), "path_sum": tilt_mc.tolist(),
                    "max_abs_diff": float(np.max(np.abs(tilt - tilt_mc)))},
           "transport_events_per_photon": mc["events_per_photon"]}
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "fullwave.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    print("contrast", round(contrast, 3))
    print("spectral  full wave:", np.round(spec[:12], 3))
    print("          path sum :", np.round(spec_mc[:12], 3), " max |diff|", round(res["spectral"]["max_abs_diff"], 3))
    print("tilt      full wave:", np.round(tilt, 3))
    print("          path sum :", np.round(tilt_mc, 3), " max |diff|", round(res["tilt"]["max_abs_diff"], 3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run")
    ap.add_argument("--realizations", type=int, default=12)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--analyze")
    ap.add_argument("--out")
    ap.add_argument("--photons", type=int, default=1_500_000)
    a = ap.parse_args()
    if a.selftest:
        selftest()
    if a.run:
        run(a.run, a.realizations, a.seed)
    if a.analyze:
        analyze(a.analyze, a.out, a.photons)
