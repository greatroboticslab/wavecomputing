"""E04: photon-path Monte Carlo that keeps what a wave simulation needs from every path.

A camera records intensity, but laser light is a wave: what reaches a pixel is the sum of the fields of all the paths
that end there, each with phase k x (path length) and its own polarization. This module follows many paths through a
scattering material and stores, for the light that leaves towards the camera,

    where it leaves (rho, phi from the entry point), how far it travelled inside (s), and its polarization
    (coherency components xx, yy, xy for light that entered polarized along x).

From these three things every second-order property of the speckle follows (see theory.py and the weekly note).

Model. Half-space z >= 0 of point scatterers, lengths in units of the scattering mean free path (so one run serves
every scattering strength), no absorption (a weight exp(-mu_a s) is applied afterwards, so one run serves every
absorption). Two scatterer models:

  rayleigh   electric-dipole scatterers. A path carries a real unit polarization vector e; at each scattering the new
             direction d' is drawn from the dipole pattern 1 - (e.d')^2 and the new polarization is the part of e
             perpendicular to d'. Exact for small particles in the far field; anisotropy g = 0.
  isotropic  scalar waves, isotropic scattering (used for checks against exact results and for fullwave.py).

Detection uses the local estimator: at every scattering event, the probability per steradian of being scattered
straight back along the surface normal and leaving without another collision is scored. That gives the radiance a
distant camera on the normal collects, with no photon wasted.

Boundary. n_rel = 1 is exact for the model. n_rel > 1 reflects paths at the surface with the unpolarized Fresnel
probability and mirrors the polarization vector; the phase difference between s and p in total internal reflection is
ignored (approximation, used only as a sensitivity run).

    python -P transport_mc.py --check          # benchmark against Chandrasekhar's exact half-space results
"""
import argparse
import json
import time

import numpy as np

C_RAYLEIGH = 3.0 / (8.0 * np.pi)
C_ISO = 1.0 / (4.0 * np.pi)


def iso_dirs(n, rng):
    mu = rng.uniform(-1.0, 1.0, n)
    ph = rng.uniform(0.0, 2.0 * np.pi, n)
    st = np.sqrt(1.0 - mu * mu)
    return np.stack([st * np.cos(ph), st * np.sin(ph), mu], 1)


def dipole_scatter(e, rng):
    """direction drawn from the dipole pattern of polarization e, and the polarization of the scattered wave"""
    n = len(e)
    d = np.empty_like(e)
    todo = np.arange(n)
    while len(todo):
        cand = iso_dirs(len(todo), rng)
        c = np.einsum("ij,ij->i", e[todo], cand)
        ok = rng.uniform(size=len(todo)) < 1.0 - c * c
        d[todo[ok]] = cand[ok]
        todo = todo[~ok]
    c = np.einsum("ij,ij->i", e, d)
    en = e - c[:, None] * d
    en /= np.maximum(np.linalg.norm(en, axis=1), 1e-12)[:, None]
    return d, en


def fresnel_unpolarized(cosi, n):
    """reflectance for light inside a medium of index n hitting the boundary to air at cos(angle) = cosi"""
    sint2 = n * n * (1.0 - cosi * cosi)
    R = np.ones_like(cosi)
    ok = sint2 < 1.0
    cost = np.sqrt(1.0 - sint2[ok])
    ci = cosi[ok]
    rs = (n * ci - cost) / (n * ci + cost)
    rp = (ci - n * cost) / (ci + n * cost)
    R[ok] = 0.5 * (rs * rs + rp * rp)
    return R


def half_space_kernel(n_photons, model="rayleigh", n_rel=1.0, s_max=4000.0, rho_min=0.01, rho_max=60.0, n_rho=96,
                      n_phi=6, s_min=0.02, n_s=244, batch=250_000, seed=0, verbose=True):
    """pencil beam at the origin along +z, polarized along x. Returns the histogram H[c, rho, phi, s] of local-estimator
    weight (per photon, per steradian towards the normal), c = xx, yy, xy (xy folded into one quadrant with its sign),
    and the escaping flux by exit radius."""
    rng = np.random.default_rng(seed)
    lr0, dlr = np.log(rho_min), (np.log(rho_max) - np.log(rho_min)) / n_rho
    ls0, dls = np.log(s_min), (np.log(s_max * 1.01) - np.log(s_min)) / n_s
    H = np.zeros((3, n_rho * n_phi * n_s))
    flux = np.zeros(n_rho)
    rayleigh = model == "rayleigh"
    c0 = C_RAYLEIGH if rayleigh else C_ISO
    if n_rel > 1.0:
        c0 *= (1.0 - ((n_rel - 1.0) / (n_rel + 1.0)) ** 2) / n_rel ** 2  # normal-incidence transmission, n^2 law
    events = 0
    escaped = 0
    t0 = time.time()
    done = 0
    while done < n_photons:
        n = min(batch, n_photons - done)
        done += n
        x = np.zeros(n); y = np.zeros(n); z = np.zeros(n); s = np.zeros(n)
        d = np.zeros((n, 3)); d[:, 2] = 1.0
        e = np.zeros((n, 3)); e[:, 0] = 1.0
        while x.size:
            t = rng.exponential(size=x.size)
            znew = z + d[:, 2] * t
            alive = np.ones(x.size, bool)
            cross = znew < 0.0
            if cross.any():
                ci = np.nonzero(cross)[0]
                tb = -z[ci] / d[ci, 2]
                xb = x[ci] + d[ci, 0] * tb
                yb = y[ci] + d[ci, 1] * tb
                if n_rel > 1.0:
                    refl = rng.uniform(size=ci.size) < fresnel_unpolarized(-d[ci, 2], n_rel)
                else:
                    refl = np.zeros(ci.size, bool)
                out = ci[~refl]
                alive[out] = False
                escaped += out.size
                rb = np.hypot(xb[~refl], yb[~refl])
                ib = np.floor((np.log(np.maximum(rb, rho_min)) - lr0) / dlr).astype(int)
                flux += np.bincount(ib[ib < n_rho], minlength=n_rho)[:n_rho]
                if refl.any():
                    ri = ci[refl]
                    d[ri, 2] *= -1.0
                    e[ri, 2] *= -1.0
                    rem = t[ri] - tb[refl]
                    x[ri] = xb[refl] + d[ri, 0] * rem
                    y[ri] = yb[refl] + d[ri, 1] * rem
                    z[ri] = d[ri, 2] * rem
                    cross[ri] = False
            go = ~cross & alive
            x[go] += d[go, 0] * t[go]
            y[go] += d[go, 1] * t[go]
            z[go] = znew[go]
            s += t
            rho = np.hypot(x, y)
            alive &= (s <= s_max) & (rho <= 1.5 * rho_max)
            if not alive.all():
                x, y, z, s, rho, d, e = x[alive], y[alive], z[alive], s[alive], rho[alive], d[alive], e[alive]
            if not x.size:
                break
            events += x.size
            # local estimator: scattered straight up and out without another collision
            wb = c0 * np.exp(-z)
            ir = np.floor((np.log(np.maximum(rho, rho_min)) - lr0) / dlr).astype(int)
            ip = np.minimum((np.arctan2(np.abs(y), np.abs(x)) * (2.0 / np.pi) * n_phi).astype(int), n_phi - 1)
            isb = np.clip(np.floor((np.log(np.maximum(s + z, s_min)) - ls0) / dls).astype(int), 0, n_s - 1)
            ok = ir < n_rho
            flat = ((ir * n_phi + ip) * n_s + isb)[ok]
            if rayleigh:
                ex, ey = e[ok, 0], e[ok, 1]
                w = wb[ok]
                H[0] += np.bincount(flat, w * ex * ex, minlength=H.shape[1])
                H[1] += np.bincount(flat, w * ey * ey, minlength=H.shape[1])
                H[2] += np.bincount(flat, w * ex * ey * np.sign(x[ok] * y[ok]), minlength=H.shape[1])
                d, e = dipole_scatter(e, rng)
            else:
                H[0] += np.bincount(flat, wb[ok], minlength=H.shape[1])
                d = iso_dirs(x.size, rng)
        if verbose:
            print(f"  {done:,} photons, {events / done:.1f} events each, {time.time() - t0:.0f} s", flush=True)
    rho_edges = np.exp(lr0 + dlr * np.arange(n_rho + 1))
    s_edges = np.exp(ls0 + dls * np.arange(n_s + 1))
    return {"model": model, "n_rel": n_rel, "n_photons": n_photons, "s_max": s_max, "rho_edges": rho_edges,
            "phi_edges": np.linspace(0, np.pi / 2, n_phi + 1), "s_edges": s_edges,
            "H": (H / n_photons).reshape(3, n_rho, n_phi, n_s), "flux": flux / n_photons,
            "events_per_photon": events / n_photons, "escaped": escaped / n_photons}


def save_kernel(k, path):
    np.savez_compressed(path, **{a: (np.asarray(b) if not isinstance(b, str) else np.array(b)) for a, b in k.items()})


def load_kernel(path):
    z = np.load(path)
    k = {a: z[a] for a in z.files}
    for a in ("model",):
        k[a] = str(k[a])
    for a in ("n_rel", "n_photons", "s_max", "events_per_photon", "escaped"):
        k[a] = float(k[a])
    return k


def radial_density(k, comp):
    """azimuthal mean of a component as radiance per unit area: array (rho, s-bin mass)"""
    re = k["rho_edges"]
    area = np.pi * (re[1:] ** 2 - re[:-1] ** 2)
    return k["H"][comp].sum(1) / area[:, None]


# ---------------------------------------------------------------------------------------------------------------------
# finite slab (cylinder), scalar isotropic scattering, far-field detection: the transport prediction for fullwave.py


def slab_farfield(n_photons, ell, R, L, w0, k0, sin2_edges, n_az, dks, qs, n_dir=2, batch=200_000, seed=0,
                  verbose=True):
    """Gaussian beam (1/e^2 intensity radius w0) entering the top face z = 0 of a cylinder (radius R, thickness L) of
    isotropic scatterers with mean free path ell. For far-field directions binned by sin^2(theta) and azimuth, sums
    the path phasors
        A[dk] = sum w exp(i dk L_p)        L_p = whole geometric path, entry plane to far field
        B[q]  = sum w exp(i dphi(q))       dphi = phase change when the beam is tilted by transverse wavenumber q
                                           along x and the far field is read at the direction shifted by the same q
    whose squared moduli are the speckle correlations a wave solution must show if the field is the sum over
    transport paths."""
    rng = np.random.default_rng(seed)
    dks = np.asarray(dks, float); qs = np.asarray(qs, float)
    n_th = len(sin2_edges) - 1
    A = np.zeros((n_th * n_az, len(dks)), complex)
    B = np.zeros((n_th * n_az, len(qs)), complex)
    Wsum = np.zeros(n_th * n_az)
    events = 0
    done = 0
    t0 = time.time()
    while done < n_photons:
        n = min(batch, n_photons - done)
        done += n
        x = rng.normal(0, w0 / 2, n); y = rng.normal(0, w0 / 2, n)
        ok = np.hypot(x, y) < R
        x, y = x[ok], y[ok]
        z = np.zeros(x.size); s = np.zeros(x.size)
        d = np.zeros((x.size, 3)); d[:, 2] = 1.0
        x1 = np.zeros(x.size); z1 = np.zeros(x.size)
        first = np.ones(x.size, bool)
        while x.size:
            t = rng.exponential(ell, x.size)
            x = x + d[:, 0] * t; y = y + d[:, 1] * t; z = z + d[:, 2] * t; s = s + t
            alive = (z > 0) & (z < L) & (np.hypot(x, y) < R)
            x, y, z, s, x1, z1, first = x[alive], y[alive], z[alive], s[alive], x1[alive], z1[alive], first[alive]
            if not x.size:
                break
            x1 = np.where(first, x, x1); z1 = np.where(first, z, z1)
            first = np.zeros(x.size, bool)
            events += x.size
            for _ in range(n_dir):
                u = rng.uniform(sin2_edges[0], sin2_edges[-1], x.size)
                st, ct = np.sqrt(u), np.sqrt(1 - u)
                ph = rng.uniform(0, 2 * np.pi, x.size)
                ux, uy = st * np.cos(ph), st * np.sin(ph)
                t_top = z / ct
                px, py = x + ux * t_top, y + uy * t_top
                side = np.hypot(px, py) > R
                tau = t_top.copy()
                if side.any():
                    a_ = u[side]; b_ = 2 * (x[side] * ux[side] + y[side] * uy[side])
                    c_ = x[side] ** 2 + y[side] ** 2 - R * R
                    tau[side] = (-b_ + np.sqrt(b_ * b_ - 4 * a_ * c_)) / (2 * a_)
                w = np.exp(-tau / ell) * C_ISO
                ib = (np.minimum(np.searchsorted(sin2_edges, u) - 1, n_th - 1) * n_az
                      + np.minimum((ph / (2 * np.pi) * n_az).astype(int), n_az - 1))
                Lp = s - (x * ux + y * uy) + z * ct
                Wsum += np.bincount(ib, w, minlength=Wsum.size)
                for j, dk in enumerate(dks):
                    pz = w * np.exp(1j * dk * Lp)
                    A[:, j] += np.bincount(ib, pz.real, minlength=Wsum.size) + 1j * np.bincount(ib, pz.imag, minlength=Wsum.size)
                kx, ky = k0 * ux, k0 * uy
                kz = k0 * ct
                for j, q in enumerate(qs):
                    kz2 = np.sqrt(np.maximum(k0 * k0 - (kx + q) ** 2 - ky ** 2, 0.0))
                    dphi = q * x1 + (np.sqrt(k0 * k0 - q * q) - k0) * z1 - q * x + (kz2 - kz) * z
                    pz = w * np.exp(1j * dphi)
                    B[:, j] += np.bincount(ib, pz.real, minlength=Wsum.size) + 1j * np.bincount(ib, pz.imag, minlength=Wsum.size)
            d = iso_dirs(x.size, rng)
        if verbose:
            print(f"  slab: {done:,} photons, {events / done:.2f} events each, {time.time() - t0:.0f} s", flush=True)
    return {"A": A / n_photons, "B": B / n_photons, "W": Wsum / n_photons, "events_per_photon": events / n_photons}


# ---------------------------------------------------------------------------------------------------------------------


def check():
    """isotropic half-space against exact transport theory (Chandrasekhar, Radiative Transfer, 1960):
    normal-exit radiant intensity for normal incidence = H(1)^2 / (8 pi) = 0.33643 per steradian per unit power;
    far from the entry point the radiance / flux ratio tends to the Milne value sqrt(3) H(1) / (4 pi) = 0.40079."""
    H1 = 2.9078
    k = half_space_kernel(400_000, model="isotropic", s_max=1500.0, seed=1)
    total = float(k["H"][0].sum())
    re = k["rho_edges"]
    rc = np.sqrt(re[1:] * re[:-1])
    rad = k["H"][0].sum((1, 2))
    far = (rc > 8) & (rc < 25)
    ratio = float(rad[far].sum() / k["flux"][far].sum())
    r = half_space_kernel(200_000, model="rayleigh", s_max=1500.0, seed=2)
    single = C_RAYLEIGH / 2
    out = {"isotropic_normal_radiance": total, "exact": H1 ** 2 / (8 * np.pi),
           "isotropic_far_radiance_over_flux": ratio, "milne_exact": np.sqrt(3) * H1 / (4 * np.pi),
           "isotropic_escaped_fraction_within_s_max": k["escaped"], "events_per_photon": k["events_per_photon"],
           "rayleigh_normal_radiance_co": float(r["H"][0].sum()), "rayleigh_normal_radiance_cross": float(r["H"][1].sum()),
           "rayleigh_single_scattering_exact": single}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--kernel", help="output .npz for a half-space kernel")
    ap.add_argument("--model", default="rayleigh")
    ap.add_argument("--n-rel", type=float, default=1.0)
    ap.add_argument("--photons", type=int, default=2_000_000)
    ap.add_argument("--s-max", type=float, default=4000.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.check:
        check()
    if a.kernel:
        save_kernel(half_space_kernel(a.photons, a.model, a.n_rel, a.s_max, seed=a.seed), a.kernel)
        print("saved", a.kernel)
