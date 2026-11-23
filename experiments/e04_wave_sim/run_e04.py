"""E04 driver: the checks and simulated experiments planned in lab notebook (r).

    python -P run_e04.py --kernel ../../data/e04/kernel_rayleigh_n1.npz --out ../../results/e04 --parts checks,m,wavelength,beam

  checks      machinery against exact transport results; diffusion with complex absorption against Monte Carlo (F2);
              polarization against distance
  m           method 1A: speckle contrast ring by ring in synthesized frames against F1
  wavelength  method 1C: speckle correlation against wavelength step in frames against F3
  beam        method 4B: speckle correlation against beam step, flat and curved wavefront, in frames against F4

Every number in the result files is computed here; none is typed by hand.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.fft as sfft

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth as sy  # noqa: E402
import theory as th  # noqa: E402
from transport_mc import half_space_kernel, load_kernel  # noqa: E402

EDGES = np.round(np.arange(0.25, 3.51, 0.25), 2)
VOLUME = [(0.05, 0.10), (0.2, 0.05), (0.8, 0.02)]          # transport length mm, absorption per mm (made up)
MIN_SIGNAL = 2e-3                                           # ring mean over spot-centre mean, below which a ring is not scored


def dump(obj, path):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1)


def bin_phasor(q, se):
    """mean of exp(i q s) over each path-length bin (uniform within the bin): array (len(q), bins)"""
    q = np.asarray(q, float)[:, None]
    s1, s2 = se[None, :-1], se[None, 1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        v = (np.exp(1j * q * s2) - np.exp(1j * q * s1)) / (1j * q * (s2 - s1))
    return np.where(q == 0, 1.0, v)


def ring_grains(rig):
    area_c = rig.lam ** 2 / (np.pi * rig.na ** 2)           # speckle correlation area of a circular pupil
    return (np.pi * (EDGES[1:] ** 2 - EDGES[:-1] ** 2) / area_c).tolist()


# ---------------------------------------------------------------------------------------------------------------------


def part_checks(kraw, out):
    res = {}
    k = half_space_kernel(600_000, model="isotropic", s_max=3000.0, seed=21, verbose=False)
    re = k["rho_edges"]
    rc = np.sqrt(re[1:] * re[:-1])
    rad = k["H"][0].sum((1, 2))
    tot = float(k["H"][0].sum())
    far = rc > 30
    res["machinery"] = {
        "normal_radiance_paths_finished": tot, "share_of_paths_unfinished": 1 - k["escaped"],
        "normal_radiance_all_paths": tot + (1 - k["escaped"]) * th.MILNE_RADIANCE_OVER_FLUX,
        "exact_chandrasekhar": th.EXACT_NORMAL_RADIANCE,
        "radiance_over_flux_beyond_30_mfp": float(rad[far].sum() / k["flux"][far].sum()),
        "milne_exact": th.MILNE_RADIANCE_OVER_FLUX}
    m = res["machinery"]
    m["pass"] = bool(abs(m["normal_radiance_all_paths"] / m["exact_chandrasekhar"] - 1) < 0.01)

    H, re, se = kraw["H"], kraw["rho_edges"], kraw["s_edges"]
    rc = np.sqrt(re[1:] * re[:-1])
    sc = np.sqrt(se[1:] * se[:-1])
    ds = np.diff(se)
    area = np.pi * (re[1:] ** 2 - re[:-1] ** 2)
    tot_rs = (H[0] + H[1]).sum(1)
    res["kernel"] = {"photons": kraw["n_photons"], "events_per_photon": kraw["events_per_photon"],
                     "normal_radiance_co": float(H[0].sum()), "normal_radiance_cross": float(H[1].sum()),
                     "cross_over_co": float(H[1].sum() / H[0].sum()),
                     "single_scattering_share_of_co": float(th.np.float64(3 / (16 * np.pi)) / H[0].sum())}
    # radial profiles (per unit area) and diffusion
    prof = {"rho": rc.tolist(), "co": (H[0].sum((1, 2)) / area).tolist(), "cross": (H[1].sum((1, 2)) / area).tolist(),
            "diffusion_total": th.diffusion(rc, 0.0, 1.0)[0].tolist()}
    res["profile"] = prof
    # F2: diffusion with complex absorption against the Monte Carlo path-length Fourier transform
    mua = 0.002
    q = np.linspace(0, 0.6, 2401)
    PH = bin_phasor(q, se)
    res["spectral"] = {}
    for rho0 in (2.0, 10.0, 20.0):
        i = int(np.argmin(np.abs(rc - rho0)))
        h = tot_rs[i] * np.exp(-mua * sc)
        c_mc = (PH @ h) / h.sum()
        rq = th.diffusion(rc[i], mua, 1.0, 1.0, n_dk=q)[0]
        c_df = rq / rq[0]
        sfine = np.exp(np.linspace(np.log(0.05), np.log(2e4), 4000))
        pd = th.diffusion_path_density(rc[i], sfine, mua, 1.0)
        mean_df = float(np.trapezoid(pd * sfine, sfine) / np.trapezoid(pd, sfine))
        pd_bins = th.diffusion_path_density(rc[i], sc, mua, 1.0)
        res["spectral"][f"{rho0:g}"] = {
            "rho": float(rc[i]), "half_width_mc": th.half_width(q, c_mc), "half_width_diffusion": th.half_width(q, c_df),
            "mean_path_mc": float((h * sc).sum() / h.sum()), "mean_path_diffusion": mean_df,
            "q": q[::8].tolist(), "abs_mc": np.abs(c_mc)[::8].tolist(), "abs_diffusion": np.abs(c_df)[::8].tolist(),
            "s": sc.tolist(), "path_density_mc": (h / ds / h.sum()).tolist(),
            "path_density_diffusion": (pd_bins / np.trapezoid(pd, sfine)).tolist()}
    for key in ("10", "20"):
        r = res["spectral"][key]
        r["ratio"] = r["half_width_diffusion"] / r["half_width_mc"]
        r["pass"] = bool(abs(r["ratio"] - 1) < 0.15)
    # polarization against distance from a pencil beam
    J = H.sum(3)
    P = np.sqrt((J[0] - J[1]) ** 2 + 4 * J[2] ** 2) / np.maximum(J[0] + J[1], 1e-300)
    ring_P = np.abs(J[0].sum(1) - J[1].sum(1)) / (J[0].sum(1) + J[1].sum(1))
    sel = (rc > 5) & (rc <= 40)
    res["polarization"] = {"rho": rc.tolist(), "P_max_over_azimuth": P.max(1).tolist(), "P_ring_mean": ring_P.tolist(),
                           "max_P_5_to_40_mfp": float(P.max(1)[sel].max()),
                           "single_frequency_M_min_5_to_40_mfp": float(2 / (1 + P.max(1)[sel].max() ** 2)),
                           "pass": bool(P.max(1)[sel].max() < 0.1)}
    dump(res, os.path.join(out, "checks.json"))
    print("machinery:", {a: (round(b, 4) if isinstance(b, float) else b) for a, b in m.items()})
    print("kernel: cross/co", round(res["kernel"]["cross_over_co"], 4))
    for key in ("2", "10", "20"):
        r = res["spectral"][key]
        print(f"F2 at rho {r['rho']:.1f}: half width MC {r['half_width_mc']:.4f} diffusion {r['half_width_diffusion']:.4f}; "
              f"mean path MC {r['mean_path_mc']:.1f} diffusion {r['mean_path_diffusion']:.1f}")
    print("polarization: max P for 5 < rho <= 40:", round(res["polarization"]["max_P_5_to_40_mfp"], 3))


# ---------------------------------------------------------------------------------------------------------------------


def materials(kern, rig, rng):
    """(name, scene) for the bare metal and the volume scatterers with 0% and 30% surface light"""
    out = [("metal (surface only)", sy.Scene(kern, rig, sy.Material("metal"), rng))]
    for ell, mua in VOLUME:
        base = sy.Scene(kern, rig, sy.Material("v", ell, mua, 0.0), rng)
        out.append((f"transport length {ell:g} mm", base))
        out.append((f"transport length {ell:g} mm, 30% surface light", base.with_surface(0.3)))
    return out


def part_m(kern, out, data, n_obj, seed):
    rng = np.random.default_rng(seed)
    rig = sy.Rig()
    sources = {"single-frequency": (np.zeros(1), np.ones(1)), "multimode": th.multimode_spectrum(rig.lam)}
    res = {"ring_edges_mm": EDGES.tolist(), "grains_per_ring": ring_grains(rig), "objects": n_obj, "sources": {},
           "rig": {"wavelength_mm": rig.lam, "grid_mm": rig.dx, "pixel_mm": rig.dx * rig.pix, "grain_mm": rig.grain,
                   "beam_radius_mm": rig.w, "index": rig.n, "NA": rig.na}}
    for sname, (dk, wts) in sources.items():
        res["sources"][sname] = {"lines": int(len(dk))}
    frames_keep = {}
    t0 = time.time()
    mats = materials(kern, rig, rng)
    print(f"scenes built, {time.time() - t0:.0f} s; path nodes:", [len(s.s) for _, s in mats], flush=True)
    res["materials"] = {}
    for name, sc in mats:
        mean = sc.mean_image()
        peak = float(rig.to_pixels(mean).max())
        entry = {"path_nodes": int(len(sc.s)), "signal": (sy.ring_mean(mean, rig, EDGES) / mean.max()).tolist(), "sources": {}}
        for sname, (dk, wts) in sources.items():
            k2t = sy.ring_mean(sc.k2_theory(dk, wts), rig, EDGES)
            phases = np.outer(dk, rig.n * sc.s)
            meas = []
            for o in range(n_obj):
                E = sc.fields(phases, rng)
                I = sum(w * rig.image(E[m]) for m, w in enumerate(wts))
                fr = sy.camera(I, rig, peak, rng)
                p, _ = sy.ring_products(fr, fr, rig, (0, 0), (0, 0), EDGES)
                meas.append(p)
                if o == 0:
                    frames_keep[f"{name}|{sname}"] = fr[0].astype(np.float32)
            meas = np.array(meas)
            entry["sources"][sname] = {"k2_theory": k2t.tolist(), "k2_measured_mean": meas.mean(0).tolist(),
                                       "k2_measured_sem": (meas.std(0, ddof=1) / np.sqrt(n_obj)).tolist()}
        res["materials"][name] = entry
        print(f"  {name}: done, {time.time() - t0:.0f} s", flush=True)
    # the bare-metal reference gives the factor by which pixels and the estimator scale every K^2
    mid = 0.5 * (EDGES[1:] + EDGES[:-1])
    for sname in sources:
        ref = np.array(res["materials"]["metal (surface only)"]["sources"][sname]["k2_measured_mean"])
        sig = np.array(res["materials"]["metal (surface only)"]["signal"])
        beta = float(np.median(ref[(mid <= 1.5) & (sig > MIN_SIGNAL)]))
        res["sources"][sname]["metal_reference_k2"] = beta
        worst = 0.0
        for name, entry in res["materials"].items():
            e = entry["sources"][sname]
            rel = np.array(e["k2_measured_mean"]) / beta
            e["k2_relative_to_metal"] = rel.tolist()
            e["M_theory"] = (1 / np.array(e["k2_theory"])).tolist()
            e["M_measured"] = (1 / np.maximum(rel, 1e-6)).tolist()
            ok = (np.array(res["grains_per_ring"]) >= 200) & (np.array(entry["signal"]) > MIN_SIGNAL)
            dev = np.abs(rel / np.array(e["k2_theory"]) - 1)
            e["scored_rings"] = ok.tolist()
            e["max_relative_deviation"] = float(dev[ok].max()) if ok.any() else None
            if ok.any():
                worst = max(worst, float(dev[ok].max()))
        res["sources"][sname]["worst_relative_deviation"] = worst
        res["sources"][sname]["pass"] = bool(worst < 0.15)
        print(f"{sname}: metal reference K^2 {beta:.3f}; worst deviation from F1 {worst:.3f}", flush=True)
    dump(res, os.path.join(out, "m_profiles.json"))
    np.savez_compressed(os.path.join(data, "example_frames.npz"), **frames_keep)


def part_wavelength(kern, out, n_obj, seed):
    rng = np.random.default_rng(seed)
    rig = sy.Rig()
    steps_nm = np.array([0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    dk = -2 * np.pi * steps_nm * 1e-6 / rig.lam ** 2
    res = {"ring_edges_mm": EDGES.tolist(), "grains_per_ring": ring_grains(rig), "steps_nm": steps_nm.tolist(),
           "objects": n_obj, "materials": {}}
    worst = 0.0
    t0 = time.time()
    for name, sc in materials(kern, rig, rng):
        mean = sc.mean_image()
        peak = float(rig.to_pixels(mean).max())
        sig = sy.ring_mean(mean, rig, EDGES) / mean.max()
        phases = np.outer(dk, rig.n * sc.s)
        P0a = np.zeros((len(dk), len(EDGES) - 1))
        Paa = np.zeros_like(P0a)
        for o in range(n_obj):
            E = sc.fields(phases, rng)
            fr = [sy.camera(rig.image(E[a]), rig, peak, rng) for a in range(len(dk))]
            for a in range(len(dk)):
                P0a[a] += sy.ring_products(fr[0], fr[a], rig, (0, 0), (0, 0), EDGES)[0]
                Paa[a] += sy.ring_products(fr[a], fr[a], rig, (0, 0), (0, 0), EDGES)[0]
        meas = P0a / np.sqrt(np.maximum(Paa[:1] * Paa, 1e-30))
        theory = []
        for a in range(len(dk)):
            num, den = sc.step_theory(dk[0], dk[a])
            theory.append(sy.ring_mean(num / mean ** 2, rig, EDGES) / sy.ring_mean(den / mean ** 2, rig, EDGES))
        theory = np.array(theory)
        ok = (np.array(res["grains_per_ring"]) >= 200) & (sig > MIN_SIGNAL)
        dev = float(np.abs(meas - theory)[:, ok].max())
        worst = max(worst, dev)
        res["materials"][name] = {"signal": sig.tolist(), "measured": meas.tolist(), "theory": theory.tolist(),
                                  "scored_rings": ok.tolist(), "max_abs_deviation": dev}
        print(f"  {name}: max |frames - F3| {dev:.3f}, {time.time() - t0:.0f} s", flush=True)
    res["worst_abs_deviation"] = worst
    res["pass"] = bool(worst < 0.05)
    dump(res, os.path.join(out, "wavelength_steps.json"))


def beam_step(kern, rig, mat, positions, Rc, n_obj, rng):
    N, n = rig.N, len(positions)
    A = [rig.beam((d, 0.0), Rc) for d in positions]
    H = np.zeros((2, n, n, N, N), complex)
    spread = None
    if mat.ell is not None:
        km = sy.KernelMaps(kern, rig, mat, rng)
        Kt = km.total()
        off = (np.arange(2 * N) - N) * rig.dx
        tot = Kt[0] + Kt[1]
        spread = float(np.sqrt((tot * off[None, :] ** 2).sum() / tot.sum()))      # rms sideways travel per axis
        FK = sfft.fft2(sfft.ifftshift(Kt[:2], axes=(-2, -1)), axes=(-2, -1), workers=sy.WORKERS)
        for a in range(n):
            for b in range(a, n):
                h = sfft.ifft2(FK * sy.padded_fft(rig, A[a] * np.conj(A[b])), axes=(-2, -1),
                               workers=sy.WORKERS)[..., :N, :N] * rig.dx ** 2
                H[:, a, b] = h
                if b != a:
                    H[:, b, a] = np.conj(h)
        p_vol = float((H[0, 0, 0] + H[1, 0, 0]).real.sum()) * rig.dx ** 2
        cs = mat.f_surf / (1 - mat.f_surf) * p_vol
    else:
        cs = 1.0
    # theory (F4) in the image plane
    G = np.empty_like(H)
    for a in range(n):
        for b in range(n):
            G[0, a, b] = rig.blur(H[0, a, b] + cs * A[a] * np.conj(A[b]))
            G[1, a, b] = rig.blur(H[1, a, b])
    Ibar = [(G[0, a, a] + G[1, a, a]).real for a in range(n)]
    peak = [float(rig.to_pixels(Ibar[a]).max()) for a in range(n)]
    self_t = [sy.ring_mean((np.abs(G[0, a, a]) ** 2 + np.abs(G[1, a, a]) ** 2) / Ibar[a] ** 2, rig, EDGES) for a in range(n)]
    theory = np.array([sy.ring_mean((np.abs(G[0, 0, b]) ** 2 + np.abs(G[1, 0, b]) ** 2) / (Ibar[0] * Ibar[b]), rig, EDGES)
                       / np.sqrt(self_t[0] * self_t[b]) for b in range(n)])
    sig = sy.ring_mean(Ibar[0], rig, EDGES) / Ibar[0].max()
    # frames: jointly Gaussian fields with the pointwise covariance across beam positions
    L = []
    for c in range(2):
        C = np.moveaxis(H[c], (0, 1), (-2, -1))                      # (N, N, n, n), Hermitian
        C = 0.5 * (C + np.conj(np.swapaxes(C, -1, -2)))
        ev, V = np.linalg.eigh(C)
        L.append(V * np.sqrt(np.clip(ev, 0, None))[..., None, :])    # a square root of the covariance at every point
    P0b = np.zeros((n, len(EDGES) - 1))
    Pbb = np.zeros_like(P0b)
    for _ in range(n_obj):
        Ex = np.einsum("xyab,xyb->axy", L[0], sy.cnormal((N, N, n), rng))
        Ey = np.einsum("xyab,xyb->axy", L[1], sy.cnormal((N, N, n), rng))
        gs = sy.cnormal((N, N), rng)
        fr = []
        for a in range(n):
            E = np.stack([Ex[a] + np.sqrt(cs) * A[a] * gs, Ey[a]])
            fr.append(sy.camera(rig.image(E), rig, peak[a], rng))
        for b in range(n):
            P0b[b] += sy.ring_products(fr[0], fr[b], rig, (positions[0], 0), (positions[b], 0), EDGES)[0]
            Pbb[b] += sy.ring_products(fr[b], fr[b], rig, (positions[b], 0), (positions[b], 0), EDGES)[0]
    meas = P0b / np.sqrt(np.maximum(Pbb[:1] * Pbb, 1e-30))
    return {"signal": sig.tolist(), "theory": theory.tolist(), "measured": meas.tolist(), "rms_sideways_travel_mm": spread}


def part_beam(kern, out, n_obj, seed):
    rng = np.random.default_rng(seed)
    rig = sy.Rig()
    mats = [("metal (surface only)", sy.Material("metal")),
            ("transport length 0.05 mm", sy.Material("v", 0.05, 0.10, 0.0)),
            ("transport length 0.2 mm", sy.Material("v", 0.2, 0.05, 0.0)),
            ("transport length 0.2 mm, 30% surface light", sy.Material("v", 0.2, 0.05, 0.3)),
            ("transport length 0.8 mm", sy.Material("v", 0.8, 0.02, 0.0))]
    sets = {"flat": {"Rc_mm": None, "steps_mm": [0.0, 0.1, 0.3, 0.5, 1.0]},
            "curved": {"Rc_mm": 300.0, "steps_mm": [0.0, 0.05, 0.1, 0.2, 0.4]}}
    grains = np.array(ring_grains(rig))
    res = {"ring_edges_mm": EDGES.tolist(), "grains_per_ring": grains.tolist(), "objects": n_obj, "beam_radius_mm": rig.w,
           "NA": rig.na, "sets": {}}
    worst = 0.0
    t0 = time.time()
    for sname, cfg in sets.items():
        res["sets"][sname] = {"Rc_mm": cfg["Rc_mm"], "steps_mm": cfg["steps_mm"], "materials": {}}
        for name, mat in mats:
            r = beam_step(kern, rig, mat, cfg["steps_mm"], cfg["Rc_mm"], n_obj, rng)
            ok = (grains >= 200) & (np.array(r["signal"]) > MIN_SIGNAL)
            r["scored_rings"] = ok.tolist()
            th_, me_ = np.array(r["theory"]), np.array(r["measured"])
            r["max_abs_deviation"] = float(np.abs(me_ - th_)[:, ok].max())
            # one number per step: rings pooled with equal weight over the scored rings inside 1.5 mm, and outside
            mid = 0.5 * (EDGES[1:] + EDGES[:-1])
            for zone, sel in (("inside_1p5mm", ok & (mid < 1.5)), ("outside_1p5mm", ok & (mid >= 1.5))):
                r[zone] = {"theory": th_[:, sel].mean(1).tolist() if sel.any() else None,
                           "measured": me_[:, sel].mean(1).tolist() if sel.any() else None}
            worst = max(worst, r["max_abs_deviation"])
            res["sets"][sname]["materials"][name] = r
            print(f"  {sname} / {name}: max |frames - F4| {r['max_abs_deviation']:.3f}; inside 1.5 mm theory "
                  f"{np.round(r['inside_1p5mm']['theory'], 3).tolist()}, {time.time() - t0:.0f} s", flush=True)
    res["worst_abs_deviation"] = worst
    res["pass_frames_match_formula"] = bool(worst < 0.05)
    dump(res, os.path.join(out, "beam_steps.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="../../data/e04")
    ap.add_argument("--parts", default="checks,m,wavelength,beam")
    ap.add_argument("--objects", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    kraw = load_kernel(a.kernel)
    kern = sy.Kernel(kraw)
    parts = a.parts.split(",")
    if "checks" in parts:
        part_checks(kraw, a.out)
    if "m" in parts:
        part_m(kern, a.out, a.data, a.objects, a.seed + 1)
    if "wavelength" in parts:
        part_wavelength(kern, a.out, max(4, a.objects // 2), a.seed + 2)
    if "beam" in parts:
        part_beam(kern, a.out, max(4, a.objects // 2), a.seed + 3)
