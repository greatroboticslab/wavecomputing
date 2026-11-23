"""E04 figures, drawn from the result files only.

    python -P figures.py --results ../../results/e04 --data ../../data/e04

Formulas are lines, simulations are markers. A material keeps its colour in every figure; 30% surface light is the same
colour, dashed and hollow. Colours are the first four slots of a palette checked for colour-blind separation; every
series is also named in a legend and has its own marker.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e7e6e1"
MAT = {"metal (surface only)": ("#2a78d6", "o", "bare metal (surface only)"),
       "transport length 0.05 mm": ("#eb6834", "s", r"$\ell^*$ = 0.05 mm"),
       "transport length 0.2 mm": ("#1baf7a", "^", r"$\ell^*$ = 0.2 mm"),
       "transport length 0.8 mm": ("#eda100", "D", r"$\ell^*$ = 0.8 mm")}
SURF = ", 30% surface light"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "legend.fontsize": 8, "axes.titlesize": 9.5, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "lines.linewidth": 2.0, "lines.markersize": 5.5,
                     "savefig.dpi": 220, "figure.dpi": 110, "mathtext.default": "regular"})


def style(name):
    base = name.replace(SURF, "")
    col, mk, lab = MAT[base]
    surf = name.endswith(SURF)
    return {"color": col, "marker": mk, "label": lab + (" + 30% surface" if surf else ""), "ls": "--" if surf else "-",
            "mfc": "white" if surf else col}


def marks(ax, x, y, st, yerr=None, label=None):
    ax.errorbar(x, y, yerr=yerr, fmt=st["marker"], color=st["color"], mfc=st["mfc"], mec=st["color"] if st["mfc"] == "white" else "white",
                mew=1.2 if st["mfc"] == "white" else 0.8, capsize=0, elinewidth=1.0, label=label, zorder=3)


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig_fullwave(res, out):
    r = json.load(open(os.path.join(res, "fullwave.json")))
    fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.0), constrained_layout=True)
    blue = "#2a78d6"
    e = np.array(r["intensity_hist"]["edges"])
    c = 0.5 * (e[1:] + e[:-1])
    ax[0].plot(c, np.exp(-c), color=INK, label="exponential law")
    ax[0].plot(c, r["intensity_hist"]["density"], "o", color=blue, mec="white", mew=0.8, label="exact wave solution")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("intensity / mean intensity")
    ax[0].set_ylabel("probability density")
    ax[0].set_title(f"a  Speckle statistics (contrast {r['contrast']:.2f})")
    ax[0].legend()
    s = r["spectral"]
    x = 100 * np.array(s["dk_over_k"])
    ax[1].plot(x, s["path_sum"], color=INK, label="sum over paths (F5)")
    ax[1].plot(x, s["full_wave"], "o", color=blue, mec="white", mew=0.8, label="exact wave solution")
    ax[1].set_xlabel("wavelength step (% of the wavelength)")
    ax[1].set_ylabel("speckle correlation")
    ax[1].set_title("b  Stepping the wavelength")
    ax[1].set_ylim(0, 1.03)
    ax[1].legend()
    t = r["tilt"]
    x = np.degrees(np.arcsin(np.array(t["q_over_k"])))
    ax[2].plot(x, t["path_sum"], color=INK, label="sum over paths (F5)")
    ax[2].plot(x, t["full_wave"], "o", color=blue, mec="white", mew=0.8, label="exact wave solution")
    ax[2].set_xlabel("beam tilt (degrees)")
    ax[2].set_ylabel("speckle correlation")
    ax[2].set_title("c  Tilting the beam")
    ax[2].set_ylim(0, 1.03)
    ax[2].legend()
    save(fig, out, "fig1_exact_wave")


def fig_transport(res, out):
    r = json.load(open(os.path.join(res, "checks.json")))
    fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.0), constrained_layout=True)
    p = r["profile"]
    rho = np.array(p["rho"])
    sel = (rho > 0.05) & (rho < 55)
    co, cr = np.array(p["co"]), np.array(p["cross"])
    ax[0].plot(rho[sel], np.array(p["diffusion_total"])[sel], color=INK, label="diffusion formula (total)")
    ax[0].plot(rho[sel][::3], (co + cr)[sel][::3], "o", color="#2a78d6", mec="white", mew=0.6, ms=4.5, label="Monte Carlo, total")
    ax[0].plot(rho[sel][::3], co[sel][::3], "s", color="#eb6834", mec="white", mew=0.6, ms=4.5, label="same polarization as the laser")
    ax[0].plot(rho[sel][::3], cr[sel][::3], "^", color="#1baf7a", mec="white", mew=0.6, ms=4.5, label="crossed polarization")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("distance from entry point (mean free paths)")
    ax[0].set_ylabel("light leaving towards the camera")
    ax[0].set_title("a  The glow around a pencil beam")
    ax[0].legend()
    for key, col, mk in (("10", "#2a78d6", "o"), ("20", "#eb6834", "s")):
        s = r["spectral"][key]
        q = np.array(s["q"])
        ax[1].plot(q, s["abs_diffusion"], color=col)
        ax[1].plot(q[::3], np.array(s["abs_mc"])[::3], mk, color=col, mec="white", mew=0.6, ms=4.5,
                   label=f"{s['rho']:.0f} mean free paths out")
    ax[1].plot([], [], color=INK, label="lines: diffusion with complex absorption (F2)")
    ax[1].set_xlim(0, 0.12)
    ax[1].set_ylim(0, 1.03)
    ax[1].set_xlabel("wavenumber step x mean free path")
    ax[1].set_ylabel("field correlation")
    ax[1].set_title("b  Fourier transform of the path lengths")
    ax[1].legend(loc="upper right")
    pol = r["polarization"]
    rho = np.array(pol["rho"])
    sel = (rho > 0.05) & (rho < 45)
    ax[2].plot(rho[sel], np.array(pol["P_max_over_azimuth"])[sel], "o-", color="#2a78d6", mec="white", mew=0.6, ms=4.0, lw=1.2,
               label="Monte Carlo")
    ax[2].axhline(0.1, color=MUTED, lw=1.0, ls=":")
    ax[2].set_xscale("log")
    ax[2].set_xlabel("distance from entry point (mean free paths)")
    ax[2].set_ylabel("degree of polarization")
    ax[2].set_title("c  How far the polarization survives")
    ax[2].legend(loc="upper right")
    save(fig, out, "fig2_transport")


def fig_frames(res, data, out):
    z = np.load(os.path.join(data, "example_frames.npz"))
    names = ["metal (surface only)", "transport length 0.05 mm", "transport length 0.2 mm", "transport length 0.8 mm"]
    fig, ax = plt.subplots(2, 4, figsize=(10.2, 5.4), constrained_layout=True)
    for row, src in enumerate(("single-frequency", "multimode")):
        for j, n in enumerate(names):
            im = z[f"{n}|{src}"]
            a = ax[row, j]
            a.imshow(np.log10(np.clip(im, 1e-4, None)), cmap="gray", vmin=-3.2, vmax=0.8, extent=[-4, 4, -4, 4])
            a.set_xlim(-3, 3); a.set_ylim(-3, 3)
            a.grid(False)
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
            a.plot([1.6, 2.6], [-2.6, -2.6], color="white", lw=2)
            a.text(2.1, -2.42, "1 mm", color="white", ha="center", va="bottom", fontsize=8)
            a.set_title(f"{MAT[n][2]}\n{src} laser", fontsize=8.5)
    save(fig, out, "fig3_frames")


def fig_m(res, out):
    r = json.load(open(os.path.join(res, "m_profiles.json")))
    e = np.array(r["ring_edges_mm"])
    mid = 0.5 * (e[1:] + e[:-1])
    fig, ax = plt.subplots(1, 2, figsize=(10.2, 3.6), constrained_layout=True)
    for a, src in zip(ax, ("single-frequency", "multimode")):
        for name, m in r["materials"].items():
            st = style(name)
            x = m["sources"][src]
            ok = np.array(x["scored_rings"])
            k2t = np.array(x["k2_theory"])
            rel = np.array(x["k2_relative_to_metal"])
            sem = np.array(x["k2_measured_sem"]) / r["sources"][src]["metal_reference_k2"]
            a.plot(mid[ok], 1 / k2t[ok], color=st["color"], ls=st["ls"])
            marks(a, mid[ok], 1 / rel[ok], st, yerr=sem[ok] / rel[ok] ** 2, label=st["label"])
        a.set_xlabel("distance from the spot centre (mm)")
        a.set_ylabel("speckle degrees of freedom M")
        a.set_title(("a  Single-frequency laser" if src == "single-frequency" else "b  Multimode laser diode")
                    + ": lines F1, markers frames")
        if src == "multimode":
            a.set_yscale("log")
            a.set_yticks([1, 2, 5, 10, 20])
            a.set_yticklabels(["1", "2", "5", "10", "20"])
    ax[0].legend(ncol=1, loc="lower right")
    save(fig, out, "fig4_m_profiles")


def fig_wavelength(res, out):
    r = json.load(open(os.path.join(res, "wavelength_steps.json")))
    e = np.array(r["ring_edges_mm"])
    mid = 0.5 * (e[1:] + e[:-1])
    steps = np.array(r["steps_nm"])
    fig, ax = plt.subplots(1, 2, figsize=(10.2, 3.6), constrained_layout=True)
    for a, (lo, hi, ttl) in zip(ax, ((0.5, 1.5, "a  Inside the spot (0.5 to 1.5 mm)"), (1.5, 3.0, "b  In the glow (1.5 to 3 mm)"))):
        for name, m in r["materials"].items():
            st = style(name)
            ok = np.array(m["scored_rings"]) & (mid > lo) & (mid < hi)
            if not ok.any():
                continue
            t = np.array(m["theory"])[:, ok].mean(1)
            s = np.array(m["measured"])[:, ok].mean(1)
            a.plot(steps[1:], t[1:], color=st["color"], ls=st["ls"])
            marks(a, steps[1:], s[1:], st, label=st["label"])
        a.set_xscale("log")
        a.set_ylim(-0.05, 1.05)
        a.set_xlabel("wavelength step (nm)")
        a.set_ylabel("speckle correlation with the first frame")
        a.set_title(ttl)
    h, l = ax[0].get_legend_handles_labels()
    ax[1].legend(h, l, loc="upper right")
    save(fig, out, "fig5_wavelength_steps")


def fig_beam(res, out):
    r = json.load(open(os.path.join(res, "beam_steps.json")))
    fig, ax = plt.subplots(1, 2, figsize=(10.2, 3.6), constrained_layout=True)
    for a, (key, ttl) in zip(ax, (("flat", "a  Collimated beam (flat wavefront)"),
                                  ("curved", "b  Diverging beam (wavefront radius 300 mm)"))):
        s = r["sets"][key]
        steps = np.array(s["steps_mm"])
        for name, m in s["materials"].items():
            st = style(name)
            z = m["inside_1p5mm"]
            a.plot(steps, z["theory"], color=st["color"], ls=st["ls"])
            marks(a, steps, z["measured"], st, label=st["label"])
        a.set_ylim(-0.05, 1.05)
        a.set_xlabel("beam step (mm)")
        a.set_ylabel("speckle correlation with the first frame")
        a.set_title(ttl)
    ax[0].legend(loc="lower left")
    save(fig, out, "fig6_beam_steps")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    out = os.path.join(a.results, "figs")
    os.makedirs(out, exist_ok=True)
    jobs = {"fullwave": lambda: fig_fullwave(a.results, out), "transport": lambda: fig_transport(a.results, out),
            "frames": lambda: fig_frames(a.results, a.data, out), "m": lambda: fig_m(a.results, out),
            "wavelength": lambda: fig_wavelength(a.results, out), "beam": lambda: fig_beam(a.results, out)}
    for k, f in jobs.items():
        if a.only and k not in a.only.split(","):
            continue
        try:
            f()
        except FileNotFoundError as ex:
            print("skipped", k, "-", ex)
