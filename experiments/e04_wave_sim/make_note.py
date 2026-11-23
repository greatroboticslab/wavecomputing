"""Fill the weekly note's numbers and tables from the E04 result files (no number typed by hand).

    python -P make_note.py --results ../../results/e04 --template ../../docs/weekly_notes/2026-11-22/weekly_note_template.tex \
        --out ../../docs/weekly_notes/2026-11-22/weekly_note.tex

Also prints every filled value, so the sentences around them can be checked against what they say.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VOL = ["transport length 0.05 mm", "transport length 0.2 mm", "transport length 0.8 mm"]
METAL = "metal (surface only)"
SURF = ", 30% surface light"
LAB = {METAL: "bare metal", VOL[0]: r"$\ell^*=0.05$ mm", VOL[1]: r"$\ell^*=0.2$ mm", VOL[2]: r"$\ell^*=0.8$ mm"}


def lab(name):
    return LAB[name.replace(SURF, "")] + (" + 30\\% surface" if name.endswith(SURF) else "")


def crossing(x, y, level):
    """first x at which y falls below level (linear interpolation)"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    i = int(np.nonzero(y < level)[0][0])
    return float(x[i - 1] + (y[i - 1] - level) / (y[i - 1] - y[i]) * (x[i] - x[i - 1]))


def main(res, template, out):
    J = lambda n: json.load(open(os.path.join(res, n), encoding="utf-8"))
    fw, ck, mp, wl, bm = J("fullwave.json"), J("checks.json"), J("m_profiles.json"), J("wavelength_steps.json"), J("beam_steps.json")
    import fullwave
    st = fullwave.selftest()
    m = {}
    m["fwN"] = f"{fw['n_scatterers']:,}".replace(",", "{,}")
    m["fwSamples"] = str(fw["realizations"])
    m["fwThick"] = f"{fw['optical_thickness']:g}"
    m["fwEnergy"] = f"{100 * abs(st['ratio'] - 1):.3f}\\%"
    m["fwContrast"] = f"{fw['contrast']:.2f}"
    m["fwSpecDiff"] = f"{fw['spectral']['max_abs_diff']:.2f}"
    m["fwTiltDiff"] = f"{fw['tilt']['max_abs_diff']:.2f}"
    mach = ck["machinery"]
    m["mcPhotons"] = f"{ck['kernel']['photons'] / 1e6:g}"
    m["mcNormal"] = f"{mach['normal_radiance_all_paths']:.4f}"
    m["mcExact"] = f"{mach['exact_chandrasekhar']:.4f}"
    m["mcDevPct"] = f"{100 * (mach['normal_radiance_all_paths'] / mach['exact_chandrasekhar'] - 1):+.1f}".replace("-", "$-$")
    m["dfTen"] = f"{100 * abs(ck['spectral']['10']['ratio'] - 1):.0f}"
    m["dfTwenty"] = f"{100 * abs(ck['spectral']['20']['ratio'] - 1):.0f}"
    pol = ck["polarization"]
    rho, pmax = np.array(pol["rho"]), np.array(pol["P_max_over_azimuth"])
    sel = rho > 1
    m["polDist"] = f"{crossing(rho[sel], pmax[sel], 0.1):.1f}"
    m["polAtFive"] = f"{float(np.interp(5.0, rho, pmax)):.2f}"

    # --- method 1
    e = np.array(mp["ring_edges_mm"])
    mid = 0.5 * (e[1:] + e[:-1])
    m["mObjects"] = str(mp["objects"])
    m["mWorstSingle"] = f"{100 * mp['sources']['single-frequency']['worst_relative_deviation']:.0f}"
    m["mWorstMulti"] = f"{100 * mp['sources']['multimode']['worst_relative_deviation']:.0f}"
    mmax, core = 0.0, []
    for name, mat in mp["materials"].items():
        x = mat["sources"]["multimode"]
        ok = np.array(x["scored_rings"])
        mmax = max(mmax, float(np.max(np.array(x["M_theory"])[ok])))
        if name.endswith(SURF):
            core.append(float(np.array(x["M_theory"])[ok][0]))
    m["mMaxMulti"] = f"about {mmax:.0f}"
    m["mCoreSurf"] = f"{min(core):.1f}$ to ${max(core):.1f}"
    i_in, i_out = int(np.argmin(np.abs(mid - 0.625))), int(np.argmin(np.abs(mid - 2.125)))
    rows = []
    for name, mat in mp["materials"].items():
        cells = [lab(name)]
        for src in ("single-frequency", "multimode"):
            x = mat["sources"][src]
            for i in (i_in, i_out):
                if x["scored_rings"][i]:
                    cells.append(f"{x['M_theory'][i]:.2f} ({x['M_measured'][i]:.2f})")
                else:
                    cells.append("too dim")
        rows.append(" & ".join(cells) + r" \\")
    m["mTable"] = ("\\begin{center}\\small\n\\begin{tabular}{lcccc}\n\\toprule\n"
                   " & \\multicolumn{2}{c}{single-frequency laser} & \\multicolumn{2}{c}{multimode diode}\\\\\n"
                   f"$M$: formula (frames) & at {mid[i_in]:.2f} mm & at {mid[i_out]:.2f} mm & at {mid[i_in]:.2f} mm & at {mid[i_out]:.2f} mm\\\\\n"
                   "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{center}")

    # --- wavelength steps
    steps = np.array(wl["steps_nm"])
    m["wlWorst"] = f"{wl['worst_abs_deviation']:.2f}"
    m["wlVerdict"] = "" if wl["pass"] else ", so this one is missed by a little"
    halves = []
    for name in VOL:
        mat = wl["materials"][name]
        ok = np.array(mat["scored_rings"]) & (mid > 0.5) & (mid < 1.5)
        t = np.array(mat["theory"])[:, ok].mean(1)
        halves.append(10 ** crossing(np.log10(steps[1:]), t[1:], 0.5))
    m["wlHalf"] = (f"{halves[0]:.3f}, {halves[1]:.3f} and {halves[2]:.3f}~nm for $\\ell^*=0.05$, 0.2 and 0.8~mm "
                   "(inside the spot)")

    def rms_dev(mats):
        d = np.concatenate([(np.array(x["measured"]) - np.array(x["theory"]))[:, np.array(x["scored_rings"])].ravel()
                            for x in mats.values()])
        return float(np.sqrt(np.mean(d ** 2))), float(np.mean(d)), float(np.mean(np.abs(d) > 0.05)), d.size
    r_, b_, share_, n_ = rms_dev(wl["materials"])
    m["wlRms"] = f"{r_:.3f}"
    m["wlBias"] = f"{b_:+.3f}".replace("-", "$-$")
    m["wlCount"] = str(n_)
    m["wlOver"] = str(int(round(share_ * n_)))
    m["wlObjects"] = str(wl["objects"])
    allb = {f"{s}/{k}": v for s in bm["sets"] for k, v in bm["sets"][s]["materials"].items()}
    r_, b_, share_, n_ = rms_dev(allb)
    m["beamRms"] = f"{r_:.3f}"
    m["beamBias"] = f"{b_:+.3f}".replace("-", "$-$")
    m["beamCount"] = str(n_)
    m["beamOver"] = str(int(round(share_ * n_)))

    # --- beam steps
    m["beamWorst"] = f"{bm['worst_abs_deviation']:.2f}"
    m["beamVerdict"] = "" if bm["pass_frames_match_formula"] else ", so this one is missed by a little"
    flat, cur = bm["sets"]["flat"], bm["sets"]["curved"]
    fs, cs = np.array(flat["steps_mm"]), np.array(cur["steps_mm"])
    zt = lambda s, name: np.array(s["materials"][name]["inside_1p5mm"]["theory"])
    zm = lambda s, name: np.array(s["materials"][name]["inside_1p5mm"]["measured"])
    small = fs <= 0.3 * bm["beam_radius_mm"] + 1e-9
    gap = max(float(np.max(np.abs(zt(flat, METAL)[small] - zt(flat, v)[small]))) for v in VOL)
    m["beamFlatGap"] = f"{gap:.2f}"
    iw = int(np.argmin(np.abs(fs - bm["beam_radius_mm"])))
    m["beamFlatAtW"] = (f"metal {zt(flat, METAL)[iw]:.2f}, $\\ell^*=0.8$~mm {zt(flat, VOL[2])[iw]:.2f}")
    best, bi = -1.0, 1
    for i in range(1, len(cs)):
        sep = zt(cur, METAL)[i] - max(zt(cur, VOL[1])[i], zt(cur, VOL[2])[i])
        if sep > best:
            best, bi = sep, i
    m["beamBestStep"] = f"{cs[bi]:g}"
    m["beamBestMetal"] = f"{zt(cur, METAL)[bi]:.2f}"
    m["beamBestVol"] = (f"{zt(cur, VOL[1])[bi]:.2f} ($\\ell^*=0.2$~mm) and {zt(cur, VOL[2])[bi]:.2f} ($\\ell^*=0.8$~mm)")
    it = int(np.argmin(np.abs(cs - 0.1)))
    m["beamTenthMetal"] = f"{zt(cur, METAL)[it]:.2f}"
    m["beamTenthVol"] = f"{zt(cur, VOL[1])[it]:.2f} and {zt(cur, VOL[2])[it]:.2f}"
    scored = lambda s_, n_: np.array(s_["materials"][n_]["theory"])[:, np.array(s_["materials"][n_]["scored_rings"])]
    i3 = int(np.argmin(np.abs(fs - 0.3 * bm["beam_radius_mm"])))
    m["beamFlatMin"] = f"{min(float(scored(flat, n_)[i3].min()) for n_ in flat['materials']):.2f}"
    cm = cur["materials"][METAL]
    okm = np.array(cm["scored_rings"])
    m["beamMetalBias"] = f"{float((np.array(cm['measured']) - np.array(cm['theory']))[:, okm].mean()):.2f}"
    m["beamSpread"] = ", ".join(f"{cur['materials'][v]['rms_sideways_travel_mm']:.2f}" for v in VOL[:2]) + \
        f" and {cur['materials'][VOL[2]]['rms_sideways_travel_mm']:.2f}"
    rows = []
    for name in cur["materials"]:
        cells = [lab(name)]
        for s, st_ in ((flat, fs), (cur, cs)):
            for i in range(1, len(st_)):
                cells.append(f"{zt(s, name)[i]:.2f} ({zm(s, name)[i]:.2f})")
        rows.append(" & ".join(cells) + r" \\")
    head = " & ".join([f"{v:g}" for v in fs[1:]] + [f"{v:g}" for v in cs[1:]])
    m["beamTable"] = ("\\begin{center}\\footnotesize\\setlength{\\tabcolsep}{3.5pt}\n\\begin{tabular}{l" + "c" * (len(fs) + len(cs) - 2) + "}\n\\toprule\n"
                      f" & \\multicolumn{{{len(fs) - 1}}}{{c}}{{collimated beam, step (mm)}} & \\multicolumn{{{len(cs) - 1}}}{{c}}{{$R_c=300$ mm, step (mm)}}\\\\\n"
                      "correlation: formula (frames) & " + head + "\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{center}")


    for k, v in m.items():
        if "Table" not in k:
            print(f"{k:14s} {v}")
    macros = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items())
    t = open(template, encoding="utf-8").read()
    assert "%%NUMBERS%%" in t
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(t.replace("%%NUMBERS%%", "% numbers and tables below are written by experiments/e04_wave_sim/make_note.py from results/e04\n" + macros))
    print("wrote", out)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.results, a.template, a.out)
