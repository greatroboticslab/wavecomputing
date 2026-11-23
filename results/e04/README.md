# E04 results

Simulation only. Plan: lab notebook (r). Results and reading: lab notebook (s) and
[docs/weekly_notes/2026-11-22](../../docs/weekly_notes/2026-11-22). Code: [experiments/e04_wave_sim](../../experiments/e04_wave_sim).

| File | What it holds |
|---|---|
| `fullwave.json` | Exact wave solution (3,198 scatterers, 16 samples): speckle contrast, intensity histogram, and the speckle correlation against wavelength step and beam tilt, next to the path-sum prediction |
| `checks.json` | Path Monte Carlo against exact transport theory; diffusion with complex absorption against the Monte Carlo; radial profiles by polarization; degree of polarization against distance |
| `m_profiles.json` | Speckle contrast ring by ring: formula F1 and the value measured from simulated frames, 7 materials, 2 lasers, 8 objects each |
| `wavelength_steps.json` | Speckle correlation against wavelength step: formula F3 and simulated frames |
| `beam_steps.json` | Speckle correlation against beam step, collimated and diverging beam: formula F4 and simulated frames |
| `figs/` | Figures 1 to 6 of the weekly note (PNG and PDF) |
