# Kaiser-group MOT comparison

Recreates the Letellier, Galvão de Melo, Dorne & Kaiser 399 nm Yb MOT
(Rev. Sci. Instrum. 94, 123203 (2023)) with our oven -> nozzle -> MOT pipeline
and compares against their **slowing-beam-off** data only (their coil geometry
is unknown, so only a linear quadrupole is modelled).

    cd atom_beam/kaiser_comparison
    python compare_kaiser.py

Outputs go to `runs/kaiser-comparison_<stamp>/`; `kaiser_vs_sim.csv` is the
headline table. The nozzle trace and each MOT capture map are cached in
`cache/` (first run is ~1 h on 2 cores, mostly the 12 force grids).

## Files
```
kaiser_nozzle.py    705-tube triangular micro-tube array, beamline apertures,
                    556 nm oven and 399 nm cell absorption probes
kaiser_mot.py       45 deg MOT: 22 mm beams, per-axis power, linear quadrupole;
                    reuses full_trap_sweep force grid / capture map / rates
compare_kaiser.py   runs everything, writes the CSV and figures
paper/              values digitised from the paper's figures
```

## Model choices
| item | value | source |
|---|---|---|
| nozzle to trap | 520 mm | paper Sec. II A |
| atom beam vs horizontal MOT beams | 45 deg | |
| tubes | 705 (38-row triangle, bottom 8 rows cut), l = 13 mm, d = 250 um, pitch 340 um | paper ~700, Fig. 2 |
| emission | perfect cylinders, molecular flow, nothing from the gaps | |
| apertures | 12 mm tube ending 171 mm from nozzle (35 mrad), 50 mrad cell entrance | paper, positions assumed |
| MOT beams | w0 = 22 mm, 19 mW per horizontal, 12 mW per vertical at s_delta = 0.012 | Fig. 6 caption |
| field | B = G(-x/2, -y/2, z), G = 32 G/cm vertical | coil size unknown |
| detuning | in units of 29 MHz (paper Gamma) | |
| abundance | 174Yb 31.83 % applied to all rates and absorptions | |
| nozzle wall = oven temperature | single paper temperature | |

Fig. 10 (3.1 G/cm, -1.4 Gamma) has no oven temperature and a separate free-space
beam setup: the waist (14.9 mm) is taken from P = 587 mW <-> 2.8 Isat, and the
oven temperature (~413 C) from their 1/s hot-atom loss offset via the Fig. 12 fit.
Treat it as the loosest comparison.

`N_st` from the sim is `L_sim / alpha_paper(T)` with alpha from their Fig. 12
fit (plus optical pumping 6.5 s_delta / 2); the sim has no loss model.
`sim_flux_calibrated` rescales the sim by (measured / simulated) 399 nm cell
absorption at that temperature, which isolates the capture model from the
oven-flux model.
