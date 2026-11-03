# Kaiser-group MOT comparison

Recreates the Letellier, Galvão de Melo, Dorne & Kaiser 399 nm Yb MOT
(Rev. Sci. Instrum. 94, 123203 (2023)) with our oven -> nozzle -> MOT pipeline,
using their real coils (16 x 16 turns of 2.2 mm wire, 55 mm bore, 65 mm between
inner faces) and the slowing beam.

    cd atom_beam/kaiser_comparison
    python compare_kaiser.py       # beam checks + slower-off table
    python kaiser_load_rates.py    # load-rate Figs. 7a, 8, 10, 11 (slower on/off)
    python kaiser_coils.py         # amps per G/cm and field on the atom beam

`kaiser_load_rates.py` writes `kaiser_load_rates.csv` plus one figure per paper
figure. Fig. 7a is 35 force grids, `KS_SKIP_FIG7=1` skips it.
`KS_SLOWER_WAISTS=3,8,12` adds slower waist curves to Fig. 11.
`KM_FIELD=quad` goes back to the old linear quadrupole.

Outputs go to `runs/kaiser-comparison_<stamp>/`; `kaiser_vs_sim.csv` is the
headline table. The nozzle trace and each MOT capture map are cached in
`cache/` (first run is ~1 h on 2 cores, mostly the 12 force grids).

## Files
```
kaiser_nozzle.py    705-tube triangular micro-tube array, beamline apertures,
                    556 nm oven and 399 nm cell absorption probes
kaiser_coils.py     magpylib coil model, (r, z) field table cached at 1 A
kaiser_mot.py       45 deg MOT: 22 mm beams, per-axis power, coil field,
                    slowing beam; reuses full_trap_sweep force grid / capture
compare_kaiser.py   nozzle/beam checks and slower-off table
kaiser_load_rates.py  Figs. 7a, 8, 10, 11 load rates with the slower
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
| field | real coils, current set by the vertical gradient (7.07 G/cm per A) | Kaiser group |
| slower | collimated, through trap centre, counter-propagating, MOT detuning - 220 MHz | paper Sec. II B |
| slower power | 300 mW (Figs. 6, 8, 11), 200 mW (Fig. 7) | paper |
| slower waist | 5 mm, `KM_SLOWER_WAIST_MM` | **not given**, guess |
| slower angle | 0, `KM_SLOWER_ANGLE_MRAD` (horizontal tilt) | paper: "small angle" |
| slower pol | +1, `KM_SLOWER_POL` | +1 drives the transition the fringe field pulls toward lower v |
| axial grid | 1.5 mm to 100 mm upstream, 7.5 mm beyond | coil fringe dies by ~150 mm |
| detuning | in units of 29 MHz (paper Gamma) | |
| abundance | 174Yb 31.83 % applied to all rates and absorptions | |
| nozzle wall = oven temperature | single paper temperature | |

Fig. 10 (3.1 G/cm, -1.4 Gamma) has no oven temperature and a separate free-space
beam setup: the waist (14.9 mm) is taken from P = 587 mW <-> 2.8 Isat, and the
oven temperature (~413 C) from their 1/s hot-atom loss offset via the Fig. 12 fit.
Treat it as the loosest comparison.

With the coils the fringe field on the atom beam peaks at ~48 G about 45 mm
upstream (32 G/cm) and is < 1.5 G past 200 mm, so the slower resonance drops
from 111 m/s to ~84 m/s over the last ~100 mm: the "small Zeeman slower" the
paper mentions. The old linear quadrupole gave ~830 G at the nozzle.

Fig. 7a / 8 / 11 slower-on data are in `paper/` (`fig7_maps_digitized.csv`,
`L_mot_slower`, `L_slower` rows); Fig. 7 gradient columns are the cell centres
read off the axis (15.3 ... 47.6 G/cm). `kaiser_load_rates.py` applies
`OVEN_TEMP_OFFSET_C` from compare_kaiser to every temperature. Only L is
compared with the slower on: N_st there is set by light-assisted collisions,
which the sim does not have.

`N_st` from the sim is `L_sim / alpha_paper(T)` with alpha from their Fig. 12
fit (plus optical pumping 6.5 s_delta / 2); the sim has no loss model.
`sim_flux_calibrated` rescales the sim by (measured / simulated) 399 nm cell
absorption at that temperature, which isolates the capture model from the
oven-flux model.
