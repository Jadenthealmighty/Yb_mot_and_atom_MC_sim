# Yb_mot_and_atom_MC_sim
Full Monte Carlo simulation of atom beam from oven and Magneto-Optical Trapping (MOT).

## Before running
Please run with magpylib version $\geq$ 5 since older versions use different unit conventions

Dependencies: `numpy`, `scipy`, `matplotlib`, `pylcp`, `magpylib`, `numba`, `CyRK`.
`pylcp` and `CyRK` are PyPI only, so install them with pip.
Also, CyRK uses `-mavx2 -mfma` to solve capture velocities, so it needs an x86-64 processor... use scipy solver if you want to run on a Mac/ARM processor.


## Repository structure
```
atom_beam/
├── full_trap_sweep.py      full oven -> nozzle -> MOT capture
├── beam_sweep.py           full_trap over slower saturation x beam radius
├── power-sweep-table.csv   best coil current + slower detuning vs PEAK s0
├── runlog.py               terminal progress
├── nozzle_trace_cache.npz  cached nozzle trace, reused 
├── helpers/                importable modules, one package
│   ├── __init__.py
│   ├── trap_beams.py            MOT + slowing beam geometry, misalignment
│   ├── coil_field_model.py      8-coil anti-Helmholtz field, magpylib
│   ├── yb_nozzle_beam_3d.py     3D nozzle Monte Carlo
│   ├── yb_nozzle_beam_2d.py     2D shared Yb and geometry constants
│   └── yb174_mot_simulation.py  old MOT sim, and the units/normalization
├── archived/               older copies, kept for reference, not imported
├── pre_computed/           saved outputs from nozzle trace
├── runs/                   simulation results
```



## How to run
`/Yb_mot_and_atom_MC_sim`:

    cd atom_beam
    python full_trap_sweep.py
    python beam_sweep.py

beam_sweep takes ~1 h for the default 5 x 5 grid and can be restarted 



## Going to add:
- full_trap_sweep sweeping
- Interactor for the full_sweep
- Guide for usage



## Maintainers + contact:
jaden.alaidroos@mail.utoronto.ca --- GH @Jadenthealmighty
