# Yb_mot_and_atom_MC_sim
Full Monte Carlo simulation of atom beam from oven and Magneto-Optical Trapping (MOT).

## Before running
Please run with magpylib version > 5 since older versions use different unit conventions

Dependencies: `numpy`, `scipy`, `matplotlib`, `pylcp`, `magpylib`, `numba`, `CyRK`.
`pylcp` and `CyRK` are PyPI only, so install them with pip even inside a conda env.
CyRK builds with `-mavx2 -mfma`, so it needs an x86-64 machine — it will not
compile on Apple silicon.


## Repository structure
```
atom_beam/
├── full_trap_sweep.py      entry point: oven -> nozzle -> flight -> MOT capture
├── runlog.py               terminal progress + a new runs/<name>_<timestamp>/ per run
├── nozzle_trace_cache.npz  cached nozzle trace, reused across runs (hours to rebuild)
├── helpers/                importable modules, one package
│   ├── __init__.py
│   ├── trap_beams.py            MOT + slowing beam geometry, misalignment knobs
│   ├── coil_field_model.py      8-coil anti-Helmholtz field, magpylib
│   ├── yb_nozzle_beam_3d.py     3D nozzle Monte Carlo, the real atom-beam numbers
│   ├── yb_nozzle_beam_2d.py     2D slit cross-check, and the shared constants
│   └── yb174_mot_simulation.py  trap-only MOT sim, and the units/normalization
├── archived/               older copies, kept for reference, not imported
├── pre_computed/           saved outputs from long runs
├── runs/                   created per run by full_trap_sweep and the MOT sim
└── results/                created per run by the nozzle Monte Carlo
```

`helpers` is a package, so its modules import each other with `from . import ...`
and full_trap_sweep reaches them as `helpers.trap_beams` and so on. Anything
added to `helpers/` should be listed in `__init__.py` alongside the rest.

`runs/` and `results/` are generated, and the `.npz` files in them get large —
worth a .gitignore entry if the repo starts filling up.


## How to run
Everything runs from inside `atom_beam/`:

    cd atom_beam
    python full_trap_sweep.py

If you wish to set different parameters, please edit the file, most flags are at the top
or in the parameters section in main(). Most are also settable as `FT_*` environment
variables without editing anything.

The helper modules each run on their own too, as modules rather than file paths
so their sibling imports resolve:

    python -m helpers.yb_nozzle_beam_3d     # 3D nozzle Monte Carlo
    python -m helpers.yb_nozzle_beam_2d     # 2D slit cross-check
    python -m helpers.yb174_mot_simulation  # trap-only MOT simulation
    python -m helpers.coil_field_model      # coil field self test
    python -m helpers.trap_beams            # beam geometry self test

`python helpers/yb_nozzle_beam_3d.py` will not work — run as a file path the module
has no package to resolve its sibling imports against.

A full sweep takes hours and uses every core but one, so over ssh start it under
tmux or screen.


## Going to add:
- full_trap_sweep sweeping
- Interactor for the full_sweep
- Guide for usage



## Maintainers + contact:
jaden.alaidroos@mail.utoronto.ca --- GH @Jadenthealmighty
