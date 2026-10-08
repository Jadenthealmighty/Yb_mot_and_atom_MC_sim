# Opto-optical trap (OOT)

Same oven -> nozzle -> MOT-capture pipeline as `full_trap_sweep.py`, with the
anti-Helmholtz field replaced by the effective field of a triad of 1077 nm
shift beams (Sullivan, Jackson, Ha & Braverman, *Magneto-optical trapping of
neutral atoms in a light-induced effective magnetic field*). The six MOT beams,
atom beam, nozzle trace and CyRK capture map are all reused from the MOT code.

    cd atom_beam/oot
    python oot_trap_sweep.py
    python oot_shift_beams.py     self-test against the paper's Eqs. 2-3

Outputs go to `oot/runs/yb-oot-trap-sweep_<stamp>/`. A default run takes a few
minutes: the force grid is a vectorised rate-equation solve, not pylcp.

Besides the `full_trap_sweep` figures, a run writes
```
oot_geometry                    beams and atom beam, top and side views
effective_field                 |B_eff| maps and line cuts (paper Fig. 1e)
force_cuts_through_centre       restoring force and damping, OOT vs coil MOT
phase_portrait_along_atom_beam  paper Fig. 2 along the atom beam
capture_map_diagnosis           cooling light per ray, rest point, stopping time
shift_beam_design_scan          v_c, trap frequency, depth vs shift power/waist/detuning
*_beam_paths                    copies of the cross-section maps with the MOT
                                and shift beam axes drawn on
```

## Files
```
oot_trap_sweep.py    the pipeline: shift + cooling beams, force grid, capture map,
                     temperature sweep, and a coil MOT in the same geometry
oot_shift_beams.py   shift-beam geometry (beam_profiles), 1P1 -> 6s7s data
                     (ytterbium), incoherent Doppler-shifted light shift
                     (shift_hamiltonian), effective field and gradient
oot_rate_eq.py       J=0 -> J=1 steady-state rate equations for any 3x3
                     excited-state Hamiltonian; matches pylcp.rateeq to machine precision
```

## Model
| item | value |
|---|---|
| shift beams | 3 retro-reflected quad beams at 45 deg from z, w = 10 mm, 2 W over every pass (1 W of laser) |
| shift detuning | 10 Gamma_s, sign chosen so the MOT polarizations trap (`OOT_AUTO_SIGN`) |
| light shift | second order, summed incoherently over the 12 circular components, 1/Delta regularised by Gamma_s |
| Doppler | each shift component sees Delta - k_s . v (`OOT_DOPPLER=0` gives the paper's static B_eff) |
| atom beam | 45 deg below the XY plane, coming from the top, 25.5 deg azimuth; one triad axis antiparallel to it |
| gravity | projected onto the atom beam |
| slowing beam | off by default (`FT_SLOWER=1` to turn it on) |
| MOT reference | exact coil field, current scaled to the same central gradient (`OOT_MOT_CURRENT_A` to fix it) |

Every `FT_*` variable of `full_trap_sweep.py` still applies. The nozzle trace is
read from `atom_beam/nozzle_trace_cache.npz` with its own atom count unless
`FT_NOZZLE_ATOMS` is set.

| env | default | meaning |
|---|---|---|
| `OOT_BEAM_ELEV_DEG` | -45 | atom beam angle from the XY plane |
| `OOT_SHIFT_POWER_W` | 2.0 | shift power summed over every pass (paper's P) |
| `OOT_SHIFT_WAIST_M` | 0.01 | shift beam 1/e^2 radius |
| `OOT_SHIFT_DETUNING_GAMMA` | 10 | shift detuning in units of Gamma_s = 3.0e7 /s |
| `OOT_SHIFT_THETA_DEG` | 45 | shift beam angle from z |
| `OOT_N_QUAD` | 3 | quad beams (4 is the paper's pyramid) |
| `OOT_RETRO` | 1 | retro-reflected quad beams |
| `OOT_DOPPLER` | 1 | Doppler shift on the shift beams |
| `OOT_AUTO_SIGN` | 1 | flip the detuning sign if it anti-traps |
| `OOT_COMPARE_MOT` | 0 | also run the coil MOT in the same geometry |
| `OOT_MOT_CURRENT_A` | -1 | MOT reference current, -1 matches the OOT gradient |
| `OOT_DESIGN_SCAN` | 1 | shift power / waist / detuning scan on the centre ray |
