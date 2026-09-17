# Yb_mot_and_atom_MC_sim
Full Monte Carlo simulation of atom beam from oven and Magneto-Opticla Trapping (MOT).

## Before running
Please run with magpylib version > 5 since older versions use different unit conventions


## File purposes
 - full_trap_sweep.py runs the full finely spatially resolved MOT, most important file
 - yb_nozzle_3d.py runs the full atom MC simulation through nozzles
 - trap_beams.py, coil_field_model.py are helpers for the rest of things
 - yb_nozzle_beam_2d.py and yb174_mot_simulation.py are outdated but are still used as references and for units/constants

 ## How to run
python full_trap_sweep.py
If you wish to set different parameters, please edit the file, most flags are at the top
 or in the parameters section in main().


## Going to add:
- Radius sweep
- Multi-threaded acceleration
- Optimum map sweeps



## Maintainers + contact:
jaden.al-aidroos@mail.utoronto.ca --- GH @Jadenthealmighty
