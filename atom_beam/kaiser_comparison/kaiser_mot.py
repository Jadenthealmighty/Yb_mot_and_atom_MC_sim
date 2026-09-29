"""Letellier et al. (Kaiser group) 399 nm Yb MOT, built on full_trap_sweep."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(HERE))

for _k, _v in dict(FT_BEAM_TILT_DEG=45.0, FT_FLIGHT_MM=520.0,
                   FT_TRAP_HALF_MM=25.5, FT_OFFSET_STEP_MM=3.0,
                   FT_WAIST_M=0.022, FT_SLOWER=0, FT_NOZZLE_OFFSET_C=0.0,
                   FT_S_NEAR_MM=66.0, FT_S_PAST_MM=66.0, FT_S_NEAR_STEP_MM=1.5,
                   FT_S_FAR_STEP_MM=15.0, FT_V_RES_MS=2.5,
                   FT_RESERVED_CPUS=0,
                   RUNLOG_ROOT=os.path.join(HERE, "runs")).items():
    os.environ.setdefault(_k, str(_v))

import contextlib
import hashlib
import io
import json
import math
import multiprocessing as mp
import time

import numpy as np
import pylcp
import scipy.constants as sp_const

import full_trap_sweep as fts
import helpers.coil_field_model as cfm
import helpers.trap_beams as tb
import helpers.yb174_mot_simulation as sim

GAMMA_PAPER_HZ = 29e6
ISAT_PAPER_W_M2 = 600.0
CACHE_DIR = os.path.join(HERE, "cache")


def mot_config(name, grad_G_cm, detuning_gamma, P_h_mW, P_v_mW, waist_m=0.022):
    """One MOT setting. grad is the strong (vertical) axis, detuning in paper Gamma."""
    return dict(name=name, grad_G_cm=grad_G_cm,
                detuning_hz=detuning_gamma * GAMMA_PAPER_HZ,
                P_h_mW=P_h_mW, P_v_mW=P_v_mW, waist_m=waist_m)


def s_delta(cfg):
    """Paper's total saturation parameter at the detuning."""
    i_tot = (4 * cfg["P_h_mW"] + 2 * cfg["P_v_mW"]) * 1e-3 / (
        math.pi * cfg["waist_m"] ** 2) / ISAT_PAPER_W_M2
    return i_tot / (1 + 4 * (cfg["detuning_hz"] / GAMMA_PAPER_HZ) ** 2)


def quadrupole(grad_G_cm):
    g = grad_G_cm * 1e-2

    def B(xyz):
        xyz = np.asarray(xyz, dtype=float)
        return g * np.array([-xyz[0] / 2, -xyz[1] / 2, xyz[2]])
    return B


def build_eqn(norm, cfg):
    """Rate-equation MOT: 19/12 mW style split, linear quadrupole, 45 deg beam."""
    B = cfm.make_pylcp_magfield(quadrupole(cfg["grad_G_cm"]), norm["x0"],
                                norm["Gamma_SI"], gJ_excited=sim.YB174_GJ_EXCITED)
    eps = sim.MAGFIELD_GRADIENT_EPS_UM * 1e-6 / norm["x0"]
    magField = fts.MemoMagField(pylcp.magField(B, eps=eps))
    w = cfg["waist_m"]
    delta = 2 * np.pi * cfg["detuning_hz"] / norm["Gamma_SI"]
    beams = []
    for k, p, P in zip(tb.MOT_KVECS, tb.MOT_POLS(+1),
                       [cfg["P_h_mW"]] * 4 + [cfg["P_v_mW"]] * 2):
        s = 2 * P * 1e-3 / (math.pi * w ** 2) / norm["Isat_SI"]
        beams.append(tb.OffsetGaussianBeam(kvec=k, pol=p, s=s, delta=delta,
                                           wb=w / norm["x0"]))
    ham = sim.build_hamiltonian(norm)
    a_g = -sp_const.g * norm["t0"] ** 2 / norm["x0"]
    return fts.MemoCulledRateEq(pylcp.laserBeams(beams), magField, ham,
                                a=np.array([0.0, 0.0, a_g]),
                                include_mag_forces=True)


_W = {}


def _init(spec):
    with contextlib.redirect_stdout(io.StringIO()):
        _W.update(spec, eqn=build_eqn(spec["norm"], spec["cfg"]))


def _cell(task):
    i, j, bh, bv = task
    return i, j, fts._force_cell(_W["eqn"], _W["norm"], bh, bv, _W["s_mm"],
                                 _W["v_ms"])


def force_grid(norm, cfg, v_max_ms=90.0, workers=None, say=print):
    offs = fts.build_offset_grid()[0]
    s_mm = fts.build_axial_grid()
    v_ms = np.arange(-fts.V_NEG_MS, v_max_ms + 1e-9, fts.V_RESOLUTION_MS)
    a = np.zeros((offs.size, offs.size, s_mm.size, v_ms.size))
    tasks = [(i, j, bh, bv) for i, bh in enumerate(offs)
             for j, bv in enumerate(offs)]
    t0 = time.time()
    spec = dict(norm=norm, cfg=cfg, s_mm=s_mm, v_ms=v_ms)
    with mp.get_context("spawn").Pool(workers or os.cpu_count(), _init,
                                      (spec,)) as pool:
        for n, (i, j, cell) in enumerate(pool.imap_unordered(_cell, tasks), 1):
            a[i, j] = cell
            if n % max(1, len(tasks) // 5) == 0:
                say(f"  {cfg['name']}: {n}/{len(tasks)} cells, "
                    f"{time.time() - t0:.0f} s")
    return offs, s_mm, v_ms, a


def _key(cfg, v_max_ms):
    blob = json.dumps(dict(cfg, v_max=v_max_ms, half=fts.TRAP_HALF_MM,
                           step=fts.OFFSET_STEP_MM, s=list(fts.build_axial_grid()),
                           cap=fts.CAPTURE_AXIAL_MM), sort_keys=True)
    return hashlib.md5(blob.encode()).hexdigest()[:10]


def capture_map(run, norm, cfg):
    """(offs, s, v, a_grid, vc, t_stop), cached per config; widens v if saturated."""
    v_max = 90.0
    while True:
        path = os.path.join(CACHE_DIR, f"{cfg['name']}_{_key(cfg, v_max)}.npz")
        if os.path.exists(path):
            z = np.load(path)
            out = tuple(z[k] for k in ("offs", "s_mm", "v_ms", "a", "vc", "t_stop"))
        else:
            offs, s_mm, v_ms, a = force_grid(norm, cfg, v_max, say=run.say)
            vc, ts = fts.build_capture_map(run, norm, offs, s_mm, v_ms, a,
                                           verbose=False)
            os.makedirs(CACHE_DIR, exist_ok=True)
            np.savez_compressed(path, offs=offs, s_mm=s_mm, v_ms=v_ms, a=a,
                                vc=vc, t_stop=ts)
            out = (offs, s_mm, v_ms, a, vc, ts)
        if out[4].max() < 0.85 * v_max or v_max >= 360:
            return out
        v_max *= 2


def load_rate(data, offs, vc, ts, T_c, flux_in, abundance):
    """174Yb loading rate (atoms/s) at oven temperature T_c."""
    aux = fts.make_aux(data, offs, vc, ts)
    p = fts.capture_probability(data, aux, T_c)
    return flux_in * abundance * p.sum() / data["n_launched"]
