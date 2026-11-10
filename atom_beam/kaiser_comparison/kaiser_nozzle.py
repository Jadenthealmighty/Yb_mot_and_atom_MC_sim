"""Letellier et al. RSI 94, 123203 (2023) microtube nozzle and beamline."""

import math
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import helpers.yb_nozzle_beam_2d as m2d
import helpers.yb_nozzle_beam_3d as n3

TUBE_LENGTH_M = 13e-3
TUBE_RADIUS_M = 125e-6
TUBE_PITCH_M = 340e-6
N_ROWS = 38
N_ROWS_CUT = 8

OVEN_PROBE_M = 0.110
NOZZLE_TO_CELL_M = 0.520
APERTURES_M = ((0.131, 0.171, 6e-3), (0.470, 0.470, 0.050 * 0.470))

YB174_ABUNDANCE = 0.3183
LAMBDA_BLUE_M = 398.911e-9
LAMBDA_GREEN_M = 555.802e-9
GAMMA_BLUE_HZ = 29e6
GAMMA_GREEN_HZ = 182e3
PROBE_HEIGHT_M = 2e-3

T_OFFSET = 0

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache",
                     "kaiser_nozzle_trace.npz")


def tube_centres():
    """(y, z) of every tube, triangle apex down and flattened."""
    dz = TUBE_PITCH_M * math.sqrt(3) / 2
    pts = [((i - (k - 1) / 2) * TUBE_PITCH_M, (k - 1) * dz)
           for k in range(N_ROWS_CUT + 1, N_ROWS + 1) for i in range(k)]
    pts = np.array(pts)
    return pts - pts.mean(axis=0)


def open_area_m2():
    return len(tube_centres()) * math.pi * TUBE_RADIUS_M ** 2


def _trace_chunk(args):
    seed, n = args
    rng = np.random.default_rng(seed)
    g = n3.geom_circle(TUBE_RADIUS_M, TUBE_LENGTH_M)
    cfg = dict(t_gas=673.0, t_wall=673.0, specular_frac=0.0, stick_prob=0.0,
               max_bounces=n3.MAX_BOUNCES)
    out = n3.trace_batch_3d(rng, n, g, cfg)
    tx = out["status"] == n3.ST_TRANSMIT
    c = tube_centres()[rng.integers(0, len(tube_centres()), int(tx.sum()))]
    return dict(y0=(out["exit_y"][tx] + c[:, 0]).astype(np.float32),
                z0=(out["exit_z"][tx] + c[:, 1]).astype(np.float32),
                dx=out["exit_dx"][tx], dy=out["exit_dy"][tx],
                dz=out["exit_dz"][tx], hits=out["hits"][tx].astype(np.int32),
                n=n)


def trace(n_launch=200_000_000, chunk=250_000, seed=20231220, workers=None):
    """Transmitted atoms leaving the nozzle face, cached."""
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        if int(z["n_launched"]) >= n_launch:
            return {k: z[k] for k in z.files}
    tasks = [(seed + i, chunk) for i in range(n_launch // chunk)]
    workers = workers or max(1, os.cpu_count() or 1)
    with mp.get_context("spawn").Pool(workers) as pool:
        parts = pool.map(_trace_chunk, tasks, chunksize=1)
    tr = {k: np.concatenate([p[k] for p in parts])
          for k in ("y0", "z0", "dx", "dy", "dz", "hits")}
    tr["n_launched"] = np.asarray(sum(p["n"] for p in parts))
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **tr)
    return tr


def position_at(tr, x_m):
    return (tr["y0"] + x_m * tr["dy"] / tr["dx"],
            tr["z0"] + x_m * tr["dz"] / tr["dx"])


def passes_apertures(tr):
    ok = np.ones(tr["dx"].size, bool)
    for a, b, r in APERTURES_M:
        for x in (a, b):
            y, z = position_at(tr, x)
            ok &= y * y + z * z <= r * r
    return ok


def trap_data(tr, half_mm):
    """Arrivals at the trap plane in the dict layout full_trap_sweep expects."""
    ok = passes_apertures(tr)
    y, z = position_at(tr, NOZZLE_TO_CELL_M)
    y, z = y * 1e3, z * 1e3
    sel = ok & (np.abs(y) <= half_mm) & (np.abs(z) <= half_mm)
    rng = np.random.default_rng(1)
    speed = m2d.sample_flux_speed(rng, int(sel.sum()), 673.0)
    n_l = int(tr["n_launched"])
    return dict(y=y[sel].astype(np.float32), z=z[sel].astype(np.float32),
                dx=tr["dx"][sel].astype(np.float32),
                dperp=np.hypot(tr["dy"][sel], tr["dz"][sel]).astype(np.float32),
                hits=tr["hits"][sel], speed=speed.astype(np.float32),
                n_launched=n_l, n_tx=int(tr["dx"].size),
                n_accept=int(sel.sum()), W=tr["dx"].size / n_l,
                n_aperture=int(ok.sum()), open_area_m2=open_area_m2(),
                t_gas_ref=673.0, t_wall_ref=673.0)


def vapour_density(T_c):
    T = T_c - T_OFFSET + 273.15
    p = 1e5 * 10.0 ** (9.111 - 8111.0 / T - 1.0849 * np.log10(T))
    return p / (m2d.KB * T)


def flux_into_tubes(T_c):
    T = T_c - T_OFFSET + 273.15
    vbar = math.sqrt(8 * m2d.KB * T / (math.pi * m2d.M_YB))
    return 0.25 * vapour_density(T_c) * vbar * open_area_m2()


def _slab(tr, T_c, x_m, mask=None, reps=8, seed=7):
    """Atoms in a thin horizontal slab at x_m: (weight / v_x, v_y) per sample."""
    y, z = position_at(tr, x_m)
    sel = np.abs(z) <= PROBE_HEIGHT_M / 2
    if mask is not None:
        sel &= mask
    dx, dy = np.repeat(tr["dx"][sel], reps), np.repeat(tr["dy"][sel], reps)
    v = m2d.sample_flux_speed(np.random.default_rng(seed), dx.size, T_c + 273.15)
    w = flux_into_tubes(T_c) * YB174_ABUNDANCE / (float(tr["n_launched"]) * reps)
    return w / (v * dx * PROBE_HEIGHT_M), v * dy


def oven_spectrum(tr, T_c, nu_mhz):
    """556 nm absorption (%) vs detuning at the oven cross (Fig. 4a)."""
    dens, vy = _slab(tr, T_c, OVEN_PROBE_M)
    edges = np.concatenate([nu_mhz - np.diff(nu_mhz)[0] / 2,
                            [nu_mhz[-1] + np.diff(nu_mhz)[0] / 2]])
    col, _ = np.histogram(vy / LAMBDA_GREEN_M / 1e6, bins=edges, weights=dens)
    sigma0 = 3 * LAMBDA_GREEN_M ** 2 / (2 * math.pi)
    od = col * sigma0 * math.pi * GAMMA_GREEN_HZ / 2 / (np.diff(edges) * 1e6)
    return 100 * (1 - np.exp(-od))


def cell_spectrum(tr, T_c, nu_mhz):
    """399 nm absorption (%) vs detuning at the cell centre (Fig. 5)."""
    dens, vy = _slab(tr, T_c, NOZZLE_TO_CELL_M, mask=passes_apertures(tr))
    sigma0 = 3 * LAMBDA_BLUE_M ** 2 / (2 * math.pi)
    shift = vy / LAMBDA_BLUE_M / 1e6
    od = np.array([np.sum(dens / (1 + 4 * ((f - shift) * 1e6 / GAMMA_BLUE_HZ) ** 2))
                   for f in nu_mhz]) * sigma0
    return 100 * (1 - np.exp(-od))


def cell_vperp_rms(tr, T_c):
    dens, vy = _slab(tr, T_c, NOZZLE_TO_CELL_M, mask=passes_apertures(tr))
    return math.sqrt(np.sum(dens * vy ** 2) / np.sum(dens))


def angular_density(tr, edges_rad):
    """Atoms per solid angle vs polar angle, normalised to the first bin."""
    th = np.arccos(np.clip(tr["dx"], -1, 1))
    h, _ = np.histogram(th, bins=edges_rad)
    dom = 2 * math.pi * (np.cos(edges_rad[:-1]) - np.cos(edges_rad[1:]))
    j = h / dom
    return j / j[0]


def hwhm(x, y):
    y = np.asarray(y) / np.max(y)
    i = int(np.argmax(y < 0.5))
    return float(np.interp(0.5, [y[i], y[i - 1]], [x[i], x[i - 1]]))
