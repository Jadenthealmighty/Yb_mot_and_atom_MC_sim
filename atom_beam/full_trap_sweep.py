"""End-to-end Yb-174 trap-loading model: oven reservoir -> nozzle array ->
310 mm of free flight -> 399 nm MOT capture, swept over oven
temperatures from 350 to 500 C.

important note:

    u_hat = (cos 25.5, sin 25.5, 0)    along the atom beam
    e_h   = (-sin 25.5, cos 25.5, 0)   horizontal, across it
  e_v   = (0, 0, 1)                  vertical, across it

"""

import os


def _pin_blas_threads():
    """One BLAS thread per process, set BEFORE numpy is imported...

    Every grid point is a small linear solve, so BLAS threading doesn't make faster"""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")


os.environ.setdefault("MPLBACKEND", "Agg")
_pin_blas_threads()

import contextlib
import io
import math
import multiprocessing as mp
import time

import CyRK
import matplotlib.pyplot as plt
import numba
import numpy as np
import scipy.constants as sp_const
from matplotlib.colors import LogNorm
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

import pylcp
import runlog

import helpers.trap_beams as tb
import helpers.yb_nozzle_beam_3d as n3
import helpers.yb174_mot_simulation as sim

KB = sp_const.k


def _envf(name, default):
    return float(os.environ.get(name, default))


def _envi(name, default):
    return int(float(os.environ.get(name, default)))

T_MIN_C = _envf("FT_T_MIN_C", 350.0)
T_MAX_C = _envf("FT_T_MAX_C", 500.0)
T_STEP_C = _envf("FT_T_STEP_C", 1.0)

# based on Ziqing's simulations
NOZZLE_OFFSET_C = _envf("FT_NOZZLE_OFFSET_C", 30.0)


# roughly what I found on fusion model
NOZZLE_TO_TRAP_MM = _envf("FT_FLIGHT_MM", 344.0)
TRAP_HALF_MM = _envf("FT_TRAP_HALF_MM", 10.0)

BEAM_TILT_DEG = _envf("FT_BEAM_TILT_DEG", 25.5)

BEAM_WAIST_M = _envf("FT_WAIST_M", 0.005)
BEAM_AVG_SAT = _envf("FT_AVG_SAT", 0.1)

NOZZLE_ATOMS = _envi("FT_NOZZLE_ATOMS", 1_200_000_000)
NOZZLE_CHUNKS = _envi("FT_NOZZLE_CHUNKS", 1500)
NOZZLE_SEED = _envi("FT_NOZZLE_SEED", 20260903)

ACCEPT_MARGIN = _envf("FT_ACCEPT_MARGIN", 1.6)

NOZZLE_CACHE = os.environ.get("FT_NOZZLE_CACHE", "nozzle_trace_cache.npz")
NOZZLE_CACHE_ENABLED = bool(_envi("FT_NOZZLE_CACHE_ON", 1))

# Skip beam if saturation is below this fraction of nominal Isat
CULL_RTOL = _envf("FT_CULL_RTOL", 1e-9)

# Trajectory integrator, passed to CyRK. RK23 measured 9.2x over scipy RK45
# with identical capture velocities; "RK45", "DOP853", "LSODA" also valid.
IVP_METHOD = os.environ.get("FT_IVP_METHOD", "RK23")

SLOWER_ENABLED = bool(_envi("FT_SLOWER", 1))
SLOWER_OPTIC_MM = _envf("FT_SLOWER_OPTIC_MM", 500.0)
SLOWER_W_OPTIC_M = _envf("FT_SLOWER_W_OPTIC_M", 0.005)
SLOWER_DIVERGENCE_MRAD = _envf("FT_SLOWER_DIV_MRAD", 0.0)
SLOWER_FOCUS_MM = _envf("FT_SLOWER_FOCUS_MM", 0.0)
SLOWER_AVG_SAT = _envf("FT_SLOWER_AVG_SAT", 0.5)
SLOWER_DETUNING_HZ = _envf("FT_SLOWER_DETUNING_HZ", -104048000)
SLOWER_POL = _envi("FT_SLOWER_POL", -1)
SLOWER_TILT_MDEG = (_envf("FT_SLOWER_TILT1_MDEG", 0.0),
                    _envf("FT_SLOWER_TILT2_MDEG", 0.0))
SLOWER_OFFSET_MM = (_envf("FT_SLOWER_OFF_X_MM", 0.0),
                    _envf("FT_SLOWER_OFF_Y_MM", 0.0),
                    _envf("FT_SLOWER_OFF_Z_MM", 0.0))

MOT_BEAM_OFFSET_MM = {
}
MOT_BEAM_TILT_MDEG = {
}


OPT_CURRENT_A = _envf("FT_COIL_CURRENT_A", 42.0)

TOLERANCE_ENABLED = bool(_envi("FT_TOLERANCE", 0))
TOL_MARGIN_MM = _envf("FT_TOL_MARGIN_MM", 3.0)
TIME_BUDGET_H = _envf("FT_TIME_BUDGET_H", 1)
TOL_N_V = _envi("FT_TOL_N_V", 41)
TOL_TEMP_C = _envf("FT_TOL_TEMP_C", 425.0)
TOL_MOT_OFFSET_MM = tuple(float(x) for x in os.environ.get(
    "FT_TOL_MOT_OFFSETS", "0,0.05,0.1,0.2,0.4,0.8,1.6").split(","))
TOL_MOT_BEAM_INDEX = _envi("FT_TOL_MOT_BEAM", 0)
TOL_SLOWER_DIV_MRAD = tuple(float(x) for x in os.environ.get(
    "FT_TOL_SLOWER_DIVS", "0,1,2,4,8,16,32").split(","))
SLOWER_DETUNING_SCAN = bool(_envi("FT_SLOWER_SCAN", 0))
TOL_SLOWER_DETUNING_GAMMA = tuple(float(x) for x in os.environ.get(
    "FT_SLOWER_DETUNINGS", "-0.5,-1,-2,-3,-4,-6").split(","))

OFFSET_STEP_MM = _envf("FT_OFFSET_STEP_MM", 0.75)

S_PAST_TRAP_MM = _envf("FT_S_PAST_MM", 25.0)
S_NEAR_MM = _envf("FT_S_NEAR_MM", 30.0)
S_NEAR_STEP_MM = _envf("FT_S_NEAR_STEP_MM", 1.5)
S_FAR_STEP_MM = _envf("FT_S_FAR_STEP_MM", 10.0)

V_MAX_MS = _envf("FT_V_MAX_MS", -1.0)
N_V = _envi("FT_N_V", -1)
V_RESOLUTION_MS = _envf("FT_V_RES_MS", 2.5)
V_NEG_MS = _envf("FT_V_NEG_MS", 30.0)
V_AUTO_ESCALATE = bool(_envi("FT_V_ESCALATE", 1))
V_MAX_GRID_POINTS = _envi("FT_V_MAX_POINTS", 161)

FIELD_CHUNK = _envi("FT_FIELD_CHUNK", 4000)

N_WORKERS = _envi("FT_N_WORKERS", 0)
RESERVED_CPUS = _envi("FT_RESERVED_CPUS", 2)

TRAJ_T_MAX_BAR = _envf("FT_TRAJ_T_MAX_BAR", 8.0e6)
CAPTURE_RADIUS_MM = BEAM_WAIST_M * 1000 #_envf("FT_CAPTURE_RADIUS_MM", 5.0)
CAPTURE_SPEED_MS = _envf("FT_CAPTURE_SPEED_MS", 3.0)

N_V_SCAN = _envi("FT_N_V_SCAN", 24)
N_V_BISECT = _envi("FT_N_V_BISECT", 12)

LIFETIMES_S = tuple(float(x) for x in
                    os.environ.get("FT_LIFETIMES", "1,5,20").split(","))

DETUNING_HZ = _envf("FT_DETUNING_HZ", sim.NOMINAL_DETUNING_HZ)

SEED = _envi("FT_SEED", 20260806)


TILT = math.radians(BEAM_TILT_DEG)
U_HAT = np.array([math.cos(TILT), math.sin(TILT), 0.0])
E_H = np.array([-math.sin(TILT), math.cos(TILT), 0.0])
E_V = np.array([0.0, 0.0, 1.0])

TRAP_C = "#1f77b4"
LOSS_C = "#d62728"
ACCENT = "#ff7f0e"


def yb_vapour_pressure_pa(temp_k):
    """Yb vapour pressure, from some paper, see my report....
        log10(P/bar) = 9.111 - 8111/T - 1.0849 * log10(T)

    """
    return 1e5 * 10.0 ** (9.111 - 8111.0 / temp_k
                          - 1.0849 * np.log10(temp_k))


def yb_number_density(temp_k):
    """ideal gas"""
    return yb_vapour_pressure_pa(temp_k) / (KB * temp_k)


def mean_free_path_m(temp_k, diameter_m=4.0e-10):
    """ I chose 1 since I'm a chud but 10 is the standard, also 4e-10 is my guess,
    someone please double check"""
    p = yb_vapour_pressure_pa(temp_k)
    return KB * temp_k / (math.sqrt(2.0) * math.pi * diameter_m ** 2 * p)


def mean_speed(temp_k, mass=n3.M_YB):
    return np.sqrt(8 * KB * temp_k / (math.pi * mass))


def free_molecular_limit_C(length_m):
    lo, hi = 300.0 + 273.15, 700.0 + 273.15
    if (mean_free_path_m(lo) - length_m) * (mean_free_path_m(hi) - length_m) > 0:
        return float("nan")
    for _ in range(80):


        mid = 0.5 * (lo + hi)
        if (mean_free_path_m(lo) - length_m) * (mean_free_path_m(mid) - length_m) <= 0:
            hi = mid
        else:

            lo = mid
    return 0.5 * (lo + hi) - 273.15


def trace_nozzle_to_chamber(run, rng):
    """Trace the 81-channel array once, keeping atoms that arrive within the
    trap region (plus a margin) at the chamber.
 """
    g = n3.geom_dshape()
    cfg = dict(t_gas=n3.T_GAS, t_wall=n3.T_WALL,
               specular_frac=n3.SPECULAR_FRAC, stick_prob=n3.STICK_PROB,
               max_bounces=n3.MAX_BOUNCES)

    d = NOZZLE_TO_TRAP_MM * 1e-3
    accept_mm = TRAP_HALF_MM * ACCEPT_MARGIN

    per_chunk = max(1, NOZZLE_ATOMS // NOZZLE_CHUNKS)
    keep = {k: [] for k in ("y", "z", "dx", "dperp", "speed", "hits")}
    n_launched = 0
    n_tx = 0
    n_direct = 0
    n_accept = 0
    conv_steps, conv_accept = [], []

    t_start = time.time()
    run.say(f"tracing {per_chunk * NOZZLE_CHUNKS:,} atoms through the "
            f"{n3.N_CHAN_Y}x{n3.N_CHAN_Z} channel array "
            f"(L/D_h = {n3.L / n3.D_HYD:.1f})")

    for c in range(NOZZLE_CHUNKS):
        out = n3.trace_batch_3d(rng, per_chunk, g, cfg)
        st = out["status"]
        tx = st == n3.ST_TRANSMIT

        n_launched += st.size
        k = int(tx.sum())
        n_tx += k
        hits = out["hits"][tx]
        n_direct += int((hits == 0).sum())
        if k == 0:
            continue

        dx = out["exit_dx"][tx]
        dy = out["exit_dy"][tx]

        dz = out["exit_dz"][tx]
        ey = out["exit_y"][tx]
        ez = out["exit_z"][tx]
        speed = out["exit_speed"][tx]

        iy = rng.integers(0, n3.N_CHAN_Y, size=k)
        iz = rng.integers(0, n3.N_CHAN_Z, size=k)
        oy = (iy - (n3.N_CHAN_Y - 1) / 2.0) * n3.PITCH_Y
        oz = (iz - (n3.N_CHAN_Z - 1) / 2.0) * n3.PITCH_Z


        forward = dx > 1e-6
        y_mm = (ey + oy + d * np.where(forward, dy / np.where(forward, dx, 1.0), 0.0)) * 1e3
        z_mm = (ez + oz + d * np.where(forward, dz / np.where(forward, dx, 1.0), 0.0)) * 1e3

        sel = forward & (np.abs(y_mm) <= accept_mm) & (np.abs(z_mm) <= accept_mm)
        m = int(sel.sum())
        n_accept += m
        if m:
            keep["y"].append(y_mm[sel].astype(np.float32))
            keep["z"].append(z_mm[sel].astype(np.float32))
            keep["dx"].append(dx[sel].astype(np.float32))
            keep["dperp"].append(np.hypot(dy[sel], dz[sel]).astype(np.float32))
            keep["speed"].append(speed[sel].astype(np.float32))
            keep["hits"].append(hits[sel].astype(np.int32))

        conv_steps.append(n_launched)
        conv_accept.append(n_accept / n_launched)

        run.log(step=n_launched,
                nozzle_transmission_probability=n_tx / n_launched,
                nozzle_direct_flight_fraction=n_direct / n_launched,
                fraction_launched_reaching_trap_region=n_accept / n_launched,
                fraction_transmitted_reaching_trap_region=n_accept / max(n_tx, 1))

        if (c + 1) % max(1, NOZZLE_CHUNKS // 12) == 0:
            elapsed = time.time() - t_start
            run.say(f"  {n_launched:,} launched, {n_tx:,} transmitted, "
                    f"{n_accept:,} reaching the trap region "
                    f"({elapsed:.0f} s, {n_launched / max(elapsed, 1e-9) / 1e3:.0f}k/s)")

    data = {k: (np.concatenate(v) if v else np.zeros(0, dtype=np.float32))
            for k, v in keep.items()}
    data["n_launched"] = n_launched
    data["n_tx"] = n_tx
    data["n_direct"] = n_direct
    data["n_accept"] = n_accept
    data["W"] = n_tx / n_launched
    data["t_gas_ref"] = n3.T_GAS
    data["t_wall_ref"] = n3.T_WALL
    data["conv_steps"] = np.asarray(conv_steps)
    data["conv_accept"] = np.asarray(conv_accept)
    data["open_area_m2"] = n3.N_CHAN_Y * n3.N_CHAN_Z * n3.AREA
    data["elapsed_s"] = time.time() - t_start
    return data


def _nozzle_cache_key():
    """IMPORTANT, i forgot NOZZLE_SEED does not need to change with trap seed"""
    return dict(
        atoms=NOZZLE_ATOMS, seed=NOZZLE_SEED,
        flight_mm=NOZZLE_TO_TRAP_MM, trap_half_mm=TRAP_HALF_MM,
        accept_margin=ACCEPT_MARGIN,
        width_mm=n3.RECT_WIDTH_MM, height_mm=n3.RECT_HEIGHT_MM,
        cap_mm=n3.CAP_RADIUS_MM, length_mm=n3.CHANNEL_LENGTH_MM,
        pitch_y=n3.PITCH_Y_MM, pitch_z=n3.PITCH_Z_MM,
        nchan_y=n3.N_CHAN_Y, nchan_z=n3.N_CHAN_Z,
        t_gas=n3.T_GAS, t_wall=n3.T_WALL,
        specular=n3.SPECULAR_FRAC, stick=n3.STICK_PROB,
    )


def nozzle_cache_mismatch():
    """Settings the cached nozzle trace disagrees on: [] means reusable, None
    means there is no cache. beam_sweep.py checks this before launching."""
    if not (NOZZLE_CACHE_ENABLED and os.path.exists(NOZZLE_CACHE)):
        return None
    try:
        z = np.load(NOZZLE_CACHE, allow_pickle=True)
        cached = {k[4:]: z[k].item() for k in z.files if k.startswith("key_")}
    except Exception as exc:
        return [f"<unreadable: {exc}>"]
    return [k for k, v in _nozzle_cache_key().items()
            if k not in cached or not np.isclose(float(cached[k]), float(v))]


def load_or_trace_nozzle(run, rng):
    """Added so that I would not need to rerun, please use the stuff in pre_computed"""
    key = _nozzle_cache_key()
    mismatch = nozzle_cache_mismatch()
    if mismatch is not None:
        if not mismatch:
            z = np.load(NOZZLE_CACHE, allow_pickle=True)
            data = {k: z[k] for k in ("y", "z", "dx", "dperp", "speed", "hits",
                                      "conv_steps", "conv_accept")}
            for k in ("n_launched", "n_tx", "n_direct", "n_accept", "W",
                      "t_gas_ref", "t_wall_ref", "open_area_m2", "elapsed_s"):
                data[k] = z[k].item()
            run.say(f"reusing cached nozzle trace from {NOZZLE_CACHE}: "
                    f"{data['n_accept']:,} usable atoms from "
                    f"{data['n_launched']:,} launched "
                    f"(saves ~{data['elapsed_s'] / 60:.0f} min)")
            return data
        run.say(f"nozzle cache exists but does not match on {mismatch}; "
                "retracing")

    data = trace_nozzle_to_chamber(run, rng)
    if NOZZLE_CACHE_ENABLED:
        out = {k: np.asarray(v) for k, v in data.items()}
        out.update({f"key_{k}": np.asarray(float(v)) for k, v in key.items()})
        np.savez_compressed(NOZZLE_CACHE, **out)
        run.say(f"nozzle trace cached to {NOZZLE_CACHE}; later runs will "
                "skip straight to the MOT stage")
    return data


class MemoMagField(pylcp.magField):
    """Caches grad|B| per position: asked every point, changes every n_v.
    Wraps, not patches, so pylcp's six central differences run on the inner
    field and cannot evict the entry. One instance per process.
    """

    def __init__(self, inner):
        self._inner, self.eps, self.Field = inner, inner.eps, inner.Field
        self._k = self._G = None

    def gradFieldMag(self, R=np.array([0., 0., 0.]), t=0):
        k = (R[0], R[1], R[2])
        if k != self._k:
            self._k, self._G = k, self._inner.gradFieldMag(R, t)
        return self._G


def beam_relevance(laser_beams, positions, rtol):
    """Where each beam carries light: {key: (n_positions, n_beams) bool}.
    The MOT waists sit at the origin, so most of a 369 mm ray is dark.
    """
    out = {}
    for key in laser_beams:
        I = np.array([[b.intensity(p, 0.) for b in laser_beams[key].beam_vector]
                      for p in positions])
        out[key] = I > rtol * I.max()
    return out


class MemoCulledRateEq(pylcp.rateeq):
    """Caches the position-only half of the pumping rate; skips dark beams.

        R_l = num_l / (1 + 4 (A_l - k_l . v)^2 / gamma^2)

    num_l = gamma s_l f_ijq / 2 and A_l = -(E2 - E1) + delta_l are built once
    per position, not n_v times; only k_l . v varies. Association order kept,
    so the memo is bitwise exact, only culling approximates (CULL_RTOL).
    arm() per ray, mask=None memoises only. Split by cell not velocity,
    one instance per process.
    """

    _mask = None
    _ptr = 0
    _stride = 1
    _pk = None
    _pc = None

    def arm(self, mask, stride):
        self._mask = mask
        self._stride = stride
        self._ptr = 0
        self._pk = None

    def _precompute(self, r, t, Bhat, live):
        """
        NOTE: mostly copied from pylcp.rateeq._calc_pumping_rates

        Adapted to precompute the position-dependent parts
        """
        out = {}
        for key in self.laserBeams:
            # Extract the relevant d_q matrix:
            ind = self.hamiltonian.rotated_hamiltonian.laser_keys[key]
            d_q = self.hamiltonian.rotated_hamiltonian.blocks[ind].matrix
            gamma = self.hamiltonian.blocks[ind].parameters['gamma']

            # Extract the energies:
            E1 = np.diag(self.hamiltonian.rotated_hamiltonian.blocks[ind[0],ind[0]].matrix)
            E2 = np.diag(self.hamiltonian.rotated_hamiltonian.blocks[ind[1],ind[1]].matrix)

            E2, E1 = np.meshgrid(E2, E1)

            # Initialize the pumping matrix:  (shape only, allocated per point)
            shape = (len(self.laserBeams[key].beam_vector),) + d_q.shape[1:]

            # Grab the laser parameters:
            kvecs = self.laserBeams[key].kvec(r, t)
            intensities = self.laserBeams[key].intensity(r, t)
            deltas = self.laserBeams[key].delta(t)

            projs = self.laserBeams[key].project_pol(Bhat, R=r, t=t)

            # Loop through each laser beam driving this transition:
            rows = []
            for ll, (kvec, intensity, proj, delta) in enumerate(zip(kvecs, intensities, projs, deltas)):
                if live is not None and not live[key][ll]:
                    continue                    # carries no light here
                fijq = np.abs(d_q[0]*proj[2] + d_q[1]*proj[1] +d_q[2]*proj[0])**2

                # Finally, calculate the scattering rate the polarization
                # onto the appropriate basis:  (everything except k.v)
                rows.append((ll, kvec, gamma, gamma*intensity/2*fijq,
                             -(E2 - E1) + delta))
            out[key] = (shape, rows)
        return out

    def _pump_from_cache(self, v, cache):
        """The k.v remainder of _calc_pumping_rates, per point."""
        for key, (shape, rows) in cache.items():
            self.Rijl[key] = np.zeros(shape)
            for ll, kvec, gamma, num, A in rows:
                self.Rijl[key][ll] = num/(1 + 4*(A - np.dot(kvec, v))**2/gamma**2)

    def construct_evolution_matrix(self, r, v, t=0.,
                                   default_axis=np.array([0., 0., 1.])):
        '''
        Mostly copied from  pylcp.rateeq.construct_evolution_matrix
        '''
        pos = (r[0], r[1], r[2], t)
        if pos != self._pk:
            self._pk = pos
            B = self.magField.Field(r)

            # Calculate its magnitude:
            Bmag = np.linalg.norm(B, axis=0)

            # Calculate the Bhat direction:
            if Bmag > 1e-10:
                Bhat = B/Bmag
            else:
                Bhat = default_axis

            # Diagonalize the hamiltonian at this location:
            self.hamiltonian.diag_static_field(Bmag)

            if not np.all(self.hamiltonian.diagonal):
                # Reconstruct the decay matrix to match this new field.
                self.Rev_decay = self._calc_decay_comp_of_Rev(
                    self.hamiltonian.rotated_hamiltonian
                )

            # Recalculate the pumping rates:  (the velocity-free half)
            live = None
            if self._mask is not None:
                live = {key: self._mask[key][self._ptr // self._stride]
                        for key in self._mask}
            self._pc = self._precompute(r, t, Bhat, live)
        self._ptr += 1

        # Re-initialize the evolution matrix:
        self.Rev = np.zeros((self.hamiltonian.n, self.hamiltonian.n))

        self.Rev += self.Rev_decay

        # Recalculate the pumping rates:  (the k.v half)
        self._pump_from_cache(v, self._pc)

        # Add the pumping rates to the evolution matrix:
        self._add_pumping_rates_to_Rev()

        return self.Rev, self.Rijl


def rescale_speeds(data, t_gas_k, t_wall_k):
    fg = math.sqrt(t_gas_k / data["t_gas_ref"])
    fw = math.sqrt(t_wall_k / data["t_wall_ref"])
    return np.where(data["hits"] == 0, fg, fw).astype(np.float32)


def flux_speed_cdf(v, v_p):
    """Maxwellk FLUX distribution"""
    v = np.asarray(v, dtype=float)
    v_p = np.asarray(v_p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        x = np.square(np.where(v_p > 0, v / np.where(v_p > 0, v_p, 1.0), 0.0))
    x = np.clip(np.nan_to_num(x, nan=0.0, posinf=1e3), 0.0, 1e3)
    return -np.expm1(-x) - x * np.exp(-x)


def capture_probability(data, aux, T_c):
    """Capture probability per stored atom, integrating the speed
    distribution ANALYTICALLY.

    two capture conditions:

        axial       v * cos(theta) < v_c(y, z)
        transverse  v * sin(theta) * t_stop(y, z) < capture radius

    An atom is caught if v is below the smaller of the two, so the joint
    probability is a single CDF evaluation
    """
    t_gas = T_c + 273.15
    t_wall = T_c + NOZZLE_OFFSET_C + 273.15
    v_p = np.where(data["hits"] == 0,
                   math.sqrt(2 * KB * t_gas / n3.M_YB),
                   math.sqrt(2 * KB * t_wall / n3.M_YB))

    dx = np.maximum(data["dx"], 1e-9)
    dperp = np.maximum(data["dperp"], 1e-12)

    v_lim_axial = aux["vc_atom"] / dx
    denom = dperp * aux["ts_atom"]
    v_lim_trans = np.where(denom > 0,
                           CAPTURE_RADIUS_MM * 1e-3 / np.maximum(denom, 1e-30),
                           np.inf)

    v_max = np.minimum(v_lim_axial, v_lim_trans)
    v_max = np.where(aux["vc_atom"] > 0, v_max, 0.0)

    p = flux_speed_cdf(v_max, v_p)
    return np.where(aux["inside"], p, 0.0)


def build_offset_grid(step_mm=None, half_mm=None):
    """THE transverse sampling lattice, shared by every capture map """
    step_mm = OFFSET_STEP_MM if step_mm is None else step_mm
    half_mm = TRAP_HALF_MM if half_mm is None else half_mm
    n = max(1, int(round(2 * half_mm / step_mm)))
    w = 2 * half_mm / n
    return -half_mm + w * (np.arange(n) + 0.5), w


def offset_edges(offs_mm):
    """Bin edges derived frim the offset centres. Building edges as linspace(-half, half, n+1) while
    tabulating at linspace(-half, half, n) does NOT match, aghhh
    """
    offs_mm = np.asarray(offs_mm, dtype=float)
    if offs_mm.size == 1:
        w = 2 * TRAP_HALF_MM
    else:
        w = offs_mm[1] - offs_mm[0]
    return np.concatenate([offs_mm - 0.5 * w, [offs_mm[-1] + 0.5 * w]])


def bin_atoms(data, offs_mm):
    """Assign every stored atom to a cell of the capture map"""
    edges = offset_edges(offs_mm)
    n = offs_mm.size
    inside = ((data["y"] >= edges[0]) & (data["y"] <= edges[-1])
              & (data["z"] >= edges[0]) & (data["z"] <= edges[-1]))
    iy = np.clip(np.digitize(data["y"], edges) - 1, 0, n - 1)
    iz = np.clip(np.digitize(data["z"], edges) - 1, 0, n - 1)
    return inside, iy, iz


def make_aux(data, offs_mm, vc, t_stop):
    """Per-atom capture lookups. Shared by both paths, for the same reason."""
    inside, iy, iz = bin_atoms(data, offs_mm)
    return dict(inside=inside, iy=iy, iz=iz,
                vc_atom=vc[iy, iz], ts_atom=t_stop[iy, iz],
                v_axial_ref=data["speed"] * data["dx"],
                v_perp_ref=data["speed"] * data["dperp"])


def build_axial_grid():
    """fine near the trap, coarse over the upstream drift. """
    near = np.arange(-S_NEAR_MM, S_PAST_TRAP_MM + 1e-9, S_NEAR_STEP_MM)
    far = np.arange(-NOZZLE_TO_TRAP_MM, -S_NEAR_MM, S_FAR_STEP_MM)
    s = np.unique(np.concatenate([far, near]))
    return s


def json_safe(obj):
    """Replace non-finite floats with None so a summary is JSON-serialisable, 
    so much easier in python than Java lmao"""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def auto_velocity_grid(norm, slower_detuning_hz):
    """Size the velocity grid: returns (v_min, v_max, n_v, v_res).

    Three competing requirements, all enforced explicitly:

      -REACH. The grid must extend past the largest capture velocity, or
    RegularGridInterpolator extrapolates and manufactures captures. 

      -Spacing must be small against the Doppler resonance width
    Gamma/k = 12.05 m/s. A 30 m/s grid put the resonance between points,
    the optical force vanished, and every far-detuned scan point came
    back as exactly zero , an artefact, not physics.

    """
    k_SI = norm["k_SI"]
    v_res = abs(2 * np.pi * slower_detuning_hz) / k_SI if slower_detuning_hz else 0.0
    if V_MAX_MS > 0:
        v_max = V_MAX_MS
    else:
        floor = 200.0 if SLOWER_ENABLED else 90.0
        v_max = max(floor, 1.5 * v_res + 60.0)
    v_min = -abs(V_NEG_MS)
    if N_V > 0:
        n_v = N_V
    else:
        n_v = int((v_max - v_min) / V_RESOLUTION_MS) + 1
    n_v = max(n_v, 41)
    return v_min, v_max, n_v, v_res


def worker_count(n_tasks):
    """How many processes to fan out over, never more than there are tasks.

    FT_N_WORKERS sets it outright. Otherwise every logical CPU is used except
    FT_RESERVED_CPUS, which defaults to 2 so one full physical core (both its
    SMT threads) stays free for whatever else shares the machine.
    """
    n = N_WORKERS if N_WORKERS > 0 else (os.cpu_count() or 1) - RESERVED_CPUS
    return max(1, min(n, n_tasks))


_FIELD_TABLE = None


def extended_field_table():
    """Sample the 1 A coil field on a grid covering the whole flight path.

    Returns (points_m, B_tesla, info), cached after the first call and handed
    to worker processes so they do not each repeat the 128-loop sum.
    """
    global _FIELD_TABLE
    if _FIELD_TABLE is not None:
        return _FIELD_TABLE

    from helpers.coil_field_model import oswald_coil_bfield, REFERENCE_CURRENT_A

    b_max = TRAP_HALF_MM * 1.4 * 1e-3
    s_lo, s_hi = -NOZZLE_TO_TRAP_MM * 1e-3, S_PAST_TRAP_MM * 1e-3
    corners = []
    for s in (s_lo, s_hi):
        f = (s + NOZZLE_TO_TRAP_MM * 1e-3) / (NOZZLE_TO_TRAP_MM * 1e-3)
        for sh in (-1, 1):
            for sv in (-1, 1):
                corners.append(s * U_HAT + f * b_max * (sh * E_H + sv * E_V))
    corners = np.array(corners)
    lo = corners.min(axis=0) - 5e-3
    hi = corners.max(axis=0) + 5e-3

    def axis(a, b, fine=0.0015, coarse=0.010, near=0.030):
        parts = [np.arange(max(a, -near), min(b, near) + 1e-9, fine)]
        if a < -near:
            parts.append(np.arange(a, -near, coarse))
        if b > near:
            parts.append(np.arange(near, b + 1e-9, coarse))
        return np.unique(np.round(np.concatenate(parts), 9))

    xs = axis(lo[0], hi[0])
    ys = axis(lo[1], hi[1])
    zs = axis(lo[2], hi[2])
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    t0 = time.time()
    B_unit = np.empty_like(pts)
    for lo_i in range(0, pts.shape[0], FIELD_CHUNK):
        hi_i = min(lo_i + FIELD_CHUNK, pts.shape[0])
        B_unit[lo_i:hi_i] = oswald_coil_bfield(pts[lo_i:hi_i],
                                               REFERENCE_CURRENT_A)
    info = dict(n_points=pts.shape[0], build_s=time.time() - t0,
                shape=(xs.size, ys.size, zs.size),
                extent_mm=[[lo[i] * 1e3, hi[i] * 1e3] for i in range(3)])
    _FIELD_TABLE = (pts, B_unit, info)
    return _FIELD_TABLE


def build_extended_magfield(norm, table=None):
    """Tabulate the coil field over the WHOLE flight path, not just the trap

    Pass a table from extended_field_table() to skip the sampling"""
    from helpers.coil_field_model import (REFERENCE_CURRENT_A, grid_field_interpolator,
                                  make_pylcp_magfield)

    pts, B_unit, table_info = (extended_field_table() if table is None
                               else table)
    interp_unit = grid_field_interpolator(pts, B_unit)

    def B_interp(xyz_meters):
        return interp_unit(xyz_meters) * (OPT_CURRENT_A / REFERENCE_CURRENT_A)

    B_bar_func = make_pylcp_magfield(B_interp, norm["x0"], norm["Gamma_SI"],
                                     gJ_excited=sim.YB174_GJ_EXCITED)
    eps_bar = (sim.MAGFIELD_GRADIENT_EPS_UM * 1e-6) / norm["x0"]
    magField = MemoMagField(pylcp.magField(B_bar_func, eps=eps_bar))

    dz = 1e-4
    grad = ((B_interp(np.array([0., 0., dz]))[2]
             - B_interp(np.array([0., 0., -dz]))[2]) / (2 * dz) * 1e4 * 1e-2)
    return magField, dict(table_info, gradient_G_per_cm=grad)


def build_mot(norm, alignment=None, slower_detuning_hz=None,
              slower_divergence_mrad=None, magField=None):
    """Assemble the rate-equation object for a given alignment state """
    isat = norm["Isat_SI"]
    power_w = BEAM_AVG_SAT * isat * math.pi * BEAM_WAIST_M ** 2

    sim.LASER_BEAM_MODE = "gaussian"
    sim.GAUSSIAN_WAIST_M = BEAM_WAIST_M
    sim.GAUSSIAN_POWER_W = power_w

    hamiltonian = sim.build_hamiltonian(norm)
    if magField is None:
        magField, _ = build_extended_magfield(norm)

    delta_bar = 2 * np.pi * DETUNING_HZ / norm["Gamma_SI"]

    slower_cfg = None
    if SLOWER_ENABLED:
        d_hz = (slower_detuning_hz if slower_detuning_hz is not None
                else (DETUNING_HZ if math.isnan(SLOWER_DETUNING_HZ)
                      else SLOWER_DETUNING_HZ))
        avg_sat = SLOWER_AVG_SAT if SLOWER_AVG_SAT >= 0 else BEAM_AVG_SAT
        s_power = avg_sat * isat * math.pi * SLOWER_W_OPTIC_M ** 2
        div_mrad = (slower_divergence_mrad if slower_divergence_mrad is not None
                    else SLOWER_DIVERGENCE_MRAD)
        slower_cfg = dict(
            khat=-U_HAT,
            delta_bar=2 * np.pi * d_hz / norm["Gamma_SI"],
            power_w=s_power,
            w_optic_m=SLOWER_W_OPTIC_M,
            optic_distance_m=SLOWER_OPTIC_MM * 1e-3,
            divergence_rad=div_mrad * 1e-3,
            focus_distance_m=SLOWER_FOCUS_MM * 1e-3,
            pol=SLOWER_POL,
        )

    if alignment is None:
        alignment = tb.BeamAlignment(
            mot_offsets_m={i: np.asarray(v, dtype=float) * 1e-3
                           for i, v in MOT_BEAM_OFFSET_MM.items()},
            mot_tilts_rad={i: tuple(math.radians(a / 1e3) for a in v)
                           for i, v in MOT_BEAM_TILT_MDEG.items()},
            slower_tilt_rad=tuple(math.radians(a / 1e3)
                                  for a in SLOWER_TILT_MDEG),
            slower_offset_m=np.asarray(SLOWER_OFFSET_MM, dtype=float) * 1e-3,
        )

    laserBeams, beam_info = tb.build_trap_beams(
        norm, delta_bar, waist_m=BEAM_WAIST_M, power_w=power_w,
        alignment=alignment, slower=slower_cfg,
        wavelength_m=sim.YB174_WAVELENGTH_M)

    a_grav_bar = -sp_const.g * norm["t0"] ** 2 / norm["x0"]
    eqn = MemoCulledRateEq(laserBeams, magField, hamiltonian,
                           a=np.array([0.0, 0.0, a_grav_bar]),
                           include_mag_forces=True)
    beam_info.update(power_w=power_w, a_grav_bar=a_grav_bar)
    return eqn, beam_info


def mot_spec(norm, alignment=None, slower_detuning_hz=None,
             slower_divergence_mrad=None):
    """description of build_mot() call, have this cuz need all workers to have same

    pylcp objects hold closures over the field interpolator and do not pickle,
    so a worker gets the arguments and builds its own copy.
    """
    return dict(norm=norm, field_table=extended_field_table(),
                alignment=alignment, slower_detuning_hz=slower_detuning_hz,
                slower_divergence_mrad=slower_divergence_mrad)


_force_worker = {}


def _force_worker_init(spec):
    """Rebuild one rate-equation object per worker process"""
    with contextlib.redirect_stdout(io.StringIO()):
        magField, _ = build_extended_magfield(spec["norm"],
                                              table=spec["field_table"])
        eqn, _ = build_mot(spec["norm"], alignment=spec["alignment"],
                           slower_detuning_hz=spec["slower_detuning_hz"],
                           slower_divergence_mrad=spec["slower_divergence_mrad"],
                           magField=magField)
    _force_worker.update(spec)
    _force_worker["eqn"] = eqn


def _force_worker_cell(task):
    """One (offset h, offset v) plane of the force grid, in a worker."""
    i, j, bh, bv = task
    st = _force_worker
    return i, j, _force_cell(st["eqn"], st["norm"], bh, bv, st["s_mm"],
                             st["v_ms"])


def _force_cell(eqn, norm, bh_mm, bv_mm, s_mm, v_ms):
    """Axial acceleration over the (s, v_s) plane at one transverse offset."""
    x0, v0 = norm["x0"], norm["v0"]
    S, V = np.meshgrid(s_mm * 1e-3 / x0, v_ms / v0, indexing="ij")
    L_m = NOZZLE_TO_TRAP_MM * 1e-3
    F_S = np.repeat(((s_mm * 1e-3 + L_m) / L_m)[:, None], v_ms.size, axis=1)
    base = (bh_mm * 1e-3 * E_H + bv_mm * 1e-3 * E_V) / x0
    R = np.array([S * U_HAT[c] + F_S * base[c] for c in range(3)])
    Vv = np.array([V * U_HAT[c] for c in range(3)])

    # Per ray: cull dark beams, reset the position cache.
    # R[:, :, 0] is the ray's positions, independent of velocity
    if isinstance(eqn, MemoCulledRateEq):
        eqn.arm(beam_relevance(eqn.laserBeams, R[:, :, 0].T, CULL_RTOL)
                if CULL_RTOL > 0 else None, v_ms.size)

    eqn.generate_force_profile(R, Vv, name="axial", progress_bar=False)
    F = eqn.profile["axial"].F
    return sum(F[c] * U_HAT[c] for c in range(3)) / norm["mass_bar"]



def _grid_progress(run, n_done, n_tasks, per_cell, t_start, verbose):
    """Progress for a cell-by-cell grid"""
    rate = n_done * per_cell / max(time.time() - t_start, 1e-9)
    tabulate_axial_force.last_rate = rate
    if not verbose:
        return
    run.log(step=n_done * per_cell, force_grid_points_per_second=rate)
    if n_done % max(1, n_tasks // 10) == 0 or n_done == n_tasks:
        run.say(f"  {n_done}/{n_tasks} offset cells, "
                f"{n_done * per_cell:,} points, {rate:.0f} pts/s")


def tabulate_axial_force(run, eqn, norm, n_offset=None, v_max_ms=None,
                         n_v=None, verbose=True, offsets_mm=None,
                         v_min_ms=None, spec=None):
    """Tabulate the steady-state optical force projected onto the atom-beam
    axis, on a 4D grid of (offset h, offset v, axial position s, axial
    velocity v_s)

    phase_space_plots.build_phase_space_force_grid generalised from one
    (z, v) plane on the trap axis to a stack of (s, v) planes at different
    transverse offsets."""
    if v_max_ms is None or n_v is None:
        auto_lo, auto_hi, auto_n, _ = auto_velocity_grid(norm,
                                                         _slower_detuning_hz())
        v_max_ms = auto_hi if v_max_ms is None else v_max_ms
        n_v = auto_n if n_v is None else n_v
        if v_min_ms is None:

            v_min_ms = auto_lo

    if offsets_mm is None:
        offsets_mm = build_offset_grid()[0]
    offs_mm = np.asarray(offsets_mm, dtype=float)
    n_offset = offs_mm.size
    s_mm = build_axial_grid()
    v_ms = np.linspace(v_min_ms if v_min_ms is not None else -abs(V_NEG_MS),
                       v_max_ms, n_v)
    n_s = s_mm.size

    a_grid = np.zeros((n_offset, n_offset, n_s, n_v))

    tasks = [(i, j, bh, bv) for i, bh in enumerate(offs_mm)
             for j, bv in enumerate(offs_mm)]
    per_cell = n_s * n_v
    total = len(tasks) * per_cell
    n_workers = worker_count(len(tasks)) if spec is not None else 1
    if verbose:
        run.say(f"tabulating the equilibrium force on a "
                f"{n_offset}x{n_offset}x{n_s}x{n_v} = {total:,} point grid, "
                f"axial span {s_mm[0]:.0f} to {s_mm[-1]:.0f} mm "
                f"(each point a steady-state solve, no time integration), "
                f"over {n_workers} process{'es' if n_workers > 1 else ''}")

    t_start = time.time()
    if n_workers > 1:
        init = dict(spec, s_mm=s_mm, v_ms=v_ms)
        with mp.get_context("spawn").Pool(
                processes=n_workers, initializer=_force_worker_init,
                initargs=(init,)) as pool:
            for n, (i, j, cell) in enumerate(
                    pool.imap_unordered(_force_worker_cell, tasks,
                                       chunksize=1), start=1):
                a_grid[i, j] = cell
                _grid_progress(run, n, len(tasks), per_cell, t_start, verbose)
    else:
        for n, (i, j, bh, bv) in enumerate(tasks, start=1):
            a_grid[i, j] = _force_cell(eqn, norm, bh, bv, s_mm, v_ms)
            _grid_progress(run, n, len(tasks), per_cell, t_start, verbose)

    return offs_mm, s_mm, v_ms, a_grid


@numba.njit(cache=True)
def bilinear(x, y, ax, ay, V):
    """a(s, v) off the force table, on scipy's clip-then-extrapolate indexing """
    i = np.searchsorted(ax, x) - 1
    if i < 0:
        i = 0
    elif i > ax.size - 2:
        i = ax.size - 2
    j = np.searchsorted(ay, y) - 1
    if j < 0:
        j = 0
    elif j > ay.size - 2:
        j = ay.size - 2

    tx = (x - ax[i]) / (ax[i + 1] - ax[i])
    ty = (y - ay[j]) / (ay[j + 1] - ay[j])
    return (V[i, j] * (1 - tx) * (1 - ty) + V[i + 1, j] * tx * (1 - ty)
            + V[i, j + 1] * (1 - tx) * ty + V[i + 1, j + 1] * tx * ty)


def capture_velocity(a_tab, norm, v_try_ms, s_mm, v_max_ms):
    """Launch one atom AT THE NOZZLE FACE at v_try_ms, integrate the whole
    flight, and report whether it is caught.

    Equation of motion in pylcp's normalized units"""
    x0, v0, t0 = norm["x0"], norm["v0"], norm["t0"]
    s_lo_bar = s_mm[0] * 1e-3 / x0
    s_hi_bar = s_mm[-1] * 1e-3 / x0
    s0_bar = s_lo_bar * 0.999

    ax, ay, V = a_tab           # unpack once, not 4.5M times

    def rhs(t, y):
        return np.array((y[1], bilinear(y[0], y[1], ax, ay, V)))

    def left_far(t, y):
        return y[0] - s_lo_bar
    left_far.terminal = True
    left_far.direction = -1

    def left_near(t, y):
        return y[0] - s_hi_bar
    left_near.terminal = True
    left_near.direction = 1

    def entered_trap(t, y):
        return y[0] + TRAP_HALF_MM * 1e-3 / x0

    entered_trap.direction = 1

    max_step = min(TRAJ_T_MAX_BAR / 200.0,
                   (2e-3 / x0) / max(v_max_ms / v0, 1e-9))

    # See: https://github.com/jrenaud90/CyRK
    sol = CyRK.pysolve_ivp(rhs, (0.0, TRAJ_T_MAX_BAR),
                           np.array([s0_bar, v_try_ms / v0]),
                           method=IVP_METHOD,
                           events=(left_far, left_near, entered_trap),
                           rtol=1e-7, atol=1e-9, max_step=max_step)

    s_end = sol.y[0, -1] * x0 * 1e3
    v_end = sol.y[1, -1] * v0
    t_ev = [np.asarray(e) for e in sol.t_events]
    escaped = (t_ev[0].size > 0) or (t_ev[1].size > 0)
    caught = ((not escaped) and abs(s_end) < CAPTURE_RADIUS_MM
              and abs(v_end) < CAPTURE_SPEED_MS)

    if t_ev[2].size > 0:
        t_stop = (sol.t[-1] - t_ev[2].ravel()[0]) * t0
    else:
        t_stop = 0.0
    return caught, max(t_stop, 0.0)


def _capture_cell(norm, s_bar, v_bar, a_cell, s_mm, v_max_ms):
    """Largest catchable launch speed at one transverse offset, and the time
    the marginal atom spends decelerating."""
    interp = (s_bar, v_bar, a_cell)
    ladder = np.linspace(v_max_ms / N_V_SCAN, v_max_ms * 0.95, N_V_SCAN)

    best, best_t = 0.0, 0.0
    worst_escape = None
    for vt in ladder:
        caught, ts = capture_velocity(interp, norm, vt, s_mm, v_max_ms)
        if caught:
            best, best_t = vt, ts
            worst_escape = None
        elif worst_escape is None and best > 0:
            worst_escape = vt

    if worst_escape is not None:
        lo, hi, lo_t = best, worst_escape, best_t
        for _ in range(N_V_BISECT):
            mid = 0.5 * (lo + hi)
            caught, ts = capture_velocity(interp, norm, mid, s_mm, v_max_ms)
            if caught:
                lo, lo_t = mid, ts
            else:
                hi = mid
        best, best_t = lo, lo_t

    return best, best_t


_capture_worker = {}


def _capture_worker_init(state):
    """Constants every capture-map cell needs, stored once per worker."""
    _capture_worker.update(state)


def _capture_worker_cell(task):
    """One cell of the capture map, in a worker."""
    i, j, a_cell = task
    st = _capture_worker
    best, best_t = _capture_cell(st["norm"], st["s_bar"], st["v_bar"], a_cell,
                                 st["s_mm"], st["v_max_ms"])
    return i, j, best, best_t


def build_capture_map(run, norm, offs_mm, s_mm, v_ms, a_grid, verbose=True):
   
    x0, v0 = norm["x0"], norm["v0"]
    s_bar = s_mm * 1e-3 / x0
    v_bar = v_ms / v0
    n_offset = offs_mm.size
    v_max_ms = float(v_ms[-1])

    vc = np.zeros((n_offset, n_offset))
    t_stop = np.zeros((n_offset, n_offset))

    tasks = [(i, j, a_grid[i, j]) for i in range(n_offset)
             for j in range(n_offset)]
    n_workers = worker_count(len(tasks))
    state = dict(norm=norm, s_bar=s_bar, v_bar=v_bar, s_mm=s_mm,
                 v_max_ms=v_max_ms)

    def record(n, i, j, best, best_t):
        vc[i, j], t_stop[i, j] = best, best_t
        if verbose:
            run.log(step=n, capture_map_cells_done=n,
                    capture_velocity_max_so_far_m_per_s=float(vc.max()))

    t_start = time.time()
    if verbose:
        run.say(f"capture map: {len(tasks)} cells over {n_workers} "
                f"process{'es' if n_workers > 1 else ''}")
    if n_workers > 1:
        with mp.get_context("spawn").Pool(
                processes=n_workers, initializer=_capture_worker_init,
                initargs=(state,)) as pool:
            for n, (i, j, best, best_t) in enumerate(
                    pool.imap_unordered(_capture_worker_cell, tasks,
                                       chunksize=1), start=1):
                record(n, i, j, best, best_t)
    else:
        for n, (i, j, a_cell) in enumerate(tasks, start=1):
            best, best_t = _capture_cell(norm, s_bar, v_bar, a_cell, s_mm,
                                         v_max_ms)
            record(n, i, j, best, best_t)

    if verbose:
        run.say(f"capture map done in {time.time() - t_start:.0f} s: "
                f"v_c ranges {vc.min():.1f} to {vc.max():.1f} m/s over the "
                f"+-{TRAP_HALF_MM:.0f} mm region")

        headroom = v_max_ms / max(vc.max(), 1e-9)
        if headroom < 1.5:
            run.say(f"  WARNING: peak v_c = {vc.max():.1f} m/s against a grid "
                    f"edge of {v_max_ms:.1f} m/s (only {headroom:.2f}x "
                    "headroom). Raise FT_V_MAX_MS.")
    return vc, t_stop


def sweep(run, data, offs_mm, vc, t_stop):
    """Convolve the arriving atom beam with the capture map at every oven
    temperature   """
    temps_c = np.arange(T_MIN_C, T_MAX_C + 0.5 * T_STEP_C, T_STEP_C)

    aux = make_aux(data, offs_mm, vc, t_stop)
    inside = aux["inside"]

    rows = []
    for T_c in temps_c:
        t_gas = T_c + 273.15

        n_dens = yb_number_density(t_gas)
        vbar = mean_speed(t_gas)
        flux_in = 0.25 * n_dens * vbar * data["open_area_m2"]
        flux_out = flux_in * data["W"]

        p_atom = capture_probability(data, aux, T_c)

        p_land = inside.sum() / data["n_launched"]
        p_catch = p_atom.sum() / data["n_launched"]
        rate = flux_in * p_catch
        rows.append(dict(
            T_c=T_c, n_dens=n_dens, pressure_pa=yb_vapour_pressure_pa(t_gas),
            mfp_mm=mean_free_path_m(t_gas) * 1e3, vbar=vbar,
            flux_in=flux_in, flux_out=flux_out,
            flux_landing=flux_in * p_land, rate=rate,
            effective_captured_atoms=float(p_atom.sum()),
            frac_of_landing_caught=(p_atom.sum() / max(inside.sum(), 1)),
            grams_per_day=flux_out * n3.M_YB * 1000.0 * 86400.0,
        ))

        run.log(step=int(round(T_c)),
                oven_temperature_C=T_c,
                reservoir_density_per_m3=n_dens,
                mot_load_rate_atoms_per_s=rate,
                nozzle_output_flux_atoms_per_s=flux_out,
                fraction_of_arriving_beam_captured=rows[-1]["frac_of_landing_caught"],
                knudsen_lambda_over_L=mean_free_path_m(t_gas) / n3.L,
                yb_consumption_g_per_day=rows[-1]["grams_per_day"])

    return temps_c, rows, aux


def rate_for_configuration(run, norm, data, magField, *, alignment=None,
                           slower_detuning_hz=None, slower_divergence_mrad=None,
                           n_offset=None, v_max_ms=None, n_v=None, T_c=None,
                           label="", offsets_mm=None, v_min_ms=None):
    """Full pipeline for ONE beam configuration: build beams, tabulate the
    force, extract the capture map, return the load rate at one temperature, useful sweep"""
    T_c = TOL_TEMP_C if T_c is None else T_c
    if v_max_ms is not None and n_v is not None:
        lo = v_min_ms if v_min_ms is not None else -abs(V_NEG_MS)
        if (v_max_ms - lo) / (n_v - 1) > norm["v0"] / 3.0:
            print(f"[full_trap_sweep] WARNING: '{label}' runs on a velocity "
                  f"grid of {(v_max_ms - lo) / (n_v - 1):.1f} m/s against a "
                  f"{norm['v0']:.1f} m/s resonance width; its rate is not "
                  "trustworthy.")
    eqn, _ = build_mot(norm, alignment=alignment,
                       slower_detuning_hz=slower_detuning_hz,
                       slower_divergence_mrad=slower_divergence_mrad,
                       magField=magField)
    offs_mm, s_mm, v_ms, a_grid = tabulate_axial_force(
        run, eqn, norm, n_offset=n_offset, v_max_ms=v_max_ms, n_v=n_v,
        verbose=False, offsets_mm=offsets_mm, v_min_ms=v_min_ms,
        spec=mot_spec(norm, alignment=alignment,
                      slower_detuning_hz=slower_detuning_hz,
                      slower_divergence_mrad=slower_divergence_mrad))
    vc, t_stop = build_capture_map(run, norm, offs_mm, s_mm, v_ms, a_grid,
                                   verbose=False)

    aux = make_aux(data, offs_mm, vc, t_stop)

    p_atom = capture_probability(data, aux, T_c)
    t_gas = T_c + 273.15
    flux_in = 0.25 * yb_number_density(t_gas) * mean_speed(t_gas) \
        * data["open_area_m2"]
    rate = flux_in * p_atom.sum() / data["n_launched"]
    edge_live = bool(vc[0, :].max() > 0 or vc[-1, :].max() > 0
                     or vc[:, 0].max() > 0 or vc[:, -1].max() > 0)
    return dict(rate=rate, vc_max=float(vc.max()),
                vc_centre=float(vc[offs_mm.size // 2, offs_mm.size // 2]),
                edge_live=edge_live, label=label)


def capture_window(offs_mm, vc, margin_mm=None):
    """A contiguous SUBSET of the shared lattice that still contains the
    whole capture region, for the tolerance sweeps to run"""



    margin_mm = TOL_MARGIN_MM if margin_mm is None else margin_mm
    live = vc > 0
    if not live.any():
        return np.asarray(offs_mm, dtype=float)
    w = offs_mm[1] - offs_mm[0] if offs_mm.size > 1 else 1.0
    pad = int(math.ceil(margin_mm / w))
    idx = np.where(live.any(axis=1) | live.any(axis=0))[0]
    lo = max(0, idx[0] - pad)
    hi = min(offs_mm.size - 1, idx[-1] + pad)
    return np.asarray(offs_mm[lo:hi + 1], dtype=float)


def _thin(seq, frac):
    """Keep a subset of a sweep list, always retaining both endpoints."""
    seq = list(seq)
    if frac >= 1.0 or len(seq) <= 2:
        return tuple(seq)
    keep = max(2, int(round(len(seq) * frac)))
    idx = sorted(set(np.linspace(0, len(seq) - 1, keep).round().astype(int)))
    return tuple(seq[i] for i in idx)


def _thin_sweeps(frac):
    """Shrink every sweep list by the same factor to fit the time budget."""
    global TOL_MOT_OFFSET_MM, TOL_SLOWER_DIV_MRAD, TOL_SLOWER_DETUNING_GAMMA
    TOL_MOT_OFFSET_MM = _thin(TOL_MOT_OFFSET_MM, frac)
    TOL_SLOWER_DIV_MRAD = _thin(TOL_SLOWER_DIV_MRAD, frac)
    TOL_SLOWER_DETUNING_GAMMA = TOL_SLOWER_DETUNING_GAMMA




def run_tolerance_sweeps(run, norm, data, magField, window_mm=None,
                         v_min_ms=None, v_max_ms=None, n_v=None):
    results = {}
    n_v = TOL_N_V
    clipped = []
    v_max = v_max_ms if v_max_ms is not None else \
        auto_velocity_grid(norm, _slower_detuning_hz())[1]
    if n_v is not None:
        n_v = max(n_v, TOL_N_V)
    if window_mm is not None:
        run.say(f"   tolerance sweeps run on an offset window "
                f"{window_mm[0]:+.1f} to {window_mm[-1]:+.1f} mm "
                f"({window_mm.size} points), taken from where the converged "
                "baseline map actually captures anything")

    beam = TOL_MOT_BEAM_INDEX
    rows = []
    t0 = time.time()
    run.say(f"tolerance sweep 1/2: displacing the {tb.MOT_NAMES[beam]} MOT "
            f"beam over {TOL_MOT_OFFSET_MM} mm "
            f"({len(TOL_MOT_OFFSET_MM)} force-grid rebuilds)")
    for d_mm in TOL_MOT_OFFSET_MM:
        khat = tb.MOT_KVECS[beam]
        e1, _ = tb._perp_basis(khat)
        al = tb.BeamAlignment(mot_offsets_m={beam: e1 * d_mm * 1e-3})
        r = rate_for_configuration(run, norm, data, magField, alignment=al,
                                   v_max_ms=v_max, n_v=n_v,
                                   label=f"{d_mm} mm", offsets_mm=window_mm,
                                   v_min_ms=v_min_ms)
        rows.append((d_mm, r["rate"], r["vc_max"]))
        if r["edge_live"]:
            clipped.append(f"MOT offset {d_mm} mm")
        run.log(step=int(d_mm * 1000), mot_offset_mm=d_mm,
                load_rate_vs_mot_offset=r["rate"])
        run.say(f"   {d_mm:5.2f} mm -> {r['rate']:.4e} atoms/s "
                f"(v_c max {r['vc_max']:.1f} m/s)")
    results["mot_offset"] = dict(x=[r[0] for r in rows],
                                 rate=[r[1] for r in rows],
                                 vc=[r[2] for r in rows],
                                 beam=tb.MOT_NAMES[beam],
                                 seconds=time.time() - t0)

    rows = []
    t0 = time.time()
    if SLOWER_ENABLED:
        run.say(f"tolerance sweep 2/2: slowing-beam divergence over "
                f"{TOL_SLOWER_DIV_MRAD} mrad")
        for div in TOL_SLOWER_DIV_MRAD:
            r = rate_for_configuration(run, norm, data, magField,
                                       slower_divergence_mrad=div,
                                       v_max_ms=v_max, n_v=n_v,
                                       label=f"{div} mrad",
                                       offsets_mm=window_mm,
                                       v_min_ms=v_min_ms)
            w_trap = math.sqrt(SLOWER_W_OPTIC_M ** 2
                               + (div * 1e-3 * SLOWER_OPTIC_MM * 1e-3) ** 2)
            rows.append((div, r["rate"], w_trap * 1e3))
            if r["edge_live"]:
                clipped.append(f"divergence {div} mrad")
            run.log(step=int(div * 100), slower_divergence_mrad=div,
                    load_rate_vs_slower_divergence=r["rate"])
            run.say(f"   {div:5.1f} mrad -> {r['rate']:.4e} atoms/s "
                    f"(beam is {w_trap * 1e3:.2f} mm at the trap)")
    results["slower_divergence"] = dict(x=[r[0] for r in rows],
                                        rate=[r[1] for r in rows],
                                        w_trap_mm=[r[2] for r in rows],
                                        seconds=time.time() - t0)
    if clipped:
        run.say(f"   WARNING: capture reached the window edge for "
                f"{clipped}. Raise "
                "FT_TOL_MARGIN_MM and rerun the sweeps if the curve shape "
                "matters there.")
    results["clipped"] = clipped
    return results


def run_slower_detuning_scan(run, norm, data, magField, window_mm=None,
                             v_floor_ms=None):
    rows = []
    gamma_hz = sim.YB174_LINEWIDTH_HZ

    def _grid_for(detuning_hz):
        v_lo, v_hi, n, v_r = auto_velocity_grid(norm, detuning_hz)
        if v_floor_ms is not None:
            v_hi = max(v_hi, v_floor_ms)
        n = int((v_hi - v_lo) / V_RESOLUTION_MS) + 1
        n = min(max(n, TOL_N_V), V_MAX_GRID_POINTS)
        return v_lo, v_hi, n, v_r

    run.say(f"slowing-beam detuning scan over {TOL_SLOWER_DETUNING_GAMMA} "
            "Gamma, each on its own velocity grid at a fixed "
            f"{V_RESOLUTION_MS:.1f} m/s resolution")

    global SLOWER_ENABLED
    _was = SLOWER_ENABLED
    SLOWER_ENABLED = False
    nom_lo, nom_hi, nom_n, _ = _grid_for(_slower_detuning_hz())
    off = rate_for_configuration(run, norm, data, magField,
                                 v_max_ms=nom_hi,
                                 n_v=nom_n, label="slower off",
                                 offsets_mm=window_mm, v_min_ms=nom_lo)
    SLOWER_ENABLED = _was
    run.say(f"   slowing beam OFF -> {off['rate']:.4e} atoms/s "
            "(the number to beat)")
    for g in TOL_SLOWER_DETUNING_GAMMA:
        d_hz = g * gamma_hz
        g_lo, g_hi, g_n, _ = _grid_for(d_hz)
        r = rate_for_configuration(run, norm, data, magField,
                                   slower_detuning_hz=d_hz,
                                   v_max_ms=g_hi,
                                   n_v=g_n, label=f"{g} Gamma",
                                   offsets_mm=window_mm, v_min_ms=g_lo)
        v_res = abs(2 * np.pi * d_hz) / norm["k_SI"]
        rows.append((g, r["rate"], v_res, r["vc_max"]))
        run.log(step=int(abs(g) * 100), slower_detuning_gamma=g,
                load_rate_vs_slower_detuning=r["rate"],
                slower_resonant_velocity_m_per_s=v_res)
        dv = (g_hi - g_lo) / (g_n - 1)
        flag = "  <, UNRESOLVED, ignore" if dv > norm["v0"] / 3.0 else ""
        run.say(f"   {g:+6.1f} Gamma (resonant at {v_res:5.1f} m/s) -> "
                f"{r['rate']:.4e} atoms/s  [dv = {dv:.1f} m/s]{flag}")
    return dict(gamma=[r[0] for r in rows], rate=[r[1] for r in rows],
                v_res=[r[2] for r in rows], vc=[r[3] for r in rows],
                rate_slower_off=off["rate"])


def _slower_detuning_hz():
    if not SLOWER_ENABLED:
        return 0.0
    return DETUNING_HZ if math.isnan(SLOWER_DETUNING_HZ) else SLOWER_DETUNING_HZ


def trapped_fraction_map(data, aux, T_c, offs_mm):
    """Per-bin arrival and capture weight at one temperature."""
    p_atom = capture_probability(data, aux, T_c)
    ins = aux["inside"]

    n = np.asarray(offs_mm).size
    arrive = np.zeros((n, n))
    trap = np.zeros((n, n))
    np.add.at(arrive, (aux["iy"][ins], aux["iz"][ins]), 1.0)
    np.add.at(trap, (aux["iy"][ins], aux["iz"][ins]), p_atom[ins])
    return arrive, trap


def _cell_extent(centres):
    """imshow's `extent` spans the OUTER EDGES, but a coordinate vector gives
    cell CENTRES, took me so long to figure out"""
    c = np.asarray(centres, dtype=float)
    d = (c[1] - c[0]) if c.size > 1 else 1.0
    return [c[0] - d / 2, c[-1] + d / 2, c[0] - d / 2, c[-1] + d / 2]


def _mm_box(ax, half, **kw):
    """Outline the trapping region."""
    kw.setdefault("color", "k")
    kw.setdefault("lw", 1.2)
    kw.setdefault("ls", "--")
    ax.plot([-half, half, half, -half, -half],
            [-half, -half, half, half, -half], **kw)


def fig_geometry():
    """Schematic of the slowing geometry in case I did something wrong"""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 6.2),
                                  constrained_layout=True)

    for vec, lbl, col in ((np.array([1.0, 0.0]), "MOT beams, x pair", TRAP_C),
                          (np.array([0.0, 1.0]), "MOT beams, y pair", ACCENT)):
        for sgn in (+1, -1):
            ax.annotate("", xy=(0, 0), xytext=tuple(sgn * vec * 1.15),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2.4))
        ax.plot([], [], color=col, lw=2.4, label=lbl)

    ax.annotate("", xy=tuple(U_HAT[:2] * 1.25), xytext=tuple(-U_HAT[:2] * 1.25),
                arrowprops=dict(arrowstyle="-|>", color=LOSS_C, lw=3.0))
    ax.plot([], [], color=LOSS_C, lw=3.0, label="atom beam from the oven")

    arc = np.linspace(0, TILT, 60)
    ax.plot(0.45 * np.cos(arc), 0.45 * np.sin(arc), color="k", lw=1.0)
    ax.text(0.52 * math.cos(TILT / 2), 0.52 * math.sin(TILT / 2),
            f"{BEAM_TILT_DEG:.1f}$^\\circ$", fontsize=11)
    arc2 = np.linspace(TILT, math.pi / 2, 60)
    ax.plot(0.72 * np.cos(arc2), 0.72 * np.sin(arc2), color="k", lw=1.0)
    ax.text(0.80 * math.cos((TILT + math.pi / 2) / 2),
            0.80 * math.sin((TILT + math.pi / 2) / 2),
            f"{90 - BEAM_TILT_DEG:.1f}$^\\circ$", fontsize=11)

    ax.set_aspect("equal")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("MOT Beam layout in XY plane")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.2)

    ax2.annotate("", xy=(0, 1.15), xytext=(0, -1.15),
                 arrowprops=dict(arrowstyle="<|-|>", color="#2ca02c", lw=2.4))
    ax2.plot([], [], color="#2ca02c", lw=2.4, label="MOT beams, z pair")
    ax2.annotate("", xy=(1.25, 0), xytext=(-1.25, 0),
                 arrowprops=dict(arrowstyle="-|>", color=LOSS_C, lw=3.0))
    ax2.plot([], [], color=LOSS_C, lw=3.0, label="atom beam (horizontal)")
    ax2.annotate("", xy=(0.0, -0.55), xytext=(0.0, -0.15),
                 arrowprops=dict(arrowstyle="-|>", color="k", lw=1.6))
    ax2.text(0.05, -0.42, "gravity", fontsize=10)
    ax2.plot([0.0, 0.16, 0.16], [0.0, 0.0, 0.16], color="k", lw=1.0)
    ax2.text(0.20, 0.06, "$90^\\circ$", fontsize=10)
    ax2.set_aspect("equal")
    ax2.set_xlim(-1.45, 1.45)
    ax2.set_ylim(-1.45, 1.45)
    ax2.set_xlabel("along the atom beam")
    ax2.set_ylabel("z")
    ax2.set_title("MOT Beam layout in XZ plane")
    ax2.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax2.grid(alpha=0.2)
    return fig


def fig_oven_thermodynamics(temps_c, rows, t_free_mol_c):
    """Reservoir conditions across the sweep of temps"""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    T = np.asarray([r["T_c"] for r in rows])

    ax = axes[0, 0]
    ax.semilogy(T, [r["pressure_pa"] for r in rows], color=TRAP_C, lw=2)
    ax.set_ylabel("reservoir vapour pressure [Pa]")
    ax.set_title("Vapour pressure\n"
                 r"$\log_{10}(P/\mathrm{bar}) = 9.111 - 8111/T - 1.0849\log_{10}T$")

    ax = axes[0, 1]
    ax.semilogy(T, [r["n_dens"] for r in rows], color=TRAP_C, lw=2)
    ax.set_ylabel("reservoir number density [m$^{-3}$]")
    ax.set_title("Number density")

    ax = axes[1, 0]
    ax.semilogy(T, [r["flux_out"] for r in rows], color=TRAP_C, lw=2,
                label="out of the nozzle")
    ax.semilogy(T, [r["flux_in"] for r in rows], color="0.6", lw=1.6, ls="--",
                label="into the channels")
    ax.set_ylabel("flux [atoms s$^{-1}$]")
    ax.set_xlabel(r"oven temperature [$^\circ$C]")
    ax.set_title("Effusive flux")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    lam_over_L = np.asarray([r["mfp_mm"] for r in rows]) * 1e-3 / n3.L
    ax.semilogy(T, lam_over_L, color=TRAP_C, lw=2)
    ax.axhline(1.0, color=LOSS_C, lw=1.4, ls="--")
    ax.set_ylabel(r"$\lambda / L$")
    ax.set_xlabel(r"oven temperature [$^\circ$C]")
    ax.set_title("Knudsen number: is free-molecular flow still valid?")
    if np.isfinite(t_free_mol_c):
        for a in axes.ravel():
            a.axvspan(t_free_mol_c, T.max(), color=LOSS_C, alpha=0.10)
        ax.axvline(t_free_mol_c, color=LOSS_C, lw=1.4)
        ax.text(t_free_mol_c, lam_over_L.max(),
                f"  $\\lambda = L$ at {t_free_mol_c:.0f} $^\\circ$C\n"
                "  shaded: model is an\n  optimistic bound",
                color=LOSS_C, fontsize=9, va="top")

    for a in axes.ravel():
        a.grid(alpha=0.25, which="both")
    fig.suptitle("Oven reservoir conditions across the sweep", fontsize=13)
    return fig


def fig_beam_at_chamber(data):
    """Where atoms land at chamber"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    half = TRAP_HALF_MM * ACCEPT_MARGIN

    ax = axes[0]
    h = ax.hist2d(data["y"], data["z"], bins=121,
                  range=[[-half, half], [-half, half]],
                  norm=LogNorm(), cmap="viridis")
    fig.colorbar(h[3], ax=ax, label="atoms per bin")
    _mm_box(ax, TRAP_HALF_MM, color="w")
    ax.set_aspect("equal")
    ax.set_xlabel("horizontal offset [mm]")
    ax.set_ylabel("vertical offset [mm]")
    ax.set_title(f"Beam cross-section at {NOZZLE_TO_TRAP_MM:.0f} mm\n"
                 "dashed: trapping region")

    ax = axes[1]
    for arr, lbl, col in ((data["y"], "horizontal", TRAP_C),
                          (data["z"], "vertical", ACCENT)):
        ax.hist(arr, bins=161, range=(-half, half), histtype="step",
                lw=1.8, color=col, label=lbl)
    ax.axvspan(-TRAP_HALF_MM, TRAP_HALF_MM, color="k", alpha=0.08)
    ax.set_xlabel("offset [mm]")
    ax.set_ylabel("atoms per bin")
    ax.set_title("Line profiles of atom beam")
    ax.legend(fontsize=9)

    ax = axes[2]
    direct = data["hits"] == 0
    ax.hist(data["speed"][~direct], bins=120, histtype="step", lw=1.8,
            color=LOSS_C, label=f"wall-scattered ({(~direct).mean() :.3f} fraction)")
    if direct.any():
        ax.hist(data["speed"][direct], bins=120, histtype="step", lw=1.8,
                color=TRAP_C, label=f"never touched a wall ({direct.mean():.5f} fraction)")
    ax.set_xlabel("speed at the reference trace [m s$^{-1}$]")
    ax.set_ylabel("atoms per bin")
    ax.set_yscale("log")
    ax.set_title("Speed of the atoms that reach the trap region")
    ax.legend(fontsize=9)

    for a in axes:
        a.grid(alpha=0.2)
    return fig


def fig_capture_map(offs_mm, vc, t_stop):
    """The MOT's acceptance, as a function of where the atom crosses."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
    ext = _cell_extent(offs_mm)

    ax = axes[0]
    im = ax.imshow(vc.T, origin="lower", extent=ext, cmap="magma",
                   aspect="equal")
    fig.colorbar(im, ax=ax, label="capture velocity [m s$^{-1}$]")
    cs = ax.contour(offs_mm, offs_mm, vc.T, colors="w", linewidths=0.8)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.0f")
    ax.set_xlabel("horizontal offset [mm]")
    ax.set_ylabel("vertical offset [mm]")
    ax.set_title("Capture velocity $v_c$ along the atom beam\n"
                 f"peak {vc.max():.1f} m s$^{{-1}}$ "
                 f"(grid edge {V_MAX_MS:.0f}, "
                 f"{V_MAX_MS / max(vc.max(), 1e-9):.1f}x headroom)")

    ax = axes[1]
    im = ax.imshow(t_stop.T * 1e3, origin="lower", extent=ext, cmap="viridis",
                   aspect="equal")
    fig.colorbar(im, ax=ax, label="stopping time [ms]")
    ax.set_xlabel("horizontal offset [mm]")
    ax.set_ylabel("vertical offset [mm]")
    ax.set_title("Time to stop the marginal atom\n"
                 "sets how far it drifts sideways before it is caught")
    return fig


def fig_phase_portrait(norm, offs_mm, s_mm, v_ms, a_grid):
    """(position, velocity) portrait along the tilted MOT beam axis."""
    c = offs_mm.size // 2
    a = a_grid[c, c]
    x0, v0, t0 = norm["x0"], norm["v0"], norm["t0"]
    a_si = a * x0 / t0 ** 2

    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    lim = np.abs(a_si).max()
    im = ax.pcolormesh(s_mm, v_ms, a_si.T, cmap="RdBu_r",
                       vmin=-lim, vmax=lim, shading="auto")
    fig.colorbar(im, ax=ax, label="axial acceleration [m s$^{-2}$]")
    ax.contour(s_mm, v_ms, a_si.T, levels=[0.0], colors="k", linewidths=1.2)

    interp = RegularGridInterpolator(
        (s_mm * 1e-3 / x0, v_ms / v0), a, bounds_error=False, fill_value=None)

    def rhs(t, y):
        return [y[1], float(interp([[y[0], y[1]]])[0])]

    s_lo_bar = s_mm[0] * 1e-3 / x0
    s_hi_bar = s_mm[-1] * 1e-3 / x0

    def escaped(t, y):
        return min(y[0] - s_lo_bar, s_hi_bar - y[0])
    escaped.terminal = True
    escaped.direction = -1

    s0 = s_lo_bar * 0.999
    v_edge = float(v_ms[-1])
    for v_launch in np.linspace(5.0, v_edge * 0.9, 12):
        sol = solve_ivp(rhs, (0.0, TRAJ_T_MAX_BAR), [s0, v_launch / v0],
                        events=escaped, rtol=1e-6, atol=1e-8,
                        max_step=TRAJ_T_MAX_BAR / 200, dense_output=True)
        ts = np.linspace(0, sol.t[-1], 900)
        yy = sol.sol(ts)
        caught = (abs(yy[0, -1] * x0 * 1e3) < CAPTURE_RADIUS_MM
                  and abs(yy[1, -1] * v0) < CAPTURE_SPEED_MS)
        ax.plot(yy[0] * x0 * 1e3, yy[1] * v0,
                color="k" if caught else "0.45",
                ls="-" if caught else ":", lw=1.6 if caught else 1.1)

    ax.plot([], [], color="k", lw=1.6, label="captured")
    ax.plot([], [], color="0.45", ls=":", lw=1.1, label="escaped")
    ax.axvspan(-TRAP_HALF_MM, TRAP_HALF_MM, color="k", alpha=0.06)
    ax.set_xlim(-S_NEAR_MM, S_PAST_TRAP_MM)
    ax.set_ylim(0, v_ms[-1])
    ax.set_xlabel("position along the atom beam [mm]")
    ax.set_ylabel("axial velocity [m s$^{-1}$]")
    ax.set_title("Phase portrait on the atom-beam axis at the trap centre\n"
                 "background: axial acceleration; black line: zero-force contour")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.2)
    return fig


def fig_cross_section(data, aux, vc, offs_mm, T_c):
    """THE headline figure: a plane cross-section of the atom beam at the
    trap, my fav"""
    arrive, trap = trapped_fraction_map(data, aux, T_c, offs_mm)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(arrive > 0, trap / np.maximum(arrive, 1), np.nan)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11), constrained_layout=True)
    ext = _cell_extent(offs_mm)

    ax = axes[0, 0]
    im = ax.imshow(arrive.T, origin="lower", extent=ext, cmap="viridis",
                   aspect="equal")
    fig.colorbar(im, ax=ax, label="atoms arriving [MC counts]")
    ax.set_title("(a) atoms arriving in the trap plane")

    ax = axes[0, 1]
    im = ax.imshow(frac.T, origin="lower", extent=ext, cmap="magma",
                   aspect="equal")
    fig.colorbar(im, ax=ax, label="captured fraction")
    ax.set_title("(b) fraction of the arriving atoms that get trapped")

    ax = axes[1, 0]
    im = ax.imshow(trap.T, origin="lower", extent=ext, cmap="inferno",
                   aspect="equal")
    fig.colorbar(im, ax=ax, label="expected captures [weighted counts]")
    ax.set_title("(c) capture fraction weighted by distribution")

    ax = axes[1, 1]
    im = ax.imshow(vc.T, origin="lower", extent=ext, cmap="cividis",
                   aspect="equal")
    fig.colorbar(im, ax=ax, label="$v_c$ [m s$^{-1}$]")
    cs = ax.contour(offs_mm, offs_mm, vc.T, colors="w", linewidths=0.8)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.0f")
    ax.set_title("(d) capture velocity, for reference")

    for a in axes.ravel():
        a.set_xlabel("horizontal offset [mm]")
        a.set_ylabel("vertical offset [mm]")

    fig.suptitle(f"Atom-beam cross-section at the trap, "
                 f"oven at {T_c:.0f} $^\\circ$C\n"
                 f"plane is perpendicular to the beam, "
                 f"{NOZZLE_TO_TRAP_MM:.0f} mm from the nozzle face",
                 fontsize=13)
    return fig


def fig_cross_section_vs_T(data, aux, temps, offs_mm):
    """The same trapped-fraction map at several temperatures."""
    fig, axes = plt.subplots(1, len(temps), figsize=(5.0 * len(temps), 5.0),
                             constrained_layout=True)
    ext = _cell_extent(offs_mm)
    axes = np.atleast_1d(axes)

    maps = []
    for T_c in temps:
        arrive, trap = trapped_fraction_map(data, aux, T_c, offs_mm)
        with np.errstate(invalid="ignore", divide="ignore"):
            maps.append(np.where(arrive > 0, trap / np.maximum(arrive, 1), np.nan))
    vmax = np.nanmax([np.nanmax(m) for m in maps])

    for ax, T_c, m in zip(axes, temps, maps):
        im = ax.imshow(m.T, origin="lower", extent=ext, cmap="magma",
                       aspect="equal", vmin=0, vmax=vmax)
        ax.set_title(f"{T_c:.0f} $^\\circ$C")
        ax.set_xlabel("horizontal offset [mm]")
        ax.set_ylabel("vertical offset [mm]")
        fig.colorbar(im, ax=ax, label="captured fraction")
    fig.suptitle("Trapped fraction across the beam cross-section vs. temperature",
                 fontsize=12)
    return fig


def fig_rate_vs_temperature(temps_c, rows, t_free_mol_c):
    """Load rate and steady-state number vs oven temperature."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), constrained_layout=True)
    T = np.asarray([r["T_c"] for r in rows])
    R = np.asarray([r["rate"] for r in rows])

    ax = axes[0]
    ax.semilogy(T, R, color=TRAP_C, lw=2.4)
    ax.set_xlabel(r"oven temperature [$^\circ$C]")
    ax.set_ylabel("MOT load rate [atoms s$^{-1}$]")
    ax.set_title("Capture rate")

    ax = axes[1]
    for tau in LIFETIMES_S:
        ax.semilogy(T, R * tau, lw=2.0, label=rf"$\tau$ = {tau:g} s")
    ax.set_xlabel(r"oven temperature [$^\circ$C]")
    ax.set_ylabel("steady-state trapped number $N = R\\tau$")
    ax.set_title("Steady-state number\n"
                 "pick the curve matching your vacuum-limited lifetime")
    ax.legend(fontsize=9)

    for a in axes:
        a.grid(alpha=0.25, which="both")
        if np.isfinite(t_free_mol_c):
            a.axvspan(t_free_mol_c, T.max(), color=LOSS_C, alpha=0.10)
            a.axvline(t_free_mol_c, color=LOSS_C, lw=1.3, ls="--")
    if np.isfinite(t_free_mol_c):
        axes[0].text(t_free_mol_c, R.max(),
                     f" free-molecular model\n breaks down above\n"
                     f" {t_free_mol_c:.0f} $^\\circ$C",
                     color=LOSS_C, fontsize=9, va="top")
    return fig


def fig_competing_trends(rows, t_free_mol_c):
    fig, ax = plt.subplots(figsize=(11, 6.4), constrained_layout=True)
    T = np.asarray([r["T_c"] for r in rows])

    def norm1(a):
        a = np.asarray(a, dtype=float)
        ref = a[0] if a[0] > 0 else (a[a > 0][0] if np.any(a > 0) else 1.0)
        return a / ref

    ax.semilogy(T, norm1([r["flux_out"] for r in rows]), color=ACCENT, lw=2.2,
                label="nozzle output flux")
    ax.semilogy(T, norm1([r["frac_of_landing_caught"] for r in rows]),
                color=LOSS_C, lw=2.2,
                label="fraction of arriving atoms slow enough")
    ax.semilogy(T, norm1([r["rate"] for r in rows]), color=TRAP_C, lw=2.8,
                label="load rate (the product)")
    ax.axhline(1.0, color="0.7", lw=1.0)
    ax.set_xlabel(r"oven temperature [$^\circ$C]")
    ax.set_ylabel(f"relative to {T[0]:.0f} $^\\circ$C")
    ax.set_title("Capture rate vs. temperature")
    if np.isfinite(t_free_mol_c):
        ax.axvspan(t_free_mol_c, T.max(), color=LOSS_C, alpha=0.10)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25, which="both")
    return fig


def fig_flux_cascade(rows, t_free_mol_c):
    """Where the atoms go, at every temperature."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), constrained_layout=True)
    T = np.asarray([r["T_c"] for r in rows])

    ax = axes[0]
    for key, lbl, col in (("flux_in", "into the channels", "0.55"),
                          ("flux_out", "out of the nozzle", ACCENT),
                          ("flux_landing", "arriving in the trap region", "#2ca02c"),
                          ("rate", "actually captured", TRAP_C)):
        ax.semilogy(T, [r[key] for r in rows], lw=2.2, color=col, label=lbl)
    ax.set_xlabel(r"oven temperature [$^\circ$C]")
    ax.set_ylabel("atoms s$^{-1}$")
    ax.set_title("Flux rates")
    ax.legend(fontsize=9)

    ax = axes[1]
    mid = rows[len(rows) // 2]
    stages = ["into\nchannels", "out of\nnozzle", "in trap\nregion", "captured"]
    vals = [mid["flux_in"], mid["flux_out"], mid["flux_landing"], mid["rate"]]
    ax.bar(stages, vals, color=["0.55", ACCENT, "#2ca02c", TRAP_C])
    ax.set_yscale("log")
    ax.set_ylabel("atoms s$^{-1}$")
    ax.set_title(f"Surviving fraction at {mid['T_c']:.0f} $^\\circ$C")
    for i, v in enumerate(vals):
        ax.text(i, v, f"  {v:.2e}\n  {v / vals[0]:.2e} of input",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylim(min(vals) * 0.2, vals[0] * 30)

    for a in axes:
        a.grid(alpha=0.25, which="both", axis="y")
    if np.isfinite(t_free_mol_c):
        axes[0].axvspan(t_free_mol_c, T.max(), color=LOSS_C, alpha=0.10)
    return fig


def fig_speed_vs_capture(data, aux, vc, temps):
    """The arriving speed distribution against the capture velocity"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), constrained_layout=True)
    inside = aux["inside"]

    ax = axes[0]
    bins = np.linspace(0, 900, 200)
    for T_c in temps:
        f = rescale_speeds(data, T_c + 273.15,
                           T_c + NOZZLE_OFFSET_C + 273.15)
        ax.hist(aux["v_axial_ref"][inside] * f[inside], bins=bins,
                histtype="step", lw=1.8, label=f"{T_c:.0f} $^\\circ$C")
    ax.axvspan(0, vc.max(), color=TRAP_C, alpha=0.18)
    ax.text(vc.max(), ax.get_ylim()[1] * 0.5,
            f"  capturable\n  ($v < v_c$, peak {vc.max():.0f} m s$^{{-1}}$)",
            fontsize=9, color=TRAP_C, va="center")
    ax.set_xlabel("axial speed on arrival [m s$^{-1}$]")
    ax.set_ylabel("atoms per bin")
    ax.set_title("Arriving speed distribution vs the capture window")
    ax.legend(fontsize=9)

    ax = axes[1]
    bins = np.linspace(0, max(80.0, vc.max() * 1.6), 120)
    for T_c in temps:
        f = rescale_speeds(data, T_c + 273.15,
                           T_c + NOZZLE_OFFSET_C + 273.15)
        ax.hist(aux["v_axial_ref"][inside] * f[inside], bins=bins,
                histtype="step", lw=1.8, label=f"{T_c:.0f} $^\\circ$C")
    ax.axvline(vc.max(), color=TRAP_C, lw=1.6, ls="--", label="peak $v_c$")
    ax.axvline(vc.min(), color=LOSS_C, lw=1.6, ls="--", label="worst-corner $v_c$")
    ax.set_xlabel("axial speed on arrival [m s$^{-1}$]")
    ax.set_ylabel("atoms per bin")
    ax.set_title("Zoomed on the capturable tail\n"
                 r"flux distribution goes as $v^3$, so this tail is thin")
    ax.legend(fontsize=9)

    for a in axes:
        a.grid(alpha=0.2)
    return fig


def fig_tilt_asymmetry(norm, offs_mm, s_mm, v_ms, a_grid, vc):
    """Boris asked about this"""
    jc = vc.shape[1] // 2
    i_s0 = int(np.argmin(np.abs(s_mm)))
    i_v0 = int(np.argmin(np.abs(v_ms)))
    a_si = a_grid[:, jc, i_s0, i_v0] * norm["x0"] / norm["t0"] ** 2

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), constrained_layout=True)

    ax = axes[0]
    ax.plot(offs_mm, a_si, "o-", color=TRAP_C, lw=2)
    ax.axhline(0, color="0.6", lw=1.0)
    ax.axvline(0, color="0.6", lw=1.0)
    ax.set_xlabel("transverse offset along the beam-perpendicular "
                  "horizontal axis [mm]")
    ax.set_ylabel("axial acceleration of a stationary atom [m s$^{-2}$]")
    ax.set_title("Acceleration at trap center")
    ax.grid(alpha=0.25)
    asym = float(np.max(np.abs(a_si + a_si[::-1])) / max(np.abs(a_si).max(), 1e-30))

    ax = axes[1]
    vc_line = vc[:, jc]
    ax.plot(offs_mm, vc_line, "o-", color=LOSS_C, lw=2)
    i_best = int(np.argmax(vc_line))
    ax.axvline(offs_mm[i_best], color="k", ls="--", lw=1.2)
    ax.axvline(0.0, color="0.6", lw=1.0)
    ax.annotate(f"best at {offs_mm[i_best]:+.1f} mm\n"
                f"$v_c$ = {vc_line[i_best]:.1f} m s$^{{-1}}$\n",
                xy=(offs_mm[i_best], vc_line[i_best]),
                xytext=(0.05, 0.72), textcoords="axes fraction", fontsize=10,
                arrowprops=dict(arrowstyle="->", color="k"))
    ax.set_xlabel("transverse offset along the beam-perpendicular "
                  "horizontal axis [mm]")
    ax.set_ylabel("capture velocity [m s$^{-1}$]")
    ax.set_title("Capture velocity vs cross section")
    ax.grid(alpha=0.25)
    return fig


def _half_point(x, y):
    """ Useless but too lazy to remove"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.size < 2 or y[0] <= 0:
        return float("nan")
    half = 0.5 * y[0]
    for i in range(1, y.size):
        if y[i] <= half:
            if y[i - 1] == y[i]:
                return x[i]
            f = (y[i - 1] - half) / (y[i - 1] - y[i])
            return x[i - 1] + f * (x[i] - x[i - 1])
    return float("nan")


def fig_tolerances(tol):
    have_slow = bool(tol.get("slower_divergence", {}).get("x"))
    n = 2 if have_slow else 1
    fig, axes = plt.subplots(1, n, figsize=(7.0 * n, 5.6),
                             constrained_layout=True, squeeze=False)
    axes = axes[0]

    d = tol["mot_offset"]
    ax = axes[0]
    x, y = np.asarray(d["x"]), np.asarray(d["rate"])
    ax.plot(x, y / y[0], "o-", color=TRAP_C, lw=2)
    ax.axhline(0.5, color=LOSS_C, ls="--", lw=1.2)
    ax.set_xlabel(f"transverse displacement of the {d['beam']} MOT beam [mm]")
    ax.set_ylabel("load rate, relative to perfect alignment")
    ax.set_title("MOT beam position tolerance")
    ax.grid(alpha=0.25)

    if have_slow:
        d = tol["slower_divergence"]
        ax = axes[1]
        x, y = np.asarray(d["x"]), np.asarray(d["rate"])
        ax.plot(x, y / y[0], "o-", color=ACCENT, lw=2)
        ax.axhline(1.0, color="0.6", lw=1.0)
        ax.set_xlabel("slowing-beam far-field half-divergence [mrad]")
        ax.set_ylabel("load rate, relative to a collimated slowing beam")
        ax.set_title("Slowing beam collimation")
        ax.grid(alpha=0.25)
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"{w:.1f}" for w in d["w_trap_mm"]], fontsize=8)
        ax2.set_xlabel("1/e$^2$ radius at the trap [mm]", fontsize=9)
    return fig


def fig_slower_detuning(scan, nominal_gamma):
    fig, ax = plt.subplots(figsize=(11, 6.2), constrained_layout=True)
    g = np.asarray(scan["gamma"])
    r = np.asarray(scan["rate"])
    ax.plot(g, r, "o-", color=TRAP_C, lw=2.2)
    ax.set_yscale("log")
    ax.set_xlabel(r"slowing-beam detuning [$\Gamma$]")
    ax.set_ylabel("MOT load rate [atoms s$^{-1}$]")
    ax.grid(alpha=0.25, which="both")

    i_nom = int(np.argmin(np.abs(g - nominal_gamma)))
    ax.axvline(g[i_nom], color=LOSS_C, ls="--", lw=1.3)
    if scan.get("rate_slower_off"):
        ax.axhline(scan["rate_slower_off"], color="0.4", ls="-.", lw=1.5)
        ax.text(g.min(), scan["rate_slower_off"],
                "  slowing beam OFF", fontsize=9, color="0.3", va="bottom")
    i_best = int(np.argmax(r))

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(g)
    ax2.set_xticklabels([f"{v:.0f}" for v in scan["v_res"]], fontsize=8)
    ax2.set_xlabel("atom velocity that slower beam is resonant with [m s$^{-1}$]",
                   fontsize=9)
    ax.set_title("Atom resonance", pad=32)
    return fig


def fig_slower_profile(norm, eqn_info):
    """Slowing-beam geometry along the flight: width and surviving intensity
    from the optic through the trap to the nozzle.
    """
    fig, ax = plt.subplots(figsize=(11, 5.6), constrained_layout=True)
    s_mm = np.linspace(-NOZZLE_TO_TRAP_MM, S_PAST_TRAP_MM, 400)
    zeta_mm = SLOWER_OPTIC_MM - s_mm
    for div in sorted({0.0, SLOWER_DIVERGENCE_MRAD, 2.0, 8.0, 32.0}):
        w = np.sqrt(SLOWER_W_OPTIC_M ** 2
                    + (div * 1e-3 * (zeta_mm - SLOWER_FOCUS_MM) * 1e-3) ** 2)
        lw = 2.6 if abs(div - SLOWER_DIVERGENCE_MRAD) < 1e-9 else 1.4
        ax.plot(s_mm, w * 1e3, lw=lw,
                label=f"{div:g} mrad" + (" (in use)" if lw > 2 else ""))
    ax.axvline(0.0, color="k", lw=1.2)
    ax.text(2, ax.get_ylim()[1] * 0.95, "trap", fontsize=9, va="top")
    ax.axvline(-NOZZLE_TO_TRAP_MM, color="0.5", lw=1.2, ls="--")
    ax.text(-NOZZLE_TO_TRAP_MM + 4, ax.get_ylim()[1] * 0.95, "nozzle",
            fontsize=9, va="top", color="0.4")
    ax.axvspan(-TRAP_HALF_MM, TRAP_HALF_MM, color="k", alpha=0.08)
    ax.set_xlabel("position along the atom beam [mm]  "
                  "(0 = trap, negative = toward the oven)")
    ax.set_ylabel("slowing beam 1/e$^2$ radius [mm]")
    ax.set_title(f"Slowing beam geometry\n"
                 f"optic {SLOWER_OPTIC_MM:.0f} mm past the trap, "
                 f"{SLOWER_W_OPTIC_M * 1e3:.1f} mm  waist at fiber")
    ax.legend(fontsize=9, title="divergence")
    ax.grid(alpha=0.25)
    return fig


def fig_convergence(data):
    fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
    n = data["conv_steps"]
    p = data["conv_accept"]
    ax.plot(n, p, color=TRAP_C, lw=1.6)
    if p.size:
        final = p[-1]
        band = np.sqrt(np.maximum(final * (1 - final), 0) / np.maximum(n, 1))
        ax.fill_between(n, final - 3 * band, final + 3 * band,
                        color=TRAP_C, alpha=0.18,
                        label=r"$\pm 3\sigma$ binomial about the final value")
        ax.axhline(final, color="k", lw=1.0, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("atoms launched")
    ax.set_ylabel("fraction landing in the trap region")
    ax.set_title("Convergence of the nozzle Monte Carlo\n"
                 f"{data['n_accept']:,} usable atoms from "
                 f"{data['n_launched']:,} launched")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, which="both")
    return fig


def main():
    rng = np.random.default_rng(NOZZLE_SEED)
    norm = sim.build_normalization()

    isat = norm["Isat_SI"]
    power_w = BEAM_AVG_SAT * isat * math.pi * BEAM_WAIST_M ** 2
    t_free_mol_c = free_molecular_limit_C(n3.L)

    params = {
        "T_min_C": T_MIN_C, "T_max_C": T_MAX_C, "T_step_C": T_STEP_C,
        "nozzle_wall_offset_C": NOZZLE_OFFSET_C,
        "nozzle_to_trap_mm": NOZZLE_TO_TRAP_MM,
        "trap_half_width_mm": TRAP_HALF_MM,
        "atom_beam_tilt_deg": BEAM_TILT_DEG,
        "atom_beam_tilt_to_other_pair_deg": 90.0 - BEAM_TILT_DEG,
        "beam_waist_m": BEAM_WAIST_M,
        "beam_average_saturation": BEAM_AVG_SAT,
        "beam_power_per_arm_W": power_w,
        "beam_peak_saturation": 2.0 * BEAM_AVG_SAT,
        "Isat_mW_per_cm2": norm["Isat_mW_cm2"],
        "detuning_Hz": DETUNING_HZ,
        "detuning_over_Gamma": DETUNING_HZ / sim.YB174_LINEWIDTH_HZ,
        "coil_current_A": OPT_CURRENT_A,
        "nozzle_atoms": NOZZLE_ATOMS,
        "n_channels": n3.N_CHAN_Y * n3.N_CHAN_Z,
        "channel_length_mm": n3.CHANNEL_LENGTH_MM,
        "hydraulic_diameter_mm": n3.D_HYD * 1e3,
        "capture_offset_cells": int(build_offset_grid()[0].size),
        "capture_offset_cell_mm": float(build_offset_grid()[1]),
        "axial_grid_step_near_mm": S_NEAR_STEP_MM,
        "axial_grid_step_far_mm": S_FAR_STEP_MM,
        "free_molecular_limit_C": t_free_mol_c,
        "vapour_pressure_model": "oven_code_rate.ipynb 3-parameter fit",
        "slowing_beam_enabled": int(SLOWER_ENABLED),
        "slowing_optic_distance_mm": SLOWER_OPTIC_MM,
        "slowing_w_at_optic_mm": SLOWER_W_OPTIC_M * 1e3,
        "slowing_divergence_mrad": SLOWER_DIVERGENCE_MRAD,
        "mot_beam_offsets_mm": str(MOT_BEAM_OFFSET_MM),
        "mot_beam_tilts_mdeg": str(MOT_BEAM_TILT_MDEG),
        "seed": NOZZLE_SEED,
        "logical_cpus": os.cpu_count(),
        "worker_processes": worker_count(build_offset_grid()[0].size ** 2),
    }

    with runlog.start("yb-full-trap-sweep", params=json_safe(params),
                      note="oven -> nozzle -> 310 mm -> MOT capture, "
                           "350-500 C") as run:
        run.say(f"beams: w = {BEAM_WAIST_M * 1e3:.1f} mm, "
                f"P = {power_w * 1e3:.3f} mW per arm "
                f"-> I_avg = {BEAM_AVG_SAT:.2f} Isat, peak s0 = "
                f"{2 * BEAM_AVG_SAT:.2f} (Isat = "
                f"{norm['Isat_mW_cm2']:.2f} mW/cm^2)")
        run.say(f"atom beam at {BEAM_TILT_DEG:.1f} deg to the x MOT pair and "
                f"{90 - BEAM_TILT_DEG:.1f} deg to the y pair; z pair square-on")
        if np.isfinite(t_free_mol_c):
            run.say(f"free-molecular flow (lambda > L) holds below "
                    f"{t_free_mol_c:.0f} C , results above that are an "
                    "OPTIMISTIC BOUND, not a prediction")

        data = load_or_trace_nozzle(run, rng)
        run.say(f"nozzle: W = {data['W']:.5f}, "
                f"{data['n_accept']:,} of {data['n_launched']:,} launched "
                f"atoms reach the trap region "
                f"({data['n_accept'] / data['n_launched']:.3e}), "
                f"in {data['elapsed_s']:.0f} s")
        run.figure(fig_convergence(data), "nozzle_convergence", close=True)
        run.figure(fig_beam_at_chamber(data), "atom_beam_at_chamber", close=True)
        run.figure(fig_geometry(), "beam_and_laser_geometry", close=True)

        magField, field_info = build_extended_magfield(norm)
        run.say(f"coil field tabulated on {field_info['shape']} = "
                f"{field_info['n_points']:,} points spanning "
                f"{field_info['extent_mm'][0][0]:.0f}..{field_info['extent_mm'][0][1]:.0f} mm in x "
                f"({field_info['build_s']:.0f} s); axial gradient "
                f"{field_info['gradient_G_per_cm']:.2f} G/cm")

        d_slow = _slower_detuning_hz()
        v_min_ms, v_max_ms, n_v, v_res = auto_velocity_grid(norm, d_slow)
        if True:
            run.say(f"slowing beam ON: detuning {d_slow / 1e6:+.2f} MHz "
                    f"({d_slow / sim.YB174_LINEWIDTH_HZ:+.2f} Gamma), "
                    f"resonant with {v_res:.1f} m/s atoms; "
                    f"velocity grid +-{v_max_ms:.0f} m/s in {n_v} points "
                    f"({2 * v_max_ms / (n_v - 1):.2f} m/s, vs Gamma/k = "
                    f"{norm['v0']:.1f} m/s resonance width)")

        dv = 2 * v_max_ms / (n_v - 1)
        if dv > norm["v0"] / 3.0:
            run.say(f"WARNING: velocity grid spacing {dv:.2f} m/s is coarse "
                    f"against the {norm['v0']:.1f} m/s Doppler resonance "
                    "width. Capture velocities will be unreliable; lower "
                    "FT_V_RES_MS or raise FT_N_V.")

        eqn, beam_info = build_mot(norm, magField=magField)
        run.say(f"beams: {beam_info['n_beams']} total, "
                f"alignment = {beam_info['alignment']}")
        if SLOWER_ENABLED:
            run.figure(fig_slower_profile(norm, beam_info),
                       "slowing_beam_geometry", close=True)

        offs_mm, s_mm, v_ms, a_grid = tabulate_axial_force(
            run, eqn, norm, v_max_ms=v_max_ms, n_v=n_v, v_min_ms=v_min_ms,
            spec=mot_spec(norm))
        grid_rate = getattr(tabulate_axial_force, "last_rate", 500.0)
        run.figure(fig_phase_portrait(norm, offs_mm, s_mm, v_ms, a_grid),
                   "phase_portrait_along_atom_beam", close=True)

        vc, t_stop = build_capture_map(run, norm, offs_mm, s_mm, v_ms, a_grid)

        escalations = 0
        while (V_AUTO_ESCALATE and escalations < 2
               and vc.max() > 0.85 * v_max_ms):
            escalations += 1
            v_max_ms *= 2.0
            n_v = min(int((v_max_ms - v_min_ms) / V_RESOLUTION_MS) + 1,
                      V_MAX_GRID_POINTS)
            run.say(f"v_c saturated the velocity grid; re-tabulating with "
                    f"v up to {v_max_ms:.0f} m/s ({n_v} points, "
                    f"{(v_max_ms - v_min_ms) / (n_v - 1):.2f} m/s spacing) "
                    f", escalation {escalations}/2")
            offs_mm, s_mm, v_ms, a_grid = tabulate_axial_force(
                run, eqn, norm, v_max_ms=v_max_ms, n_v=n_v, v_min_ms=v_min_ms,
                spec=mot_spec(norm))
            vc, t_stop = build_capture_map(run, norm, offs_mm, s_mm, v_ms,
                                           a_grid)
        if vc.max() > 0.85 * v_max_ms:
            run.say("WARNING: capture velocity is STILL at the grid edge "
                    "after escalation. Treat every absolute rate below as "
                    "an upper bound and rerun with a larger FT_V_MAX_MS.")
        run.figure(fig_capture_map(offs_mm, vc, t_stop), "capture_velocity_map",
                   close=True)
        run.figure(fig_tilt_asymmetry(norm, offs_mm, s_mm, v_ms, a_grid, vc),
                   "tilt_induced_capture_asymmetry", close=True)

        jc = vc.shape[1] // 2
        i_best = int(np.argmax(vc[:, jc]))
        best_offset_mm = float(offs_mm[i_best])
        if vc.max() <= 0:
            run.say("WARNING: nothing is captured anywhere on the map. With "
                    "the slowing beam on at a near-MOT detuning this can be "
                    "physical (it stops trappable atoms upstream), but check "
                    "the velocity-grid resolution warning above first.")
        elif vc[i_best, jc] > vc[jc, jc]:
            run.say(f"tilt asymmetry: v_c peaks at a transverse offset of "
                    f"{best_offset_mm:+.1f} mm ({vc[i_best, jc]:.1f} m/s) "
                    f"rather than on axis ({vc[jc, jc]:.1f} m/s) , aiming "
                    "the atom beam there should load faster")

        temps_c, rows, aux = sweep(run, data, offs_mm, vc, t_stop)

        show_T = [T_MIN_C, 0.5 * (T_MIN_C + T_MAX_C), T_MAX_C]
        mid_T = show_T[1]

        run.figure(fig_cross_section(data, aux, vc, offs_mm, mid_T),
                   "beam_cross_section_trapped_fraction", close=True)
        run.figure(fig_cross_section_vs_T(data, aux, show_T, offs_mm),
                   "trapped_fraction_vs_temperature", close=True)
        run.figure(fig_rate_vs_temperature(temps_c, rows, t_free_mol_c),
                   "load_rate_and_trapped_number", close=True)
        run.figure(fig_competing_trends(rows, t_free_mol_c),
                   "competing_temperature_trends", close=True)
        run.figure(fig_flux_cascade(rows, t_free_mol_c), "flux_cascade",
                   close=True)
        run.figure(fig_speed_vs_capture(data, aux, vc, show_T),
                   "arriving_speed_vs_capture_velocity", close=True)
        run.figure(fig_oven_thermodynamics(temps_c, rows, t_free_mol_c),
                   "oven_thermodynamics", close=True)

        tol_window = capture_window(offs_mm, vc)
        n_reb = (len(TOL_MOT_OFFSET_MM)
                 + (len(TOL_SLOWER_DIV_MRAD) if SLOWER_ENABLED else 0)
                 + ((1 + len(TOL_SLOWER_DETUNING_GAMMA))
                    if (SLOWER_ENABLED and SLOWER_DETUNING_SCAN) else 0))
        tol_v_max = max(90.0, 1.6 * float(vc.max()))
        tol_n_v = min(int((tol_v_max - v_min_ms) / V_RESOLUTION_MS) + 1,
                      V_MAX_GRID_POINTS)
        pts_each = tol_window.size ** 2 * s_mm.size * tol_n_v
        est_h = n_reb * (pts_each / max(grid_rate, 1.0)) / 3600.0
        run.say(f"tolerance + detuning stage: {n_reb} force-grid rebuilds of "
                f"{tol_window.size}x{tol_window.size}x{s_mm.size}x{tol_n_v} "
                f"= {pts_each:,} points each, at the measured "
                f"{grid_rate:.0f} pts/s -> about {est_h:.1f} h")
        if TIME_BUDGET_H > 0 and est_h > TIME_BUDGET_H:
            run.say(f"   that exceeds FT_TIME_BUDGET_H = {TIME_BUDGET_H:.1f} h. "
                    "Thinning the sweep lists to fit; set FT_TIME_BUDGET_H=0 "
                    "to disable this, or FT_TOLERANCE=0 to skip the stage.")
            _thin_sweeps(TIME_BUDGET_H / max(est_h, 1e-9))

        tol = None
        if TOLERANCE_ENABLED:
            tol = run_tolerance_sweeps(run, norm, data, magField,
                                       window_mm=tol_window,
                                       v_min_ms=v_min_ms,
                                       v_max_ms=tol_v_max, n_v=tol_n_v)
            run.figure(fig_tolerances(tol), "alignment_tolerances", close=True)
            hp = _half_point(tol["mot_offset"]["x"], tol["mot_offset"]["rate"])
            run.say(f"ALIGNMENT SPEC: displacing the "
                    f"{tol['mot_offset']['beam']} MOT beam halves the load "
                    f"rate at {hp:.2f} mm" if np.isfinite(hp) else
                    "ALIGNMENT SPEC: load rate never halved over the swept "
                    "MOT displacement range")

        scan = None
        if SLOWER_ENABLED and SLOWER_DETUNING_SCAN:
            scan = run_slower_detuning_scan(
                run, norm, data, magField,
                window_mm=tol_window, v_floor_ms=tol_v_max)
            nominal_gamma = d_slow / sim.YB174_LINEWIDTH_HZ
            run.figure(fig_slower_detuning(scan, nominal_gamma),
                       "slowing_beam_detuning_scan", close=True)

        np.savetxt(
            run.out("full_trap_sweep.csv"),
            np.column_stack([
                [r["T_c"] for r in rows],
                [r["pressure_pa"] for r in rows],
                [r["n_dens"] for r in rows],
                [r["mfp_mm"] for r in rows],
                [r["flux_in"] for r in rows],
                [r["flux_out"] for r in rows],
                [r["flux_landing"] for r in rows],
                [r["rate"] for r in rows],
                [r["frac_of_landing_caught"] for r in rows],
                [r["grams_per_day"] for r in rows],
            ] + [[r["rate"] * tau for r in rows] for tau in LIFETIMES_S]),
            delimiter=",",
            header="oven_T_C,vapour_pressure_Pa,reservoir_density_m3,"
                   "mean_free_path_mm,flux_into_channels_per_s,"
                   "flux_out_of_nozzle_per_s,flux_into_trap_region_per_s,"
                   "mot_load_rate_per_s,fraction_of_arrivals_captured,"
                   "yb_consumption_g_per_day,"
                   + ",".join(f"N_steady_tau_{tau:g}s" for tau in LIFETIMES_S),
            comments="")
        run.file("full_trap_sweep.csv", "sweep-table")

        np.savez_compressed(
            run.out("capture_map.npz"),
            offsets_mm=offs_mm, capture_velocity_m_per_s=vc,
            stopping_time_s=t_stop, s_mm=s_mm, v_ms=v_ms,
            axial_acceleration_bar=a_grid)
        run.file("capture_map.npz", "capture-map")

        best = max(rows, key=lambda r: r["rate"])
        safe = [r for r in rows
                if not np.isfinite(t_free_mol_c) or r["T_c"] <= t_free_mol_c]
        best_safe = max(safe, key=lambda r: r["rate"]) if safe else best

        v_typ = float(np.median(aux["v_axial_ref"]))
        t_flight = NOZZLE_TO_TRAP_MM * 1e-3 / max(v_typ, 1e-9)
        sag_um = 0.5 * sp_const.g * t_flight ** 2 * 1e6

        for r in (rows[0], rows[len(rows) // 2], rows[-1]):
            run.say(f"{r['T_c']:.0f} C: load rate {r['rate']:.3e} atoms/s, "
                    f"{r['frac_of_landing_caught']:.3e} of arrivals, "
                    f"N = " + ", ".join(
                        f"{r['rate'] * tau:.2e} (tau={tau:g}s)"
                        for tau in LIFETIMES_S))

        run.finish(summary=json_safe({
            "nozzle_transmission_W": data["W"],
            "nozzle_atoms_launched": data["n_launched"],
            "atoms_reaching_trap_region": data["n_accept"],
            "fraction_launched_reaching_trap_region":
                data["n_accept"] / data["n_launched"],
            "capture_velocity_peak_m_per_s": float(vc.max()),
            "best_transverse_offset_mm": best_offset_mm,
            "capture_velocity_at_best_offset_m_per_s": float(vc[i_best, jc]),
            "capture_velocity_centre_m_per_s":
                float(vc[vc.shape[0] // 2, vc.shape[1] // 2]),
            "capture_velocity_min_m_per_s": float(vc.min()),
            "velocity_grid_headroom": V_MAX_MS / max(float(vc.max()), 1e-9),
            "beam_power_per_arm_mW": power_w * 1e3,
            **({"mot_offset_halving_load_rate_mm":
                _half_point(tol["mot_offset"]["x"], tol["mot_offset"]["rate"])}
               if tol else {}),
            **({"best_slower_detuning_Gamma":
                float(scan["gamma"][int(np.argmax(scan["rate"]))]),
                "best_slower_gain_over_nominal":
                float(np.max(scan["rate"])
                      / scan["rate"][int(np.argmin(np.abs(
                          np.asarray(scan["gamma"])
                          - d_slow / sim.YB174_LINEWIDTH_HZ)))])}
               if scan else {}),
            "beam_peak_saturation": 2.0 * BEAM_AVG_SAT,
            "axial_gradient_G_per_cm": field_info["gradient_G_per_cm"],
            "flight_time_ms": t_flight * 1e3,
            "gravitational_sag_over_flight_um": sag_um,
            "free_molecular_limit_C": t_free_mol_c,
            "slowing_beam_enabled": int(SLOWER_ENABLED),
            "slowing_beam_detuning_Gamma": (d_slow / sim.YB174_LINEWIDTH_HZ
                                            if SLOWER_ENABLED else 0.0),
            "slowing_beam_resonant_velocity_m_per_s": v_res,
            "slowing_beam_divergence_mrad": SLOWER_DIVERGENCE_MRAD,
            "velocity_grid_max_m_per_s": v_max_ms,
            "velocity_grid_step_m_per_s": 2 * v_max_ms / (n_v - 1),
            "axial_grid_points": int(s_mm.size),
            "axial_span_mm": float(s_mm[-1] - s_mm[0]),
            "best_load_rate_atoms_per_s": best["rate"],
            "best_load_rate_at_C": best["T_c"],
            "best_load_rate_within_free_molecular_atoms_per_s": best_safe["rate"],
            "best_load_rate_within_free_molecular_at_C": best_safe["T_c"],
            "load_rate_at_nearest_400C":
                min(rows, key=lambda r: abs(r["T_c"] - 400.0))["rate"],
            "load_rate_reported_at_C":
                min(rows, key=lambda r: abs(r["T_c"] - 400.0))["T_c"],
            **{f"N_steady_at_{best_safe['T_c']:.0f}C_tau_{tau:g}s":
               best_safe["rate"] * tau for tau in LIFETIMES_S},
        }))

if __name__ == "__main__":
    main()
