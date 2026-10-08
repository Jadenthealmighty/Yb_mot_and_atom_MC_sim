"""Yb-174 opto-optical trap (OOT): oven -> nozzle -> free flight -> capture,
on full_trap_sweep's atom beam, MOT beams, slower and CyRK capture map, with
the coil field replaced by the 1077 nm shift-beam triad.


    u_hat  along the atom beam, horizontal, FT_BEAM_TILT_DEG from the x MOT pair
           (full_trap_sweep's U_HAT)
    e_h    horizontal, across it
    e_v    vertical

"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ATOM_BEAM = os.path.dirname(HERE)
sys.path.append(ATOM_BEAM)

# Slower off: with no coil field to tune it out of resonance near the trap it pushes
# slow atoms back out, leaving only a ~2 m/s wide capture band near 100 m/s
for _k, _v in dict(FT_NOZZLE_CACHE=os.path.join(ATOM_BEAM, "nozzle_trace_cache.npz"),
                   FT_SLOWER=0, RUNLOG_ROOT=os.path.join(HERE, "runs")).items():
    os.environ.setdefault(_k, str(_v))

import full_trap_sweep as fts

import math
import time

import CyRK
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sp_const

import runlog

import helpers.coil_field_model as cfm
import helpers.trap_beams as tb
import helpers.yb_nozzle_beam_3d as n3
import helpers.yb174_mot_simulation as sim

import oot_rate_eq as req
import oot_shift_beams as osb

# Extra turn of the triad about z; 0 puts one axis in the atom beam's vertical plane
TRIAD_ROT_DEG = fts._envf("OOT_TRIAD_ROT_DEG", 0.0)

SHIFT_POWER_W = fts._envf("OOT_SHIFT_POWER_W", 2.0)
SHIFT_WAIST_M = fts._envf("OOT_SHIFT_WAIST_M", 0.01)
SHIFT_DETUNING_GAMMA = fts._envf("OOT_SHIFT_DETUNING_GAMMA", 10.0)
SHIFT_THETA_DEG = fts._envf("OOT_SHIFT_THETA_DEG", 45.0)
N_QUAD = fts._envi("OOT_N_QUAD", 3)
RETRO = bool(fts._envi("OOT_RETRO", 1))

# Doppler shift of the 1077 nm light; 0 gives the paper's static B_eff
DOPPLER = bool(fts._envi("OOT_DOPPLER", 1))
# Flip the shift detuning if the given sign anti-traps with the MOT polarizations
AUTO_SIGN = bool(fts._envi("OOT_AUTO_SIGN", 1))

# Coil MOT in this same geometry; -1 picks the current matching the OOT gradient
COMPARE_MOT = bool(fts._envi("OOT_COMPARE_MOT", 0))
MOT_CURRENT_A = fts._envf("OOT_MOT_CURRENT_A", -1.0)

# Centre-ray capture velocity over shift power, waist and detuning
DESIGN_SCAN = bool(fts._envi("OOT_DESIGN_SCAN", 1))

TILT = math.radians(fts.BEAM_TILT_DEG)
U_HAT = fts.U_HAT
E_H = fts.E_H
E_V = fts.E_V
Z_HAT = np.array([0.0, 0.0, 1.0])

GJ = sim.YB174_GJ_EXCITED
TRAP_C = fts.TRAP_C
LOSS_C = fts.LOSS_C
ACCENT = fts.ACCENT
SHIFT_C = "#9467bd"
MOT_C = "0.35"
MOT_PAIRS = ((np.array([1.0, 0, 0]), "MOT x pair", TRAP_C),
             (np.array([0, 1.0, 0]), "MOT y pair", ACCENT),
             (Z_HAT, "MOT z pair", "#2ca02c"))


class _Quiet:
    """Stands in for a runlog.Run when a stage should not log."""

    def say(self, *a, **k):
        pass

    def log(self, *a, **k):
        pass


def reuse_nozzle_cache():
    """Adopt the cached trace's atom count unless FT_NOZZLE_ATOMS is set."""
    if "FT_NOZZLE_ATOMS" in os.environ or not os.path.exists(fts.NOZZLE_CACHE):
        return
    with np.load(fts.NOZZLE_CACHE, allow_pickle=True) as z:
        if "key_atoms" in z.files:
            fts.NOZZLE_ATOMS = int(z["key_atoms"].item())


def triad_phi0():
    """Azimuth that puts one triad axis in the atom beam's vertical plane, its
    upward pass against the atoms and its return coming down along them."""
    return -TILT + math.radians(TRIAD_ROT_DEG)


def build_shift_lasers(sign=+1, power_w=SHIFT_POWER_W, waist_m=SHIFT_WAIST_M,
                       detuning_gamma=SHIFT_DETUNING_GAMMA):
    n = 4 * N_QUAD
    I = osb.intensity_for_power(power_w, waist_m, n)
    return osb.pyramid_beams(I, waist_m, math.radians(SHIFT_THETA_DEG),
                             sign * detuning_gamma * osb.GAMMA_S, N_QUAD,
                             triad_phi0(), RETRO)


def build_cooling(norm):
    """The MOT's six beams and slower, as build_mot makes them, slower along -u_hat."""
    isat = norm["Isat_SI"]
    power_w = fts.BEAM_AVG_SAT * isat * math.pi * fts.BEAM_WAIST_M ** 2

    slower_cfg = None
    if fts.SLOWER_ENABLED:
        d_hz = fts._slower_detuning_hz()
        avg_sat = fts.SLOWER_AVG_SAT if fts.SLOWER_AVG_SAT >= 0 else fts.BEAM_AVG_SAT
        slower_cfg = dict(
            khat=-U_HAT,
            delta_bar=2 * np.pi * d_hz / norm["Gamma_SI"],
            power_w=avg_sat * isat * math.pi * fts.SLOWER_W_OPTIC_M ** 2,
            w_optic_m=fts.SLOWER_W_OPTIC_M,
            optic_distance_m=fts.SLOWER_OPTIC_MM * 1e-3,
            divergence_rad=fts.SLOWER_DIVERGENCE_MRAD * 1e-3,
            focus_distance_m=fts.SLOWER_FOCUS_MM * 1e-3,
            pol=fts.SLOWER_POL,
        )

    alignment = tb.BeamAlignment(
        mot_offsets_m={i: np.asarray(v, dtype=float) * 1e-3
                       for i, v in fts.MOT_BEAM_OFFSET_MM.items()},
        mot_tilts_rad={i: tuple(math.radians(a / 1e3) for a in v)
                       for i, v in fts.MOT_BEAM_TILT_MDEG.items()},
        slower_tilt_rad=tuple(math.radians(a / 1e3) for a in fts.SLOWER_TILT_MDEG),
        slower_offset_m=np.asarray(fts.SLOWER_OFFSET_MM, dtype=float) * 1e-3,
    )
    beams, info = tb.build_trap_beams(
        norm, 2 * np.pi * fts.DETUNING_HZ / norm["Gamma_SI"],
        waist_m=fts.BEAM_WAIST_M, power_w=power_w, alignment=alignment,
        slower=slower_cfg, wavelength_m=sim.YB174_WAVELENGTH_M)
    info.update(power_w=power_w)
    return req.CoolingBeams(beams), info


def oot_force(model, R, V, doppler=None):
    """Equilibrium force (3, N) in hbar k Gamma at pylcp-unit R, V."""
    doppler = DOPPLER if doppler is None else doppler
    H = osb.shift_hamiltonian(model["lasers"], R * model["x0"],
                              V * model["v0"] if doppler else None)
    return req.beam_force(model["cool"], H / model["gamma"], R, V)


def coil_field(model, R):
    """Coil field (3, N) in pylcp units at R, at the MOT reference current."""
    B = np.atleast_2d(cfm.oswald_coil_bfield(np.asarray(R).T * model["x0"],
                                             model["mot_current_A"]))
    return model["b_conv"] * B.T


def mot_force(model, R, V, B=None):
    B = coil_field(model, R) if B is None else B
    return req.beam_force(model["cool"], req.zeeman(B), R, V)


def stiffness(model, axis, force=oot_force, h_mm=0.5):
    """dF/dr along `axis` at the centre, v = 0, hbar k Gamma / x0. Negative traps."""
    h = h_mm * 1e-3 / model["x0"]
    R = np.outer(axis, [h, -h])
    F = force(model, R, np.zeros((3, 2)))
    return float(axis @ (F[:, 0] - F[:, 1])) / (2 * h)


def trap_freq_hz(model, k):
    return math.sqrt(-k / model["mass"]) * model["gamma"] / (2 * np.pi) if k < 0 else 0.0


def build_oot(norm):
    """Cooling beams, shift lasers, units, and the MOT reference current."""
    cool, cool_info = build_cooling(norm)
    model = dict(cool=cool, cool_info=cool_info, x0=norm["x0"], v0=norm["v0"],
                 gamma=norm["Gamma_SI"], mass=norm["mass_bar"],
                 a_g=-sp_const.g * norm["t0"] ** 2 / norm["x0"] * U_HAT[2],
                 b_conv=GJ * osb.MU_B / (osb.HBAR * norm["Gamma_SI"]),
                 flipped=False)
    model["lasers"] = build_shift_lasers(+1)
    if AUTO_SIGN and stiffness(model, Z_HAT) > 0:
        model.update(lasers=build_shift_lasers(-1), flipped=True)

    beta = osb.gradient_tensor(model["lasers"], gF=GJ) * 1e2
    coil_ref = cfm.axial_gradient_G_per_cm(fts.OPT_CURRENT_A)
    current = (MOT_CURRENT_A if MOT_CURRENT_A > 0
               else fts.OPT_CURRENT_A * abs(beta[2, 2]) / abs(coil_ref))
    model.update(beta_G_cm=beta, mot_current_A=current,
                 coil_gradient_G_cm=coil_ref * current / fts.OPT_CURRENT_A,
                 delta_s=model["lasers"][0]["delta"],
                 I_shift=model["lasers"][0]["I"])
    return model


def ray(norm, bh_mm, bv_mm, s_mm, v_ms):
    """Positions and velocities (3, n_s, n_v) on one nozzle ray, pylcp units."""
    x0, v0 = norm["x0"], norm["v0"]
    S, Vs = np.meshgrid(s_mm * 1e-3 / x0, v_ms / v0, indexing="ij")
    L_m = fts.NOZZLE_TO_TRAP_MM * 1e-3
    F_S = ((s_mm * 1e-3 + L_m) / L_m)[:, None]
    base = (bh_mm * E_H + bv_mm * E_V) * 1e-3 / x0
    R = U_HAT[:, None, None] * S + base[:, None, None] * F_S
    return R, U_HAT[:, None, None] * Vs


def axial_cell(model, norm, bh_mm, bv_mm, s_mm, v_ms, B=None, doppler=None):
    """Axial acceleration (n_s, n_v) on one ray: OOT, or MOT given its coil field (3, n_s)."""
    R, V = ray(norm, bh_mm, bv_mm, s_mm, v_ms)
    R, V = R.reshape(3, -1), V.reshape(3, -1)
    if B is None:
        F = oot_force(model, R, V, doppler)
    else:
        F = mot_force(model, R, V, np.repeat(B, v_ms.size, axis=1))
    return ((U_HAT @ F) / model["mass"] + model["a_g"]).reshape(s_mm.size, v_ms.size)


def coil_field_on_rays(model, norm, offs_mm, s_mm):
    """Coil field (n, n, 3, n_s) on every ray, one magpylib call."""
    n = offs_mm.size
    R = np.array([[ray(norm, bh, bv, s_mm, np.zeros(1))[0][:, :, 0]
                   for bv in offs_mm] for bh in offs_mm])
    B = coil_field(model, R.transpose(2, 0, 1, 3).reshape(3, -1))
    return B.reshape(3, n, n, s_mm.size).transpose(1, 2, 0, 3)


def tabulate_axial_force(run, model, norm, offs_mm, s_mm, v_ms, mot=False,
                         verbose=True):
    """a(offset h, offset v, s, v_s) for the OOT, or the coil MOT if mot=True."""
    n = offs_mm.size
    B = coil_field_on_rays(model, norm, offs_mm, s_mm) if mot else None
    a = np.zeros((n, n, s_mm.size, v_ms.size))
    per_cell = s_mm.size * v_ms.size
    label = "MOT reference" if mot else "OOT"
    if verbose:
        run.say(f"tabulating the {label} force on a {n}x{n}x{s_mm.size}x{v_ms.size} "
                f"= {n * n * per_cell:,} point grid, axial span {s_mm[0]:.0f} to "
                f"{s_mm[-1]:.0f} mm")
    t_start = time.time()
    for k, (i, j) in enumerate(np.ndindex(n, n), start=1):
        a[i, j] = axial_cell(model, norm, offs_mm[i], offs_mm[j], s_mm, v_ms,
                             None if B is None else B[i, j])
        rate = k * per_cell / max(time.time() - t_start, 1e-9)
        if verbose:
            run.log(step=k * per_cell, force_grid_points_per_second=rate)
            if k % max(1, n * n // 10) == 0 or k == n * n:
                run.say(f"  {k}/{n * n} offset cells, {k * per_cell:,} points, "
                        f"{rate:.0f} pts/s")
    return a


def capture_map(run, model, norm, offs_mm, s_mm, v_min_ms, v_max_ms, n_v,
                mot=False, verbose=True):
    """Force grid and CyRK capture map, widening v like full_trap_sweep if v_c saturates."""
    escalations = 0
    while True:
        v_ms = np.linspace(v_min_ms, v_max_ms, n_v)
        a_grid = tabulate_axial_force(run, model, norm, offs_mm, s_mm, v_ms,
                                      mot=mot, verbose=verbose)
        vc, t_stop = fts.build_capture_map(run, norm, offs_mm, s_mm, v_ms, a_grid,
                                           verbose=verbose)
        if not (fts.V_AUTO_ESCALATE and escalations < 2 and vc.max() > 0.85 * v_max_ms):
            break
        escalations += 1
        v_max_ms *= 2.0
        n_v = min(int((v_max_ms - v_min_ms) / fts.V_RESOLUTION_MS) + 1,
                  fts.V_MAX_GRID_POINTS)
        run.say(f"v_c saturated the velocity grid; re-tabulating with v up to "
                f"{v_max_ms:.0f} m/s ({n_v} points), escalation {escalations}/2")
    if vc.max() > 0.85 * v_max_ms:
        run.say("WARNING: capture velocity is STILL at the grid edge after "
                "escalation; treat the rates as upper bounds.")
    return v_ms, a_grid, vc, t_stop


def trajectory(norm, a_cell, s_mm, v_ms, v_launch_ms):
    """CyRK flight from the nozzle face along one ray: (s mm, v m/s, caught)."""
    x0, v0 = norm["x0"], norm["v0"]
    ax = s_mm * 1e-3 / x0
    ay = v_ms / v0
    V = np.ascontiguousarray(a_cell)
    s_lo, s_hi = ax[0], ax[-1]
    s_cap = fts.CAPTURE_AXIAL_MM * 1e-3 / x0
    v_cap = fts.CAPTURE_SPEED_MS / v0

    def rhs(t, y):
        return np.array((y[1], fts.bilinear(y[0], y[1], ax, ay, V)))

    def escaped(t, y):
        return min(y[0] - s_lo, s_hi - y[0])
    escaped.terminal = True
    escaped.direction = -1

    def settled(t, y):
        return max(abs(y[0]) / (0.05 * s_cap), abs(y[1]) / (0.05 * v_cap)) - 1.0
    settled.terminal = True
    settled.direction = -1

    max_step = min(fts.TRAJ_T_MAX_BAR / 200.0,
                   (2e-3 / x0) / max(abs(v_ms[-1]) / v0, 1e-9))
    sol = CyRK.pysolve_ivp(rhs, (0.0, fts.TRAJ_T_MAX_BAR),
                           np.array([0.999 * s_lo, v_launch_ms / v0]),
                           method=fts.IVP_METHOD, events=(escaped, settled),
                           rtol=1e-7, atol=1e-9, max_step=max_step)
    s, v = np.asarray(sol.y)
    caught = abs(s[-1]) < s_cap and abs(v[-1]) < v_cap
    return s * x0 * 1e3, v * v0, caught


def ray_light(model, norm, offs_mm):
    """Cooling light along each ray near the trap, sum_l s_l |k_l . u| ds in mm."""
    s = np.linspace(-fts.S_NEAR_MM, fts.S_PAST_TRAP_MM, 221)
    proj = np.abs(model["cool"].khat @ U_HAT)
    out = np.zeros((offs_mm.size, offs_mm.size))
    for i, bh in enumerate(offs_mm):
        for j, bv in enumerate(offs_mm):
            R = ray(norm, bh, bv, s, np.zeros(1))[0][:, :, 0]
            out[i, j] = np.sum(proj @ model["cool"].intensity(R)) * (s[1] - s[0])
    return out


def settle_position(s_mm, v_ms, a_grid):
    """Stable zero of a(s, v = 0) nearest the centre on each ray, mm; nan if none."""
    a = a_grid[..., int(np.argmin(np.abs(v_ms)))]
    hit = (a[..., :-1] > 0) & (a[..., 1:] <= 0)
    step = np.where(hit, a[..., :-1] - a[..., 1:], 1.0)
    s0 = np.where(hit, s_mm[:-1] + np.diff(s_mm) * a[..., :-1] / step, np.inf)
    k = np.argmin(np.abs(s0), axis=-1)
    out = np.take_along_axis(s0, k[..., None], axis=-1)[..., 0]
    return np.where(np.isfinite(out), out, np.nan)


def potential_depth_mK(model):
    """Escape barrier from the centre at v = 0, weakest of x, y, z and the atom beam, mK."""
    r = np.linspace(0.0, 3 * max(SHIFT_WAIST_M, fts.BEAM_WAIST_M), 151) / model["x0"]
    depth = np.inf
    for axis in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), Z_HAT, U_HAT):
        for sgn in (+1, -1):
            d = sgn * axis
            F = d @ oot_force(model, np.outer(d, r), np.zeros((3, r.size)), False)
            U = -np.concatenate([[0.0], np.cumsum(0.5 * (F[1:] + F[:-1]) * np.diff(r))])
            depth = min(depth, U.max())
    return depth * osb.HBAR * model["gamma"] / sp_const.k * 1e3


def design_scan(model, norm, s_mm, v_ms):
    """Centre-ray v_c, trap frequency and depth over shift power x waist; v_c over detuning."""
    sign = np.sign(model["delta_s"])
    s_bar, v_bar = s_mm * 1e-3 / norm["x0"], v_ms / norm["v0"]

    def vc(lasers, doppler=None):
        a = axial_cell(dict(model, lasers=lasers), norm, 0.0, 0.0, s_mm, v_ms,
                       doppler=doppler)
        return fts._capture_cell(norm, s_bar, v_bar, a, s_mm, v_ms[-1])[0]

    P = SHIFT_POWER_W * 2.0 ** np.arange(-3, 3)
    w = SHIFT_WAIST_M * np.linspace(0.4, 1.6, 7)
    vc_pw, grad, freq, depth = (np.zeros((P.size, w.size)) for _ in range(4))
    for i, p in enumerate(P):
        for j, wj in enumerate(w):
            lasers = build_shift_lasers(sign, p, wj)
            m = dict(model, lasers=lasers)
            vc_pw[i, j] = vc(lasers)
            grad[i, j] = osb.gradient_tensor(lasers, gF=GJ)[2, 2] * 1e2
            freq[i, j] = trap_freq_hz(m, stiffness(m, U_HAT))
            depth[i, j] = potential_depth_mK(m)

    D = np.array([2, 3, 5, 7, 10, 15, 20, 30], dtype=float)
    vc_d = np.array([[vc(build_shift_lasers(sign, detuning_gamma=d), doppler)
                      for d in D] for doppler in (True, False)])
    return dict(P=P, w=w, vc_pw=vc_pw, grad=grad, freq=freq, depth=depth, D=D,
                vc_d=vc_d)


def _project(vec, e1, e2):
    return np.array([vec @ e1, vec @ e2])


def _views():
    u_h = U_HAT[:2] / max(np.linalg.norm(U_HAT[:2]), 1e-12)
    u_h = np.array([u_h[0], u_h[1], 0.0])
    return [(np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), "x", "y",
             "Top view (XY plane)"),
            (u_h, Z_HAT, "horizontal, along the atom beam", "z",
             "Vertical plane containing the atom beam")]


def fig_oot_geometry(model):
    """MOT beams, shift-beam passes and the atom beam, in units of the shift waist."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), constrained_layout=True)
    w = SHIFT_WAIST_M
    mot_axes = ((np.array([1.0, 0, 0]), "MOT x pair", TRAP_C),
                (np.array([0, 1.0, 0]), "MOT y pair", ACCENT),
                (Z_HAT, "MOT z pair", "#2ca02c"))
    for ax, (e1, e2, l1, l2, title) in zip(axes, _views()):
        for vec, lbl, col in mot_axes:
            p = _project(vec, e1, e2)
            if np.linalg.norm(p) < 0.2:
                continue
            for sgn in (+1, -1):
                ax.annotate("", xy=(0, 0), xytext=tuple(sgn * p * 2.3),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0,
                                            alpha=0.7))
            ax.plot([], [], color=col, lw=2.0, label=lbl)

        for n, b in enumerate(model["lasers"][::2]):
            c = _project(b["Rmat"] @ b["r0"] / w, e1, e2)
            k = _project(b["k_hat"], e1, e2)
            ax.annotate("", xy=tuple(c + 2.0 * k), xytext=tuple(c - 2.0 * k),
                        arrowprops=dict(arrowstyle="-|>", color=SHIFT_C, lw=1.6,
                                        ls="-" if n % 2 == 0 else "--"))
        ax.plot([], [], color=SHIFT_C, lw=1.6, label="shift beam, first pass")
        ax.plot([], [], color=SHIFT_C, lw=1.6, ls="--",
                label="shift beam, return pass" if RETRO else "shift beam, second")

        p = _project(U_HAT, e1, e2)
        ax.annotate("", xy=tuple(-0.35 * p), xytext=tuple(-2.6 * p),
                    arrowprops=dict(arrowstyle="-|>", color=LOSS_C, lw=3.0))
        ax.plot([], [], color=LOSS_C, lw=3.0, label="atom beam")
        if e2 is Z_HAT:
            ax.annotate("", xy=(2.2, -2.2), xytext=(2.2, -1.6),
                        arrowprops=dict(arrowstyle="-|>", color="k", lw=1.6))
            ax.text(2.28, -1.95, "g", fontsize=11)

        ax.set_aspect("equal")
        ax.set_xlim(-2.7, 2.7)
        ax.set_ylim(-2.7, 2.7)
        ax.set_xlabel(f"{l1} [units of shift waist w = {w * 1e3:.0f} mm]")
        ax.set_ylabel(f"{l2} [w]")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.suptitle(f"OOT geometry: {N_QUAD} quad beams at {SHIFT_THETA_DEG:.0f}$^\\circ$ "
                 f"from z, horizontal atom beam {fts.BEAM_TILT_DEG:.1f}$^\\circ$ "
                 "from x", fontsize=12)
    return fig


def fig_effective_field(model):
    """|B_eff| and its in-plane direction (paper Fig. 1e), and line cuts against the coil."""
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), constrained_layout=True)
    w = SHIFT_WAIST_M
    a = np.linspace(-2 * w, 2 * w, 121)
    A, Bv = np.meshgrid(a, a, indexing="xy")
    for ax, (e1, e2, l1, l2, title) in zip(axes[:2], _views()):
        R = np.outer(e1, A.ravel()) + np.outer(e2, Bv.ravel())
        B = osb.effective_field(model["lasers"], R, gF=GJ) * 1e4
        b1 = (e1 @ B).reshape(A.shape)
        b2 = (e2 @ B).reshape(A.shape)
        mag = np.linalg.norm(B, axis=0).reshape(A.shape)
        im = ax.imshow(mag, origin="lower", extent=[a[0] / w, a[-1] / w] * 2,
                       cmap="viridis")
        ax.streamplot(a / w, a / w, b1, b2, color="w", linewidth=0.6, density=1.1,
                      arrowsize=0.8)
        p = _project(U_HAT, e1, e2)
        ax.plot([-2.0 * p[0], 0], [-2.0 * p[1], 0], color=LOSS_C, lw=2.0,
                label="atom beam")
        fig.colorbar(im, ax=ax, label="|B$_{eff}$| [G]")
        ax.set_xlim(a[0] / w, a[-1] / w)
        ax.set_ylim(a[0] / w, a[-1] / w)
        ax.set_xlabel(f"{l1} [w]")
        ax.set_ylabel(f"{l2} [w]")
        ax.set_title(title)
    axes[0].legend(loc="lower left", fontsize=9)

    ax = axes[2]
    r = np.linspace(-2 * w, 2 * w, 201)
    for vec, lbl, col in ((Z_HAT, "z", "#2ca02c"), (U_HAT, "atom beam", LOSS_C),
                          (E_H, "$e_h$", TRAP_C)):
        R = np.outer(vec, r)
        ax.plot(r * 1e3, vec @ osb.effective_field(model["lasers"], R, gF=GJ) * 1e4,
                color=col, lw=2.0, label=f"OOT, along {lbl}")
        if COMPARE_MOT:
            Bc = coil_field(model, R / model["x0"]) / model["b_conv"] * 1e4
            ax.plot(r * 1e3, vec @ Bc, color=col, lw=1.2, ls="--")
    if COMPARE_MOT:
        ax.plot([], [], color="k", lw=1.2, ls="--",
                label=f"coil at {model['mot_current_A']:.1f} A")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("distance from trap centre [mm]")
    ax.set_ylabel("field component along the cut [G]")
    ax.set_title("Line cuts through the centre")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.suptitle(f"Effective field of the shift beams at v = 0 (g = {GJ:g}): "
                 f"{SHIFT_POWER_W:g} W, w = {w * 1e3:.0f} mm, "
                 f"$\\Delta_s$ = {model['delta_s'] / osb.GAMMA_S:+.1f} $\\Gamma_s$",
                 fontsize=12)
    return fig


def fig_force_cuts(model, norm):
    """Axial acceleration through the centre vs position, and vs velocity off axis."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), constrained_layout=True)
    acc = norm["x0"] / norm["t0"] ** 2
    w = SHIFT_WAIST_M
    r = np.linspace(-2.5 * w, 2.5 * w, 241) / model["x0"]
    v = np.linspace(-40.0, 40.0, 321) / model["v0"]
    zero = lambda n: np.zeros((3, n))

    for ax, vec, lbl in ((axes[0], U_HAT, "atom beam"), (axes[1], Z_HAT, "z")):
        R = np.outer(vec, r)
        ax.plot(r * model["x0"] * 1e3,
                vec @ oot_force(model, R, zero(r.size)) / model["mass"] * acc,
                color=SHIFT_C, lw=2.2, label="OOT")
        if COMPARE_MOT:
            ax.plot(r * model["x0"] * 1e3,
                    vec @ mot_force(model, R, zero(r.size)) / model["mass"] * acc,
                    color=MOT_C, lw=1.6, ls="--", label="coil MOT")
        ax.set_xlabel(f"position along the {lbl} through the centre [mm]")
        ax.set_ylabel(f"acceleration along the {lbl} [m s$^{{-2}}$], v = 0")
        ax.set_title(f"Restoring force along the {lbl}")

    ax = axes[2]
    V = np.outer(U_HAT, v)
    r0 = (-3.0 * U_HAT + 3.0 * E_V) * 1e-3 / model["x0"]
    R = np.outer(r0, np.ones(v.size))
    ax.plot(v * model["v0"], U_HAT @ oot_force(model, R, V) / model["mass"] * acc,
            color=SHIFT_C, lw=2.2, label="OOT, Doppler-shifted shift beams")
    ax.plot(v * model["v0"],
            U_HAT @ oot_force(model, R, V, doppler=False) / model["mass"] * acc,
            color=SHIFT_C, lw=1.4, ls=":", label="OOT, static $B_{eff}$")
    if COMPARE_MOT:
        ax.plot(v * model["v0"], U_HAT @ mot_force(model, R, V) / model["mass"] * acc,
                color=MOT_C, lw=1.6, ls="--", label="coil MOT")
    ax.set_xlabel("velocity along the atom beam [m s$^{-1}$]")
    ax.set_ylabel("acceleration along the atom beam [m s$^{-2}$]")
    ax.set_title("Damping 3 mm upstream, 3 mm off axis ($e_v$)")
    for a in axes:
        a.axhline(0, color="0.6", lw=0.8)
        a.axvline(0, color="0.6", lw=0.8)
        a.legend(fontsize=9)
        a.grid(alpha=0.25)
    return fig


def fig_phase_portrait(norm, s_mm, v_ms, cells, vc_centre):
    """Paper Fig. 2 along the atom beam: a(s, v) on the centre ray with CyRK flights."""
    acc = norm["x0"] / norm["t0"] ** 2
    v_top = min(v_ms[-1], max(40.0, 2.2 * max(vc_centre)))
    near = (s_mm >= -fts.S_NEAR_MM) & (s_mm <= fts.S_PAST_TRAP_MM)
    vis = v_ms <= v_top
    lim = max(np.abs(a[np.ix_(near, vis)]).max() for _, a in cells) * acc

    fig, axes = plt.subplots(len(cells), 1, figsize=(11, 4.8 * len(cells)),
                             sharex=True, constrained_layout=True, squeeze=False)
    for ax, (label, a), vc in zip(axes[:, 0], cells, vc_centre):
        im = ax.pcolormesh(s_mm, v_ms, a.T * acc, cmap="RdBu_r", vmin=-lim,
                           vmax=lim, shading="auto")
        ax.contour(s_mm, v_ms, a.T, levels=[0.0], colors="k", linewidths=1.0)
        launches = (np.linspace(0.4, 1.6, 10) * vc if vc > 0
                    else np.linspace(5.0, 0.9 * v_top, 10))
        for vl in launches:
            s, v, caught = trajectory(norm, a, s_mm, v_ms, vl)
            ax.plot(s, v, color="k" if caught else "0.4",
                    ls="-" if caught else ":", lw=1.4 if caught else 1.0)
        ax.plot([], [], color="k", lw=1.4, label="captured")
        ax.plot([], [], color="0.4", ls=":", lw=1.0, label="escaped")
        ax.axvspan(-fts.TRAP_HALF_MM, fts.TRAP_HALF_MM, color="k", alpha=0.05)
        ax.set_ylim(-5.0, v_top)
        ax.set_ylabel("axial velocity [m s$^{-1}$]")
        ax.set_title(f"{label}: centre-ray capture velocity {vc:.1f} m s$^{{-1}}$")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.2)
        fig.colorbar(im, ax=ax, label="axial acceleration [m s$^{-2}$]")
    axes[-1, 0].set_xlim(-fts.S_NEAR_MM, fts.S_PAST_TRAP_MM)
    axes[-1, 0].set_xlabel("position along the atom beam [mm]")
    return fig


def fig_capture_maps(offs_mm, maps):
    """v_c over the beam cross-section, one panel per trap, shared scale."""
    fig, axes = plt.subplots(1, len(maps), figsize=(6.6 * len(maps), 5.6),
                             constrained_layout=True, squeeze=False)
    ext = fts._cell_extent(offs_mm)
    vmax = max(vc.max() for _, vc in maps)
    for ax, (label, vc) in zip(axes[0], maps):
        im = ax.imshow(vc.T, origin="lower", extent=ext, cmap="magma",
                       aspect="equal", vmin=0, vmax=vmax)
        if vc.max() > 0:
            cs = ax.contour(offs_mm, offs_mm, vc.T, colors="w", linewidths=0.8)
            ax.clabel(cs, inline=True, fontsize=8, fmt="%.0f")
        fig.colorbar(im, ax=ax, label="capture velocity [m s$^{-1}$]")
        ax.set_xlabel("horizontal offset $e_h$ [mm]")
        ax.set_ylabel("offset $e_v$ [mm]")
        ax.set_title(f"{label}: peak {vc.max():.1f} m s$^{{-1}}$")
    return fig


def beam_paths(ax, model):
    """Where nozzle rays cross each MOT pair and shift pass axis, on an (e_h, e_v) map."""
    xl, yl = ax.get_xlim(), ax.get_ylim()
    b = np.array([-3.0, 3.0]) * fts.TRAP_HALF_MM
    halo = [pe.withStroke(linewidth=3.4, foreground="w", alpha=0.75)]
    dash = ((0, (5, 5)), (5, (5, 5)), (0, (5, 5)))
    lines = [(np.zeros(3), k, col, ls, lbl, 4)
             for (k, lbl, col), ls in zip(MOT_PAIRS, dash)]
    for n, sb in enumerate(model["lasers"][::2]):
        first = n % 2 == 0
        lines.append((sb["Rmat"] @ sb["r0"] * 1e3, sb["k_hat"], SHIFT_C,
                      "-" if first else ":",
                      "shift beam, first pass" if first else "shift beam, return pass", 3))
    seen = set()
    for c, k, col, ls, lbl, z in lines:
        n_vec = np.cross(U_HAT, k)
        if np.linalg.norm(n_vec) < 1e-6:
            lbl = "shift beam intersects atom beam"
            ax.plot(c @ E_H, c @ E_V, marker="X", ms=11, color=col, mec="w", mew=1.2,
                    ls="none", label=None if lbl in seen else lbl)
        else:
            nh, nv, d = E_H @ n_vec, E_V @ n_vec, c @ n_vec
            x, y = ((d - b * nv) / nh, b) if abs(nh) > abs(nv) else (b, (d - b * nh) / nv)
            ax.plot(x, y, color=col, ls=ls, lw=1.6, path_effects=halo, zorder=z,
                    label=None if lbl in seen else lbl)
        seen.add(lbl)
    ax.set_xlim(xl)
    ax.set_ylim(yl)


def with_beam_paths(fig, model):
    """Overlay beam_paths on every image panel of `fig`, legend on the first."""
    panels = [ax for ax in fig.axes if ax.images]
    for ax in panels:
        beam_paths(ax, model)
    if panels:
        panels[0].legend(loc="lower left", fontsize=7, framealpha=0.85)
    return fig


def fig_capture_diagnosis(model, norm, offs_mm, s_mm, v_ms, a_grid, vc, t_stop):
    """What sets the v_c map: cooling light per ray, where atoms settle, how long it takes."""
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11.5), constrained_layout=True)
    ext = fts._cell_extent(offs_mm)
    light = ray_light(model, norm, offs_mm)
    settle = settle_position(s_mm, v_ms, a_grid)
    live = vc > 0

    ax = axes[0, 0]
    im = ax.imshow(vc.T, origin="lower", extent=ext, cmap="magma", aspect="equal")
    fig.colorbar(im, ax=ax, label="capture velocity [m s$^{-1}$]")
    ax.set_title("(a) capture velocity and the beam axes each ray crosses")

    ax = axes[0, 1]
    im = ax.imshow(light.T, origin="lower", extent=ext, cmap="viridis", aspect="equal")
    fig.colorbar(im, ax=ax, label=r"$\int \sum_l s_l\,|\hat k_l\cdot\hat u|\,ds$ [mm]")
    r = np.corrcoef(vc[live], light[live])[0, 1] if live.sum() > 2 else float("nan")
    ax.set_title(f"(b) cooling light experienced by each ray (correlation with $v_c$ {r:.2f})")

    ax = axes[1, 0]
    lim = max(np.nanmax(np.abs(settle)), fts.CAPTURE_AXIAL_MM) if np.isfinite(settle).any() else 1.0
    im = ax.imshow(settle.T, origin="lower", extent=ext, cmap="RdBu_r", aspect="equal",
                   vmin=-lim, vmax=lim)
    fig.colorbar(im, ax=ax, label="rest point along the ray [mm]")
    if np.isfinite(settle).any():
        ax.contour(offs_mm, offs_mm, np.abs(np.nan_to_num(settle.T, nan=99.0)),
                   levels=[fts.CAPTURE_AXIAL_MM], colors="k", linewidths=1.2)
    ax.set_title(f"(c) where atom settles along the ray; black: $\\pm${fts.CAPTURE_AXIAL_MM:g} mm "
                 "capture box edge")

    ax = axes[1, 1]
    im = ax.imshow(np.where(live, t_stop * 1e3, np.nan).T, origin="lower", extent=ext,
                   cmap="cividis", aspect="equal")
    fig.colorbar(im, ax=ax, label="stopping time [ms]")
    ax.set_title(f"(d) stopping time of an atom at {fts.STOP_FRAC:g} $v_c$, "
                 "used by the drift cut")

    for a in axes.ravel():
        a.set_xlabel("horizontal offset $e_h$ [mm]")
        a.set_ylabel("offset $e_v$ [mm]")
    with_beam_paths(fig, model)
    for a in axes.ravel()[1:]:
        if a.get_legend():
            a.get_legend().remove()
    return fig


def fig_design_scan(scan, model):
    """Shift power, waist and detuning against what they buy: v_c, stiffness, depth."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11.5), constrained_layout=True)
    P, w = scan["P"], scan["w"] * 1e3
    maps = ((axes[0, 0], scan["vc_pw"], "magma", "centre-ray capture velocity [m s$^{-1}$]",
             "(a) capture velocity; white: z gradient [G/cm]", 0.0),
            (axes[0, 1], scan["freq"], "viridis", "trap frequency along the atom beam [Hz]",
             "(b) stiffness at the centre", 0.0),
            (axes[1, 0], scan["depth"], "cividis", "potential depth [mK]",
             "(c) escape barrier, weakest of x, y, z and the atom beam", None))
    for ax, data, cmap, label, title, vmin in maps:
        im = ax.pcolormesh(P, w, data.T, cmap=cmap, shading="nearest", vmin=vmin)
        fig.colorbar(im, ax=ax, label=label)
        ax.plot(SHIFT_POWER_W, SHIFT_WAIST_M * 1e3, marker="D", ms=9, color=SHIFT_C,
                mec="w", mew=1.5, ls="none", label="this run")
        ax.set_xscale("log")
        ax.set_xticks(P, [f"{p:.3g}" for p in P])
        ax.minorticks_off()
        ax.set_xlabel("shift power summed over every pass [W]")
        ax.set_ylabel("shift beam waist w [mm]")
        ax.set_title(title)
    cs = axes[0, 0].contour(P, w, np.abs(scan["grad"]).T,
                            levels=[2, 5, 10, 20, 50, 100, 200], colors="w",
                            linewidths=0.9)
    axes[0, 0].clabel(cs, inline=True, fontsize=8, fmt="%g")
    axes[0, 0].legend(loc="upper left", fontsize=9)

    ax = axes[1, 1]
    ax.plot(scan["D"], scan["vc_d"][0], "o-", color=SHIFT_C, lw=2.2,
            label="Doppler-shifted shift beams")
    ax.plot(scan["D"], scan["vc_d"][1], "s--", color=MOT_C, lw=1.6,
            label="static $B_{eff}$")
    ax.axvline(abs(model["delta_s"]) / osb.GAMMA_S, color="0.6", lw=1.0, ls=":")
    ax.set_xscale("log")
    ax.set_xticks(scan["D"], [f"{d:g}" for d in scan["D"]])
    ax.minorticks_off()
    ax.set_xlabel("|shift detuning| [$\\Gamma_s$]")
    ax.set_ylabel("centre-ray capture velocity [m s$^{-1}$]")
    ax.set_title(f"(d) capture velocity vs detuning at {SHIFT_POWER_W:g} W, "
                 f"w = {SHIFT_WAIST_M * 1e3:g} mm")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, which="both")
    fig.suptitle(f"Shift-beam design scan at |$\\Delta_s$| = "
                 f"{abs(model['delta_s']) / osb.GAMMA_S:g} $\\Gamma_s$ (a-c)", fontsize=13)
    return fig


def fig_oot_vs_mot(rows, rows_mot, t_free_mol_c, label_mot):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), constrained_layout=True)
    T = np.asarray([r["T_c"] for r in rows])
    R = np.asarray([r["rate"] for r in rows])
    Rm = np.asarray([r["rate"] for r in rows_mot])

    ax = axes[0]
    ax.semilogy(T, R, color=SHIFT_C, lw=2.4, label="OOT")
    ax.semilogy(T, Rm, color=MOT_C, lw=2.0, ls="--", label=label_mot)
    ax.set_ylabel("load rate [atoms s$^{-1}$]")
    ax.set_title("Capture rate, same atom beam and cooling beams")
    ax.legend(fontsize=9)

    ax = axes[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ax.plot(T, R / Rm, color=SHIFT_C, lw=2.4)
    ax.axhline(1.0, color="0.6", lw=1.0)
    ax.set_ylabel("OOT / MOT load rate")
    ax.set_title("Ratio")
    for a in axes:
        a.set_xlabel(r"oven temperature [$^\circ$C]")
        a.grid(alpha=0.25, which="both")
        if np.isfinite(t_free_mol_c):
            a.axvspan(t_free_mol_c, T.max(), color=LOSS_C, alpha=0.10)
    return fig


def main():
    reuse_nozzle_cache()
    rng = np.random.default_rng(fts.NOZZLE_SEED)
    norm = sim.build_normalization()
    model = build_oot(norm)
    t_free_mol_c = fts.free_molecular_limit_C(n3.L)

    beta = model["beta_G_cm"]
    beta_u = float(U_HAT @ beta @ U_HAT)
    reg = model["delta_s"] ** 2 / (model["delta_s"] ** 2 + osb.GAMMA_S ** 2 / 4)
    beta_paper = osb.paper_gradient(SHIFT_POWER_W, SHIFT_WAIST_M,
                                    math.radians(SHIFT_THETA_DEG),
                                    model["delta_s"], gF=GJ)[2, 2] * 1e2 * reg
    k_back = max((b["k_hat"] for b in model["lasers"]),
                 key=lambda k: -(k[:2] @ U_HAT[:2]) / max(np.linalg.norm(k[:2]), 1e-12))
    align = -(k_back[:2] @ U_HAT[:2]) / np.linalg.norm(k_back[:2])
    k_z = stiffness(model, Z_HAT)
    k_u = stiffness(model, U_HAT)
    laser_power = osb.total_power(model["lasers"], retro_reflected=RETRO)
    cool = model["cool_info"]

    params = {
        "atom_beam_azimuth_deg": fts.BEAM_TILT_DEG,
        "triad_extra_rotation_deg": TRIAD_ROT_DEG,
        "atom_beam_u_hat": U_HAT.tolist(),
        "nozzle_to_trap_mm": fts.NOZZLE_TO_TRAP_MM,
        "trap_half_width_mm": fts.TRAP_HALF_MM,
        "nozzle_atoms": fts.NOZZLE_ATOMS,
        "shift_power_all_passes_W": SHIFT_POWER_W,
        "shift_laser_power_W": laser_power,
        "shift_waist_m": SHIFT_WAIST_M,
        "shift_theta_deg": SHIFT_THETA_DEG,
        "shift_n_quad": N_QUAD,
        "shift_retro_reflected": int(RETRO),
        "shift_detuning_requested_gamma_s": SHIFT_DETUNING_GAMMA,
        "shift_detuning_used_gamma_s": model["delta_s"] / osb.GAMMA_S,
        "shift_detuning_auto_flipped": int(model["flipped"]),
        "shift_peak_intensity_per_component_W_m2": model["I_shift"],
        "shift_peak_intensity_per_component_Isat_s": model["I_shift"] / osb.ISAT_S,
        "shift_doppler": int(DOPPLER),
        "triad_phi0_deg": math.degrees(triad_phi0()),
        "gamma_s_per_s": osb.GAMMA_S,
        "lambda_s_nm": osb.LAMBDA_S * 1e9,
        "mot_beam_waist_m": fts.BEAM_WAIST_M,
        "mot_beam_average_saturation": fts.BEAM_AVG_SAT,
        "mot_beam_power_per_arm_W": cool["power_w"],
        "cooling_detuning_Hz": fts.DETUNING_HZ,
        "slowing_beam_enabled": int(fts.SLOWER_ENABLED),
        "slowing_detuning_Hz": fts._slower_detuning_hz(),
        "compare_mot": int(COMPARE_MOT),
        "mot_reference_current_A": model["mot_current_A"],
        "capture_offset_cells": int(fts.build_offset_grid()[0].size),
        "seed": fts.NOZZLE_SEED,
        "logical_cpus": os.cpu_count(),
    }

    with runlog.start("yb-oot-trap-sweep", params=fts.json_safe(params),
                      note="oven -> nozzle -> 344 mm -> OOT capture, "
                           "shift-beam triad instead of coils") as run:
        run.say(f"atom beam horizontal, {fts.BEAM_TILT_DEG:.1f} deg from the x pair; "
                f"shift pass most against it: k = ({k_back[0]:+.3f}, {k_back[1]:+.3f}, "
                f"{k_back[2]:+.3f}), {math.degrees(math.asin(k_back[2])):+.0f} deg "
                f"elevation, horizontal part antiparallel to cos = {align:.4f}")
        run.say(f"shift beams: {N_QUAD} quad beams, {4 * N_QUAD} circular components, "
                f"{SHIFT_POWER_W:g} W over every pass ({laser_power:.2f} W of laser"
                f"{', retro-reflected' if RETRO else ''}), w = {SHIFT_WAIST_M * 1e3:.1f} mm, "
                f"peak {model['I_shift'] / osb.ISAT_S:.0f} Isat_s per component")
        run.say(f"shift detuning {model['delta_s'] / osb.GAMMA_S:+.1f} Gamma_s "
                f"({model['delta_s'] / (2 * np.pi * 1e6):+.1f} MHz)"
                + (" , sign FLIPPED from the requested one so the MOT polarizations "
                   "trap" if model["flipped"] else ""))
        run.say(f"effective gradient (g = {GJ:g}): diag = "
                f"({beta[0, 0]:+.2f}, {beta[1, 1]:+.2f}, {beta[2, 2]:+.2f}) G/cm, "
                f"paper Eq. 3 gives {beta_paper:+.2f} on z; {beta_u:+.2f} G/cm along "
                f"the atom beam. Coil at {fts.OPT_CURRENT_A:g} A: "
                f"{cfm.axial_gradient_G_per_cm(fts.OPT_CURRENT_A):.2f} G/cm")
        run.say(f"centre stiffness: trap frequency {trap_freq_hz(model, k_z):.0f} Hz "
                f"along z, {trap_freq_hz(model, k_u):.0f} Hz along the atom beam"
                f"{'' if k_z < 0 and k_u < 0 else '  , WARNING: NOT TRAPPING'}")
        run.say(f"shift-beam Doppler {'ON' if DOPPLER else 'OFF'}: the counter-"
                f"propagating pass is resonant for atoms at "
                f"{abs(model['delta_s']) / osb.K_S:.0f} m/s")

        data = fts.load_or_trace_nozzle(run, rng)
        run.say(f"nozzle: W = {data['W']:.5f}, {data['n_accept']:,} of "
                f"{data['n_launched']:,} launched atoms reach the trap region")
        run.figure(fts.fig_beam_at_chamber(data), "atom_beam_at_chamber", close=True)
        run.figure(fig_oot_geometry(model), "oot_geometry", close=True)
        run.figure(fig_effective_field(model), "effective_field", close=True)
        run.figure(fig_force_cuts(model, norm), "force_cuts_through_centre",
                   close=True)

        v_min_ms, v_max_ms, n_v, v_res = fts.auto_velocity_grid(
            norm, fts._slower_detuning_hz())
        offs_mm = fts.build_offset_grid()[0]
        s_mm = fts.build_axial_grid()
        c = offs_mm.size // 2

        v_ms, a_grid, vc, t_stop = capture_map(run, model, norm, offs_mm, s_mm,
                                               v_min_ms, v_max_ms, n_v)
        run.say(f"OOT capture velocity: peak {vc.max():.1f} m/s, centre "
                f"{vc[c, c]:.1f} m/s")

        mot = None
        if COMPARE_MOT:
            run.say(f"coil MOT reference in the same geometry at "
                    f"{model['mot_current_A']:.2f} A ({model['coil_gradient_G_cm']:.2f} "
                    "G/cm axial)")
            v_ms_m, a_mot, vc_m, ts_m = capture_map(run, model, norm, offs_mm, s_mm,
                                                    v_min_ms, v_max_ms, n_v,
                                                    mot=True, verbose=False)
            _, rows_m, _ = fts.sweep(_Quiet(), data, offs_mm, vc_m, ts_m)
            mot = dict(v_ms=v_ms_m, a=a_mot, vc=vc_m, t_stop=ts_m, rows=rows_m)
            run.say(f"MOT reference capture velocity: peak {vc_m.max():.1f} m/s, "
                    f"centre {vc_m[c, c]:.1f} m/s")

        cells = [("OOT", a_grid[c, c])]
        vcs = [vc[c, c]]
        if mot is not None and np.allclose(mot["v_ms"], v_ms):
            cells.append((f"coil MOT, {model['coil_gradient_G_cm']:.1f} G/cm",
                          mot["a"][c, c]))
            vcs.append(mot["vc"][c, c])
        run.figure(fig_phase_portrait(norm, s_mm, v_ms, cells, vcs),
                   "phase_portrait_along_atom_beam", close=True)
        maps = [("OOT", vc)] + ([("coil MOT", mot["vc"])] if mot else [])
        run.figure(fig_capture_maps(offs_mm, maps), "capture_velocity_map",
                   close=True)
        run.figure(with_beam_paths(fig_capture_maps(offs_mm, maps), model),
                   "capture_velocity_map_beam_paths", close=True)
        run.figure(fig_capture_diagnosis(model, norm, offs_mm, s_mm, v_ms, a_grid,
                                         vc, t_stop),
                   "capture_map_diagnosis", close=True)
        run.figure(fts.fig_tilt_asymmetry(norm, offs_mm, s_mm, v_ms, a_grid, vc),
                   "tilt_induced_capture_asymmetry", close=True)

        temps_c, rows, aux = fts.sweep(run, data, offs_mm, vc, t_stop)
        show_T = [fts.T_MIN_C, 0.5 * (fts.T_MIN_C + fts.T_MAX_C), fts.T_MAX_C]
        run.figure(fts.fig_cross_section(data, aux, vc, offs_mm, show_T[1]),
                   "beam_cross_section_trapped_fraction", close=True)
        run.figure(with_beam_paths(fts.fig_cross_section(data, aux, vc, offs_mm,
                                                         show_T[1]), model),
                   "beam_cross_section_trapped_fraction_beam_paths", close=True)
        run.figure(fts.fig_cross_section_vs_T(data, aux, show_T, offs_mm),
                   "trapped_fraction_vs_temperature", close=True)
        run.figure(with_beam_paths(fts.fig_cross_section_vs_T(data, aux, show_T,
                                                              offs_mm), model),
                   "trapped_fraction_vs_temperature_beam_paths", close=True)
        run.figure(fts.fig_rate_vs_temperature(temps_c, rows, t_free_mol_c),
                   "load_rate_and_trapped_number", close=True)
        run.figure(fts.fig_speed_vs_capture(data, aux, vc, show_T),
                   "arriving_speed_vs_capture_velocity", close=True)
        scan = None
        if DESIGN_SCAN:
            t_scan = time.time()
            scan = design_scan(model, norm, s_mm, v_ms)
            run.figure(fig_design_scan(scan, model), "shift_beam_design_scan",
                       close=True)
            np.savez_compressed(run.out("design_scan.npz"), **scan)
            run.file("design_scan.npz", "design-scan")
            i, j = np.unravel_index(np.argmax(scan["vc_pw"]), scan["vc_pw"].shape)
            run.say(f"design scan ({time.time() - t_scan:.0f} s): centre-ray v_c "
                    f"{scan['vc_pw'].min():.1f} to {scan['vc_pw'].max():.1f} m/s, best at "
                    f"{scan['P'][i]:.2f} W, w = {scan['w'][j] * 1e3:.1f} mm")
        if mot is not None:
            run.figure(fig_oot_vs_mot(rows, mot["rows"], t_free_mol_c,
                                      f"coil MOT, {model['coil_gradient_G_cm']:.1f} G/cm"),
                       "oot_vs_mot_load_rate", close=True)

        cols = [[r["T_c"] for r in rows], [r["flux_out"] for r in rows],
                [r["flux_landing"] for r in rows], [r["rate"] for r in rows],
                [r["frac_of_landing_caught"] for r in rows]]
        cols += [[r["rate"] * tau for r in rows] for tau in fts.LIFETIMES_S]
        header = ("oven_T_C,flux_out_of_nozzle_per_s,flux_into_trap_region_per_s,"
                  "oot_load_rate_per_s,fraction_of_arrivals_captured,"
                  + ",".join(f"N_steady_tau_{tau:g}s" for tau in fts.LIFETIMES_S))
        if mot is not None:
            cols.append([r["rate"] for r in mot["rows"]])
            header += ",mot_reference_load_rate_per_s"
        np.savetxt(run.out("oot_trap_sweep.csv"), np.column_stack(cols),
                   delimiter=",", header=header, comments="")
        run.file("oot_trap_sweep.csv", "sweep-table")

        extra = {}
        if mot is not None:
            extra = dict(mot_v_ms=mot["v_ms"], mot_capture_velocity_m_per_s=mot["vc"],
                         mot_stopping_time_s=mot["t_stop"],
                         mot_axial_acceleration_bar=mot["a"])
        np.savez_compressed(run.out("capture_map.npz"), offsets_mm=offs_mm,
                            capture_velocity_m_per_s=vc, stopping_time_s=t_stop,
                            s_mm=s_mm, v_ms=v_ms, axial_acceleration_bar=a_grid,
                            **extra)
        run.file("capture_map.npz", "capture-map")

        best = max(rows, key=lambda r: r["rate"])
        safe = [r for r in rows
                if not np.isfinite(t_free_mol_c) or r["T_c"] <= t_free_mol_c]
        best_safe = max(safe, key=lambda r: r["rate"]) if safe else best
        at_400 = min(rows, key=lambda r: abs(r["T_c"] - 400.0))
        for r in (rows[0], rows[len(rows) // 2], rows[-1]):
            run.say(f"{r['T_c']:.0f} C: OOT load rate {r['rate']:.3e} atoms/s, "
                    f"{r['frac_of_landing_caught']:.3e} of arrivals")

        summary = {
            "shift_laser_power_W": laser_power,
            "shift_detuning_used_gamma_s": model["delta_s"] / osb.GAMMA_S,
            "shift_detuning_auto_flipped": int(model["flipped"]),
            "effective_gradient_z_G_per_cm": beta[2, 2],
            "effective_gradient_xy_G_per_cm": beta[0, 0],
            "effective_gradient_along_atom_beam_G_per_cm": beta_u,
            "paper_eq3_gradient_z_G_per_cm": beta_paper,
            "trap_frequency_z_Hz": trap_freq_hz(model, k_z),
            "trap_frequency_along_atom_beam_Hz": trap_freq_hz(model, k_u),
            "capture_velocity_peak_m_per_s": float(vc.max()),
            "capture_velocity_centre_m_per_s": float(vc[c, c]),
            "velocity_grid_max_m_per_s": float(v_ms[-1]),
            "best_load_rate_atoms_per_s": best["rate"],
            "best_load_rate_at_C": best["T_c"],
            "best_load_rate_within_free_molecular_atoms_per_s": best_safe["rate"],
            "best_load_rate_within_free_molecular_at_C": best_safe["T_c"],
            "load_rate_at_nearest_400C": at_400["rate"],
            "load_rate_reported_at_C": at_400["T_c"],
            "free_molecular_limit_C": t_free_mol_c,
            "median_stopping_time_ms": float(np.median(t_stop[vc > 0]) * 1e3)
            if (vc > 0).any() else float("nan"),
        }
        if scan is not None:
            i, j = np.unravel_index(np.argmax(scan["vc_pw"]), scan["vc_pw"].shape)
            summary.update(design_scan_best_centre_vc_m_per_s=float(scan["vc_pw"][i, j]),
                           design_scan_best_power_W=float(scan["P"][i]),
                           design_scan_best_waist_mm=float(scan["w"][j] * 1e3))
        if mot is not None:
            m400 = min(mot["rows"], key=lambda r: abs(r["T_c"] - 400.0))
            summary.update(
                mot_reference_current_A=model["mot_current_A"],
                mot_reference_gradient_G_per_cm=model["coil_gradient_G_cm"],
                mot_reference_capture_velocity_peak_m_per_s=float(mot["vc"].max()),
                mot_reference_capture_velocity_centre_m_per_s=float(mot["vc"][c, c]),
                mot_reference_load_rate_at_nearest_400C=m400["rate"],
                oot_over_mot_load_rate_at_nearest_400C=(
                    at_400["rate"] / m400["rate"] if m400["rate"] > 0 else float("nan")))
        run.finish(summary=fts.json_safe(summary))


if __name__ == "__main__":
    main()
