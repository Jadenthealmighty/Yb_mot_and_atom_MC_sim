"""MOT trap depth in uK over coil current vs MOT detuning, no slowing beam.

"""

import contextlib
import io
import itertools
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "oot"))

import beam_sweep as bs
import mot_sweep as ms
import runlog

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sp_const

import helpers.coil_field_model as cfm
import helpers.trap_beams as tb
import helpers.yb174_mot_simulation as sim
import oot_rate_eq as req

from beam_sweep import _envf, _floats, INK, INK2, MUTED, SURFACE

CURRENTS_A = _floats("MD_CURRENTS_A", "20,40,60,80,100,120,140")
DETUNINGS_GAMMA = _floats("MD_DETUNINGS_GAMMA",
                          "-0.25,-0.5,-0.75,-1,-1.25,-1.5,-2,-2.5,-3")
MOT_RADIUS_MM = _envf("MD_MOT_RADIUS_MM", 5)
TOTAL_MW = _envf("MD_TOTAL_MW", 30.0)
V_RATIO = _envf("MD_V_RATIO", 2.0)
# Escaped once past here; light is gone at ~3 w except along the beam axes
R_MAX_MM = _envf("MD_R_MAX_MM", 3.0 * MOT_RADIUS_MM)
DS_MM = _envf("MD_DS_MM", 0.05)
GAMMA_HZ = sim.YB174_LINEWIDTH_HZ


def beam_powers_mW():
    """(horizontal, vertical) power of one MOT beam, TOTAL_MW over the 3."""
    p_h = TOTAL_MW / (2.0 + V_RATIO)
    return p_h, V_RATIO * p_h


def directions():
    """26 unit vectors to the neighbours of a cube, with labels like +x-z."""
    out, names = [], []
    for c in itertools.product((-1, 0, 1), repeat=3):
        if any(c):
            out.append(np.array(c, dtype=float) / np.linalg.norm(c))
            names.append("".join(f"{'+' if ci > 0 else '-'}{ax}"
                                 for ci, ax in zip(c, "xyz") if ci))
    return np.array(out), names


def build_model():
    """Units, cooling beams at zero detuning, and the field along every ray."""
    with contextlib.redirect_stdout(io.StringIO()):
        norm = sim.build_normalization()
    p_h, p_v = beam_powers_mW()
    beams, _ = tb.build_trap_beams(
        norm, 0.0, waist_m=MOT_RADIUS_MM * 1e-3,
        power_w=(p_h * 1e-3,) * 4 + (p_v * 1e-3,) * 2,
        wavelength_m=sim.YB174_WAVELENGTH_M)
    cool = req.CoolingBeams(beams)

    centre_m = np.array([0.0, 0.0, cfm.field_zero_z_mm() * 1e-3])
    dhat, names = directions()
    n_steps = int(round(R_MAX_MM / DS_MM))
    s_m = np.linspace(0.0, R_MAX_MM * 1e-3, 2 * n_steps + 1)   # RK4 half steps
    R_m = centre_m[None, None, :] + dhat[:, None, :] * s_m[None, :, None]
    flat = R_m.reshape(-1, 3)

    return dict(
        norm=norm, cool=cool, dhat=dhat, names=names, s_m=s_m,
        centre_m=centre_m,
        B1=cfm.oswald_coil_bfield(flat, 1.0).reshape(R_m.shape),   # T at 1 A
        sat=cool.intensity(flat.T / norm["x0"]).reshape(
            len(cool.beams), *R_m.shape[:2]),
        b_conv=(sim.YB174_GJ_EXCITED * sp_const.value("Bohr magneton")
                / (sp_const.hbar * norm["Gamma_SI"])),
        a_unit=sp_const.hbar * norm["k_SI"] * norm["Gamma_SI"] / norm["mass_SI"])


def outward_accel(m, j, v_ms, I, d):
    """d_hat . a (n_dir, n_I, n_d) in m/s^2 at node j, outward speed v_ms."""
    n_dir = m["dhat"].shape[0]
    shape = (n_dir, I.size, d.size)
    N = int(np.prod(shape))
    B = (m["b_conv"] * m["B1"][:, j, None, None, :]
         * I[None, :, None, None]) * np.ones(shape + (3,))
    # MOT detuning folded into H: delta - kv - E = -kv - (E - delta)
    delta = np.broadcast_to(2 * np.pi * d * GAMMA_HZ / m["norm"]["Gamma_SI"],
                            shape).reshape(N, 1, 1)
    H = req.zeeman(B.reshape(N, 3).T) - delta * np.eye(3)
    s = np.broadcast_to(m["sat"][:, :, j, None, None],
                        (len(m["cool"].beams),) + shape).reshape(-1, N)
    V = (np.broadcast_to(v_ms, shape)[..., None] * m["dhat"][:, None, None, :]
         / m["norm"]["v0"]).reshape(N, 3).T
    F = req.steady_force(H, s, m["cool"].khat, m["cool"].pol,
                         np.zeros(len(m["cool"].beams)), V)
    Fd = np.einsum("kn,nk->n", F, np.repeat(m["dhat"], I.size * d.size, 0))
    return m["a_unit"] * Fd.reshape(shape)


def escape_energy(m, I, d):
    """Escape kinetic energy per mass (n_dir, n_I, n_d), m^2/s^2.

    RK4 on E = v^2 / 2, dE/ds = a(s, v), inward from R_MAX where E = 0. The
    least E that still escapes from s is max(0, ...): ODE paths in (s, E)
    never cross, so this is the critical path without bisection.
    """
    h = m["s_m"][2] - m["s_m"][0]
    n = (m["s_m"].size - 1) // 2
    E = np.zeros((m["dhat"].shape[0], I.size, d.size))
    f = lambda j, E: outward_accel(m, j, np.sqrt(2 * np.maximum(E, 0.0)), I, d)
    for k in range(n, 0, -1):
        j = 2 * k
        k1 = f(j, E)
        k2 = f(j - 1, E - 0.5 * h * k1)
        k3 = f(j - 1, E - 0.5 * h * k2)
        k4 = f(j - 2, E - h * k3)
        E = np.maximum(E - h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), 0.0)
    return E


def static_barrier(m, I, d):
    """max_s of -int_0^s a(s', v = 0) ds' per mass (n_dir, n_I, n_d), m^2/s^2."""
    a = np.array([outward_accel(m, j, 0.0, I, d)
                  for j in range(m["s_m"].size)])
    U = -np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(
        0.5 * (a[1:] + a[:-1]) * np.diff(m["s_m"])[:, None, None, None], 0)])
    return U.max(axis=0)


def to_uK(m, e_per_mass):
    return m["norm"]["mass_SI"] * e_per_mass / sp_const.k * 1e6


def trap_depth_uK(current_A, detuning_gamma, m=None):
    """Escape depth in uK for one point, for use from other scripts."""
    m = build_model() if m is None else m
    E = escape_energy(m, np.atleast_1d(float(current_A)),
                      np.atleast_1d(float(detuning_gamma)))
    return float(to_uK(m, E.min()))


def fig_maps(g):
    """Escape depth and static barrier over the grid."""
    scale = bs._scale(g["depth"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.8))
    ms._heatmap(fig, a1, g, g["depth"] / scale,
                rf"trap depth [{bs._sci(scale)} $\mu$K]")
    ms._heatmap(fig, a2, g, g["static"] / scale,
                rf"static barrier [{bs._sci(scale)} $\mu$K]")
    a1.set_title(r"Escape depth, $\frac{1}{2} m v_{esc}^2$, weakest direction",
                 loc="left")
    a2.set_title(r"Static barrier, $-\int F(r, v=0)\,dr$, weakest direction",
                 loc="left")
    fig.suptitle(_title(), x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    return fig


def fig_cuts(g):
    """Depth against each knob separately."""
    scale = bs._scale(g["depth"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, x, Y, labs, xlab, leg in (
            (a1, g["I"], g["depth"], g["d"], "coil current [A]",
             r"detuning [$\Gamma$]"),
            (a2, g["d"], g["depth"].T, g["I"], r"MOT detuning [$\Gamma$]",
             "current [A]")):
        for k, lab in enumerate(labs):
            color = bs.ORDINAL(k / max(labs.size - 1, 1))
            ax.plot(x, Y[k] / scale, "o-", color=color, label=f"{lab:g}",
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_xlabel(xlab)
        ax.set_ylim(bottom=0)
        ax.legend(title=leg, title_fontsize=9, loc="best")
    a2.invert_xaxis()
    a1.set_ylabel(rf"trap depth [{bs._sci(scale)} $\mu$K]")
    a1.set_title("Raising current, one line per detuning", loc="left")
    a2.set_title("Raising detuning, one line per current", loc="left")
    fig.tight_layout()
    return fig


def _title():
    p_h, p_v = beam_powers_mW()
    return (f"MOT only, {TOTAL_MW:g} mW ({p_h:.3g} / {p_h:.3g} / {p_v:.3g} mW "
            f"x / y / z), w = {MOT_RADIUS_MM:g} mm, escape at "
            f"{R_MAX_MM:g} mm, $\\Gamma/2\\pi$ = {GAMMA_HZ / 1e6:g} MHz")


def text_table(g):
    lines = ["trap depth [uK]",
             "  d/G \\ I A " + "".join(f"{i:>11g}" for i in g["I"])]
    for k, d in enumerate(g["d"]):
        lines.append(f"  {d:>9g}  " + "".join(f"{v:>11.3e}" for v in g["depth"][k]))
    return lines


def main():
    I = np.array(sorted(CURRENTS_A))
    d = np.array(sorted(DETUNINGS_GAMMA, reverse=True))   # ms._heatmap row order
    p_h, p_v = beam_powers_mW()
    bs._style()

    params = dict(
        currents_A=I, detunings_gamma=d, mot_radius_mm=MOT_RADIUS_MM,
        total_power_mW=TOTAL_MW, vertical_power_ratio=V_RATIO,
        horizontal_power_mW=p_h, vertical_power_mW=p_v, r_max_mm=R_MAX_MM,
        ds_mm=DS_MM, linewidth_MHz=GAMMA_HZ / 1e6)

    with runlog.start("yb-mot-depth", params=params,
                      note=f"{I.size * d.size} points, slowing beam off") as run:
        m = build_model()
        run.say(f"trap centre at z = {m['centre_m'][2] * 1e3:+.2f} mm (B = 0), "
                f"{len(m['names'])} escape directions, "
                f"{(m['s_m'].size - 1) // 2} steps to {R_MAX_MM:g} mm")
        E = escape_energy(m, I, d)
        U = static_barrier(m, I, d)
        weak = E.argmin(axis=0)

        depth = to_uK(m, E.min(axis=0)).T          # (n_d, n_I)
        static = to_uK(m, U.min(axis=0)).T
        grad = np.array([cfm.axial_gradient_G_per_cm(i) for i in I])
        g = dict(I=I, d=d, grad=grad, depth=depth, static=static)

        rows = [dict(current_A=I[a], gradient_G_per_cm=grad[a],
                     detuning_gamma=d[b], detuning_MHz=d[b] * GAMMA_HZ / 1e6,
                     trap_depth_uK=depth[b, a],
                     escape_speed_m_per_s=math.sqrt(2 * E[weak[a, b], a, b]),
                     weakest_direction=m["names"][weak[a, b]],
                     static_barrier_uK=static[b, a],
                     horizontal_power_mW=p_h, vertical_power_mW=p_v)
                for b in range(d.size) for a in range(I.size)]
        bs.write_csv(run.out("trap_depth.csv"), rows)
        run.file("trap_depth.csv", "one row per point")
        run.figure(fig_maps(g), "trap_depth_map", close=True)
        run.figure(fig_cuts(g), "trap_depth_cuts", close=True)
        for line in text_table(g):
            run.say(line)

        best = max(rows, key=lambda r: r["trap_depth_uK"])
        run.finish(summary=dict(
            best_trap_depth_uK=best["trap_depth_uK"],
            best_current_A=best["current_A"],
            best_gradient_G_per_cm=best["gradient_G_per_cm"],
            best_detuning_gamma=best["detuning_gamma"],
            best_escape_speed_m_per_s=best["escape_speed_m_per_s"],
            best_weakest_direction=best["weakest_direction"], points=len(rows)))


if __name__ == "__main__":
    main()
