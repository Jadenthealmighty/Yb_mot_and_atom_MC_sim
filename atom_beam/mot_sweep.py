"""sweep coil current vs MOT detuning, no slowing beam
"""

import contextlib
import glob
import hashlib
import io
import json
import math
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["FT_NOZZLE_CACHE"] = os.path.join(
    HERE, os.environ.get("FT_NOZZLE_CACHE", "nozzle_trace_cache.npz"))

import full_trap_sweep as ft
import beam_sweep as bs
import runlog

import matplotlib.pyplot as plt
import numpy as np

from beam_sweep import _envf, _floats, INK, INK2, MUTED, SURFACE

CURRENTS_A = _floats("MS_CURRENTS_A", "100,110,120")
DETUNINGS_GAMMA = _floats("MS_DETUNINGS_GAMMA",
                          "-1,-1.5,-2")
MOT_RADIUS_MM = _envf("MS_MOT_RADIUS_MM", 5.0)
TOTAL_MW = _envf("MS_TOTAL_MW", 30.0)
V_RATIO = _envf("MS_V_RATIO", 2.0)
REF_TEMP_C = _envf("MS_REF_TEMP_C", 400.0)
POINTS_ROOT = os.path.join(HERE, os.environ.get(
    "MS_POINTS_DIR", os.path.join("runs", "mot_sweep_points")))
ALLOW_RETRACE = bool(int(os.environ.get("MS_ALLOW_RETRACE", "0")))
GAMMA_HZ = ft.sim.YB174_LINEWIDTH_HZ


def isat_mW_cm2():
    with contextlib.redirect_stdout(io.StringIO()):
        return ft.sim.build_normalization()["Isat_mW_cm2"]


def beam_powers_mW():
    """(horizontal, vertical) power of one MOT beam, TOTAL_MW over the 3."""
    p_h = TOTAL_MW / (2.0 + V_RATIO)
    return p_h, V_RATIO * p_h


def point_env(current_A, gamma, avg_sat):
    """FT_* overrides that turn full_trap_sweep into one sweep point."""
    return {
        "FT_SLOWER": "0",
        "FT_WAIST_M": f"{MOT_RADIUS_MM * 1e-3:.6g}",
        "FT_AVG_SAT": f"{avg_sat:.6g}",
        "FT_MOT_V_RATIO": f"{V_RATIO:.6g}",
        "FT_DETUNING_HZ": f"{gamma * GAMMA_HZ:.8g}",
        "FT_COIL_CURRENT_A": f"{current_A:.6g}",
        "FT_TOLERANCE": "0",
        "FT_SLOWER_SCAN": "0",
    }


def code_hash():
    """Short hash of the code and FT_* overrides a point depends on."""
    h = hashlib.sha1()
    for path in [os.path.join(HERE, "full_trap_sweep.py"),
                 *sorted(glob.glob(os.path.join(HERE, "helpers", "*.py")))]:
        with open(path, "rb") as fh:
            h.update(fh.read())
    h.update(repr(sorted((k, v) for k, v in os.environ.items()
                         if k.startswith("FT_"))).encode())
    return h.hexdigest()[:10]


def point_dir(root, current_A, gamma, env):
    env_hash = hashlib.sha1(json.dumps(env, sort_keys=True).encode())
    return os.path.join(root, f"I{current_A:05.1f}A_d{gamma:+.2f}G_"
                              f"{env_hash.hexdigest()[:6]}")


def read_point(run_dir):
    """Load rate at REF_TEMP_C plus the diagnostics worth tabulating."""
    with open(os.path.join(run_dir, "summary.json")) as fh:
        summary = json.load(fh)
    sweep = np.genfromtxt(os.path.join(run_dir, "full_trap_sweep.csv"),
                          delimiter=",", names=True)
    vc = np.load(os.path.join(run_dir, "capture_map.npz"))[
        "capture_velocity_m_per_s"]
    rim = np.concatenate([vc[0], vc[-1], vc[:, 0], vc[:, -1]])
    return dict(
        load_rate=float(np.interp(REF_TEMP_C, np.atleast_1d(sweep["oven_T_C"]),
                                  np.atleast_1d(sweep["mot_load_rate_per_s"]))),
        best_load_rate=summary["best_load_rate_atoms_per_s"],
        best_load_rate_T_C=summary["best_load_rate_at_C"],
        gradient_G_per_cm=summary["axial_gradient_G_per_cm"],
        capture_velocity_peak_m_per_s=summary["capture_velocity_peak_m_per_s"],
        capture_clipped=bool(rim.max() > 0),
        run_dir=os.path.relpath(run_dir, HERE))


def grid(rows):
    """(detuning x current) grids of the results."""
    d = np.array(sorted({r["detuning_gamma"] for r in rows}, reverse=True))
    I = np.array(sorted({r["current_A"] for r in rows}))
    at = {(r["detuning_gamma"], r["current_A"]): r for r in rows}
    g = lambda key: np.array([[at[(di, ii)][key] for ii in I] for di in d],
                             dtype=float)
    return dict(d=d, I=I, R=g("load_rate"),
                vc=g("capture_velocity_peak_m_per_s"),
                clipped=g("capture_clipped").astype(bool),
                grad=g("gradient_G_per_cm").mean(axis=0))


def _grid_axes(ax, g):
    ax.set_xticks(range(g["I"].size), [f"{x:g}" for x in g["I"]])
    ax.set_yticks(range(g["d"].size), [f"{x:g}" for x in g["d"]])
    ax.set_xlabel("coil current [A]")
    ax.set_ylabel(r"MOT detuning [$\Gamma$]")
    top = ax.secondary_xaxis("top")
    top.set_xticks(range(g["I"].size), [f"{x:.1f}" for x in g["grad"]])
    top.set_xlabel("axial gradient [G/cm]", color=INK2)
    top.tick_params(colors=MUTED)
    ax.grid(False)
    for spine in (*ax.spines.values(), *top.spines.values()):
        spine.set_visible(False)


def _heatmap(fig, ax, g, Z, label):
    """Grid with a star on the best cell."""
    im = ax.imshow(Z, origin="lower", cmap=bs.SEQUENTIAL, aspect="auto",
                   vmin=0, vmax=np.nanmax(Z) if np.nanmax(Z) > 0 else 1)
    i, j = np.unravel_index(np.nanargmax(Z), Z.shape)
    ax.plot(j, i, "*", color=SURFACE, markersize=13, markeredgecolor=INK,
            markeredgewidth=0.8)
    _grid_axes(ax, g)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label(label, color=INK2)
    cb.outline.set_visible(False)


def _title():
    p_h, p_v = beam_powers_mW()
    return (f"MOT only, {TOTAL_MW:g} mW ({p_h:.3g} / {p_h:.3g} / {p_v:.3g} mW "
            f"x / y / z), w = {MOT_RADIUS_MM:g} mm")


def fig_maps(g):
    """Load rate and peak capture velocity over the grid."""
    scale = bs._scale(g["R"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.8))
    _heatmap(fig, a1, g, g["R"] / scale,
             f"load rate at {REF_TEMP_C:.0f} C [{bs._sci(scale)} atoms/s]")
    _heatmap(fig, a2, g, g["vc"], "peak capture velocity [m/s]")
    a1.set_title("Load rate", loc="left")
    a2.set_title("Capture velocity", loc="left")
    fig.suptitle(_title(), x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    return fig


def fig_cuts(g):
    """Load rate against each knob separately."""
    scale = bs._scale(g["R"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, x, Y, F, labs, xlab, leg in (
            (a1, g["I"], g["R"], g["clipped"], g["d"], "coil current [A]",
             r"detuning [$\Gamma$]"),
            (a2, g["d"], g["R"].T, g["clipped"].T, g["I"],
             r"MOT detuning [$\Gamma$]", "current [A]")):
        for k, lab in enumerate(labs):
            color = bs.ORDINAL(k / max(labs.size - 1, 1))
            y, f = Y[k] / scale, F[k]
            ax.plot(x, y, color=color, label=f"{lab:g}", zorder=2)
            ax.plot(x[~f], y[~f], "o", color=color, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.plot(x[f], y[f], "o", color=SURFACE, zorder=3,
                    markeredgecolor=color, markeredgewidth=1.8)
        ax.set_xlabel(xlab)
        ax.set_ylim(bottom=0)
        ax.legend(title=leg, title_fontsize=9, loc="best")
    a2.invert_xaxis()
    a1.set_ylabel(f"load rate at {REF_TEMP_C:.0f} C [{bs._sci(scale)} atoms/s]")
    a1.set_title("Raising current, one line per detuning", loc="left")
    a2.set_title("Raising detuning, one line per current", loc="left")
    fig.tight_layout()
    return fig


def text_table(g):
    """The load-rate grid as terminal text, for reading over ssh."""
    lines = [f"load rate at {REF_TEMP_C:.0f} C [atoms/s]",
             "  d/G \\ I A " + "".join(f"{i:>11g}" for i in g["I"])]
    for k, d in enumerate(g["d"]):
        lines.append(f"  {d:>9g}  " + "".join(
            f"{v:>10.3e}{bs.DAGGER if c else ' '}"
            for v, c in zip(g["R"][k], g["clipped"][k])))
    return lines


def main():
    points = [(i, d) for d in DETUNINGS_GAMMA for i in CURRENTS_A]
    mismatch = ft.nozzle_cache_mismatch()
    if mismatch is None or mismatch:
        why = ("there is no nozzle cache" if mismatch is None
               else f"the nozzle cache does not match on {mismatch}")
        msg = (f"{why} ({os.environ['FT_NOZZLE_CACHE']}), so the first point "
               "would retrace the nozzle, which takes hours.")
        if not ALLOW_RETRACE:
            raise SystemExit(msg + " Fix the cache, or set MS_ALLOW_RETRACE=1 "
                             "to let the first point rebuild it.")
        print("WARNING: " + msg, flush=True)

    isat = isat_mW_cm2()
    p_h, p_v = beam_powers_mW()
    s_h = p_h / float(bs.beam_mW(1.0, MOT_RADIUS_MM, isat))
    root = os.path.join(POINTS_ROOT, code_hash())
    bs._style()
    params = dict(
        currents_A=CURRENTS_A, detunings_gamma=DETUNINGS_GAMMA,
        mot_radius_mm=MOT_RADIUS_MM, total_power_mW=TOTAL_MW,
        vertical_power_ratio=V_RATIO, horizontal_power_mW=p_h,
        vertical_power_mW=p_v, horizontal_avg_sat=s_h,
        vertical_avg_sat=V_RATIO * s_h, ref_temp_C=REF_TEMP_C,
        points_dir=os.path.relpath(root, HERE), n_points=len(points),
        logical_cpus=os.cpu_count(), workers_per_point=ft.worker_count(10**6))

    with runlog.start("yb-mot-sweep", params=params,
                      note=f"{len(points)} full_trap_sweep points, "
                           "slowing beam off") as run:
        run.say(f"MOT beams: {p_h:.3g} mW horizontal, {p_v:.3g} mW vertical "
                f"at w = {MOT_RADIUS_MM:g} mm -> avg {s_h:.3f} / "
                f"{V_RATIO * s_h:.3f} Isat")
        run.say(f"points kept in {os.path.relpath(root, HERE)}")
        plan = []
        for i, d in points:
            env = point_env(i, d, s_h)
            plan.append((i, d, env, point_dir(root, i, d, env)))
        n_done = sum(1 for p in plan if bs.finished_run(p[-1]))
        if n_done:
            run.say(f"{n_done} of {len(plan)} points already finished and "
                    "will be reused")

        rows, spent, n_ran = [], 0.0, 0
        for k, (i, d, env, pdir) in enumerate(plan, start=1):
            run.say(f"[{k}/{len(plan)}] {i:g} A, {d:+g} Gamma "
                    f"({d * GAMMA_HZ / 1e6:+.1f} MHz)")
            t0 = time.time()
            run_dir, reused = bs.run_point(env, pdir)
            if not reused:
                spent += time.time() - t0
                n_ran += 1
            row = dict(current_A=i, detuning_gamma=d,
                       detuning_MHz=d * GAMMA_HZ / 1e6,
                       horizontal_power_mW=p_h, vertical_power_mW=p_v,
                       horizontal_avg_sat=s_h, vertical_avg_sat=V_RATIO * s_h,
                       **read_point(run_dir))
            rows.append(row)
            left = sum(1 for p in plan[k:] if not bs.finished_run(p[-1]))
            eta = (f", ~{left * spent / n_ran / 60:.0f} min left"
                   if n_ran and left else "")
            run.say(f"      -> {row['load_rate']:.3e} atoms/s, v_c "
                    f"{row['capture_velocity_peak_m_per_s']:.1f} m/s"
                    f"{' (reused)' if reused else f' in {time.time() - t0:.0f} s'}"
                    f"{eta}")
            run.log(step=k, load_rate=row["load_rate"])

        bs.write_csv(run.out("sweep_results.csv"), rows)
        run.file("sweep_results.csv", "one row per point")
        g = grid(rows)
        run.figure(fig_maps(g), "load_rate_map", close=True)
        run.figure(fig_cuts(g), "load_rate_cuts", close=True)
        for line in text_table(g):
            run.say(line)

        best = max(rows, key=lambda r: r["load_rate"])
        run.finish(summary=dict(
            best_load_rate_atoms_per_s=best["load_rate"],
            best_current_A=best["current_A"],
            best_gradient_G_per_cm=best["gradient_G_per_cm"],
            best_detuning_gamma=best["detuning_gamma"],
            best_detuning_MHz=best["detuning_MHz"],
            ref_temp_C=REF_TEMP_C, points=len(rows), points_run_now=n_ran,
            points_capture_clipped=sum(r["capture_clipped"] for r in rows)))


if __name__ == "__main__":
    main()
