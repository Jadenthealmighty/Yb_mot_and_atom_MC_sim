"""sweep radii of MOT and slowing vs saturation
"""

import csv
import glob
import hashlib
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["FT_NOZZLE_CACHE"] = os.path.join(
    HERE, os.environ.get("FT_NOZZLE_CACHE", "nozzle_trace_cache.npz"))

import full_trap_sweep as ft
import runlog

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MaxNLocator


def _envf(name, default):
    return float(os.environ.get(name, default))


def _floats(name, default):
    raw = os.environ.get(name, default).strip()
    return tuple(float(x) for x in raw.split(",")) if raw else ()


SLOWER_AVG_SATS = _floats("SW_SLOWER_AVG_SATS", "0.2,0.4,0.6,0.8,1.0")
RADII_MM = _floats("SW_RADII_MM", "5,7.5,10,12.5,15")
MOT_RADII_MM = _floats("SW_MOT_RADII_MM", "")
SLOWER_POL = int(os.environ.get("SW_SLOWER_POL", "-1"))
TABLE_CSV = os.path.join(HERE, os.environ.get("SW_TABLE",
                                              "power-sweep-table.csv"))
REF_TEMP_C = _envf("SW_REF_TEMP_C", 400.0)
MOT_ARMS = int(os.environ.get("SW_MOT_ARMS", "6"))
POINTS_ROOT = os.path.join(HERE, os.environ.get(
    "SW_POINTS_DIR", os.path.join("runs", "beam_sweep_points")))
ALLOW_RETRACE = bool(int(os.environ.get("SW_ALLOW_RETRACE", "0")))

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
ORDINAL = LinearSegmentedColormap.from_list(
    "ordinal", ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"])
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "sequential", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "diverging", ["#1c5cab", "#86b6ef", "#f0efec", "#f09a99", "#b32f2f"])
LEVER_CAP = 1.0
DAGGER = "\u2020"


def load_table(path, pol):
    """The optimal-settings rows for one slowing-beam polarization, by s0."""
    with open(path) as fh:
        rows = [r for r in csv.DictReader(fh) if r["branch"] == f"pol={pol:+d}"]
    if not rows:
        raise SystemExit(f"{path} has no rows for pol={pol:+d}")
    rows.sort(key=lambda r: float(r["s0"]))
    col = lambda k: np.array([float(r[k]) for r in rows])
    return dict(s0=col("s0"), current_A=col("optimum_current_A"),
                detuning_MHz=col("optimum_detuning_MHz"),
                gradient=col("optimum_gradient_G_per_cm"))


def table_settings(table, s0_peak):
    """Coil current and slower detuning at peak saturation s0_peak."""
    s0 = table["s0"]
    return dict(
        coil_current_A=float(np.interp(s0_peak, s0, table["current_A"])),
        slower_detuning_MHz=float(np.interp(s0_peak, s0,
                                            table["detuning_MHz"])),
        table_gradient_G_per_cm=float(np.interp(s0_peak, s0,
                                                table["gradient"])),
        table_clamped=bool(not s0[0] <= s0_peak <= s0[-1]))


def sweep_points():
    """(slower avg saturation, slower radius mm, MOT radius mm) per run."""
    if not MOT_RADII_MM:
        return [(s, r, r) for s in SLOWER_AVG_SATS for r in RADII_MM]
    return [(s, r, m) for s in SLOWER_AVG_SATS for r in RADII_MM
            for m in MOT_RADII_MM]


def point_env(s_avg, r_mm, mot_mm, settings):
    """FT_* overrides that turn full_trap_sweep into one sweep point."""
    return {
        "FT_SLOWER": "1",
        "FT_SLOWER_POL": str(SLOWER_POL),
        "FT_SLOWER_AVG_SAT": f"{s_avg:.6g}",
        "FT_SLOWER_W_OPTIC_M": f"{r_mm * 1e-3:.6g}",
        "FT_WAIST_M": f"{mot_mm * 1e-3:.6g}",
        "FT_SLOWER_DETUNING_HZ": f"{settings['slower_detuning_MHz'] * 1e6:.8g}",
        "FT_COIL_CURRENT_A": f"{settings['coil_current_A']:.6g}",
        "FT_TOLERANCE": "0",
        "FT_SLOWER_SCAN": "0",
    }


def code_hash():
    """Short hash of everything besides a point's own env that its result
    depends on: the code, the table and any FT_* overrides. The nozzle cache
    is covered too, since its contents follow from those same settings."""
    h = hashlib.sha1()
    paths = [os.path.join(HERE, "full_trap_sweep.py"),
             *sorted(glob.glob(os.path.join(HERE, "helpers", "*.py"))),
             TABLE_CSV]
    for path in paths:
        with open(path, "rb") as fh:
            h.update(fh.read())
    h.update(repr(sorted((k, v) for k, v in os.environ.items()
                         if k.startswith("FT_"))).encode())
    return h.hexdigest()[:10]


def point_dir(root, s_avg, r_mm, mot_mm, env):
    """Folder for one point: readable settings plus a hash of its env."""
    env_hash = hashlib.sha1(json.dumps(env, sort_keys=True).encode())
    name = (f"s{s_avg:.3f}_w{r_mm:05.2f}mm_mot{mot_mm:05.2f}mm_"
            f"{env_hash.hexdigest()[:6]}")
    return os.path.join(root, name)


def finished_run(pdir):
    """Newest full_trap run folder under pdir that reached its summary."""
    done = sorted(glob.glob(os.path.join(pdir, "*", "summary.json")))
    return os.path.dirname(done[-1]) if done else None


def run_point(env, pdir):
    """Run one point unless it already finished. Returns (run folder,
    whether it was reused)."""
    done = finished_run(pdir)
    if done:
        return done, True
    os.makedirs(pdir, exist_ok=True)
    with open(os.path.join(pdir, "env.json"), "w") as fh:
        json.dump(env, fh, indent=2, sort_keys=True)
    log_path = os.path.join(pdir, "stdout.log")
    with open(log_path, "w") as fh:
        proc = subprocess.run(
            [sys.executable, os.path.join(HERE, "full_trap_sweep.py")],
            cwd=HERE, env={**os.environ, **env, "RUNLOG_ROOT": pdir},
            stdout=fh, stderr=subprocess.STDOUT)
    done = finished_run(pdir)
    if proc.returncode != 0 or done is None:
        with open(log_path) as fh:
            tail = "".join(fh.readlines()[-25:])
        raise RuntimeError(f"point failed (exit {proc.returncode}), see "
                           f"{log_path}\n{tail}")
    return done, False


def read_point(run_dir):
    """Load rate at REF_TEMP_C plus the diagnostics worth tabulating."""
    with open(os.path.join(run_dir, "summary.json")) as fh:
        summary = json.load(fh)
    with open(os.path.join(run_dir, "params.json")) as fh:
        params = json.load(fh)
    sweep = np.genfromtxt(os.path.join(run_dir, "full_trap_sweep.csv"),
                          delimiter=",", names=True)
    temps = np.atleast_1d(sweep["oven_T_C"])
    rates = np.atleast_1d(sweep["mot_load_rate_per_s"])
    vc = np.load(os.path.join(run_dir, "capture_map.npz"))[
        "capture_velocity_m_per_s"]
    rim = np.concatenate([vc[0], vc[-1], vc[:, 0], vc[:, -1]])
    return dict(
        load_rate=float(np.interp(REF_TEMP_C, temps, rates)),
        best_load_rate=summary["best_load_rate_atoms_per_s"],
        best_load_rate_T_C=summary["best_load_rate_at_C"],
        model_gradient_G_per_cm=summary["axial_gradient_G_per_cm"],
        capture_velocity_peak_m_per_s=summary["capture_velocity_peak_m_per_s"],
        capture_clipped=bool(rim.max() > 0),
        isat_mW_cm2=params["Isat_mW_per_cm2"],
        mot_avg_sat=params["beam_average_saturation"],
        run_dir=os.path.relpath(run_dir, HERE))


def beam_mW(avg_sat, r_mm, isat):
    """Power of one beam at average saturation avg_sat and 1/e^2 radius r_mm,
    the same P = s Isat pi w^2 full_trap_sweep uses."""
    return isat * np.asarray(avg_sat) * math.pi * (np.asarray(r_mm) / 10.0) ** 2


def power_mW(s_avg, r_mm, mot_mm, isat, mot_avg_sat):
    """Slowing beam plus MOT_ARMS MOT arms."""
    return (beam_mW(s_avg, r_mm, isat)
            + MOT_ARMS * beam_mW(mot_avg_sat, mot_mm, isat))


def slices(rows):
    """2D (saturation x radius) grids of the results, one per MOT radius when
    the radii are swept independently."""
    mots = sorted({r["mot_radius_mm"] for r in rows}) if MOT_RADII_MM else [None]
    out = []
    for mot in mots:
        sel = [r for r in rows if mot is None or r["mot_radius_mm"] == mot]
        s = np.array(sorted({r["slower_avg_sat"] for r in sel}))
        w = np.array(sorted({r["slower_radius_mm"] for r in sel}))
        at = {(r["slower_avg_sat"], r["slower_radius_mm"]): r for r in sel}
        grid = lambda key: np.array([[at[(si, wi)][key] for wi in w]
                                     for si in s], dtype=float)
        isat, mot_sat = sel[0]["isat_mW_cm2"], sel[0]["mot_avg_sat"]
        pfn = (lambda S, W, m=mot: power_mW(S, W, W if m is None else m,
                                            isat, mot_sat))
        sl = dict(s=s, w=w, R=grid("load_rate"), P=pfn(*np.meshgrid(
                  s, w, indexing="ij")), power_fn=pfn,
                  clamped=grid("table_clamped").astype(bool),
                  clipped=grid("capture_clipped").astype(bool),
                  label="" if mot is None else f"_mot{mot:g}mm",
                  mot=mot)
        sl["gain_s"], sl["gain_w"] = levers(sl)
        out.append(sl)
    return out


def levers(sl):
    """Load-rate gain per extra mW from raising saturation vs raising radius,
    by finite differences on the grid (one-sided at the edges)."""
    if sl["s"].size < 2 or sl["w"].size < 2:
        nan = np.full(sl["R"].shape, np.nan)
        return nan, nan
    dR_ds, dR_dw = np.gradient(sl["R"], sl["s"], sl["w"])
    dP_ds, dP_dw = np.gradient(sl["P"], sl["s"], sl["w"])
    return dR_ds / dP_ds, dR_dw / dP_dw


def lever_labels(gain_s, gain_w):
    """Cell text for the lever map: which knob wins, and by how much."""
    def one(gs, gw):
        if not gs > 0 and not gw > 0:
            return "neither"
        if not gs > 0:
            return "radius only"
        if not gw > 0:
            return "sat only"
        r = gw / gs
        if r >= 10 ** LEVER_CAP:
            return f"radius ≥{10 ** LEVER_CAP:.0f}×"
        if r <= 10 ** -LEVER_CAP:
            return f"sat ≥{10 ** LEVER_CAP:.0f}×"
        if abs(math.log10(r)) < 0.02:
            return "equal"
        return f"radius ×{r:.1f}" if r > 1 else f"sat ×{1 / r:.1f}"
    return [[one(a, b) for a, b in zip(ra, rb)] for ra, rb in zip(gain_s, gain_w)]


def lever_preference(gain_s, gain_w):
    """log10(radius gain / saturation gain), capped at +-LEVER_CAP; the cap
    where only one lever pays, nan where neither does."""
    out = np.full(gain_s.shape, np.nan)
    both = (gain_s > 0) & (gain_w > 0)
    out[both] = np.log10(gain_w[both] / gain_s[both])
    out[(gain_w > 0) & ~(gain_s > 0)] = LEVER_CAP
    out[(gain_s > 0) & ~(gain_w > 0)] = -LEVER_CAP
    return np.clip(out, -LEVER_CAP, LEVER_CAP)


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2, "text.color": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.titlesize": 11,
        "axes.titlecolor": INK, "lines.linewidth": 2.0,
        "lines.markersize": 7, "font.size": 10, "legend.frameon": False,
        "legend.fontsize": 9})


def _scale(values):
    """Power of ten to divide by so the largest value reads as 1-999."""
    top = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 0.0
    return 10 ** int(math.floor(math.log10(top))) if top > 0 else 1.0


def _sci(exp_scale):
    return rf"$10^{{{int(round(math.log10(exp_scale)))}}}$"


def _grid_axes(ax, sl):
    ax.set_xticks(range(sl["w"].size), [f"{x:g}" for x in sl["w"]])
    ax.set_yticks(range(sl["s"].size),
                  [f"{x:g}*" if c else f"{x:g}"
                   for x, c in zip(sl["s"], sl["clamped"].any(axis=1))])
    ax.set_xlabel("beam 1/e$^2$ radius [mm]" if sl["mot"] is None
                  else f"slowing-beam 1/e$^2$ radius [mm] (MOT {sl['mot']:g} mm)")
    ax.set_ylabel("slowing beam average saturation [I$_{sat}$]")
    ax.grid(False)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def _heatmap(ax, sl, Z, cmap, norm, text, dark_at):
    """Grid annotated with the strings in text; clipped-capture cells get a
    corner dagger."""
    im = ax.imshow(Z, origin="lower", cmap=cmap, norm=norm, aspect="auto",
                   extent=(-0.5, sl["w"].size - 0.5, -0.5, sl["s"].size - 0.5))
    for i in range(sl["s"].size):
        for j in range(sl["w"].size):
            z = Z[i, j]
            dark = np.isfinite(z) and dark_at(norm(z))
            # ax.text(j, i, text[i][j], ha="center", va="center", fontsize=8.5,
            #         color="white" if dark else INK)
            # if sl["clipped"][i, j]:
            #     ax.text(j + 0.42, i + 0.40, "†", ha="right", va="top",
            #             fontsize=8, color="white" if dark else INK2)
    _grid_axes(ax, sl)
    return im


def _footnote(fig, sl, extra=""):
    notes = []
    # if sl["clamped"].any():
    #     notes.append("* beyond the table's last row: current and detuning "
    #                  "held at s0 = 1.0, not optimised")
    # if sl["clipped"].any():
    #     notes.append(f"† capture reaches the ±{ft.TRAP_HALF_MM:g} mm "
    #                  "window edge: rate is a lower bound")
    # if extra:
    #     notes.append(extra)
    # if notes:
    #     fig.text(0.01, 0.005, "\n".join(notes), ha="left", va="bottom",
    #              fontsize=8, color=MUTED)


def _edge_interp(idx, values):
    """Index -> value, extended half a cell past each end so lines drawn over
    a heatmap reach its outer edges."""
    v = np.asarray(values, dtype=float)
    ext_i = np.r_[-0.5, np.arange(v.size), v.size - 0.5]
    ext_v = np.r_[v[0] - (v[1] - v[0]) / 2, v, v[-1] + (v[-1] - v[-2]) / 2]
    return np.interp(idx, ext_i, ext_v)


def _iso_power(ax, sl):
    """Lines of equal total optical power, labelled above the top edge."""
    n_s, n_w = sl["s"].size, sl["w"].size
    ii, jj = np.meshgrid(np.linspace(-0.5, n_s - 0.5, 200),
                         np.linspace(-0.5, n_w - 0.5, 200), indexing="ij")
    P = sl["power_fn"](_edge_interp(ii, sl["s"]), _edge_interp(jj, sl["w"]))
    levels = MaxNLocator(nbins=6).tick_values(P.min(), P.max())[1:-1]
    ax.contour(jj, ii, P, levels=levels, colors=INK2, linewidths=0.8,
               alpha=0.6)
    top = P[-1]
    for level in levels:
        if top.min() < level < top.max():
            # ax.text(np.interp(level, top, jj[-1]), n_s - 0.5, f"{level:.0f} mW",
            #         ha="center", va="bottom", fontsize=7.5, color=INK2,
            #         clip_on=False)
            pass


def fig_load_rate_map(sl):
    """Load rate over the grid, with lines of equal total optical power."""
    scale = _scale(sl["R"])
    Z = sl["R"] / scale
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    norm = Normalize(0, np.nanmax(Z) if np.nanmax(Z) > 0 else 1)
    im = _heatmap(ax, sl, Z, SEQUENTIAL, norm,
                  [[f"{z:.3g}" for z in row] for row in Z], lambda v: v > 0.6)
    if sl["s"].size > 1 and sl["w"].size > 1:
        _iso_power(ax, sl)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label(f"load rate at {REF_TEMP_C:.0f} C [{_sci(scale)} atoms/s]",
                 color=INK2)
    cb.outline.set_visible(False)
    ax.set_title(f"MOT load rate, pol={SLOWER_POL:+d} slowing beam", loc="left",
                 pad=20)
    _footnote(fig, sl, f"contours: total optical power, slowing beam + "
                       f"{MOT_ARMS} MOT arms")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def fig_cuts(sl):
    """Load rate against each knob separately: diminishing returns at a glance."""
    scale = _scale(sl["R"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    flag = sl["clamped"] | sl["clipped"]
    for ax, x, n_lines, xlab, leg in (
            (a1, sl["s"], sl["w"].size,
             "slowing beam average saturation [I$_{sat}$]", "radius"),
            (a2, sl["w"], sl["s"].size, "beam 1/e$^2$ radius [mm]",
             "avg saturation")):
        for k in range(n_lines):
            if ax is a1:
                y, f, lab = sl["R"][:, k], flag[:, k], f"{sl['w'][k]:g} mm"
            else:
                y, f, lab = sl["R"][k, :], flag[k, :], f"{sl['s'][k]:g}"
            color = ORDINAL(k / max(n_lines - 1, 1))
            ax.plot(x, y / scale, color=color, label=lab, zorder=2)
            ax.plot(x[~f], y[~f] / scale, "o", color=color, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.plot(x[f], y[f] / scale, "o", color=SURFACE, zorder=3,
                    markeredgecolor=color, markeredgewidth=1.8)
        ax.set_xlabel(xlab)
        ax.set_ylim(bottom=0)
        ax.legend(title=leg, title_fontsize=9, loc="upper left")
    a1.set_ylabel(f"load rate at {REF_TEMP_C:.0f} C [{_sci(scale)} atoms/s]")
    a1.set_title("Raising saturation, one line per radius", loc="left")
    a2.set_title("Raising radius, one line per saturation", loc="left")
    _footnote(fig, sl, "open markers: not table-optimised (*) or capture "
                       "clipped (†)")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    return fig


def fig_vs_power(sl):
    """Load rate against total optical power: which setup buys the most
    atoms for the light available."""
    scale = _scale(sl["R"])
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    flag = sl["clamped"] | sl["clipped"]
    for k in range(sl["w"].size):
        color = ORDINAL(k / max(sl["w"].size - 1, 1))
        P, R, f = sl["P"][:, k], sl["R"][:, k] / scale, flag[:, k]
        ax.plot(P, R, color=color, label=f"{sl['w'][k]:g} mm", zorder=2)
        ax.plot(P[~f], R[~f], "o", color=color, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.plot(P[f], R[f], "o", color=SURFACE, zorder=3,
                markeredgecolor=color, markeredgewidth=1.8)
    P, R = sl["P"].ravel(), sl["R"].ravel() / scale
    order = np.argsort(P)
    front = np.maximum.accumulate(R[order])
    keep = np.r_[True, np.diff(front) > 0]
    ax.step(P[order][keep], front[keep], where="post", color=MUTED, lw=1.0,
            zorder=1, label="best for the power")
    ax.set_xscale("log")
    ticks = [t for t in (10, 20, 50, 100, 200, 500, 1000, 2000, 5000)
             if P.min() / 1.3 <= t <= P.max() * 1.3]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_ylim(bottom=0)
    ax.set_xlabel(f"total optical power, slowing beam + {MOT_ARMS} MOT arms "
                  "[mW]")
    ax.set_ylabel(f"load rate at {REF_TEMP_C:.0f} C [{_sci(scale)} atoms/s]")
    ax.set_title("Load rate for mW spent", loc="left")
    ax.legend(title="radius", title_fontsize=9, loc="upper left")
    _footnote(fig, sl, "saturation rises left to right along each line; open "
                       "markers: not table-optimised or capture clipped")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def fig_levers(sl):
    """Where the next milliwatt should go: into saturation or into radius."""
    payoff = np.fmax(np.fmax(sl["gain_s"], sl["gain_w"]), 0.0)
    scale = _scale(payoff)
    pref = lever_preference(sl["gain_s"], sl["gain_w"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.8))

    top = np.nanmax(payoff / scale) if np.isfinite(payoff).any() else 1.0
    n1 = Normalize(0, top if top > 0 else 1)
    im1 = _heatmap(a1, sl, payoff / scale, SEQUENTIAL, n1,
                   [["–" if not np.isfinite(z) else f"{z:.2g}"
                     for z in row] for row in payoff / scale],
                   lambda v: v > 0.6)
    cb1 = fig.colorbar(im1, ax=a1, fraction=0.045, pad=0.03)
    cb1.set_label(f"gain from the better lever [{_sci(scale)} atoms/s per mW]",
                  color=INK2)
    cb1.outline.set_visible(False)
    a1.set_title("Gain per mW", loc="left")

    n2 = Normalize(-LEVER_CAP, LEVER_CAP)
    im2 = _heatmap(a2, sl, pref, DIVERGING, n2,
                   lever_labels(sl["gain_s"], sl["gain_w"]),
                   lambda v: abs(v - 0.5) > 0.32)
    if np.isfinite(pref).sum() >= 4 and np.nanmin(pref) < 0 < np.nanmax(pref):
        a2.contour(np.arange(sl["w"].size), np.arange(sl["s"].size),
                   np.ma.masked_invalid(pref), levels=[0.0], colors=INK,
                   linewidths=1.2)
    cb2 = fig.colorbar(im2, ax=a2, fraction=0.045, pad=0.03,
                       ticks=[-LEVER_CAP, 0, LEVER_CAP])
    cb2.ax.set_yticklabels([f"saturation\n≥{10 ** LEVER_CAP:.0f}×",
                            "equal", f"radius\n≥{10 ** LEVER_CAP:.0f}×"])
    cb2.outline.set_visible(False)
    a2.set_title("Radius vs Saturation gain (line: break-even)", loc="left")
    # _footnote(fig, sl, "finite differences on the grid, one-sided on the "
    #                    "edge cells; radius lever grows every beam sharing it")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def write_csv(path, rows):
    cols = list(rows[0])
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def text_table(sl):
    """The load-rate grid as terminal text, for reading over ssh."""
    lines = [f"load rate at {REF_TEMP_C:.0f} C [atoms/s]"
             + (f", MOT {sl['mot']:g} mm" if sl["mot"] is not None else "")]
    lines.append("  sat \\ w mm " + "".join(f"{w:>11g}" for w in sl["w"]))
    for i, s in enumerate(sl["s"]):
        mark = "*" if sl["clamped"][i].any() else " "
        lines.append(f"  {s:>9g}{mark} " + "".join(
            f"{v:>10.3e}{DAGGER if c else ' '}"
            for v, c in zip(sl["R"][i], sl["clipped"][i])))
    return lines


def main():
    table = load_table(TABLE_CSV, SLOWER_POL)
    points = sweep_points()
    mismatch = ft.nozzle_cache_mismatch()
    if mismatch is None or mismatch:
        why = ("there is no nozzle cache" if mismatch is None
               else f"the nozzle cache does not match on {mismatch}")
        msg = (f"{why} ({os.environ['FT_NOZZLE_CACHE']}), so the first point "
               "would retrace the nozzle, which takes hours.")
        if not ALLOW_RETRACE:
            raise SystemExit(msg + " Fix the cache, or set SW_ALLOW_RETRACE=1 "
                             "to let the first point rebuild it.")
        print("WARNING: " + msg, flush=True)

    root = os.path.join(POINTS_ROOT, code_hash())
    _style()
    params = dict(
        slower_avg_sats=SLOWER_AVG_SATS, radii_mm=RADII_MM,
        mot_radii_mm=MOT_RADII_MM or "same as slowing beam",
        slower_pol=SLOWER_POL, table=os.path.relpath(TABLE_CSV, HERE),
        ref_temp_C=REF_TEMP_C, mot_arms=MOT_ARMS,
        points_dir=os.path.relpath(root, HERE), n_points=len(points),
        logical_cpus=os.cpu_count(), workers_per_point=ft.worker_count(10**6))

    with runlog.start("yb-beam-sweep", params=params,
                      note=f"{len(points)} full_trap_sweep points, "
                           f"pol={SLOWER_POL:+d}") as run:
        run.say(f"table {os.path.basename(TABLE_CSV)}: pol={SLOWER_POL:+d}, "
                f"peak s0 {table['s0'][0]:g}..{table['s0'][-1]:g} "
                f"(avg {table['s0'][0] / 2:g}..{table['s0'][-1] / 2:g})")
        beyond = sorted({s for s in SLOWER_AVG_SATS
                         if table_settings(table, 2 * s)["table_clamped"]})
        if beyond:
            run.say(f"avg saturation {', '.join(f'{s:g}' for s in beyond)} "
                    "lies beyond the table: those points reuse the end row's "
                    "current and detuning and are marked * on every plot")
        run.say(f"points kept in {os.path.relpath(root, HERE)}")

        plan = []
        for s, r, m in points:
            st = table_settings(table, 2 * s)
            env = point_env(s, r, m, st)
            plan.append((s, r, m, st, env, point_dir(root, s, r, m, env)))
        n_done = sum(1 for p in plan if finished_run(p[-1]))
        if n_done:
            run.say(f"{n_done} of {len(plan)} points already finished and "
                    "will be reused")

        rows, spent, n_ran = [], 0.0, 0
        for k, (s, r, m, st, env, pdir) in enumerate(plan, start=1):
            run.say(f"[{k}/{len(points)}] slower {s:g} Isat avg, w {r:g} mm, "
                    f"MOT {m:g} mm: {st['coil_current_A']:.1f} A, "
                    f"{st['slower_detuning_MHz']:+.1f} MHz")
            t0 = time.time()
            run_dir, reused = run_point(env, pdir)
            if not reused:
                spent += time.time() - t0
                n_ran += 1
            row = dict(slower_avg_sat=s, slower_peak_s0=2 * s,
                       slower_radius_mm=r, mot_radius_mm=m, **st,
                       **read_point(run_dir))
            row["slower_power_mW"] = float(beam_mW(s, r, row["isat_mW_cm2"]))
            row["mot_power_per_arm_mW"] = float(beam_mW(
                row["mot_avg_sat"], m, row["isat_mW_cm2"]))
            row["total_power_mW"] = (row["slower_power_mW"]
                                     + MOT_ARMS * row["mot_power_per_arm_mW"])
            rows.append(row)
            left = sum(1 for p in plan[k:] if not finished_run(p[-1]))
            eta = (f", ~{left * spent / n_ran / 60:.0f} min left"
                   if n_ran and left else "")
            run.say(f"      -> {row['load_rate']:.3e} atoms/s"
                    f"{' (reused)' if reused else f' in {time.time() - t0:.0f} s'}"
                    f"{eta}")
            run.log(step=k, load_rate=row["load_rate"],
                    total_power_mW=row["total_power_mW"])

        grids = slices(rows)
        for sl in grids:
            idx = {(r["slower_avg_sat"], r["slower_radius_mm"]): r for r in rows
                   if sl["mot"] is None or r["mot_radius_mm"] == sl["mot"]}
            for i, s in enumerate(sl["s"]):
                for j, w in enumerate(sl["w"]):
                    idx[(s, w)]["gain_saturation_per_mW"] = float(sl["gain_s"][i, j])
                    idx[(s, w)]["gain_radius_per_mW"] = float(sl["gain_w"][i, j])
        write_csv(run.out("sweep_results.csv"), rows)
        run.file("sweep_results.csv", "one row per point")

        for sl in grids:
            for name, fn in (("load_rate_map", fig_load_rate_map),
                             ("load_rate_cuts", fig_cuts),
                             ("load_rate_vs_power", fig_vs_power),
                             ("power_levers", fig_levers)):
                run.figure(fn(sl), name + sl["label"], close=True)
            for line in text_table(sl):
                run.say(line)

        best = max(rows, key=lambda r: r["load_rate"])
        grad_err = max(abs(r["model_gradient_G_per_cm"]
                           / r["table_gradient_G_per_cm"] - 1) for r in rows)
        if grad_err > 0.02:
            run.say(f"WARNING: full_trap's axial gradient differs from the "
                    f"table's by up to {100 * grad_err:.1f}%; the table may "
                    "have been made with a different coil model")
        run.finish(summary=dict(
            best_load_rate_atoms_per_s=best["load_rate"],
            best_slower_avg_sat=best["slower_avg_sat"],
            best_slower_radius_mm=best["slower_radius_mm"],
            best_mot_radius_mm=best["mot_radius_mm"],
            best_total_power_mW=best["total_power_mW"],
            ref_temp_C=REF_TEMP_C,
            points=len(rows), points_run_now=n_ran,
            points_beyond_table=sum(r["table_clamped"] for r in rows),
            points_capture_clipped=sum(r["capture_clipped"] for r in rows),
            max_gradient_mismatch_vs_table=grad_err))


if __name__ == "__main__":
    main()
