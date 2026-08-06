#!/usr/bin/env python3
"""2D free-molecular Monte Carlo of the Yb oven nozzle channel array.

Refs: Senaratne et al., RSI 86, 023105 (2015); Giordmaine & Wang, JAP 31, 463
(1960); Beijerinck & Verster, JAP 46, 2083 (1975).
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")  # BEFORE importing pyplot (no display needed)

import json
import math
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter


RESULTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


class Run:

    def __init__(self, name, params=None, note=""):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.name = name
        self.dir = os.path.join(RESULTS_ROOT, f"{name}_{stamp}")
        os.makedirs(self.dir, exist_ok=True)
        self.t0 = time.time()
        self.params = dict(params or {})
        self.note = note

    def __enter__(self):
        print("=" * 78)
        print(f"{self.name}  {time.strftime('%Y-%m-%d %H:%M:%S')}")
        if self.note:
            print(self.note)
        print(f"output -> {self.dir}")
        print("=" * 78)
        if self.params:
            width = max(len(k) for k in self.params)
            for k, v in self.params.items():
                print(f"  {k:<{width}}  {_fmt(v)}")
            self._dump("params.json", self.params)
            print("-" * 78)
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            print(f"\nrun failed after {self._elapsed()}: {exc_type.__name__}: {exc}")
        print(f"\noutput written to {self.dir}")
        return False

    def _elapsed(self):
        return f"{time.time() - self.t0:.1f}s"

    def _dump(self, filename, obj):
        with open(self.path(filename), "w") as fh:
            json.dump(obj, fh, indent=2, default=float)

    def path(self, filename):
        """Full path to `filename` inside this run's results folder."""
        return os.path.join(self.dir, filename)

    def say(self, msg):
        print(f"[{self._elapsed():>8}] {msg}", flush=True)

    def log(self, step=None, **metrics):
        """Append one row of per-chunk metrics to metrics.csv (no printing).
        """
        row = {"step": step, **metrics}
        first = not hasattr(self, "_metrics_fh")
        if first:
            self._metrics_fh = open(self.path("metrics.csv"), "w")
            self._metrics_fh.write(",".join(row) + "\n")
        self._metrics_fh.write(",".join(f"{v:.10g}" for v in row.values()) + "\n")

    def figure(self, fig, name, close=True):
        out = self.path(f"{name}.png")
        fig.savefig(out, dpi=150)
        if close:
            plt.close(fig)
        print(f"[{self._elapsed():>8}] figure  {name}.png", flush=True)

    def file(self, filename, label=""):
        """Announce a data file already written into the results folder."""
        p = self.path(filename)
        size = os.path.getsize(p) / 1e6 if os.path.exists(p) else 0.0
        print(f"[{self._elapsed():>8}] data    {filename}  ({size:.1f} MB)"
              f"{'  ' + label if label else ''}", flush=True)

    def finish(self, summary=None):
        if hasattr(self, "_metrics_fh"):
            self._metrics_fh.close()
        summary = dict(summary or {})
        if summary:
            print("\n" + "=" * 78)
            print("SUMMARY")
            print("=" * 78)
            width = max(len(k) for k in summary)
            for k, v in summary.items():
                print(f"  {k:<{width}}  {_fmt(v)}")
            self._dump("summary.json", summary)
        print(f"\ndone in {self._elapsed()}")


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def start(name, params=None, note=""):
    return Run(name, params=params, note=note)


# constants
KB = 1.380649e-23           # J/K
AMU = 1.66053906660e-27     # kg
M_YB = 173.045 * AMU # 173.938 * AMU        # kg, natural-abundance-weighted Yb or just 174

# IMPORTANT STUFF
def _envf(name, default):
    return float(os.environ.get(name, default))


def _envi(name, default):
    return int(float(os.environ.get(name, default)))


CHANNEL_WIDTH_MM = _envf("NOZZLE_WIDTH_MM", 0.305)
CHANNEL_LENGTH_MM = _envf("NOZZLE_LENGTH_MM", 12.5)
SEPTUM_MM = _envf("NOZZLE_SEPTUM_MM", 0.295)
N_CHANNELS = _envi("NOZZLE_N_CHANNELS", 9)

RESERVOIR_TEMP_C = _envf("NOZZLE_RESERVOIR_C", 450.0)
NOZZLE_TEMP_C = _envf("NOZZLE_WALL_C", RESERVOIR_TEMP_C + 30.0)

N_ATOMS = _envi("NOZZLE_N_ATOMS", 40_000_000)
N_CHUNKS = _envi("NOZZLE_N_CHUNKS", 1000)      # about 1000 logged points per metric i found works
PROGRESS_EVERY = _envi("NOZZLE_PROGRESS_EVERY", 100)  # chunks between progress lines
SEED = _envi("NOZZLE_SEED", 20260724)

# I found that most papers have these at zero
SPECULAR_FRAC = _envf("NOZZLE_SPECULAR_FRAC", 0.0)
STICK_PROB = _envf("NOZZLE_STICK_PROB", 0.0)
PAIR_COLLISION_TEST = bool(_envi("NOZZLE_PAIR_COLLISIONS", 0))

MAX_BOUNCES = _envi("NOZZLE_MAX_BOUNCES", 5000)
N_PATHS_PER_CHANNEL = _envi("NOZZLE_N_PATHS", 60)
DOWNSTREAM_MM = _envf("NOZZLE_DOWNSTREAM_MM", 9.0)

# Geometries of thing nozzle
W = CHANNEL_WIDTH_MM * 1e-3
L = CHANNEL_LENGTH_MM * 1e-3
PITCH = (CHANNEL_WIDTH_MM + SEPTUM_MM) * 1e-3
T_GAS = RESERVOIR_TEMP_C + 273.15
T_WALL = NOZZLE_TEMP_C + 273.15

GEOM_HALF_ANGLE = math.degrees(math.atan(W / L))

# Status codes
ST_ACTIVE, ST_TRANSMIT, ST_RETURN, ST_STUCK, ST_MAXBOUNCE = 0, 1, 2, 3, 4


# Yb vapour pressure
# Can update later, not super necessary

_VP_B = 1.0 / (1.0 / 813.0 - 1.0 / 736.0)
_VP_A = -_VP_B / 736.0


def yb_vapour_pressure_pa(temp_k):
    return 10.0 ** (_VP_A + _VP_B / temp_k)


def mean_free_path_m(temp_k, diameter_m=4.0e-10):
    p = yb_vapour_pressure_pa(temp_k)
    return KB * temp_k / (math.sqrt(2.0) * math.pi * diameter_m**2 * p)



def sample_cosine_angle(rng, n):
    """Angle from the normal, p(psi) = (1/2)cos(psi); inverse CDF arcsin(2u-1)."""
    return np.arcsin(2.0 * rng.random(n) - 1.0)


def sample_flux_speed(rng, n, temp_k, mass=M_YB):
    """Speed from the Maxwell *flux* distribution p(v) ~ v^3 exp(-mv^2/2kT)."""
    u1 = 1.0 - rng.random(n)  # in (0, 1], keeps log finite
    u2 = 1.0 - rng.random(n)
    s = -np.log(u1) - np.log(u2)
    return np.sqrt(2.0 * KB * temp_k * s / mass)


def direct_flight_fraction(w, length):
    """Exact zero-bounce fraction, (sqrt(L^2 + w^2) - L)/w."""
    return (math.sqrt(length**2 + w**2) - length) / w


def direct_flight_angular_density(psi, w, length):
    """
        p(psi) = (1/2)cos(psi) * max(0, 1 - (L/w)|tan psi|),

    supported on |psi| < atan(w/L) and integrating to direct_flight_fraction().
    """
    t = np.tan(psi)
    return 0.5 * np.cos(psi) * np.clip(1.0 - (length / w) * np.abs(t), 0.0, None)


def cosine_density(psi):
    """The expectation: p(psi) = (1/2) cos(psi)."""
    return 0.5 * np.cos(psi)


def trace_batch(rng, n, w, length, cfg):
    """Trace n atoms through one channel. Returns a dict of per-atom results."""
    y0 = w * rng.random(n)
    psi0 = sample_cosine_angle(rng, n)
    v0 = sample_flux_speed(rng, n, cfg["t_gas"])

    status = np.zeros(n, dtype=np.int8)
    hits = np.zeros(n, dtype=np.int32)
    exit_y = np.full(n, np.nan)
    exit_psi = np.full(n, np.nan)
    exit_speed = np.full(n, np.nan)

    oid = np.arange(n)
    xa = np.zeros(n)
    ya = y0.copy()
    vx = np.cos(psi0)
    vy = np.sin(psi0)
    sp = v0.copy()
    nh = np.zeros(n, dtype=np.int32)

    spec_frac = cfg["specular_frac"]
    stick_p = cfg["stick_prob"]
    pair = cfg["pair_collisions"]

    for _ in range(cfg["max_bounces"]):
        if oid.size == 0:
            break
        m = oid.size


        t_wall = np.full(m, np.inf)
        up = vy > 0.0
        dn = vy < 0.0
        t_wall[up] = (w - ya[up]) / vy[up]
        t_wall[dn] = (0.0 - ya[dn]) / vy[dn]

        t_exit = np.full(m, np.inf)
        fw = vx > 0.0
        t_exit[fw] = (length - xa[fw]) / vx[fw]

        t_ret = np.full(m, np.inf)
        bw = vx < 0.0
        t_ret[bw] = (0.0 - xa[bw]) / vx[bw]

        t_event = np.minimum(t_wall, np.minimum(t_exit, t_ret))

        if pair is not None:
            t_event, scattered = _background_collide(
                rng, xa, vx, vy, sp, t_event, pair
            )
        else:
            scattered = None

        xa = xa + vx * t_event
        ya = ya + vy * t_event

        is_wall = (t_wall <= t_exit) & (t_wall <= t_ret)
        if scattered is not None:
            is_wall = is_wall & ~scattered
        is_exit = (~is_wall) & (t_exit <= t_ret)
        if scattered is not None:
            is_exit = is_exit & ~scattered
        is_ret = ~is_wall & ~is_exit
        if scattered is not None:
            is_ret = is_ret & ~scattered

        if is_exit.any():
            g = oid[is_exit]
            status[g] = ST_TRANSMIT
            exit_y[g] = ya[is_exit]
            exit_psi[g] = np.arctan2(vy[is_exit], vx[is_exit])
            exit_speed[g] = sp[is_exit]
            hits[g] = nh[is_exit]
        if is_ret.any():
            g = oid[is_ret]
            status[g] = ST_RETURN
            hits[g] = nh[is_ret]

        keep = ~(is_exit | is_ret)

        # change this if you're a chud istg

        if stick_p > 0.0:
            stuck = is_wall & (rng.random(m) < stick_p)
            if stuck.any():
                g = oid[stuck]
                status[g] = ST_STUCK
                hits[g] = nh[stuck] + 1
                keep = keep & ~stuck

        if not keep.all():
            oid, xa, ya = oid[keep], xa[keep], ya[keep]
            vx, vy, sp, nh = vx[keep], vy[keep], sp[keep], nh[keep]
            is_wall = is_wall[keep]
            if scattered is not None:
                scattered = scattered[keep]
        if oid.size == 0:
            break

        m = oid.size

        # wall reemissionnnn I don't know why works
        if is_wall.any():
            idx = np.flatnonzero(is_wall)
            k = idx.size
            nh[idx] += 1
            upper = ya[idx] > 0.5 * w # NEEDS THE 0.5 I'm so stupid


            # phi is measured from the inward normal
            phi = sample_cosine_angle(rng, k)
            new_vx = np.sin(phi)
            new_vy = np.where(upper, -np.cos(phi), np.cos(phi))
            new_sp = sample_flux_speed(rng, k, cfg["t_wall"])

            if spec_frac > 0.0:
                is_spec = rng.random(k) < spec_frac
                new_vx = np.where(is_spec, vx[idx], new_vx)
                new_vy = np.where(is_spec, -vy[idx], new_vy)
                new_sp = np.where(is_spec, sp[idx], new_sp)

            vx[idx] = new_vx
            vy[idx] = new_vy
            sp[idx] = new_sp

        
        if scattered is not None and scattered.any():
            idx = np.flatnonzero(scattered)
            k = idx.size
            #Boris said not needed
            ang = 2.0 * math.pi * rng.random(k)
            vx[idx] = np.cos(ang)
            vy[idx] = np.sin(ang)
            sp[idx] = sample_flux_speed(rng, k, cfg["t_gas"])

    if oid.size:
        status[oid] = ST_MAXBOUNCE
        hits[oid] = nh

    return {
        "status": status,
        "hits": hits,
        "entry_psi": psi0,
        "entry_speed": v0,
        "exit_y": exit_y,
        "exit_psi": exit_psi,
        "exit_speed": exit_speed,
    }


def _background_collide(rng, xa, vx, vy, sp, t_event, pair_cfg):
    """Optional test-particle-in-static-background scattering."""
    lam0 = pair_cfg["lambda_entrance"]
    length = pair_cfg["length"]
    frac = np.clip(1.0 - xa / length, 0.0, 1.0)
    lam = np.where(frac > 1e-6, lam0 / np.maximum(frac, 1e-6), np.inf)
    s = -np.log(1.0 - rng.random(xa.size)) * lam
    scattered = s < t_event
    return np.where(scattered, s, t_event), scattered

# Scalar tests for tracer, I wanna see if this makes it faster
def trace_single(rng, w, length, spec_frac=0.0, max_bounces=MAX_BOUNCES):
    """Python records every vertex."""
    x, y = 0.0, w * rng.random()
    psi = float(math.asin(2.0 * rng.random() - 1.0))
    vx, vy = math.cos(psi), math.sin(psi)
    verts = [(x, y)]
    nh = 0

    while nh < max_bounces:
        if vy > 0.0:
            tw = (w - y) / vy
        elif vy < 0.0:
            tw = (0.0 - y) / vy
        else:
            tw = math.inf
        te = (length - x) / vx if vx > 0.0 else math.inf
        tr = (0.0 - x) / vx if vx < 0.0 else math.inf

        t = min(tw, te, tr)
        x += vx * t
        y += vy * t
        verts.append((x, y))

        if tw <= te and tw <= tr:
            nh += 1
            upper = y > 0.5 * w
            if spec_frac > 0.0 and rng.random() < spec_frac:
                vy = -vy
            else:
                phi = float(math.asin(2.0 * rng.random() - 1.0))
                vx = math.sin(phi)
                vy = -math.cos(phi) if upper else math.cos(phi)
            continue

        st = ST_TRANSMIT if te <= tr else ST_RETURN
        return {"status": st, "verts": verts, "hits": nh,
                "psi": math.atan2(vy, vx), "y": y}

    return {"status": ST_MAXBOUNCE, "verts": verts, "hits": nh,
            "psi": math.atan2(vy, vx), "y": y}


# --------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------
TRANSMIT_C = "#d1495b"
RETURN_C = "#2e86ab"
ACCENT = "#f4a259"

# Internally everything is in metres; label axes in mm without touching data.
_MM_FMT = FuncFormatter(lambda v, _pos: f"{v * 1e3:g}")


def _mm_axes(ax, x=True, y=True):
    if x:
        ax.xaxis.set_major_formatter(_MM_FMT)
    if y:
        ax.yaxis.set_major_formatter(_MM_FMT)


def _channel_offsets():
    """y-offset of each channel centre, channel 0 at the bottom."""
    return (np.arange(N_CHANNELS) - (N_CHANNELS - 1) / 2.0) * PITCH


# Display categories for the trajectory plots, drawn back to front.
PATH_STYLE = {
    "many": dict(color="#8d99ae", lw=0.35, alpha=0.35, zorder=2,
                 label="transmitted, >20 wall hits"),
    "returned": dict(color=RETURN_C, lw=0.6, alpha=0.55, zorder=3,
                     label="recirculated to the reservoir"),
    "few": dict(color=ACCENT, lw=0.8, alpha=0.80, zorder=4,
                label="transmitted, 1-20 wall hits"),
    "direct": dict(color=TRANSMIT_C, lw=1.15, alpha=0.95, zorder=5,
                   label="transmitted, 0 wall hits (the collimated core)"),
}
DRAW_ORDER = ["many", "returned", "few", "direct"]


def _categorise(r):
    if r["status"] == ST_TRANSMIT:
        if r["hits"] == 0:
            return "direct"
        return "few" if r["hits"] <= 20 else "many"
    if r["status"] == ST_RETURN:
        # Cap the drawn recirculated paths so the entrance does not turn into
        # a solid blue block.
        return "returned" if r["hits"] <= 14 else None
    return None


def _collect_paths(rng, w, length, quotas, max_tries=100_000):
    """Rejection-sample trajectories into display categories.

    Direct-flight atoms are only ~1.2% of launches and transmitted atoms average
    ~160 wall collisions, so a uniform sample is unreadable spaghetti that hides
    the population forming the collimated core. The counts drawn are therefore
    NOT proportional to the real populations; that is what the histograms are
    for. This plot is about mechanism.
    """
    buckets = {k: [] for k in quotas}
    need = dict(quotas)
    tries = 0
    while tries < max_tries and any(v > 0 for v in need.values()):
        tries += 1
        r = trace_single(rng, w, length, SPECULAR_FRAC)
        cat = _categorise(r)
        if cat is not None and need.get(cat, 0) > 0:
            buckets[cat].append(r)
            need[cat] -= 1
    return buckets


def _draw_path(ax, r, base_y, style, downstream):
    v = np.asarray(r["verts"])
    xs, ys = v[:, 0], v[:, 1] + base_y
    if r["status"] == ST_TRANSMIT and downstream > 0:
        xs = np.append(xs, xs[-1] + downstream)
        ys = np.append(ys, ys[-1] + downstream * math.tan(r["psi"]))
    ax.plot(xs, ys, solid_joinstyle="round", **style)


def fig_trajectories(rng):
    """The hero plot: all 9 channels stacked, with trajectories."""
    offs = _channel_offsets()
    fig, ax = plt.subplots(figsize=(15, 8.5))

    span_lo = offs[0] - W / 2.0
    span_hi = offs[-1] + W / 2.0

    # Septa and outer body, drawn as solid blocks.
    body_pad = 0.35e-3
    edges = []
    for o in offs:
        edges.append((o - W / 2.0, o + W / 2.0))
    ax.add_patch(Rectangle((0, span_hi), L, body_pad, facecolor="#3d3d3d",
                           edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((0, span_lo - body_pad), L, body_pad,
                           facecolor="#3d3d3d", edgecolor="none", zorder=3))
    for i in range(N_CHANNELS - 1):
        lo = edges[i][1]
        hi = edges[i + 1][0]
        ax.add_patch(Rectangle((0, lo), L, hi - lo, facecolor="#3d3d3d",
                               edgecolor="none", zorder=3))

    quotas = {"direct": 7, "few": 5, "many": 2, "returned": 9}
    down = DOWNSTREAM_MM * 1e-3
    for off in offs:
        base = off - W / 2.0
        buckets = _collect_paths(rng, W, L, quotas)
        for cat in DRAW_ORDER:
            style = {k: v for k, v in PATH_STYLE[cat].items() if k != "label"}
            for r in buckets[cat]:
                _draw_path(ax, r, base, style, down)

    ax.axvline(0.0, color="0.35", lw=1.0, ls=":", zorder=4)
    ax.axvline(L, color="0.35", lw=1.0, ls=":", zorder=4)
    ax.set_xlim(-0.6e-3, L + DOWNSTREAM_MM * 1e-3)
    ax.set_ylim(span_lo - 1.6e-3, span_hi + 1.6e-3)
    ax.set_xlabel("axial position  x  [mm]")
    ax.set_ylabel("transverse position  y  [mm]")
    ax.set_title(
        f"Yb nozzle, {RESERVOIR_TEMP_C:.0f} C: {N_CHANNELS} channels of "
        f"{CHANNEL_WIDTH_MM} mm x {CHANNEL_LENGTH_MM} mm, "
        "(paths sampled by category, not in proportion to population)"
    )
    ax.set_xticks(np.arange(0, (CHANNEL_LENGTH_MM + DOWNSTREAM_MM) * 1e-3, 2e-3))
    ax.set_yticks(offs)
    _mm_axes(ax)

    from matplotlib.lines import Line2D
    ax.legend(
        handles=[Line2D([], [], color=PATH_STYLE[c]["color"], lw=1.6,
                        label=PATH_STYLE[c]["label"])
                 for c in ("direct", "few", "many", "returned")],
        loc="upper left", framealpha=0.93, fontsize=8.5,
    )
    fig.tight_layout()
    return fig


def fig_single_channel(rng):
    """Zoomed view of one channel, aspect-distorted so bounces are visible."""
    fig, ax = plt.subplots(figsize=(14, 4.6))
    quotas = {"direct": 14, "few": 9, "many": 3, "returned": 12}
    buckets = _collect_paths(rng, W, L, quotas)
    down = DOWNSTREAM_MM * 1e-3
    for cat in DRAW_ORDER:
        style = {k: v for k, v in PATH_STYLE[cat].items() if k != "label"}
        for r in buckets[cat]:
            _draw_path(ax, r, 0.0, style, down)

    ax.axhline(0.0, color="#3d3d3d", lw=3)
    ax.axhline(W, color="#3d3d3d", lw=3)
    ax.axvline(L, color="0.35", lw=1.0, ls=":")
    ax.set_xlim(-0.3e-3, L + DOWNSTREAM_MM * 1e-3)
    ax.set_ylim(-0.35e-3, W + 0.35e-3)
    ax.set_xlabel("axial position  x  [mm]")
    ax.set_ylabel("y [mm]")
    _mm_axes(ax)
    ax.set_title("Single channel, transverse axis heavily stretched. The "
                 "recirculated atoms (blue) mostly turn around near the "
                 "entrance:\nthat is the mechanism that both collimates the "
                 "beam and saves the ytterbium.")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=PATH_STYLE[c]["color"], lw=1.6,
                              label=PATH_STYLE[c]["label"])
                       for c in ("direct", "few", "many", "returned")],
              loc="upper left", ncol=2, framealpha=0.93, fontsize=8)
    fig.tight_layout()
    return fig


def fig_exit_angle(exit_psi, n_launched, hits_tx, log_y=False):
    """Exit-angle distribution against the two analytic reference curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    lim = 25.0
    bins = np.linspace(-lim, lim, 601)
    deg = np.degrees(exit_psi)

    # Per degree per LAUNCHED atom, so the curve integrates to the transmission
    # probability and is directly comparable to the analytic densities.
    wgt = np.full(deg.size, 1.0 / n_launched)
    ax.hist(deg, bins=bins, weights=wgt / (bins[1] - bins[0]),
            color="0.3", histtype="stepfilled", alpha=0.8, label="Monte Carlo, all exits")

    direct = hits_tx == 0
    ax.hist(deg[direct], bins=bins,
            weights=np.full(direct.sum(), 1.0 / n_launched) / (bins[1] - bins[0]),
            color=TRANSMIT_C, histtype="stepfilled", alpha=0.65,
            label="MC, zero wall collisions")

    psi = np.radians(bins)
    ax.plot(bins, direct_flight_angular_density(psi, W, L) * math.pi / 180.0,
            color=ACCENT, lw=2.0, ls="--",
            label=r"analytic direct core  $\frac{1}{2}\cos\psi\,(1-\frac{L}{w}|\tan\psi|)$")

    for s in (-1, 1):
        ax.axvline(s * GEOM_HALF_ANGLE, color=ACCENT, lw=1.0, alpha=0.7)
    ax.annotate(f"aspect-ratio half-angle\n"
                f"$\\arctan(w/L) = {GEOM_HALF_ANGLE:.2f}^\\circ$",
                xy=(0.5, 0.02), xycoords="axes fraction", ha="center",
                fontsize=9, color=ACCENT,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ACCENT,
                          alpha=0.85))

    ax.set_xlabel(r"exit angle $\psi$ from the nozzle axis [deg]")
    ax.set_ylabel("probability density per launched atom [1/deg]")
    if log_y:
        ax.set_yscale("log")
        ax.set_ylim(1e-7, None)
        ax.set_title("Exit angle distribution (log scale: the wall-scattered shoulders)")
    else:
        ax.set_xlim(-6, 6)
        ax.set_title("Exit angle distribution vs analytic expectation")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def fig_entrance_vs_exit(entry_psi, exit_psi):
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(-90, 90, 361)
    ax.hist(np.degrees(entry_psi), bins=bins, density=True, color=RETURN_C,
            histtype="stepfilled", alpha=0.55, label="entrance angles (all launched)")
    ax.hist(np.degrees(exit_psi), bins=bins, density=True, color=TRANSMIT_C,
            histtype="stepfilled", alpha=0.75, label="exit angles (transmitted)")
    psi = np.radians(bins)
    ax.plot(bins, cosine_density(psi) * math.pi / 180.0, "k--", lw=1.6,
            label=r"$\frac{1}{2}\cos\psi$ (Knudsen cosine law)")
    ax.set_yscale("log")
    ax.set_xlabel(r"angle $\psi$ [deg]")
    ax.set_ylabel("normalised density [1/deg]")
    ax.set_title("Collimation: the channel converts a cosine distribution into a "
                 "peak plus shoulders")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def fig_polar(exit_psi, n_launched):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="polar")
    lim = 30.0
    bins = np.linspace(-lim, lim, 361)
    h, e = np.histogram(np.degrees(exit_psi), bins=bins)
    dens = h / n_launched / (e[1] - e[0])
    ctr = 0.5 * (e[1:] + e[:-1])
    floor = 1e-7
    r = np.log10(np.maximum(dens, floor)) - math.log10(floor)
    ax.plot(np.radians(ctr), r, color=TRANSMIT_C, lw=1.4)
    ax.fill_between(np.radians(ctr), 0, r, color=TRANSMIT_C, alpha=0.3)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(-lim)
    ax.set_thetamax(lim)
    ax.set_rlabel_position(0)
    decades = int(np.ceil(r.max())) + 1
    ax.set_yticks(range(decades + 1))
    ax.set_yticklabels([f"$10^{{{int(v + math.log10(floor))}}}$"
                        for v in range(decades + 1)], fontsize=8)
    ax.set_title("Beam pattern (radius = log10 density per launched atom per deg)", pad=24)
    fig.tight_layout()
    return fig


def fig_farfield(rng, exit_y, exit_psi, distances_mm=(76.2, 300.0)):
    """Transverse intensity profile downstream, with all 9 channels superposed."""
    offs = _channel_offsets()
    k = rng.integers(0, N_CHANNELS, size=exit_y.size)
    y_at_exit = exit_y - W / 2.0 + offs[k]

    fig, axes = plt.subplots(1, len(distances_mm), figsize=(13, 5.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, d_mm in zip(axes, distances_mm):
        d = d_mm * 1e-3
        y_far = (y_at_exit + d * np.tan(exit_psi)) * 1e3
        # Percentile-based span. A fixed window silently truncates the wide
        # shoulders and draws a hard edge that looks like real structure.
        span = float(np.percentile(np.abs(y_far), 99.0)) * 1.05
        inside = float((np.abs(y_far) <= span).mean())
        bins = np.linspace(-span, span, 401)
        ax.hist(y_far, bins=bins, density=True, color=TRANSMIT_C,
                histtype="stepfilled", alpha=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("transverse position [mm]")
        ax.set_title(f"{d_mm:.0f} mm downstream "
                     f"({100*inside:.0f}% of the beam shown)")
        ax.grid(alpha=0.25)
        # Geometric shadow of the nozzle face plus the aspect-ratio cone: this
        # is where the direct-flight core lands.
        edge = (offs[-1] + W / 2.0) * 1e3 + d * math.tan(
            math.radians(GEOM_HALF_ANGLE)) * 1e3
        for s in (-1, 1):
            ax.axvline(s * edge, color=ACCENT, ls="--", lw=1.2)
        core = float((np.abs(y_far) <= edge).mean())
        ax.annotate(f"{100*core:.0f}% inside the\naspect-ratio cone",
                    xy=(0.03, 0.06), xycoords="axes fraction", fontsize=8.5,
                    color=ACCENT)
    axes[0].set_ylabel("normalised flux density [1/mm]")
    axes[0].annotate("dashed: nozzle face + aspect-ratio cone",
                     xy=(0.03, 0.94), xycoords="axes fraction", fontsize=9,
                     color=ACCENT)
    fig.suptitle("Far-field beam profile, all 9 channels superposed")
    fig.tight_layout()
    return fig


def _ccdf(vals):
    """Return (k, P(X >= k)) for an integer sample."""
    v = np.sort(vals)
    ks, first = np.unique(v, return_index=True)
    return ks, 1.0 - first / v.size


# Bands for the collimation-vs-bounce-count panel. Transmitted atoms turn out
# to be strongly bimodal - either zero collisions or of order a hundred - so
# evenly spaced integer bins waste the whole axis.
HIT_BANDS = [(0, 0, "0"), (1, 5, "1-5"), (6, 20, "6-20"), (21, 100, "21-100"),
             (101, 500, "101-500"), (501, 10**9, ">500")]


def fig_wall_hits(hits_all, status_all, exit_psi_tx, hits_tx):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    rt = status_all == ST_RETURN

    # Survival function, not a histogram: the bounce count spans four decades
    # because wall-scattered atoms perform a Levy-flight random walk along the
    # channel (the axial step w*tan(phi) has a heavy tail, index 2).
    for vals, color, lab in ((hits_all[rt], RETURN_C, "recirculated"),
                             (hits_tx, TRANSMIT_C, "transmitted")):
        k, s = _ccdf(vals)
        ax1.step(k, s, where="post", color=color, lw=1.8,
                 label=f"{lab}  (mean {vals.mean():.1f}, "
                       f"{100*(vals == 0).mean():.1f}% with none)")
    ax1.set_xscale("symlog", linthresh=1)
    ax1.set_yscale("log")
    ax1.set_xlabel("number of wall collisions, n")
    ax1.set_ylabel(r"$P(\mathrm{collisions} \geq n)$")
    ax1.set_title("Wall collisions before termination")
    ax1.legend(fontsize=8.5)
    ax1.grid(alpha=0.25, which="both")

    # Collimation in one panel: an atom that touches a wall is essentially
    # uncollimated, so the sharp core IS the zero-bounce population.
    abs_deg = np.abs(np.degrees(exit_psi_tx))
    labels, counts, medians, p90s = [], [], [], []
    for lo_h, hi_h, lab in HIT_BANDS:
        sel = (hits_tx >= lo_h) & (hits_tx <= hi_h)
        if sel.sum() < 30:
            continue
        labels.append(lab)
        counts.append(int(sel.sum()))
        medians.append(float(np.median(abs_deg[sel])))
        p90s.append(float(np.percentile(abs_deg[sel], 90)))

    xpos = np.arange(len(labels))
    ax2.bar(xpos - 0.2, medians, width=0.4, color=TRANSMIT_C,
            label=r"median $|\psi|$")
    ax2.bar(xpos + 0.2, p90s, width=0.4, color="0.45",
            label=r"90th percentile $|\psi|$")
    ax2.axhline(GEOM_HALF_ANGLE, color=ACCENT, ls="--", lw=1.3,
                label=f"arctan(w/L) = {GEOM_HALF_ANGLE:.2f}$^\\circ$")
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(labels)
    ax2.set_yscale("log")
    ax2.set_xlabel("wall collisions before exiting")
    ax2.set_ylabel("exit angle [deg]")
    ax2.set_title("Collimation vs wall-collision count\n"
                  "(a single bounce is enough to destroy it)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25, axis="y")
    for x, c in zip(xpos, counts):
        ax2.annotate(f"n={c:,}", xy=(x, 0.02), xycoords=("data", "axes fraction"),
                     ha="center", fontsize=7, color="0.3")
    fig.tight_layout()
    return fig


def fig_phase_space(exit_y, exit_psi):
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    lim = 8.0
    sel = np.abs(np.degrees(exit_psi)) < lim
    h = ax.hist2d(exit_y[sel] * 1e3, np.degrees(exit_psi[sel]),
                  bins=[130, 110], cmap="magma",
                  norm=matplotlib_lognorm())
    fig.colorbar(h[3], ax=ax, label="transmitted atoms per bin")
    # The zero-bounce population occupies the parallelogram bounded by these.
    yy = np.array([0.0, W * 1e3])
    for y0 in (0.0, W * 1e3):
        ax.plot(yy, np.degrees(np.arctan((yy - y0) * 1e-3 / L)), color="cyan",
                lw=1.2, ls="--", alpha=0.8)
    ax.set_xlabel("exit position within channel  y  [mm]")
    ax.set_ylabel(r"exit angle $\psi$ [deg]")
    ax.set_title("Exit phase space: direct-flight parallelogram (dashed) "
                 )
    fig.tight_layout()
    return fig


def matplotlib_lognorm():
    from matplotlib.colors import LogNorm
    return LogNorm()


def fig_cumulative(exit_psi, n_launched):
    fig, ax = plt.subplots(figsize=(9.5, 6))
    a = np.sort(np.abs(np.degrees(exit_psi)))
    frac_launched = np.arange(1, a.size + 1) / n_launched
    frac_tx = np.arange(1, a.size + 1) / a.size
    ax.plot(a, frac_tx, color=TRANSMIT_C, lw=2.0,
            label="fraction of transmitted flux")
    ax.plot(a, frac_launched, color=RETURN_C, lw=2.0,
            label="fraction of ALL launched atoms (material efficiency)")
    ax.axvline(GEOM_HALF_ANGLE, color=ACCENT, lw=1.3, ls="--",
               label=f"arctan(w/L) = {GEOM_HALF_ANGLE:.2f}$^\\circ$")
    ax.set_xscale("log")
    ax.set_xlim(0.05, 90)
    ax.set_xlabel(r"acceptance half-angle $\theta$ [deg]")
    ax.set_ylabel(r"fraction within $|\psi| < \theta$")
    ax.set_title("Capture efficiency vs downstream acceptance angle\n"
                 "(read off your differential-pumping / Zeeman slower acceptance)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def fig_speeds(entry_speed, exit_speed, exit_psi, hits_tx):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.linspace(0, 900, 200)
    ax1.hist(entry_speed, bins=bins, density=True, color=RETURN_C, alpha=0.55,
             label=f"entrance, {RESERVOIR_TEMP_C:.0f} C")
    ax1.hist(exit_speed, bins=bins, density=True, color=TRANSMIT_C, alpha=0.7,
             label="transmitted")
    v = 0.5 * (bins[1:] + bins[:-1])
    for temp, c, lab in ((T_GAS, "k", "flux Maxwellian, gas T"),
                         (T_WALL, "0.5", "flux Maxwellian, wall T")):
        pv = v**3 * np.exp(-M_YB * v**2 / (2 * KB * temp))
        pv /= np.trapezoid(pv, v) if hasattr(np, "trapezoid") else np.trapz(pv, v)
        ax1.plot(v, pv, color=c, ls="--", lw=1.4, label=lab)
    ax1.set_xlabel("speed [m/s]")
    ax1.set_ylabel("density [s/m]")
    ax1.set_title("Speed distributions")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)

    lim = 10.0
    sel = np.abs(np.degrees(exit_psi)) < lim
    bins2 = np.linspace(-lim, lim, 81)
    idx = np.digitize(np.degrees(exit_psi[sel]), bins2) - 1
    ctr = 0.5 * (bins2[1:] + bins2[:-1])
    means = np.array([exit_speed[sel][idx == i].mean() if (idx == i).sum() > 30
                      else np.nan for i in range(ctr.size)])
    ax2.plot(ctr, means, color=TRANSMIT_C, lw=1.8)
    ax2.axhline(np.nanmean(exit_speed), color="k", ls=":", lw=1.2,
                label="beam mean")
    for s in (-1, 1):
        ax2.axvline(s * GEOM_HALF_ANGLE, color=ACCENT, lw=1.0, ls="--")
    ax2.set_xlabel(r"exit angle $\psi$ [deg]")
    ax2.set_ylabel("mean speed [m/s]")
    ax2.set_title("Mean speed vs exit angle\n(flat unless wall T differs from gas T)")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def fig_convergence(hist_n, hist_w):
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    n = np.asarray(hist_n, dtype=float)
    w = np.asarray(hist_w, dtype=float)
    final = w[-1]
    ax.plot(n, w, color=TRANSMIT_C, lw=1.3, label="running transmission probability")
    err = np.sqrt(final * (1 - final) / n)
    ax.fill_between(n, final - 3 * err, final + 3 * err, color="0.7", alpha=0.5,
                    label=r"$\pm 3\sigma$ binomial band")
    ax.axhline(final, color="k", ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("atoms launched")
    ax.set_ylabel("transmission probability  W")
    ax.set_ylim(final - 12 * err[-1], final + 12 * err[-1])
    ax.set_title("Monte Carlo convergence of the 2D transmission probability")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_validation(rng, results):
    """Three independent checks, drawn as one panel."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.6))

    # (1) direct-flight fraction vs the closed form, across aspect ratios
    ratios = np.array([0.5, 1, 2, 5, 10, 20, 41, 80])
    mc, an = [], []
    for r in ratios:
        length = W * r
        out = trace_batch(rng, 40000, W, length,
                          dict(t_gas=T_GAS, t_wall=T_WALL, specular_frac=0.0,
                               stick_prob=0.0, pair_collisions=None,
                               max_bounces=MAX_BOUNCES))
        d = ((out["status"] == ST_TRANSMIT) & (out["hits"] == 0)).mean()
        mc.append(d)
        an.append(direct_flight_fraction(W, length))
    ax1.loglog(ratios, an, "k-", lw=1.6, label=r"$(\sqrt{L^2+w^2}-L)/w$")
    ax1.loglog(ratios, mc, "o", color=TRANSMIT_C, ms=6, label="Monte Carlo")
    ax1.axvline(L / W, color=ACCENT, ls="--", lw=1.0, label="our channel")
    ax1.set_xlabel("aspect ratio L/w")
    ax1.set_ylabel("zero-bounce fraction")
    ax1.set_title("(1) direct flight vs closed form")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")

    # (2) L -> 0 must reproduce the cosine law exactly
    out = trace_batch(rng, 300000, W, W * 1e-4,
                      dict(t_gas=T_GAS, t_wall=T_WALL, specular_frac=0.0,
                           stick_prob=0.0, pair_collisions=None,
                           max_bounces=MAX_BOUNCES))
    p = out["exit_psi"][out["status"] == ST_TRANSMIT]
    bins = np.linspace(-90, 90, 181)
    ax2.hist(np.degrees(p), bins=bins, density=True, color=TRANSMIT_C, alpha=0.75,
             label="MC, L/w = 1e-4")
    ax2.plot(bins, cosine_density(np.radians(bins)) * math.pi / 180.0, "k--",
             lw=1.8, label=r"$\frac{1}{2}\cos\psi$")
    ax2.set_xlabel(r"$\psi$ [deg]")
    ax2.set_ylabel("density [1/deg]")
    ax2.set_title("(2) bare-orifice limit")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # (3) vectorised tracer vs the independent scalar tracer
    n_ref = 150000
    ref = [trace_single(rng, W, L, SPECULAR_FRAC) for _ in range(n_ref)]
    ref_psi = np.array([r["psi"] for r in ref if r["status"] == ST_TRANSMIT])
    ref_w = len(ref_psi) / n_ref
    vec_psi = results["exit_psi_all"]
    vec_w = results["transmission"]
    bins3 = np.linspace(-15, 15, 121)
    ax3.hist(np.degrees(vec_psi), bins=bins3, density=True, color="0.35",
             histtype="stepfilled", alpha=0.7,
             label=f"vectorised  (W={vec_w:.4f})")
    ax3.hist(np.degrees(ref_psi), bins=bins3, density=True, color=TRANSMIT_C,
             histtype="step", lw=1.8,
             label=f"scalar reference  (W={ref_w:.4f})")
    # Quantify the agreement rather than eyeballing it.
    e_ref = math.sqrt(ref_w * (1 - ref_w) / n_ref)
    e_vec = math.sqrt(vec_w * (1 - vec_w) / max(results["n_launched"], 1))
    nsig = abs(vec_w - ref_w) / math.sqrt(e_ref**2 + e_vec**2)
    ax3.set_yscale("log")
    ax3.set_xlabel(r"$\psi$ [deg]")
    ax3.set_title(f"(3) two independent implementations\n"
                  f"$\\Delta W = {nsig:.2f}\\sigma$")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    fig.suptitle("Validation")
    fig.tight_layout()
    return fig, dict(direct_mc=mc, direct_analytic=an, ref_W=ref_w,
                     ref_vs_vec_sigma=nsig)


def main():
    rng = np.random.default_rng(SEED)

    p_vap = yb_vapour_pressure_pa(T_GAS)
    lam = mean_free_path_m(T_GAS)
    n_dens = p_vap / (KB * T_GAS)
    vbar = math.sqrt(8 * KB * T_GAS / (math.pi * M_YB))

    pair_cfg = None
    if PAIR_COLLISION_TEST:
        pair_cfg = {"lambda_entrance": lam, "length": L}

    cfg = dict(t_gas=T_GAS, t_wall=T_WALL, specular_frac=SPECULAR_FRAC,
               stick_prob=STICK_PROB, pair_collisions=pair_cfg,
               max_bounces=MAX_BOUNCES)

    params = {
        "channel_width_mm": CHANNEL_WIDTH_MM,
        "channel_length_mm": CHANNEL_LENGTH_MM,
        "septum_mm": SEPTUM_MM,
        "pitch_mm": PITCH * 1e3,
        "n_channels": N_CHANNELS,
        "aspect_ratio_L_over_w": L / W,
        "geometric_half_angle_deg": GEOM_HALF_ANGLE,
        "reservoir_temp_C": RESERVOIR_TEMP_C,
        "nozzle_wall_temp_C": NOZZLE_TEMP_C,
        "n_atoms": N_ATOMS,
        "seed": SEED,
        "specular_fraction": SPECULAR_FRAC,
        "stick_probability": STICK_PROB,
        "pair_collisions": int(PAIR_COLLISION_TEST),
        "yb_vapour_pressure_Pa": p_vap,
        "mean_free_path_mm": lam * 1e3,
        "knudsen_lambda_over_w": lam / W,
        "knudsen_lambda_over_L": lam / L,
        "mean_speed_m_per_s": vbar,
    }

    with start("yb-nozzle-2d", params=params,
               note="2D free-molecular MC of the 9-channel Yb oven nozzle") as run:

        run.say(f"Yb at {RESERVOIR_TEMP_C:.0f} C: P = {p_vap:.3g} Pa, "
                f"lambda = {lam*1e3:.1f} mm")
        run.say(f"lambda/w = {lam/W:.1f}, lambda/L = {lam/L:.2f} "
                f"-> transparent regime, free-molecular is justified")
        run.say(f"aspect ratio L/w = {L/W:.1f}, arctan(w/L) = {GEOM_HALF_ANGLE:.3f} deg")
        run.say(f"analytic zero-bounce fraction = {direct_flight_fraction(W, L):.6f}")

        # ---------------- main chunked run ----------------
        per_chunk = max(1, N_ATOMS // N_CHUNKS)
        total = per_chunk * N_CHUNKS

        n_launched = 0
        n_tx = 0
        n_direct = 0
        n_ret = 0
        n_stuck = 0
        n_maxb = 0
        sum_hits_tx = 0
        sum_hits_ret = 0
        n_within = 0

        # Preallocate and fill by slice. Repeated np.concatenate in the chunk
        # loop is O(n^2) and would move gigabytes for nothing.
        status_all = np.empty(total, dtype=np.int8)
        hits_all = np.empty(total, dtype=np.int32)
        entry_psi_all = np.empty(total)
        entry_speed_all = np.empty(total)
        exit_psi_all = np.empty(total)
        exit_y_all = np.empty(total)
        exit_speed_all = np.empty(total)
        hits_tx_all = np.empty(total, dtype=np.int32)
        cur = 0  # cursor into the transmitted-only arrays

        conv_n, conv_w = [], []
        accept = math.radians(GEOM_HALF_ANGLE)

        for c in range(N_CHUNKS):
            out = trace_batch(rng, per_chunk, W, L, cfg)
            st = out["status"]
            tx = st == ST_TRANSMIT
            k = int(tx.sum())

            lo, hi = c * per_chunk, (c + 1) * per_chunk
            status_all[lo:hi] = st
            hits_all[lo:hi] = out["hits"]
            entry_psi_all[lo:hi] = out["entry_psi"]
            entry_speed_all[lo:hi] = out["entry_speed"]

            psi_tx = out["exit_psi"][tx]
            exit_psi_all[cur:cur + k] = psi_tx
            exit_y_all[cur:cur + k] = out["exit_y"][tx]
            exit_speed_all[cur:cur + k] = out["exit_speed"][tx]
            hits_tx_all[cur:cur + k] = out["hits"][tx]
            cur += k

            n_launched += per_chunk
            n_tx += k
            n_ret += int((st == ST_RETURN).sum())
            n_stuck += int((st == ST_STUCK).sum())
            n_maxb += int((st == ST_MAXBOUNCE).sum())
            n_direct += int((tx & (out["hits"] == 0)).sum())
            sum_hits_tx += int(out["hits"][tx].sum())
            sum_hits_ret += int(out["hits"][st == ST_RETURN].sum())
            n_within += int((np.abs(psi_tx) < accept).sum())

            W_tx = n_tx / n_launched
            conv_n.append(n_launched)
            conv_w.append(W_tx)

            run.log(
                step=n_launched,
                transmission_probability=W_tx,
                transmission_stderr=math.sqrt(max(W_tx * (1 - W_tx), 0) / n_launched),
                direct_flight_fraction=n_direct / n_launched,
                frac_transmitted_within_geometric_cone=n_within / max(n_tx, 1),
                frac_launched_within_geometric_cone=n_within / n_launched,
                mean_abs_exit_angle_deg=float(np.degrees(np.abs(psi_tx)).mean())
                if psi_tx.size else 0.0,
                mean_wall_hits_transmitted=sum_hits_tx / max(n_tx, 1),
                mean_wall_hits_recirculated=sum_hits_ret / max(n_ret, 1),
                recirculated_fraction=n_ret / n_launched,
            )

            if (c + 1) % PROGRESS_EVERY == 0:
                run.say(f"{n_launched:,}/{total:,} atoms, W = {W_tx:.5f}")

        # result numbers I thought were useful
        exit_psi_all = exit_psi_all[:cur]
        exit_y_all = exit_y_all[:cur]
        exit_speed_all = exit_speed_all[:cur]
        hits_tx_all = hits_tx_all[:cur]

        W_tx = n_tx / n_launched
        W_err = math.sqrt(W_tx * (1 - W_tx) / n_launched)
        direct_mc = n_direct / n_launched
        direct_an = direct_flight_fraction(W, L)

        abs_psi = np.abs(np.degrees(exit_psi_all))
        srt = np.sort(abs_psi)
        # FWHM from the histogram peak
        hb = np.linspace(-10, 10, 2001)
        hh, _ = np.histogram(np.degrees(exit_psi_all), bins=hb)
        ctr = 0.5 * (hb[1:] + hb[:-1])
        half = hh.max() / 2.0
        above = np.flatnonzero(hh >= half)
        fwhm = ctr[above[-1]] - ctr[above[0]] if above.size else float("nan")

        open_area = N_CHANNELS**2 * W**2
        flux_in = 0.25 * n_dens * vbar * open_area
        flux_out = flux_in * W_tx
        flux_useful = flux_in * (n_within / n_launched)
        grams_per_day = flux_out * M_YB * 1000.0 * 86400.0

        run.say(f"transmission W = {W_tx:.5f} +/- {W_err:.5f}")
        run.say(f"direct-flight fraction: MC {direct_mc:.6f} vs analytic "
                f"{direct_an:.6f}  (ratio {direct_mc/direct_an:.4f})")
        run.say(f"FWHM = {fwhm:.3f} deg, {100*n_within/n_tx:.1f}% of the beam is "
                f"inside arctan(w/L)")

        # ---------------- figures ----------------
        plot_rng = np.random.default_rng(SEED + 1)

        run.figure(fig_trajectories(plot_rng), "nozzle_trajectories_9_channels",
                   close=True)
        run.figure(fig_single_channel(plot_rng), "single_channel_zoom", close=True)
        run.figure(fig_exit_angle(exit_psi_all, n_launched, hits_tx_all,
                                  log_y=False), "exit_angle_vs_expected", close=True)
        run.figure(fig_exit_angle(exit_psi_all, n_launched, hits_tx_all,
                                  log_y=True), "exit_angle_log_shoulders", close=True)
        run.figure(fig_entrance_vs_exit(entry_psi_all, exit_psi_all),
                   "entrance_vs_exit_angles", close=True)
        run.figure(fig_polar(exit_psi_all, n_launched), "beam_pattern_polar",
                   close=True)
        run.figure(fig_farfield(plot_rng, exit_y_all, exit_psi_all),
                   "far_field_profiles", close=True)
        run.figure(fig_wall_hits(hits_all, status_all, exit_psi_all, hits_tx_all),
                   "wall_collision_statistics", close=True)
        run.figure(fig_phase_space(exit_y_all, exit_psi_all), "exit_phase_space",
                   close=True)
        run.figure(fig_cumulative(exit_psi_all, n_launched),
                   "capture_efficiency_vs_acceptance", close=True)
        run.figure(fig_speeds(entry_speed_all, exit_speed_all, exit_psi_all,
                              hits_tx_all), "speed_distributions", close=True)
        run.figure(fig_convergence(conv_n, conv_w), "transmission_convergence",
                   close=True)

        val_fig, val = fig_validation(np.random.default_rng(SEED + 2),
                                      {"exit_psi_all": exit_psi_all,
                                       "transmission": W_tx,
                                       "n_launched": n_launched})
        run.figure(val_fig, "validation", close=True)

        # more outputs for recording
        hb2 = np.linspace(-90, 90, 3601)
        hh2, _ = np.histogram(np.degrees(exit_psi_all), bins=hb2)
        ctr2 = 0.5 * (hb2[1:] + hb2[:-1])
        dens2 = hh2 / n_launched / (hb2[1] - hb2[0])
        np.savetxt(run.path("exit_angle_distribution.csv"),
                   np.column_stack([ctr2, hh2, dens2]),
                   delimiter=",", header="angle_deg,counts,density_per_launched_per_deg",
                   comments="")
        run.file("exit_angle_distribution.csv", "exit-angle-distribution")

        sub = np.random.default_rng(SEED + 3).choice(
            exit_psi_all.size, size=min(200_000, exit_psi_all.size), replace=False)
        np.savez_compressed(run.path("exit_state.npz"),
                            exit_psi=exit_psi_all[sub], exit_y=exit_y_all[sub],
                            exit_speed=exit_speed_all[sub], wall_hits=hits_tx_all[sub])
        run.file("exit_state.npz", "transmitted-atom-state")

        run.finish(summary={
            "transmission_probability_2D": W_tx,
            "transmission_stderr": W_err,
            "direct_flight_fraction_mc": direct_mc,
            "direct_flight_fraction_analytic": direct_an,
            "direct_flight_mc_over_analytic": direct_mc / direct_an,
            "scalar_reference_transmission": val["ref_W"],
            "scalar_vs_vectorised_sigma": val["ref_vs_vec_sigma"],
            "fwhm_deg": float(fwhm),
            "frac_beam_within_geometric_cone": n_within / n_tx,
            "frac_launched_within_geometric_cone": n_within / n_launched,
            "half_angle_containing_50pct": float(srt[int(0.50 * srt.size)]),
            "half_angle_containing_90pct": float(srt[int(0.90 * srt.size)]),
            "mean_wall_hits_transmitted": sum_hits_tx / max(n_tx, 1),
            "mean_wall_hits_recirculated": sum_hits_ret / max(n_ret, 1),
            "recirculated_fraction": n_ret / n_launched,
            "maxbounce_atoms": n_maxb,
            "flux_into_nozzle_per_s": flux_in,
            "flux_out_per_s_UPPER_BOUND": flux_out,
            "flux_within_cone_per_s_UPPER_BOUND": flux_useful,
            "yb_consumption_g_per_day_UPPER_BOUND": grams_per_day,
        })


if __name__ == "__main__":
    main()
