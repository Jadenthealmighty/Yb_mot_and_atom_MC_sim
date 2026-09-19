"""
3D free-molecular Monte Carlo of the Yb oven nozzle channel array.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")  # BEFORE pyplot (no display needed)

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Wedge
from matplotlib.ticker import FuncFormatter

import atom_beam.helpers.yb_nozzle_beam_2d as m2d  # constants, the 2D cross-check, and Run

KB = m2d.KB
AMU = m2d.AMU
M_YB = m2d.M_YB
start = m2d.start  # terminal logging + results folder, defined in the 2D module


def _envf(name, default):
    return float(os.environ.get(name, default))


def _envi(name, default):
    return int(float(os.environ.get(name, default)))



RECT_WIDTH_MM = _envf("N3_WIDTH_MM", 0.3048)     # widthh
RECT_HEIGHT_MM = _envf("N3_RECT_H_MM", 0.15)     
CAP_RADIUS_MM = _envf("N3_CAP_R_MM", 0.1524)     # semicircular cap radius
CHANNEL_LENGTH_MM = _envf("N3_LENGTH_MM", 12.5)
PITCH_Y_MM = _envf("N3_PITCH_Y_MM", 0.600)       # left-right, (same below)
PITCH_Z_MM = _envf("N3_PITCH_Z_MM", 0.6024)      # up-down from facign towards the nozzle (-x direction)
N_CHAN_Y = _envi("N3_NCHAN_Y", 9)
N_CHAN_Z = _envi("N3_NCHAN_Z", 9)

A_HALF = 0.5 * RECT_WIDTH_MM * 1e-3
H_RECT = RECT_HEIGHT_MM * 1e-3
R_CAP = CAP_RADIUS_MM * 1e-3
L = CHANNEL_LENGTH_MM * 1e-3
PITCH_Y = PITCH_Y_MM * 1e-3
PITCH_Z = PITCH_Z_MM * 1e-3

HOLE_HEIGHT = H_RECT + R_CAP
AREA = 2.0 * A_HALF * H_RECT + 0.5 * math.pi * R_CAP**2

PERIMETER = 2.0 * H_RECT + 2.0 * A_HALF + math.pi * R_CAP
D_HYD = 4.0 * AREA / PERIMETER # saw this in a paper and wanted to see how accurate to compare to 2d

RESERVOIR_TEMP_C = _envf("N3_RESERVOIR_C", 360.0) #419 used to be
NOZZLE_TEMP_C = _envf("N3_WALL_C", 390.0) #450 used to be
T_GAS = RESERVOIR_TEMP_C + 273.15
T_WALL = NOZZLE_TEMP_C + 273.15



# BIG BIG BIG BIG
# Jaden don't be dumbass and forget to change this
ATOMS = 600_000_000


N_ATOMS = _envi("N3_N_ATOMS", ATOMS)
N_CHUNKS = _envi("N3_N_CHUNKS", 1000)
PROGRESS_EVERY = _envi("N3_PROGRESS_EVERY", 100)  # see 2d for explanation
SEED = _envi("N3_SEED", 20260725)


# same in 2d, should ideally be zero
SPECULAR_FRAC = _envf("N3_SPECULAR_FRAC", 0.0)
STICK_PROB = _envf("N3_STICK_PROB", 0.0)
MAX_BOUNCES = _envi("N3_MAX_BOUNCES", 200_000)

# Change this if you have lots of ram, I got like 15GB at 10M
SUBSAMPLE_MAX = _envi("N3_SUBSAMPLE", 2_000_000)

N_PATHS = _envi("N3_N_PATHS", 40)

# Saving this as npz so I can compare to Nicholas data

# "many" paths carry ~900 vertices each; "few" is the expensive one to collect

PATH_EXPORT_QUOTAS = {
    "direct": _envi("N3_EXPORT_DIRECT", 200),
    "few": _envi("N3_EXPORT_FEW", 60),
    "many": _envi("N3_EXPORT_MANY", 20),
    "returned": _envi("N3_EXPORT_RETURNED", 120),
}
DOWNSTREAM_MM = _envf("N3_DOWNSTREAM_MM", 9.0)

# ANOTHER BIG THING IF YOU WANNA CHANGE THESE FOR PLOTTING, this broke when I had more than 2, try your luck
FARFIELD_MM =  (90.0, 310.0)# used to be (88.27, 310.0)

# runs for a couple geometries so that it doesn't do a full run with a broken bounce tracer
VALIDATION_SCALE = _envf("N3_VALIDATION_SCALE", 1.0)


def _vn(n, floor=2000):
    """Scale a validation sample size, keeping it statistically meaningful."""
    return max(int(floor), int(n * VALIDATION_SCALE))

GEOM_HALF_ANGLE = math.degrees(math.atan(D_HYD / L))

EPS = 1e-12  # meters

ST_ACTIVE, ST_TRANSMIT, ST_RETURN, ST_STUCK, ST_MAXBOUNCE = 0, 1, 2, 3, 4



# Surface ids
SF_BOTTOM, SF_LEFT, SF_RIGHT, SF_CAP, SF_TOP, SF_CYL = 0, 1, 2, 3, 4, 5
SURFACE_NAMES = {SF_BOTTOM: "flat bottom", SF_LEFT: "left wall",
                 SF_RIGHT: "right wall", SF_CAP: "semicircular cap",
                 SF_TOP: "flat top", SF_CYL: "cylinder"}


def geom_dshape():
    return dict(shape="dshape", a=A_HALF, h=H_RECT, R=R_CAP, L=L, area=AREA)


def geom_rect(a, h, length):
    return dict(shape="rect", a=a, h=h, R=0.0, L=length, area=2.0 * a * h)


def geom_circle(radius, length):
    return dict(shape="circle", a=radius, h=0.0, R=radius, L=length,
                area=math.pi * radius**2)


TRANSMIT_C = m2d.TRANSMIT_C
RETURN_C = m2d.RETURN_C
ACCENT = m2d.ACCENT
_MM_FMT = FuncFormatter(lambda v, _p: f"{v * 1e3:g}")


def sample_cross_section(rng, n, g):
    """Uniform points in the cross-section."""
    shape = g["shape"]
    if shape == "circle":
        r = g["R"] * np.sqrt(rng.random(n))
        t = 2.0 * math.pi * rng.random(n)
        return r * np.cos(t), r * np.sin(t)
    if shape == "rect":
        return (2.0 * rng.random(n) - 1.0) * g["a"], rng.random(n) * g["h"]
    # dshape: pick rectangle or cap in proportion to area, then uniform within
    a, h, R = g["a"], g["h"], g["R"]
    area_rect = 2.0 * a * h
    p_rect = area_rect / g["area"]
    use_rect = rng.random(n) < p_rect
    y_r = (2.0 * rng.random(n) - 1.0) * a
    z_r = rng.random(n) * h
    rr = R * np.sqrt(rng.random(n))
    th = math.pi * rng.random(n)          # upper half only
    y_c = rr * np.cos(th)
    z_c = h + rr * np.sin(th)
    return np.where(use_rect, y_r, y_c), np.where(use_rect, z_r, z_c)


def inside_cross_section(y, z, g):
    shape = g["shape"]
    if shape == "circle":
        return y**2 + z**2 <= g["R"]**2
    if shape == "rect":
        return (np.abs(y) <= g["a"]) & (z >= 0.0) & (z <= g["h"])
    a, h, R = g["a"], g["h"], g["R"]
    in_rect = (np.abs(y) <= a) & (z >= 0.0) & (z <= h)
    in_cap = (z > h) & (y**2 + (z - h) ** 2 <= R**2)
    return in_rect | in_cap


# Direction sampling for random atoms
def sample_entrance_direction(rng, n):
    """Cosine-weighted about +x. (dy, dz) uniform in the unit disk is exact."""
    u = rng.random(n)
    r = np.sqrt(u)
    phi = 2.0 * math.pi * rng.random(n)

    return np.sqrt(np.maximum(0.0, 1.0 - u)), r * np.cos(phi), r * np.sin(phi)




def _onb(nx, ny, nz):
    """Branchless orthonormal basis around a unit normal (Duff et al. 2017)
    (I wanted to write my own but lowkey this one is crazy fast)"""
    s = np.where(nz >= 0.0, 1.0, -1.0)
    a = -1.0 / (s + nz)
    b = nx * ny * a
    t1 = (1.0 + s * nx * nx * a, s * b, -s * nx)
    t2 = (b, s + ny * ny * a, -ny)
    return t1, t2


def sample_cosine_about(rng, nx, ny, nz):
    """3D Lambertian re-emission about surface normal"""
    n = nx.size
    u = rng.random(n)
    r = np.sqrt(u)
    phi = 2.0 * math.pi * rng.random(n)
    lx, ly, lz = r * np.cos(phi), r * np.sin(phi), np.sqrt(np.maximum(0.0, 1.0 - u))
    t1, t2 = _onb(nx, ny, nz)

    dx = lx * t1[0] + ly * t2[0] + lz * nx
    dy = lx * t1[1] + ly * t2[1] + lz * ny
    dz = lx * t1[2] + ly * t2[2] + lz * nz
    return dx, dy, dz


# checking the wall intersection but you
def _wall_intersections(y, z, dy, dz, g):
    """Distance to the nearest wall and which wall it was."""
    shape = g["shape"]
    a, h, R = g["a"], g["h"], g["R"]
    m = y.size
    INF = np.inf

    if shape == "circle":
        A2 = dy * dy + dz * dz
        B2 = 2.0 * (y * dy + z * dz)
        C2 = y * y + z * z - R * R
        disc = B2 * B2 - 4.0 * A2 * C2
        ok = (A2 > 0.0) & (disc >= 0.0)
        sq = np.sqrt(np.where(ok, disc, 0.0))
        den = 2.0 * np.where(A2 > 0.0, A2, 1.0)
        # Strictly inside means C2 < 0, so the roots near zero and the forward one is always the larger.
        t = np.where(ok, (-B2 + sq) / den, INF)
        t = np.where(ok & (t > EPS), t, INF)
        return t, np.full(m, SF_CYL, dtype=np.int8)

    cands = []
    ids = []
# easy for bottom plane
    msk = dz < 0.0
    t = np.where(msk, -z / np.where(msk, dz, -1.0), INF)
    good = msk & (t > EPS) & (np.abs(y + t * dy) <= a)
    cands.append(np.where(good, t, INF))
    ids.append(SF_BOTTOM)

    for sgn, sid in ((-1.0, SF_LEFT), (1.0, SF_RIGHT)):
        msk = (dy < 0.0) if sgn < 0 else (dy > 0.0)
        t = np.where(msk, (sgn * a - y) / np.where(msk, dy, 1.0), INF)
        zz = z + t * dz
        good = msk & (t > EPS) & (zz >= 0.0) & (zz <= h)
        cands.append(np.where(good, t, INF))

        ids.append(sid)

    if shape == "rect":
        msk = dz > 0.0
        t = np.where(msk, (h - z) / np.where(msk, dz, 1.0), INF)
        good = msk & (t > EPS) & (np.abs(y + t * dy) <= a)
        cands.append(np.where(good, t, INF))
        ids.append(SF_TOP)
    else:


        # jeez



        A2 = dy * dy + dz * dz
        zc = z - h
        B2 = 2.0 * (y * dy + zc * dz)
        C2 = y * y + zc * zc - R * R
        disc = B2 * B2 - 4.0 * A2 * C2
        ok = (A2 > 0.0) & (disc >= 0.0)
        sq = np.sqrt(np.where(ok, disc, 0.0))
        den = 2.0 * np.where(A2 > 0.0, A2, 1.0)
        t_cap = np.full(m, INF)
        for root in ((-B2 - sq) / den, (-B2 + sq) / den):
            good = ok & (root > EPS) & ((z + root * dz) >= h) & (root < t_cap)
            t_cap = np.where(good, root, t_cap)
        cands.append(t_cap)
        ids.append(SF_CAP)

    T = np.stack(cands, axis=1)
    k = np.argmin(T, axis=1)
    t_wall = T[np.arange(m), k]
    id_arr = np.asarray(ids, dtype=np.int8)[k]
    return t_wall, id_arr


def _inward_normal(sid, y, z, g):
    """fin me my normal"""
    R = g["R"]
    h = g["h"]
    nx = np.zeros(y.size)
    ny = np.zeros(y.size)
    nz = np.zeros(y.size)

    m = sid == SF_BOTTOM
    nz = np.where(m, 1.0, nz)
    m = sid == SF_LEFT
    ny = np.where(m, 1.0, ny)
    m = sid == SF_RIGHT
    ny = np.where(m, -1.0, ny)
    m = sid == SF_TOP
    nz = np.where(m, -1.0, nz)
    m = sid == SF_CAP
    if m.any():
        ny = np.where(m, -y / R, ny)
        nz = np.where(m, -(z - h) / R, nz)
    m = sid == SF_CYL
    if m.any():
        rr = np.maximum(np.sqrt(y * y + z * z), 1e-30)
        ny = np.where(m, -y / rr, ny)
        nz = np.where(m, -z / rr, nz)
    return nx, ny, nz


# Vectorized tracer
def trace_batch_3d(rng, n, g, cfg):
    length = g["L"]
    y, z = sample_cross_section(rng, n, g)
    dx, dy, dz = sample_entrance_direction(rng, n)
    sp = m2d.sample_flux_speed(rng, n, cfg["t_gas"])

    status = np.zeros(n, dtype=np.int8)
    hits = np.zeros(n, dtype=np.int32)
    exit_y = np.full(n, np.nan)
    exit_z = np.full(n, np.nan)
    exit_dx = np.full(n, np.nan)
    exit_dy = np.full(n, np.nan)
    exit_dz = np.full(n, np.nan)
    exit_speed = np.full(n, np.nan)
    last_surface = np.full(n, -1, dtype=np.int8)

    oid = np.arange(n)
    xa = np.zeros(n)
    ya, za = y.copy(), z.copy()
    vx, vy, vz = dx.copy(), dy.copy(), dz.copy()
    nh = np.zeros(n, dtype=np.int32)
    lastsf = np.full(n, -1, dtype=np.int8)

    spec = cfg["specular_frac"]
    stick = cfg["stick_prob"]

    for _ in range(cfg["max_bounces"]):
        if oid.size == 0:
            break
        m = oid.size

        t_wall, sid = _wall_intersections(ya, za, vy, vz, g)

        t_exit = np.full(m, np.inf)
        fw = vx > 0.0
        t_exit[fw] = (length - xa[fw]) / vx[fw]
        t_ret = np.full(m, np.inf)
        bw = vx < 0.0
        t_ret[bw] = (0.0 - xa[bw]) / vx[bw]

        t = np.minimum(t_wall, np.minimum(t_exit, t_ret))
        xa = xa + vx * t
        ya = ya + vy * t
        za = za + vz * t

        is_wall = (t_wall <= t_exit) & (t_wall <= t_ret)
        is_exit = (~is_wall) & (t_exit <= t_ret)
        # GAHHH why is python syntax like this
        is_ret = ~is_wall & ~is_exit


        if is_exit.any():
            gidx = oid[is_exit]
            status[gidx] = ST_TRANSMIT
            exit_y[gidx] = ya[is_exit]
            exit_z[gidx] = za[is_exit]
            exit_dx[gidx] = vx[is_exit]
            exit_dy[gidx] = vy[is_exit]
            exit_dz[gidx] = vz[is_exit]
            exit_speed[gidx] = sp[is_exit]
            hits[gidx] = nh[is_exit]
            last_surface[gidx] = lastsf[is_exit]
        if is_ret.any():
            gidx = oid[is_ret]
            status[gidx] = ST_RETURN
            hits[gidx] = nh[is_ret]

        keep = is_wall
        if stick > 0.0:
            stuck = is_wall & (rng.random(m) < stick)
            if stuck.any():
                gidx = oid[stuck]
                status[gidx] = ST_STUCK
                hits[gidx] = nh[stuck] + 1
                keep = keep & ~stuck

        oid, xa, ya, za = oid[keep], xa[keep], ya[keep], za[keep]
        vx, vy, vz = vx[keep], vy[keep], vz[keep]
        sp, nh = sp[keep], nh[keep] + 1
        sid = sid[keep]
        lastsf = sid.copy()
        if oid.size == 0:
            break

        k = oid.size
        nx_, ny_, nz_ = _inward_normal(sid, ya, za, g)
        rx, ry, rz = sample_cosine_about(rng, nx_, ny_, nz_)
        rs = m2d.sample_flux_speed(rng, k, cfg["t_wall"])

        if spec > 0.0:
            is_spec = rng.random(k) < spec
            dot = vx * nx_ + vy * ny_ + vz * nz_
            sx, sy, sz = (vx - 2.0 * dot * nx_, vy - 2.0 * dot * ny_,
                          vz - 2.0 * dot * nz_)
            rx = np.where(is_spec, sx, rx)
            ry = np.where(is_spec, sy, ry)
            rz = np.where(is_spec, sz, rz)
            rs = np.where(is_spec, sp, rs)

        vx, vy, vz, sp = rx, ry, rz, rs

    if oid.size:
        status[oid] = ST_MAXBOUNCE
        hits[oid] = nh

    return {
        "status": status, "hits": hits,
        "exit_y": exit_y, "exit_z": exit_z,
        "exit_dx": exit_dx, "exit_dy": exit_dy, "exit_dz": exit_dz,
        "exit_speed": exit_speed, "last_surface": last_surface,
    }


# My proudest code...
def _scalar_wall_hit(y, z, dy, dz, g):
    """Nearest wall, 50 times faster ish...... Watch my aura"""
    a, h, R, shape = g["a"], g["h"], g["R"], g["shape"]
    best_t, best_id = math.inf, -1

    if shape == "circle":
        A2 = dy * dy + dz * dz
        if A2 > 0.0:
            B2 = 2.0 * (y * dy + z * dz)
            C2 = y * y + z * z - R * R
            disc = B2 * B2 - 4.0 * A2 * C2
            if disc >= 0.0:
                t = (-B2 + math.sqrt(disc)) / (2.0 * A2)
                if t > EPS:
                    best_t, best_id = t, SF_CYL
        return best_t, best_id


    if dz < 0.0:
        t = -z / dz
        if EPS < t < best_t and abs(y + t * dy) <= a:
            best_t, best_id = t, SF_BOTTOM
    if dy < 0.0:
        t = (-a - y) / dy
        if EPS < t < best_t and 0.0 <= z + t * dz <= h:
            best_t, best_id = t, SF_LEFT
    if dy > 0.0:
        t = (a - y) / dy
        if EPS < t < best_t and 0.0 <= z + t * dz <= h:
            best_t, best_id = t, SF_RIGHT

    if shape == "rect":
        if dz > 0.0:
            t = (h - z) / dz
            if EPS < t < best_t and abs(y + t * dy) <= a:
                best_t, best_id = t, SF_TOP
    else:
        A2 = dy * dy + dz * dz
        if A2 > 0.0:
            zc = z - h
            B2 = 2.0 * (y * dy + zc * dz)
            C2 = y * y + zc * zc - R * R
            disc = B2 * B2 - 4.0 * A2 * C2
            if disc >= 0.0:
                sq = math.sqrt(disc)
                for t in ((-B2 - sq) / (2.0 * A2), (-B2 + sq) / (2.0 * A2)):
                    if EPS < t < best_t and (z + t * dz) >= h:
                        best_t, best_id = t, SF_CAP
                        break
    return best_t, best_id


def _scalar_normal(sid, y, z, g):
    if sid == SF_BOTTOM:
        return 0.0, 0.0, 1.0
    if sid == SF_LEFT:
        return 0.0, 1.0, 0.0
    if sid == SF_RIGHT:
        return 0.0, -1.0, 0.0
    if sid == SF_TOP:
        return 0.0, 0.0, -1.0
    if sid == SF_CAP:
        return 0.0, -y / g["R"], -(z - g["h"]) / g["R"]
    r = math.hypot(y, z) or 1e-30
    return 0.0, -y / r, -z / r


def _scalar_cosine(rng, nx, ny, nz):
    u = rng.random()
    r = math.sqrt(u)
    phi = 2.0 * math.pi * rng.random()
    lx, ly, lz = r * math.cos(phi), r * math.sin(phi), math.sqrt(max(0.0, 1.0 - u))
    s = 1.0 if nz >= 0.0 else -1.0
    aa = -1.0 / (s + nz)
    b = nx * ny * aa
    t1 = (1.0 + s * nx * nx * aa, s * b, -s * nx)
    t2 = (b, s + ny * ny * aa, -ny)
    return (lx * t1[0] + ly * t2[0] + lz * nx,
            lx * t1[1] + ly * t2[1] + lz * ny,
            lx * t1[2] + ly * t2[2] + lz * nz)


def trace_single_3d(rng, g, spec_frac=0.0, max_bounces=MAX_BOUNCES):
    """One atom, recording every vertex, entirely in scalar Python."""
    length = g["L"]
    ys, zs = sample_cross_section(rng, 1, g)
    y, z = float(ys[0]), float(zs[0])
    dxa, dya, dza = sample_entrance_direction(rng, 1)
    vx, vy, vz = float(dxa[0]), float(dya[0]), float(dza[0])
    x = 0.0
    verts = [(x, y, z)]
    nh = 0

    while nh < max_bounces:
        tw, sid = _scalar_wall_hit(y, z, vy, vz, g)
        te = (length - x) / vx if vx > 0.0 else math.inf
        tr = (0.0 - x) / vx if vx < 0.0 else math.inf
        t = min(tw, te, tr)
        x += vx * t
        y += vy * t
        z += vz * t
        verts.append((x, y, z))

        if tw <= te and tw <= tr:
            nh += 1
            nx_, ny_, nz_ = _scalar_normal(sid, y, z, g)
            if spec_frac > 0.0 and rng.random() < spec_frac:
                dot = vx * nx_ + vy * ny_ + vz * nz_
                vx, vy, vz = (vx - 2 * dot * nx_, vy - 2 * dot * ny_,
                              vz - 2 * dot * nz_)
            else:
                vx, vy, vz = _scalar_cosine(rng, nx_, ny_, nz_)
            continue

        st = ST_TRANSMIT if te <= tr else ST_RETURN
        # free_path: hits is zero but can be filtered out
        return {"status": st, "verts": verts, "hits": nh,
                "free_path": nh == 0 and st == ST_TRANSMIT,
                "d": (vx, vy, vz), "y": y, "z": z}

    return {"status": ST_MAXBOUNCE, "verts": verts, "hits": nh,
            "free_path": False,
            "d": (vx, vy, vz), "y": y, "z": z}


def sample_direct_paths(rng, g, count, batch=40000, max_batches=400):
    """Zero-bounce trajectories, generated geometrically rather than traced cuz 
    i found like not a lot get transmitted if small batch
    """
    out = []
    for _ in range(max_batches):
        y, z = sample_cross_section(rng, batch, g)
        dx, dy, dz = sample_entrance_direction(rng, batch)
        y2 = y + g["L"] * dy / dx
        z2 = z + g["L"] * dz / dx
        for i in np.flatnonzero(inside_cross_section(y2, z2, g)):
            out.append({"status": ST_TRANSMIT, "hits": 0, "free_path": True,
                        "verts": [(0.0, float(y[i]), float(z[i])),
                                  (g["L"], float(y2[i]), float(z2[i]))],
                        "d": (float(dx[i]), float(dy[i]), float(dz[i])),
                        "y": float(y2[i]), "z": float(z2[i])})
            if len(out) >= count:
                return out
    return out


# 
def direct_flight_quadrature(rng, g, n=4_000_000):
    """Zero-bounce transmission by an entirely different route.

    An atom entering a prism at p with slope t clears it iff p + L t is still in
    the cross-section.

        P = (L^2 / (A pi)) * int_S int_S  dp dq / (L^2 + |p - q|^2)^2,
    """
    py, pz = sample_cross_section(rng, n, g)
    qy, qz = sample_cross_section(rng, n, g)
    d2 = (qy - py) ** 2 + (qz - pz) ** 2
    f = 1.0 / (g["L"] ** 2 + d2) ** 2
    est = g["L"] ** 2 * g["area"] / math.pi * f.mean()
    err = g["L"] ** 2 * g["area"] / math.pi * f.std(ddof=1) / math.sqrt(n)
    return est, err


def direct_flight_intensity(rng, g, theta_edges_deg, n=8_000_000):
    """Zero-bounce intensity per steradian vs polar angle, by quadrature.
   Didn't end up needing much
    """
    py, pz = sample_cross_section(rng, n, g)
    qy, qz = sample_cross_section(rng, n, g)
    rho = np.hypot(qy - py, qz - pz)

    rho_edges = g["L"] * np.tan(np.radians(theta_edges_deg))
    counts, _ = np.histogram(rho, bins=rho_edges)
    ring = math.pi * (rho_edges[1:] ** 2 - rho_edges[:-1] ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        c_mean = g["area"] ** 2 * counts / (n * ring)
    ctr = 0.5 * (theta_edges_deg[1:] + theta_edges_deg[:-1])
    return c_mean * np.cos(np.radians(ctr)) / (g["area"] * math.pi)


# Streaming accumulators (40M atoms will not fit as per-atom arrays)
THETA_BINS = np.linspace(0.0, 90.0, 1801)
ANG2D_LIM = 25.0
ANG2D_BINS = np.linspace(-ANG2D_LIM, ANG2D_LIM, 501)
HIT_BINS = np.concatenate([[-0.5, 0.5], np.geomspace(1.5, 3e4, 120)])
SPEED_BINS = np.linspace(0.0, 1000.0, 401)


class Accum:
    """Everything that has to see all 40M atoms, kept at fixed memory."""

    def __init__(self, g):
        self.g = g
        self.n_launched = 0
        self.n_tx = 0
        self.n_direct = 0
        self.n_ret = 0
        self.n_stuck = 0
        self.n_maxb = 0
        self.sum_hits_tx = 0
        self.sum_hits_ret = 0
        self.theta = np.zeros(THETA_BINS.size - 1)
        self.theta_direct = np.zeros(THETA_BINS.size - 1)
        self.ang2d = np.zeros((ANG2D_BINS.size - 1, ANG2D_BINS.size - 1))
        self.hits_tx = np.zeros(HIT_BINS.size - 1)
        self.hits_ret = np.zeros(HIT_BINS.size - 1)
        self.speed_tx = np.zeros(SPEED_BINS.size - 1)
        self.surface = np.zeros(6)
        self.pos_bins_y = np.linspace(-1.05 * A_HALF, 1.05 * A_HALF, 161)
        self.pos_bins_z = np.linspace(-0.05 * HOLE_HEIGHT, 1.05 * HOLE_HEIGHT, 161)
        self.pos = np.zeros((160, 160))
        self.far = {}
        self.far_bins = {}
        for d_mm in FARFIELD_MM:
            half = 1.4 * (d_mm * 1e-3) * math.tan(math.radians(12.0)) + 0.006
            b = np.linspace(-half, half, 321)
            self.far_bins[d_mm] = b
            self.far[d_mm] = np.zeros((320, 320))
        self.sub = []          # bounded subsample of transmitted atoms
        self.sub_target = 0

    def add(self, out, rng, keep_n):
        st = out["status"]
        tx = st == ST_TRANSMIT
        k = int(tx.sum())
        self.n_launched += st.size
        self.n_tx += k
        self.n_ret += int((st == ST_RETURN).sum())
        self.n_stuck += int((st == ST_STUCK).sum())
        self.n_maxb += int((st == ST_MAXBOUNCE).sum())
        h = out["hits"]
        self.n_direct += int((tx & (h == 0)).sum())
        self.sum_hits_tx += int(h[tx].sum())
        self.sum_hits_ret += int(h[st == ST_RETURN].sum())

        if k == 0:
            return
        dx, dy, dz = out["exit_dx"][tx], out["exit_dy"][tx], out["exit_dz"][tx]
        hh = h[tx]
        th = np.degrees(np.arccos(np.clip(dx, -1.0, 1.0)))
        self.theta += np.histogram(th, bins=THETA_BINS)[0]
        self.theta_direct += np.histogram(th[hh == 0], bins=THETA_BINS)[0]

        ay = np.degrees(np.arctan2(dy, dx))
        az = np.degrees(np.arctan2(dz, dx))
        self.ang2d += np.histogram2d(ay, az, bins=[ANG2D_BINS, ANG2D_BINS])[0]

        self.hits_tx += np.histogram(hh, bins=HIT_BINS)[0]
        self.hits_ret += np.histogram(h[st == ST_RETURN], bins=HIT_BINS)[0]
        self.speed_tx += np.histogram(out["exit_speed"][tx], bins=SPEED_BINS)[0]

        ls = out["last_surface"][tx]
        for s in range(6):
            self.surface[s] += int((ls == s).sum())

        ey, ez = out["exit_y"][tx], out["exit_z"][tx]
        self.pos += np.histogram2d(ey, ez,
                                   bins=[self.pos_bins_y, self.pos_bins_z])[0]

        # Far field and stuff that needs all channels give each atom a random 
        # hole in the 9x9 array cuz I'm not simming 81 times
        iy = rng.integers(0, N_CHAN_Y, size=k)
        iz = rng.integers(0, N_CHAN_Z, size=k)
        oy = (iy - (N_CHAN_Y - 1) / 2.0) * PITCH_Y
        oz = (iz - (N_CHAN_Z - 1) / 2.0) * PITCH_Z
        for d_mm in FARFIELD_MM:
            d = d_mm * 1e-3
            fy = ey + oy + d * dy / dx
            fz = ez + oz + d * dz / dx
            b = self.far_bins[d_mm]
            self.far[d_mm] += np.histogram2d(fy, fz, bins=[b, b])[0]

        if keep_n > 0:
            take = min(keep_n, k)
            self.sub.append(np.column_stack([
                ey[:take], ez[:take], ay[:take], az[:take], th[:take],
                out["exit_speed"][tx][:take], hh[:take]]))

    def subsample(self):
        if not self.sub:
            return np.zeros((0, 7))
        return np.vstack(self.sub)


# Plots
# I had Claude rewrite some of the plots because my plots cost too much overhead on my server

# AI Section STARTS HERE
def _hole_patch(ax, oy=0.0, oz=0.0, **kw):
    ax.add_patch(Rectangle((-A_HALF + oy, 0.0 + oz), 2 * A_HALF, H_RECT, **kw))
    ax.add_patch(Wedge((oy, H_RECT + oz), R_CAP, 0, 180, **kw))


def fig_face_geometry():
    # constrained_layout, not tight_layout: both axes are set_aspect("equal"),
    # which tight_layout does not account for, and the two-line titles end up
    # clipped off the top of the canvas.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7.4),
                                   constrained_layout=True)

    for iy in range(N_CHAN_Y):
        for iz in range(N_CHAN_Z):
            oy = (iy - (N_CHAN_Y - 1) / 2.0) * PITCH_Y
            oz = (iz - (N_CHAN_Z - 1) / 2.0) * PITCH_Z - HOLE_HEIGHT / 2.0
            _hole_patch(ax1, oy, oz, facecolor=TRANSMIT_C, edgecolor="k", lw=0.4)
    span_y = (N_CHAN_Y - 1) * PITCH_Y / 2 + A_HALF
    span_z = (N_CHAN_Z - 1) * PITCH_Z / 2 + HOLE_HEIGHT / 2
    ax1.set_xlim(-span_y * 1.12, span_y * 1.12)
    ax1.set_ylim(-span_z * 1.12, span_z * 1.12)
    ax1.set_aspect("equal")
    ax1.xaxis.set_major_formatter(_MM_FMT)
    ax1.yaxis.set_major_formatter(_MM_FMT)
    ax1.set_xlabel("y [mm]")
    ax1.set_ylabel("z [mm]")
    ax1.set_title(f"Nozzle face, {N_CHAN_Y}x{N_CHAN_Z} holes\n"
                  f"pitch {PITCH_Y_MM} mm (y) x {PITCH_Z_MM} mm (z), "
                  f"open fraction {100*81*AREA/((9*PITCH_Y)*(9*PITCH_Z)):.1f}%")

    _hole_patch(ax2, 0.0, 0.0, facecolor=TRANSMIT_C, alpha=0.35,
                edgecolor="k", lw=1.5)
    ax2.axhline(H_RECT, color="0.4", ls=":", lw=1.0)
    ax2.annotate("", xy=(-A_HALF, -6e-5), xytext=(A_HALF, -6e-5),
                 arrowprops=dict(arrowstyle="<->", color="k"))
    ax2.text(0, -8.5e-5, f"2a = {RECT_WIDTH_MM} mm", ha="center", fontsize=9)
    ax2.annotate("", xy=(A_HALF + 5e-5, 0), xytext=(A_HALF + 5e-5, H_RECT),
                 arrowprops=dict(arrowstyle="<->", color="k"))
    ax2.text(A_HALF + 7e-5, H_RECT / 2, f"h = {RECT_HEIGHT_MM} mm", fontsize=9)
    ax2.annotate("", xy=(0, H_RECT), xytext=(0, H_RECT + R_CAP),
                 arrowprops=dict(arrowstyle="<->", color="k"))
    ax2.text(1e-5, H_RECT + R_CAP / 2, f"R = {CAP_RADIUS_MM} mm", fontsize=9)
    ax2.set_xlim(-2.4 * A_HALF, 2.4 * A_HALF)
    ax2.set_ylim(-1.7e-4, HOLE_HEIGHT * 1.35)
    ax2.set_aspect("equal")
    ax2.xaxis.set_major_formatter(_MM_FMT)
    ax2.yaxis.set_major_formatter(_MM_FMT)
    ax2.set_xlabel("y [mm]")
    ax2.set_title(f"Single aperture.  R > h, so the cap dips below z = 0\n"
                  f"area {AREA*1e6:.5f} mm², hydraulic diameter "
                  f"{D_HYD*1e3:.4f} mm, L/D_h = {L/D_HYD:.1f}")
    return fig


FEW_MAX = 200 # bounce count where I wanted to limit for recording


def _categorise(r):
    if r["status"] == ST_TRANSMIT:
        if r["hits"] == 0:
            return "direct"
        return "few" if r["hits"] <= FEW_MAX else "many"
    if r["status"] == ST_RETURN:
        return "returned" if r["hits"] <= 14 else None
    return None


def _collect_paths(rng, g, quotas, max_tries=40_000):
    """
    Traced categories only, direct is by sample_direct_paths so no need for lots of 
    sim where 0.04% actually go in
    """
    buckets = {k: [] for k in quotas}
    need = dict(quotas)
    need["direct"] = 0
    tries = 0
    while tries < max_tries and any(v > 0 for v in need.values()):
        tries += 1
        r = trace_single_3d(rng, g, SPECULAR_FRAC)
        c = _categorise(r)
        if c is not None and need.get(c, 0) > 0:
            buckets[c].append(r)
            need[c] -= 1
    if quotas.get("direct", 0) > 0:
        buckets["direct"] = sample_direct_paths(rng, g, quotas["direct"])
    return buckets






def export_paths(rng, g, quotas, stem="atom_paths", max_tries=100_000):
    """Write the recorded trajectories out, tagged free-path vs bounced.

    Paths are ragged (2 vertices to well over a thousand), so the npz stores all
    vertices concatenated plus a start/stop index per path, and the CSV repeats
    the per-path fields on every vertex row:

        d = np.load("atom_paths.npz")
        v, s0, s1 = d["verts"], d["start"], d["stop"]
        for i in np.flatnonzero(d["free_path"]):
            plot(v[s0[i]:s1[i], 0], v[s0[i]:s1[i], 1])
    """
    # Transmitted atoms are bimodal (zero collisions or many hundreds), so the
    # "few" bucket needs a far bigger try budget than the figures use. It may
    # still fall short; the caller is told rather than left guessing.
    buckets = _collect_paths(rng, g, quotas, max_tries=max_tries)
    shortfall = {k: quotas[k] - len(buckets.get(k, []))
                 for k in quotas if len(buckets.get(k, [])) < quotas[k]}

    verts, start, stop = [], [], []
    free, hits, status, category = [], [], [], []
    exit_dx, exit_dy, exit_dz = [], [], []
    n = 0
    for cat in DRAW_ORDER:
        for r in buckets.get(cat, []):
            v = np.asarray(r["verts"], dtype=np.float64)
            verts.append(v)
            start.append(n)
            n += v.shape[0]
            stop.append(n)
            # Recompute rather than trusting the flag, so the exported column
            # cannot silently disagree with the trajectory it labels.
            free.append(bool(r["hits"] == 0 and r["status"] == ST_TRANSMIT))
            hits.append(int(r["hits"]))
            status.append(int(r["status"]))
            category.append(cat)
            d = r["d"]
            exit_dx.append(d[0]); exit_dy.append(d[1]); exit_dz.append(d[2])

    if not verts:
        return None

    V = np.vstack(verts)
    meta = dict(verts=V, start=np.array(start), stop=np.array(stop),
                free_path=np.array(free), wall_hits=np.array(hits),
                status=np.array(status),
                category=np.array(category, dtype="U8"),
                exit_dx=np.array(exit_dx), exit_dy=np.array(exit_dy),
                exit_dz=np.array(exit_dz),
                channel_length_m=np.array([g["L"]]))
    # Save arrays only. A dict entry here would force allow_pickle=True on the
    # reader, which is a nasty surprise to hand someone in a data file.
    np.savez_compressed(f"{stem}.npz", **meta)
    meta["_shortfall"] = shortfall

    # Flat CSV: one row per vertex, per-path fields repeated. Bigger, but it
    # drops straight into pandas/Origin without any index bookkeeping.
    rows = []
    for i in range(len(start)):
        seg = V[start[i]:stop[i]]
        pid = np.full(seg.shape[0], i)
        rows.append(np.column_stack([
            pid, np.arange(seg.shape[0]), seg[:, 0], seg[:, 1], seg[:, 2],
            np.full(seg.shape[0], int(free[i])),
            np.full(seg.shape[0], hits[i]),
            np.full(seg.shape[0], status[i])]))
    flat = np.vstack(rows)
    np.savetxt(f"{stem}.csv", flat, delimiter=",",
               fmt=["%d", "%d", "%.9e", "%.9e", "%.9e", "%d", "%d", "%d"],
               header="path_id,vertex_index,x_m,y_m,z_m,free_path,wall_hits,"
                      "status  (status: 1=transmitted 2=returned 4=bounce-capped)",
               comments="")
    return meta


PATH_STYLE = {
    "many": dict(color="#8d99ae", lw=0.35, alpha=0.35, zorder=2,
                 label=f"transmitted, >{FEW_MAX} wall hits"),
    "returned": dict(color=RETURN_C, lw=0.6, alpha=0.55, zorder=3,
                     label="recirculated to the reservoir"),
    "few": dict(color=ACCENT, lw=0.8, alpha=0.80, zorder=4,
                label=f"transmitted, 1-{FEW_MAX} wall hits"),
    "direct": dict(color=TRANSMIT_C, lw=1.15, alpha=0.95, zorder=5,
                   label="transmitted, 0 wall hits (only 0.5% of the beam)"),
}
DRAW_ORDER = ["many", "returned", "few", "direct"]


def fig_trajectories_column(rng, g):
    """Side projection of one column of 9 channels, x-z plane."""
    fig, ax = plt.subplots(figsize=(15, 8.5))
    offs = (np.arange(N_CHAN_Z) - (N_CHAN_Z - 1) / 2.0) * PITCH_Z
    quotas = {"direct": 7, "few": 5, "many": 2, "returned": 9}
    down = DOWNSTREAM_MM * 1e-3

    lo_all = offs[0] - HOLE_HEIGHT / 2
    hi_all = offs[-1] + HOLE_HEIGHT / 2
    pad = 0.35e-3
    ax.add_patch(Rectangle((0, hi_all), L, pad, facecolor="#3d3d3d", zorder=3))
    ax.add_patch(Rectangle((0, lo_all - pad), L, pad, facecolor="#3d3d3d",
                           zorder=3))
    for i in range(N_CHAN_Z - 1):
        a0 = offs[i] + HOLE_HEIGHT / 2
        a1 = offs[i + 1] - HOLE_HEIGHT / 2
        ax.add_patch(Rectangle((0, a0), L, a1 - a0, facecolor="#3d3d3d",
                               zorder=3))

    for off in offs:
        base = off - HOLE_HEIGHT / 2.0
        buckets = _collect_paths(rng, g, quotas)
        for cat in DRAW_ORDER:
            style = {k: v for k, v in PATH_STYLE[cat].items() if k != "label"}
            for r in buckets[cat]:
                v = np.asarray(r["verts"])
                xs, zs = v[:, 0], v[:, 2] + base
                if r["status"] == ST_TRANSMIT:
                    dxv, _, dzv = r["d"]
                    xs = np.append(xs, xs[-1] + down)
                    zs = np.append(zs, zs[-1] + down * dzv / dxv)
                ax.plot(xs, zs, solid_joinstyle="round", **style)

    ax.axvline(0.0, color="0.35", lw=1.0, ls=":", zorder=4)
    ax.axvline(L, color="0.35", lw=1.0, ls=":", zorder=4)
    ax.set_xlim(-0.6e-3, L + down)
    ax.set_ylim(lo_all - 1.6e-3, hi_all + 1.6e-3)
    ax.set_xticks(np.arange(0, (CHANNEL_LENGTH_MM + DOWNSTREAM_MM) * 1e-3, 2e-3))
    ax.set_yticks(offs)
    ax.xaxis.set_major_formatter(_MM_FMT)
    ax.yaxis.set_major_formatter(_MM_FMT)
    ax.set_xlabel("axial position  x  [mm]")
    ax.set_ylabel("transverse position  z  [mm]")
    ax.set_title(
        f"3D nozzle, {RESERVOIR_TEMP_C:.0f} C: one column of {N_CHAN_Z} channels, "
        f"projected into the x-z plane\n (paths sampled "
        "by category, not in proportion to population)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=PATH_STYLE[c]["color"], lw=1.6,
                              label=PATH_STYLE[c]["label"])
                       for c in ("direct", "few", "many", "returned")],
              loc="upper left", framealpha=0.93, fontsize=8.5)
    fig.tight_layout()
    return fig


def fig_trajectories_3d(rng, g):
    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(111, projection="3d")

    xs = np.linspace(0, L, 2)
    # Walk the outline as ONE continuous anticlockwise loop from the bottom
    # left, so the cap must run left -> right, i.e. theta from pi down to 0.
    # Going 0 -> pi traverses the cap backwards and leaves a stray diagonal
    # across the bottom of the box.
    th = np.linspace(math.pi, 0.0, 40)
    cap_y = R_CAP * np.cos(th)          # -R -> +R
    cap_z = H_RECT + R_CAP * np.sin(th)  # h -> h+R -> h
    outline_y = np.concatenate([[-A_HALF, -A_HALF], cap_y, [A_HALF, -A_HALF]])
    outline_z = np.concatenate([[0.0, H_RECT], cap_z, [0.0, 0.0]])
    for xv in (0.0, L):
        ax.plot(np.full_like(outline_y, xv), outline_y, outline_z,
                color="0.35", lw=1.2)
    for j in range(0, outline_y.size, 6):
        ax.plot(xs, [outline_y[j]] * 2, [outline_z[j]] * 2, color="0.75", lw=0.5)

    quotas = {"direct": 14, "few": 8, "many": 2, "returned": 8}
    buckets = _collect_paths(rng, g, quotas)
    # Short downstream stub only: wall-scattered atoms leave at tens of degrees,
    # so the full 9 mm blows the y/z range out to +-8 mm and squashes the
    # 0.3 mm channel into an invisible sliver.
    down = 1.8e-3
    for cat in DRAW_ORDER:
        st = PATH_STYLE[cat]
        for r in buckets[cat]:
            v = np.asarray(r["verts"])
            xv, yv, zv = v[:, 0], v[:, 1], v[:, 2]
            if r["status"] == ST_TRANSMIT:
                dxv, dyv, dzv = r["d"]
                xv = np.append(xv, xv[-1] + down)
                yv = np.append(yv, yv[-1] + down * dyv / dxv)
                zv = np.append(zv, zv[-1] + down * dzv / dxv)
            ax.plot(xv, yv, zv, color=st["color"], lw=st["lw"],
                    alpha=st["alpha"])

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.xaxis.set_major_formatter(_MM_FMT)
    ax.yaxis.set_major_formatter(_MM_FMT)
    ax.zaxis.set_major_formatter(_MM_FMT)
    ax.set_xlim(-0.4e-3, L + down)
    ax.set_ylim(-2.6 * A_HALF, 2.6 * A_HALF)
    ax.set_zlim(-1.3 * R_CAP, HOLE_HEIGHT + 1.3 * R_CAP)
    # The transverse ranges are sub-millimetre; matplotlib's 3D autolocator
    # crams six labels into them and they overprint into an unreadable smear.
    ax.set_yticks([-0.3e-3, 0.0, 0.3e-3])
    ax.set_zticks([0.0, 0.15e-3, 0.30e-3])
    ax.set_box_aspect((3.0, 0.75, 0.75))
    ax.view_init(elev=18, azim=-62)
    ax.set_title("Single channel\n"
                 "(paths sampled by category, not in proportion to the population)")
    fig.tight_layout()
    return fig


def fig_theta_distribution(acc, direct_theta_ref=None, log_y=False):
    fig, ax = plt.subplots(figsize=(10.5, 6))
    ctr = 0.5 * (THETA_BINS[1:] + THETA_BINS[:-1])
    dth = THETA_BINS[1] - THETA_BINS[0]
    n = acc.n_launched

    # Convert counts per theta-bin into intensity per steradian, which is the
    # quantity with a flat cosine-law reference. dOmega = 2 pi sin(theta) dtheta
    solid = 2 * math.pi * np.sin(np.radians(ctr)) * math.radians(dth)
    solid[solid <= 0] = np.nan

    ax.plot(ctr, acc.theta / n / solid, color="0.3", lw=1.2,
            label="Monte Carlo, all exits")
    ax.plot(ctr, acc.theta_direct / n / solid, color=TRANSMIT_C, lw=1.4,
            label="Monte Carlo, zero wall collisions")
    if direct_theta_ref is not None:
        ax.plot(ctr, direct_theta_ref, color=ACCENT, lw=2.0, ls="--",
                label="predicted (if no atom bouncing)")
    ax.plot(ctr, np.cos(np.radians(ctr)) / math.pi, color="#4c956c", lw=1.5,
            ls=":", label=r"bare hole, $\cos\theta/\pi$")
    ax.axvline(GEOM_HALF_ANGLE, color=ACCENT, lw=1.0, alpha=0.8)
    ax.annotate(f"$\\arctan(D_h/L) = {GEOM_HALF_ANGLE:.2f}^\\circ$",
                xy=(0.5, 0.03), xycoords="axes fraction", ha="center",
                fontsize=9, color=ACCENT,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ACCENT,
                          alpha=0.85))
    ax.set_xlabel(r"polar angle from the nozzle axis, $\theta$ [deg]")
    ax.set_ylabel("intensity per steradian, per launched atom")
    if log_y:
        ax.set_yscale("log")
        ax.set_xlim(0, 90)
        ax.set_ylim(1e-8, None)
        ax.set_title("Angular intensity vs analytic expectation (log scale)")
    else:
        ax.set_xlim(0, 8)
        ax.set_title("Angular intensity vs analytic expectation")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


# Minimum atoms required inside the on-axis cap before the peak intensity (and
# hence the FWHM) is trustworthy. 300 gives ~6% on the on-axis value.
MIN_CAP_ATOMS = 300

# --------------------------------------------------------------------------
# Transverse velocity / Doppler reference
# --------------------------------------------------------------------------
# Yb-174 1S0 -> 1P1 at 398.9 nm, the usual transverse-cooling / Zeeman line.
YB174_F0_MHZ = 751_526_533.49
C_LIGHT = 299_792_458.0
# First-order Doppler: df = f0 * v/c. Works out at 2.5068 MHz per (m/s).
MHZ_PER_MPS = YB174_F0_MHZ / C_LIGHT
# Natural linewidth of that line, ~29 MHz, drawn as a reference band. Anything
# much wider than this cannot be addressed by a single detuning.
YB399_LINEWIDTH_MHZ = 29.0
# Half-width of the "probe slice" used for the position-restricted curves.
PROBE_SLICE_MM = _envf("N3_PROBE_SLICE_MM", 2.0)

# Measured reservoir number density [m^-3]; 0 falls back on the vapour-pressure
# fit. Sets every absolute flux number, but no angle, width or transmission
# probability , those are pressure-independent in the free-molecular limit.
MEASURED_DENSITY = _envf("N3_DENSITY", 2.142e19)

# Position window for the transverse-velocity figures and the saved data.
TRANSVERSE_WINDOW_MM = _envf("N3_TRANSVERSE_WINDOW_MM", 50.0)
# Half-thickness of the z ~ 0 slab used for the on-axis line profiles.
ZSLAB_MM = _envf("N3_ZSLAB_MM", 1.0)


def adaptive_profile(vals, nbins=161, pct=99.5, smooth=3):
    """Histogram a sample on bins matched to ITS OWN width.

    A +-2 mm slice at 310 mm is ~40x narrower than the full beam, so on the
    global grid its FWHM spanned four bins and measured the bin width rather
    than the slice. Returned as a DENSITY so curves with different bin widths
    stay comparable.
    """
    vals = np.asarray(vals)
    if vals.size < 2:
        return None
    lim = float(np.percentile(np.abs(vals), pct))
    if not np.isfinite(lim) or lim <= 0.0:
        return None
    edges = np.linspace(-lim, lim, nbins + 1)
    cnt, _ = np.histogram(vals, bins=edges)
    if cnt.sum() == 0:
        return None
    ctr = 0.5 * (edges[1:] + edges[:-1])
    bw = edges[1] - edges[0]
    return dict(ctr=ctr, dens=cnt / cnt.sum() / bw, bw=bw,
                fwhm=_hist_fwhm(ctr, cnt, smooth), n=int(vals.size))


def _hist_fwhm(centers, counts, smooth=5):
    """FWHM of a peaked, roughly symmetric histogram, by half-max crossing."""
    c = np.convolve(counts.astype(float), np.ones(smooth) / smooth, mode="same")
    if c.max() <= 0:
        return float("nan")
    half = c.max() / 2.0
    above = np.flatnonzero(c >= half)
    if above.size < 2:
        return float("nan")
    return float(centers[above[-1]] - centers[above[0]])


def transverse_state(acc, rng):
    """Per-atom transverse velocity and far-field position, from the subsample.

    The subsample stores the two projected angles, not the velocity vector, so
    rebuild it from tan(ay) = dy/dx, tan(az) = dz/dx and dx = 1/sqrt(1+ty^2+tz^2).
    The source hole is re-drawn rather than stored, which is exact: the 81
    channels are identical and independent, so the choice is uniform and
    uncorrelated with velocity.
    """
    sub = acc.subsample()
    if sub.shape[0] == 0:
        return None
    ey, ez, ay, az, _th, sp, _hits = sub.T
    ty, tz = np.tan(np.radians(ay)), np.tan(np.radians(az))
    dx = 1.0 / np.sqrt(1.0 + ty * ty + tz * tz)
    vy, vz = sp * dx * ty, sp * dx * tz

    n = sub.shape[0]
    iy = rng.integers(0, N_CHAN_Y, size=n)
    iz = rng.integers(0, N_CHAN_Z, size=n)
    oy = (iy - (N_CHAN_Y - 1) / 2.0) * PITCH_Y
    oz = (iz - (N_CHAN_Z - 1) / 2.0) * PITCH_Z
    return dict(vy=vy, vz=vz, ty=ty, tz=tz, ey=ey + oy, ez=ez + oz, speed=sp)


def fig_zero_slab_profile(acc):
    """Atom distribution along the z = 0 line at each far-field plane.

    Uses the full far-field 2D histogram rather than the subsample, so this has
    every transmitted atom behind it, not the 2M kept for scatter plots.
    """
    fig, axes = plt.subplots(1, len(FARFIELD_MM), figsize=(14, 5.6))
    axes = np.atleast_1d(axes)
    out = {}
    for ax, d_mm in zip(axes, FARFIELD_MM):
        b = acc.far_bins[d_mm]
        ctr = 0.5 * (b[1:] + b[:-1])
        # far[d] is histogram2d(fy, fz) -> axis 0 is y, axis 1 is z.
        zsel = np.abs(ctr) <= ZSLAB_MM * 1e-3
        if not zsel.any():
            zsel = np.zeros(ctr.size, dtype=bool)
            zsel[ctr.size // 2] = True
        line = acc.far[d_mm][:, zsel].sum(axis=1)
        y_mm = ctr * 1e3

        # A +-1 mm z slab off a 320-bin grid is mostly ones and zeros at modest
        # N. Grouping down to ~120 points is harmless at 750M and the difference
        # between a usable curve and confetti below ~50M.
        target = 120
        fac = max(1, line.size // target)
        if fac > 1:
            n_keep = (line.size // fac) * fac
            line = line[:n_keep].reshape(-1, fac).sum(axis=1)
            y_mm = y_mm[:n_keep].reshape(-1, fac).mean(axis=1)

        ax.step(y_mm, line, where="mid", color=TRANSMIT_C, lw=1.4)
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.7)
        ax.set_xlabel("transverse position y [mm]")
        ax.set_ylabel(f"atoms per bin within $|z| <$ {ZSLAB_MM:.0f} mm")

        tot = line.sum()
        if tot > 0:
            fw = _hist_fwhm(y_mm, line, smooth=3)
            cdf = np.cumsum(line) / tot
            w50 = float(y_mm[np.searchsorted(cdf, 0.75)]
                        - y_mm[np.searchsorted(cdf, 0.25)])
            out[d_mm] = dict(fwhm_mm=fw, iqr_mm=w50)
            ax.axvline(0, color="0.6", lw=0.8, ls=":")
            for s in (-1, 1):
                ax.axvline(s * fw / 2, color=ACCENT, ls="--", lw=1.2)
            ax.set_title(f"{d_mm:.1f} mm plane, z = 0 line profile\n"
                         f"FWHM = {fw:.1f} mm,  central 50% within "
                         f"{w50:.1f} mm", fontsize=10.5)
        # Geometric shadow of the nozzle face plus the aspect-ratio cone.
        edge = ((N_CHAN_Y - 1) * PITCH_Y / 2 + A_HALF
                + d_mm * 1e-3 * math.tan(math.radians(GEOM_HALF_ANGLE))) * 1e3
        for s in (-1, 1):
            ax.axvline(s * edge, color="0.45", ls="-.", lw=1.0)
        ax.grid(alpha=0.25)
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[
        Line2D([], [], color=ACCENT, ls="--", lw=1.2, label="FWHM"),
        Line2D([], [], color="0.45", ls="-.", lw=1.0,
               label="nozzle face + aspect-ratio cone")], fontsize=8,
        loc="upper right", framealpha=0.9)
    fig.suptitle("Beam profile along z = 0")
    fig.tight_layout()
    return fig, out


def fig_doppler_vs_position(acc, rng, d_mm, nbins=61):
    """Mean Doppler shift and broadening vs transverse position, in MHz at 399 nm.

    At each y the mean of v_y sets the detuning a probe must sit at to be
    resonant there; the width sets how much of that slice one detuning reaches.
    """
    st = transverse_state(acc, rng)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    if st is None:
        return fig, {}

    d = d_mm * 1e-3
    fy = (st["ey"] + d * st["ty"]) * 1e3
    df = st["vy"] * MHZ_PER_MPS

    win = TRANSVERSE_WINDOW_MM
    edges = np.linspace(-win, win, nbins + 1)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    idx = np.digitize(fy, edges) - 1
    ok = (idx >= 0) & (idx < nbins)

    mean = np.full(nbins, np.nan)
    sig = np.full(nbins, np.nan)
    cnt = np.zeros(nbins)
    for i in range(nbins):
        v = df[ok & (idx == i)]
        cnt[i] = v.size
        if v.size >= 30:
            mean[i] = float(v.mean())
            sig[i] = float(v.std(ddof=1))

    good = np.isfinite(mean)
    # Ballistic expectation: an atom at y came in at angle ~ y/d, so its
    # transverse velocity is about vbar * y/d and the shift is linear in y.
    vbar_beam = float(np.mean(st["speed"]))
    lin = MHZ_PER_MPS * vbar_beam * (ctr * 1e-3) / d

    ax = axes[0, 0]
    ax.errorbar(ctr[good], mean[good], yerr=(sig[good] / np.sqrt(cnt[good])),
                fmt="o-", ms=3.5, color=TRANSMIT_C, lw=1.3,
                label="MC mean shift")
    ax.plot(ctr, lin, "k--", lw=1.3,
            label=r"ballistic $\bar{v}\,y/d$")
    ax.axhline(0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("transverse position y [mm]")
    ax.set_ylabel(r"mean Doppler shift $\langle\Delta f\rangle$ [MHz]")
    ax.set_title(f"Mean Doppler shift vs position, {d_mm:.1f} mm plane")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.plot(ctr[good], sig[good], "o-", ms=3.5, color=RETURN_C, lw=1.3,
            label=r"$\sigma$ (Gaussian width)")
    ax.plot(ctr[good], 2.3548 * sig[good], "s--", ms=3.0, color="0.4", lw=1.1,
            label=r"FWHM = 2.355$\sigma$")
    ax.axhline(YB399_LINEWIDTH_MHZ, color="#4c956c", lw=1.4, ls="-.",
               label=f"399 nm linewidth ({YB399_LINEWIDTH_MHZ:.0f} MHz)")
    ax.set_xlabel("transverse position y [mm]")
    ax.set_ylabel("Doppler broadening [MHz]")
    ax.set_title("Doppler broadening vs position")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25)

    # Are the local distributions actually Gaussian? Show three of them with a
    # Gaussian of the same mean and sigma laid over. Wall-scattered atoms give
    # heavy tails, so this is worth checking rather than assuming.
    ax = axes[1, 0]
    picks = [i for i in (nbins // 2, int(nbins * 0.68), int(nbins * 0.84))
             if np.isfinite(mean[i])]
    cols = [TRANSMIT_C, ACCENT, RETURN_C]
    for c, i in zip(cols, picks):
        v = df[ok & (idx == i)]
        pr = adaptive_profile(v - v.mean(), nbins=61, pct=99.0)
        if pr is None:
            continue
        ax.step(pr["ctr"] + mean[i], pr["dens"], where="mid", color=c, lw=1.5,
                label=f"y = {ctr[i]:+.0f} mm  ({int(cnt[i]):,} atoms)")
        gx = np.linspace(pr["ctr"][0], pr["ctr"][-1], 300)
        gy = np.exp(-0.5 * (gx / sig[i]) ** 2) / (sig[i] * math.sqrt(2 * math.pi))
        ax.plot(gx + mean[i], gy, color=c, ls="--", lw=1.0, alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel(r"Doppler shift $\Delta f$ [MHz]")
    ax.set_ylabel("probability density [1/MHz]")
    ax.set_title("Local line shapes (dashed = Gaussian of the same $\\sigma$)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.step(ctr, cnt, where="mid", color="0.35", lw=1.4)
    ax.set_yscale("log")
    ax.set_xlabel("transverse position y [mm]")
    ax.set_ylabel("atoms in the subsample per bin")
    ax.set_title("n Atoms")
    ax.grid(alpha=0.25)

    fig.suptitle(f"Doppler shift and broadening vs transverse position, "
                 f"{d_mm:.1f} mm plane  ,  399 nm, "
                 f"{MHZ_PER_MPS:.4f} MHz per m/s")
    fig.tight_layout()
    return fig, dict(centre_mm=ctr, mean_MHz=mean, sigma_MHz=sig, counts=cnt,
                     linear_MHz=lin)


def fig_transverse_velocity(acc, rng, doppler=False):
    """Side-to-side velocity (or Doppler shift) at the two far-field planes.

    With no forces downstream the GLOBAL v_y distribution is identical at every
    plane, fixed the moment the atom leaves. What changes is the correlation
    between position and v_y: by 310 mm position is set by v_y/v_x rather than
    by which hole the atom came from, so the beam has sorted itself. Hence the
    phase-space row and the restricted curves differ between planes while the
    "all atoms" curve does not.
    """
    st = transverse_state(acc, rng)
    scale = MHZ_PER_MPS if doppler else 1.0
    unit = "MHz" if doppler else "m/s"
    xlab = (r"Doppler shift $\Delta f$ [MHz]" if doppler
            else r"transverse velocity $v_y$ [m/s]")

    fig, axes = plt.subplots(2, len(FARFIELD_MM), figsize=(14, 9))
    if st is None:
        return fig

    vy_all, vz_all = st["vy"] * scale, st["vz"] * scale
    win = TRANSVERSE_WINDOW_MM

    for j, d_mm in enumerate(FARFIELD_MM):
        d = d_mm * 1e-3
        fy_all = (st["ey"] + d * st["ty"]) * 1e3

        # Window BEFORE picking any scale: the beam runs past +-1500 mm at the
        # 310 mm plane, so autoscaling to all of it leaves the interesting part
        # a few pixels wide. Every curve below uses this windowed set.
        keep = np.abs(fy_all) <= win
        fy, vy_s, vz_s = fy_all[keep], vy_all[keep], vz_all[keep]
        frac_win = float(keep.mean())
        lim = float(np.percentile(np.abs(vy_s), 99.5)) if vy_s.size else 1.0

        # --- top: phase space, position vs transverse velocity -------------
        ax = axes[0, j]
        h = ax.hist2d(fy, vy_s, bins=[160, 160],
                      range=[[-win, win], [-lim, lim]],
                      cmap="magma", norm=_lognorm())
        fig.colorbar(h[3], ax=ax, label="atoms per bin")
        ax.axhline(0, color="w", lw=0.6, alpha=0.5)
        for s in (-1, 1):
            ax.axvline(s * PROBE_SLICE_MM, color="cyan", ls="--", lw=1.1)
        ax.set_xlim(-win, win)
        ax.set_xlabel("transverse position y [mm]")
        ax.set_ylabel(xlab)
        ax.set_title(f"{d_mm:.1f} mm plane: phase space, $|y| \\leq$ {win:.0f} mm"
                     f"  ({100*frac_win:.1f}% of the beam)\n"
                     f"(cyan = $\\pm${PROBE_SLICE_MM:.1f} mm probe slice)",
                     fontsize=10.5)

        # --- bottom: marginal distributions --------------------------------
        # Every curve gets bins matched to its own width, and is drawn as a
        # density so the different bin widths remain directly comparable.
        ax = axes[1, j]
        p_all = adaptive_profile(vy_s)
        p_z = adaptive_profile(vz_s)
        ax.step(p_all["ctr"], p_all["dens"], where="mid", color="0.25", lw=1.6,
                label=f"all atoms with $|y| \\leq$ {win:.0f} mm")
        # v_z looks wider than v_y here purely because the window is a cut in y
        # only, so nothing restricts z. Not a beam asymmetry - say so.
        ax.step(p_z["ctr"], p_z["dens"], where="mid", color=RETURN_C, lw=1.0,
                ls="--", label=r"$v_z$ (window is in $y$ only)")

        # The slice holds a rapidly shrinking share of the beam as the plane
        # moves out, so at small N it can simply run out of atoms. Say that
        # rather than printing "nan", which reads like a failure.
        sel = np.abs(fy) < PROBE_SLICE_MM
        f_all = p_all["fwhm"]
        p_sl = adaptive_profile(vy_s[sel]) if sel.sum() >= 200 else None
        if p_sl is not None:
            ax.step(p_sl["ctr"], p_sl["dens"], where="mid", color=TRANSMIT_C,
                    lw=1.8,
                    label=f"inside $\\pm${PROBE_SLICE_MM:.1f} mm slice "
                          f"({sel.sum():,} atoms)")
            slice_txt = f"{p_sl['fwhm']:.1f} {unit} (slice)"
        else:
            slice_txt = f"slice has only {sel.sum()} atoms - raise ATOMS"
            ax.annotate(f"only {sel.sum()} atoms inside the slice at this\n"
                        f"distance; needs a larger run to resolve",
                        xy=(0.03, 0.06), xycoords="axes fraction", fontsize=8.5,
                        color=TRANSMIT_C)

        if doppler:
            ax.axvspan(-YB399_LINEWIDTH_MHZ / 2, YB399_LINEWIDTH_MHZ / 2,
                       color="#4c956c", alpha=0.18,
                       label=f"399 nm natural linewidth ({YB399_LINEWIDTH_MHZ:.0f} MHz)")
        ax.set_yscale("log")
        ax.set_xlim(-lim, lim)
        ax.set_xlabel(xlab)
        ax.set_ylabel(f"probability density [1/{unit}]")
        ax.set_title(f"{d_mm:.1f} mm plane: FWHM = {f_all:.1f} {unit} (all), "
                     f"{slice_txt}", fontsize=10.5)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    kind = "Doppler shift" if doppler else "Transverse velocity"
    ref = (f"   Yb-174 $^1S_0\\to{{}}^1P_1$, {YB174_F0_MHZ:,.2f} MHz, "
           f"{MHZ_PER_MPS:.4f} MHz per m/s" if doppler else "")
    fig.suptitle(f"{kind} side to side ($v_y$), $|y| \\leq$ "
                 f"{TRANSVERSE_WINDOW_MM:.0f} mm at each plane{ref}\n"
                 "unwindowed, the distribution would be identical at both "
                 "planes; the position window is what makes them differ",
                 fontsize=11)
    fig.tight_layout()
    return fig


def beam_profile_stats(acc, cap_deg=0.20, smooth_bins=9):
    """Angular profile plus the beam-quality scalars, in one place so the
    summary and the figures cannot drift apart.

    The on-axis value is a solid-angle-weighted mean over a small central cap,
    not a max over the innermost bins: ring solid angle vanishes at theta = 0,
    so one atom in the first bin would be divided by almost nothing.
    """
    ctr = 0.5 * (THETA_BINS[1:] + THETA_BINS[:-1])
    dth = THETA_BINS[1] - THETA_BINS[0]
    solid = 2 * math.pi * np.sin(np.radians(ctr)) * math.radians(dth)
    inten = acc.theta / acc.n_launched / np.where(solid > 0, solid, np.nan)

    # ADAPTIVE central cap. A fixed 0.2 deg cap subtends 2.2e-5 sr, about SIX
    # atoms at 1M launched, and the on-axis value then swung by 6x between seeds
    # and dragged the FWHM from 0.8 to 8 deg. Grow the cap until it means
    # something. At 150M the natural 0.15 deg cap already holds ~900, so this
    # only rescues short runs.
    cum_all = np.cumsum(acc.theta)
    i_nat = max(1, int(np.searchsorted(THETA_BINS, cap_deg)) - 1)
    i_max = max(i_nat, int(np.searchsorted(THETA_BINS, 1.5)) - 1)
    i_cap = i_nat
    while i_cap < i_max and cum_all[i_cap - 1] < MIN_CAP_ATOMS:
        i_cap += 1
    n_cap = float(cum_all[i_cap - 1])
    cap_used = float(THETA_BINS[i_cap])
    omega_cap = 2 * math.pi * (1.0 - math.cos(math.radians(cap_used)))
    peak = float(n_cap / acc.n_launched / omega_cap) if omega_cap > 0 \
        else float("nan")
    # The profile falls across the cap, so a widened cap biases `peak` LOW and
    # the FWHM correspondingly wide. Flag it rather than hiding it.
    cap_starved = n_cap < MIN_CAP_ATOMS

    kern = np.ones(smooth_bins) / smooth_bins
    smooth = np.convolve(np.nan_to_num(inten), kern, mode="same")

    # Half-maximum crossing on EQUAL-SOLID-ANGLE rings, not the uniform theta
    # grid: equal expected counts per ring keep the crossing off the mercy of a
    # few starved bins near the axis, which moved this by 60% between runs.
    e_ctr, e_int, _ = rebin_equal_solid_angle(acc, 8.0, 96)
    e_s = np.convolve(e_int, np.ones(3) / 3.0, mode="same")
    below = np.flatnonzero((e_s < peak / 2.0) & (e_ctr > cap_deg))
    hwhm = float(e_ctr[below[0]]) if below.size else float("nan")

    cum = np.cumsum(acc.theta)
    frac = cum / cum[-1] if cum[-1] > 0 else np.zeros_like(cum)
    th50 = float(ctr[min(np.searchsorted(frac, 0.50), ctr.size - 1)])
    th90 = float(ctr[min(np.searchsorted(frac, 0.90), ctr.size - 1)])
    i_geo = min(int(np.searchsorted(THETA_BINS, GEOM_HALF_ANGLE)) - 1,
                cum.size - 1)
    n_within = float(cum[i_geo])

    return dict(ctr=ctr, solid=solid, inten=inten, smooth=smooth, peak=peak,
                hwhm=hwhm, fwhm=2.0 * hwhm, th50=th50, th90=th90,
                cum=cum, frac=frac, n_within=n_within, i_cap=i_cap,
                omega_cap=omega_cap, cap_deg=cap_used, n_cap=n_cap,
                cap_starved=cap_starved)


def rebin_equal_solid_angle(acc, theta_max_deg, nbins):
    """Re-bin the theta histogram into rings of EQUAL solid angle.

    Uniform theta bins shrink in solid angle like theta, so near the axis they
    are noisy and biased low wherever a bin is empty - a spurious dip right at
    the top of the peak. Equal-solid-angle rings give every point the same
    expected count, so the core is clean without smoothing.
    """
    cum = np.concatenate([[0.0], np.cumsum(acc.theta)])
    om_max = 1.0 - math.cos(math.radians(theta_max_deg))
    edges = np.degrees(np.arccos(1.0 - np.linspace(0.0, om_max, nbins + 1)))
    c_at = np.interp(edges, THETA_BINS, cum)
    counts = np.diff(c_at)
    omega = 2.0 * math.pi * np.diff(1.0 - np.cos(np.radians(edges)))
    ctr = 0.5 * (edges[1:] + edges[:-1])
    return ctr, counts / acc.n_launched / omega, counts


def fig_beam_quality(acc, bp, w):
    """Beam width: the FWHM plot, plus where the flux actually sits."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6))
    ctr, inten, smooth = bp["ctr"], bp["inten"], bp["smooth"]
    peak, hwhm, fwhm = bp["peak"], bp["hwhm"], bp["fwhm"]

    # --- left: the core on equal-solid-angle rings --------------------------
    e_ctr, e_int, e_cnt = rebin_equal_solid_angle(acc, 4.0, 48)
    e_err = e_int / np.sqrt(np.maximum(e_cnt, 1.0))
    for s in (-1, 1):
        ax1.errorbar(s * e_ctr, e_int, yerr=e_err, fmt="o", ms=3.0,
                     color=TRANSMIT_C, lw=0.9, capsize=0,
                     label="equal-solid-angle rings" if s > 0 else None)
    # Show the uniform-bin curve only outside the starved region, otherwise its
    # divide-by-nearly-zero spikes run off the top of the panel.
    vis = ctr > 0.5
    ax1.plot(ctr[vis], inten[vis], color="0.82", lw=0.7, zorder=0,
             label=r"raw uniform bins ($\theta>0.5^\circ$)")
    ax1.axhline(peak, color="0.35", ls=":", lw=1.2)
    ax1.axhline(peak / 2.0, color=ACCENT, ls="--", lw=1.4)
    ax1.annotate(f"on-axis  {peak:.3f}", xy=(0.42, peak), fontsize=8.5,
                 color="0.35", va="bottom")
    ax1.annotate("half maximum", xy=(0.42, peak / 2.0), fontsize=8.5,
                 color=ACCENT, va="bottom")
    if np.isfinite(hwhm):
        ax1.annotate("", xy=(-hwhm, peak / 2.0), xytext=(hwhm, peak / 2.0),
                     arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.6))
        ax1.text(0.0, peak * 0.56, f"FWHM = {fwhm:.2f}$^\\circ$", ha="center",
                 fontsize=11, color=ACCENT, fontweight="bold")
        for s in (-1, 1):
            ax1.axvline(s * hwhm, color=ACCENT, lw=0.9, alpha=0.6)
    ax1.plot(-ctr[vis], inten[vis], color="0.82", lw=0.7, zorder=0)
    ax1.axvline(GEOM_HALF_ANGLE, color="0.5", ls="-.", lw=1.0)
    ax1.axvline(-GEOM_HALF_ANGLE, color="0.5", ls="-.", lw=1.0,
                label=f"$\\arctan(D_h/L)$ = {GEOM_HALF_ANGLE:.2f}$^\\circ$")
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(0, peak * 1.25)
    ax1.set_xlabel(r"polar angle $\theta$ [deg]  (mirrored)")
    ax1.set_ylabel("intensity per steradian, per launched atom")
    sub = (f"on-axis from a {bp['cap_deg']:.2f}$^\\circ$ cap holding "
           f"{bp['n_cap']:.0f} atoms")
    if bp["cap_starved"]:
        sub += ""
    ax1.set_title("Beam core and its fwhm\n",
                  fontsize=10.5)
    ax1.legend(fontsize=8.5, loc="upper right")
    ax1.grid(alpha=0.25)

    # --- right: encircled flux, log axis, the honest picture ---------------
    ax2.semilogx(ctr, bp["frac"], color=TRANSMIT_C, lw=2.0)
    for val, lab, col in ((hwhm, f"HWHM {hwhm:.2f}$^\\circ$", ACCENT),
                          (bp["th50"], f"median {bp['th50']:.1f}$^\\circ$", "0.35"),
                          (bp["th90"], f"90% {bp['th90']:.1f}$^\\circ$", "0.55")):
        if np.isfinite(val) and val > 0:
            ax2.axvline(val, color=col, ls="--", lw=1.3)
            ax2.annotate(lab, xy=(val, 0.05), rotation=90, fontsize=8.5,
                         color=col, ha="right", va="bottom")
    ax2.axvline(GEOM_HALF_ANGLE, color="0.5", ls="-.", lw=1.0)
    ax2.set_xlim(0.05, 90)
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel(r"polar angle $\theta$ [deg]")
    ax2.set_ylabel(r"fraction of transmitted flux within $\theta$")
    ax2.set_title("Encircled flux\n"
                  "")
    ax2.grid(alpha=0.3, which="both")

    inside = bp["n_within"] / max(acc.n_tx, 1)
    txt = "\n".join([
        f"W (Clausing)      = {w:.5f}",
        f"peaking factor k  = {1.0/w:.1f}",
        f"FWHM              = {fwhm:.2f} deg",
        f"median theta      = {bp['th50']:.1f} deg",
        f"90% within        = {bp['th90']:.1f} deg",
        f"inside arctan(Dh/L) = {100*inside:.1f}%",
        f"zero-bounce frac  = {acc.n_direct/acc.n_launched:.3e}",
    ])
    ax2.text(0.03, 0.97, txt, transform=ax2.transAxes, fontsize=8.2,
             family="monospace", va="top",
             bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.6",
                       alpha=0.92))
    fig.tight_layout()
    return fig


def fig_peaking_factor(acc, bp, w, dens_ref=None):
    """The standard effusive-source figure of merit.

        k(theta) = I(theta) / [ W cos(theta) / pi ]

    i.e. this source against a cosine emitter of the SAME throughput, so k = 1
    everywhere would mean the channel bought nothing. On axis k = 1/W exactly,
    since the zero-bounce on-axis intensity is 1/pi per launched atom for any
    prism. The hemisphere average of k is 1 by construction, so a peak above 1
    forces a deficit at large angle: the channel does not create on-axis atoms,
    it deletes off-axis ones.
    """
    fig, ax = plt.subplots(figsize=(11, 6.4))
    ctr = bp["ctr"]
    cosine_ref = w * np.cos(np.radians(ctr)) / math.pi
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = bp["smooth"] / cosine_ref
        kappa_direct = (acc.theta_direct / acc.n_launched
                        / np.where(bp["solid"] > 0, bp["solid"], np.nan)
                        / cosine_ref)

    ax.plot(ctr, kappa, color=TRANSMIT_C, lw=1.8, label=r"$\kappa(\theta)$, all exits")
    ax.plot(ctr, kappa_direct, color=ACCENT, lw=1.2, ls="--",
            label=r"$\kappa(\theta)$, zero-bounce only")
    ax.axhline(1.0, color="k", ls=":", lw=1.4)
    ax.annotate("no better than an orifice of the same throughput",
                xy=(12, 1.05), fontsize=8.5, color="0.3")
    ax.axhline(1.0 / w, color="0.45", ls="-.", lw=1.2)
    ax.annotate(rf"$\kappa(0) = 1/W = {1.0/w:.1f}$", xy=(0.06, 1.0 / w * 1.12),
                fontsize=10, color="0.25")
    ax.axvline(GEOM_HALF_ANGLE, color="0.6", ls="-.", lw=1.0,
               label=f"$\\arctan(D_h/L)$ = {GEOM_HALF_ANGLE:.2f}$^\\circ$")

    cross = np.flatnonzero((kappa < 1.0) & (ctr > bp["cap_deg"]))
    if cross.size:
        tc = ctr[cross[0]]
        ax.axvline(tc, color=RETURN_C, lw=1.2, alpha=0.8)
        ax.annotate(f"crossover {tc:.1f}$^\\circ$", xy=(tc, 2.0), rotation=90,
                    fontsize=8.5, color=RETURN_C, ha="right")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.05, 90)
    ax.set_xlabel(r"polar angle $\theta$ [deg]")
    ax.set_ylabel(r"peaking factor  $\kappa = I(\theta)\,\pi / (W\cos\theta)$")
    ax.set_title("Peaking factor: gain over a cosine source of equal throughput\n"
                 r"(the $\kappa$-weighted hemisphere average is exactly 1)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def fig_flux_budget(acc, bp, flux_in, flux_out, grams_per_day):
    """Where every atom that enters the channel array actually ends up."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    n = acc.n_launched
    ret = acc.n_ret / n
    core = bp["n_within"] / n
    shoulder = acc.n_tx / n - core
    other = max(0.0, 1.0 - ret - core - shoulder)

    segs = [("recirculated to reservoir", ret, RETURN_C),
            ("transmitted, outside the cone", shoulder, "0.6"),
            ("transmitted, inside arctan($D_h/L$)", core, TRANSMIT_C)]
    if other > 1e-9:
        segs.append(("stuck / bounce-capped", other, "0.3"))

    # Row 1: of everything entering. Row 2: renormalised to the transmitted
    # atoms only, because the useful core is ~0.04% of the incident flux and is
    # simply invisible on a linear bar against the 97% that turns around.
    left = 0.0
    for lab, val, col in segs:
        ax1.barh([1], [val], height=0.55, left=[left], color=col,
                 edgecolor="white", label=f"{lab}  ({100*val:.2f}%)")
        left += val

    tx = acc.n_tx / n
    if tx > 0:
        left = 0.0
        for lab, val, col in (("outside the cone", shoulder / tx, "0.6"),
                              ("inside the cone", core / tx, TRANSMIT_C)):
            ax1.barh([0], [val], height=0.55, left=[left], color=col,
                     edgecolor="white")
            ax1.text(left + val / 2, 0, f"{100*val:.1f}%", ha="center",
                     va="center", fontsize=9,
                     color="white" if col != "0.6" else "black")
            left += val

    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["of transmitted\natoms only", "of all atoms\nentering"],
                        fontsize=9)
    ax1.set_xlabel("fraction")
    ax1.set_title("Fate of every atom that enters a channel")
    ax1.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.18))

    # Absolute rates. Recirculated atoms return to the melt and are NOT
    # consumed, so consumption is the transmitted flux, not the incident one.
    rows = [
        ("entering channels", flux_in, ""),
        ("returned to reservoir (free)", flux_in - flux_out, ""),
        ("CONSUMED (transmitted)", flux_out, f"{grams_per_day:.4f} g/day"),
        ("inside the cone (useful)", flux_in * core, ""),
    ]
    ax2.axis("off")
    ax2.set_title("Absolute rates at "
                  f"{RESERVOIR_TEMP_C:.0f} $^\\circ$C", fontsize=11)
    y = 0.86
    for lab, val, extra in rows:
        bold = lab.startswith("CONSUMED")
        ax2.text(0.02, y, lab, fontsize=9.5, va="center",
                 fontweight="bold" if bold else "normal")
        ax2.text(0.72, y, f"{val:.3e} /s", fontsize=9.5, va="center",
                 family="monospace",
                 fontweight="bold" if bold else "normal")
        if extra:
            y -= 0.09
            ax2.text(0.72, y, extra, fontsize=9, va="center", color="0.35",
                     family="monospace")
        y -= 0.13
    ax2.text(0.02, y - 0.02,
             f"Yb consumed per useful atom: {acc.n_tx / max(bp['n_within'], 1):.0f}\n"
             "Recirculated atoms fall back into the melt, so they cost nothing.",
             fontsize=8.5, va="top", color="0.3")
    fig.tight_layout()
    return fig


WEDGE_THETA = np.linspace(0.0, 22.0, 45)   # 0.5 deg rings


def _wedge_profiles(acc):
    """Intensity vs polar angle in four 90-degree azimuthal wedges.

    A thin strip through the centre of the (ay, az) histogram discards >99% of
    the atoms and buries the asymmetry in shot noise. Quadrant wedges keep every
    atom inside the +-25 deg window and still resolve cap-vs-flat, since the
    asymmetry is a smooth azimuthal modulation, not a narrow feature.
    """
    ay = 0.5 * (ANG2D_BINS[1:] + ANG2D_BINS[:-1])
    AY, AZ = np.meshgrid(ay, ay, indexing="ij")
    # (ay, az) are projected angles: tan(ay) = dy/dx, tan(az) = dz/dx.
    ty, tz = np.tan(np.radians(AY)), np.tan(np.radians(AZ))
    th = np.degrees(np.arctan(np.hypot(ty, tz)))
    phi = np.degrees(np.arctan2(tz, ty))

    wedges = {
        "+y": np.abs(phi) < 45.0,
        "+z (round cap)": (phi >= 45.0) & (phi < 135.0),
        "-y": np.abs(phi) >= 135.0,
        "-z (flat bottom)": (phi <= -45.0) & (phi > -135.0),
    }
    idx = np.digitize(th, WEDGE_THETA) - 1
    valid = (idx >= 0) & (idx < WEDGE_THETA.size - 1)
    nring = WEDGE_THETA.size - 1

    # Solid angle of a quadrant slice of each ring.
    c1 = np.cos(np.radians(WEDGE_THETA[:-1]))
    c2 = np.cos(np.radians(WEDGE_THETA[1:]))
    omega = 0.5 * math.pi * (c1 - c2)

    out = {}
    for name, msk in wedges.items():
        m = msk & valid
        counts = np.bincount(idx[m], weights=acc.ang2d[m], minlength=nring)
        out[name] = counts[:nring]
    return out, omega


def fig_asymmetry(acc):
    """The D-shaped aperture makes the beam non-axisymmetric. 3D only."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))
    prof, omega = _wedge_profiles(acc)
    ctr = 0.5 * (WEDGE_THETA[1:] + WEDGE_THETA[:-1])
    norm = acc.n_launched

    styles = {"+z (round cap)": (RETURN_C, "-"), "-z (flat bottom)": (RETURN_C, "--"),
              "+y": (TRANSMIT_C, "-"), "-y": (TRANSMIT_C, "--")}
    for name, counts in prof.items():
        col, ls = styles[name]
        ax1.plot(ctr, counts / norm / omega, color=col, ls=ls, lw=1.5, label=name)
    ax1.set_yscale("log")
    ax1.set_xlabel(r"polar angle $\theta$ [deg]")
    ax1.set_ylabel("intensity per steradian, per launched atom")
    ax1.set_title("Beam intensity in four azimuthal wedges")
    ax1.legend(fontsize=8.5)
    ax1.grid(alpha=0.25)

    # Ratios, with Poisson errors. +y/-y must be 1: the aperture is mirror
    # symmetric in y, so any departure is a bug or pure shot noise, which makes
    # it a free control for the +z/-z signal.
    def ratio_with_err(a, b):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        ok = (a > 0) & (b > 0)
        r = np.where(ok, a / np.where(ok, b, 1.0), np.nan)
        e = np.where(ok, r * np.sqrt(1.0 / np.where(ok, a, 1.0)
                                     + 1.0 / np.where(ok, b, 1.0)), np.nan)
        return r, e

    r_z, e_z = ratio_with_err(prof["+z (round cap)"], prof["-z (flat bottom)"])
    r_y, e_y = ratio_with_err(prof["+y"], prof["-y"])
    ax2.errorbar(ctr, r_z, yerr=e_z, fmt="o-", color=RETURN_C, ms=3.5, lw=1.3,
                 label="+z (cap) / -z (flat)  , the real asymmetry")
    ax2.errorbar(ctr, r_y, yerr=e_y, fmt="s-", color=TRANSMIT_C, ms=3.0, lw=1.0,
                 alpha=0.75, label="+y / -y  , must be 1 (symmetry control)")
    ax2.axhline(1.0, color="k", ls=":", lw=1.2)
    ax2.set_xlim(0, WEDGE_THETA[-1])
    ax2.set_ylim(0.8, 1.25)
    ax2.set_xlabel(r"polar angle $\theta$ [deg]")
    ax2.set_ylabel("intensity ratio")
    ax2.set_title("Up/down asymmetry from the D-shaped aperture\n"
                  "(the +y/-y trace is a built-in symmetry check)")
    ax2.legend(fontsize=8.5)
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def _lognorm():
    from matplotlib.colors import LogNorm
    return LogNorm()


def fig_angular_image(acc):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for ax, lim, title in ((ax1, 6.0, "core"), (ax2, ANG2D_LIM, "full range")):
        im = ax.imshow(np.maximum(acc.ang2d.T, 0.5),
                       origin="lower", norm=_lognorm(), cmap="magma",
                       extent=[ANG2D_BINS[0], ANG2D_BINS[-1],
                               ANG2D_BINS[0], ANG2D_BINS[-1]])
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(r"$\theta_y$ [deg]")
        ax.set_ylabel(r"$\theta_z$ [deg]")
        ax.set_title(f"Far-field angular distribution, {title}")
        fig.colorbar(im, ax=ax, label="atoms per bin")
    fig.suptitle("The aperture shape is imprinted on the angular core "
                 "(flat bottom, round top)")
    fig.tight_layout()
    return fig


def fig_exit_position(acc):
    fig, ax = plt.subplots(figsize=(7.5, 7))
    im = ax.imshow(np.maximum(acc.pos.T, 0.5), origin="lower",
                   norm=_lognorm(), cmap="viridis",
                   extent=[acc.pos_bins_y[0], acc.pos_bins_y[-1],
                           acc.pos_bins_z[0], acc.pos_bins_z[-1]],
                   aspect="equal")
    _hole_patch(ax, facecolor="none", edgecolor="w", lw=1.4)
    ax.xaxis.set_major_formatter(_MM_FMT)
    ax.yaxis.set_major_formatter(_MM_FMT)
    ax.set_xlabel("y [mm]")
    ax.set_ylabel("z [mm]")
    # Quantify it rather than asserting it: compare the mean areal density over
    # the round cap against the flat rectangle. Equal densities would mean the
    # exit distribution carries no memory of the aperture shape.
    cy = 0.5 * (acc.pos_bins_y[1:] + acc.pos_bins_y[:-1])
    cz = 0.5 * (acc.pos_bins_z[1:] + acc.pos_bins_z[:-1])
    CY, CZ = np.meshgrid(cy, cz, indexing="ij")
    in_ap = inside_cross_section(CY, CZ, acc.g)
    in_cap = in_ap & (CZ > H_RECT)
    in_rect = in_ap & (CZ <= H_RECT)
    d_cap = acc.pos[in_cap].mean() if in_cap.any() else np.nan
    d_rect = acc.pos[in_rect].mean() if in_rect.any() else np.nan
    ratio = d_cap / d_rect if d_rect else np.nan
    ax.set_title("Where transmitted atoms exit\n"
                 )
    fig.colorbar(im, ax=ax, label="atoms per bin")
    fig.tight_layout()
    return fig


def fig_farfield_spots(acc):
    fig, axes = plt.subplots(1, len(FARFIELD_MM), figsize=(13.5, 6))
    axes = np.atleast_1d(axes)
    for ax, d_mm in zip(axes, FARFIELD_MM):
        b = acc.far_bins[d_mm]
        im = ax.imshow(np.maximum(acc.far[d_mm].T, 0.5), origin="lower",
                       norm=_lognorm(), cmap="inferno",
                       extent=[b[0], b[-1], b[0], b[-1]], aspect="equal")
        ax.xaxis.set_major_formatter(_MM_FMT)
        ax.yaxis.set_major_formatter(_MM_FMT)
        ax.set_xlabel("y [mm]")
        ax.set_ylabel("z [mm]")
        ax.set_title(f"{d_mm:.0f} mm downstream")
        fig.colorbar(im, ax=ax, label="atoms per bin")
    fig.suptitle("Beam profile, all 81 channels superposed")
    fig.tight_layout()
    return fig


def fig_wall_stats(acc, sub):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 5.2))
    ctr = 0.5 * (HIT_BINS[1:] + HIT_BINS[:-1])
    for h, color, lab in ((acc.hits_ret, RETURN_C, "recirculated"),
                          (acc.hits_tx, TRANSMIT_C, "transmitted")):
        tot = h.sum()
        if tot == 0:
            continue
        surv = 1.0 - np.cumsum(h) / tot + h / tot
        ax1.step(ctr, surv, where="post", color=color, lw=1.8,
                 label=f"{lab}  ({100*h[0]/tot:.1f}% with none)")
    ax1.set_xscale("symlog", linthresh=1)
    ax1.set_yscale("log")
    ax1.set_xlabel("wall collisions, n")
    ax1.set_ylabel(r"$P(\mathrm{collisions} \geq n)$")
    ax1.set_title("Wall collisions before termination")
    ax1.legend(fontsize=8.5)
    ax1.grid(alpha=0.25, which="both")

    hits = sub[:, 6]
    th = sub[:, 4]
    labels, counts, med, p90 = [], [], [], []
    for lo, hi, lab in m2d.HIT_BANDS:
        s = (hits >= lo) & (hits <= hi)
        if s.sum() < 30:
            continue
        labels.append(lab)
        counts.append(int(s.sum()))
        med.append(float(np.median(th[s])))
        p90.append(float(np.percentile(th[s], 90)))
    xp = np.arange(len(labels))
    ax2.bar(xp - 0.2, med, width=0.4, color=TRANSMIT_C, label=r"median $\theta$")
    ax2.bar(xp + 0.2, p90, width=0.4, color="0.45", label=r"90th pct $\theta$")
    ax2.axhline(GEOM_HALF_ANGLE, color=ACCENT, ls="--", lw=1.3,
                label=f"arctan($D_h$/L) = {GEOM_HALF_ANGLE:.2f}$^\\circ$")
    ax2.set_xticks(xp)
    ax2.set_xticklabels(labels)
    ax2.set_yscale("log")
    ax2.set_xlabel("wall collisions before exiting")
    ax2.set_ylabel(r"$\theta$ [deg]")
    ax2.set_title("Collimation vs wall-collision count")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25, axis="y")

    names, vals = [], []
    for s in (SF_BOTTOM, SF_LEFT, SF_RIGHT, SF_CAP):
        if acc.surface[s] > 0:
            names.append(SURFACE_NAMES[s].replace(" ", "\n"))
            vals.append(acc.surface[s])
    tot = sum(vals) if vals else 1
    ax3.bar(names, [100 * v / tot for v in vals], color=["#8d99ae", RETURN_C,
                                                         RETURN_C, TRANSMIT_C])
    ax3.set_ylabel("% of transmitted atoms")
    ax3.set_title("Last surface touched before exit\n"
                  "(direct-flight atoms excluded)")
    ax3.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    return fig


def fig_phase_space(sub):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    lim = 8.0
    for ax, ipos, iang, plab, alab in ((ax1, 0, 2, "y", r"$\theta_y$"),
                                       (ax2, 1, 3, "z", r"$\theta_z$")):
        s = np.abs(sub[:, iang]) < lim
        h = ax.hist2d(sub[s, ipos] * 1e3, sub[s, iang], bins=[150, 130],
                      cmap="magma", norm=_lognorm())
        fig.colorbar(h[3], ax=ax, label="atoms per bin")
        ax.set_xlabel(f"exit position {plab} [mm]")
        ax.set_ylabel(f"{alab} [deg]")
        ax.set_title(f"Emittance in the {plab} plane")
    fig.suptitle("Exit phase space: the bright parallelogram "
                 "are zero bounce atoms")
    fig.tight_layout()
    return fig


def fig_capture(acc):
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ctr = 0.5 * (THETA_BINS[1:] + THETA_BINS[:-1])
    cum = np.cumsum(acc.theta)
    ax.plot(ctr, cum / max(cum[-1], 1), color=TRANSMIT_C, lw=2.0,
            label="fraction of transmitted flux")
    ax.plot(ctr, cum / acc.n_launched, color=RETURN_C, lw=2.0,
            label="fraction of all launched atoms")
    ax.axvline(GEOM_HALF_ANGLE, color=ACCENT, lw=1.3, ls="--",
               label=f"arctan($D_h$/L) = {GEOM_HALF_ANGLE:.2f}$^\\circ$")
    ax.set_xscale("log")
    ax.set_xlim(0.05, 90)
    ax.set_xlabel(r"acceptance half-angle $\theta$ [deg]")
    ax.set_ylabel(r"fraction within $\theta$")
    ax.set_title("Capture efficiency vs downstream acceptance angle")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def fig_speeds(acc, sub, save_means=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    c = 0.5 * (SPEED_BINS[1:] + SPEED_BINS[:-1])
    dens = acc.speed_tx / max(acc.speed_tx.sum(), 1) / (SPEED_BINS[1] - SPEED_BINS[0])
    ax1.plot(c, dens, color=TRANSMIT_C, lw=1.6, label="transmitted")
    for temp, col, lab in ((T_GAS, "k", "flux Maxwellian, gas T"),
                           (T_WALL, "0.55", "flux Maxwellian, wall T")):
        pv = c**3 * np.exp(-M_YB * c**2 / (2 * KB * temp))
        area = np.trapezoid(pv, c) if hasattr(np, "trapezoid") else np.trapz(pv, c)
        ax1.plot(c, pv / area, color=col, ls="--", lw=1.4, label=lab)
    ax1.set_xlabel("speed [m/s]")
    ax1.set_ylabel("density [s/m]")
    ax1.set_title("Speed distribution of the transmitted beam")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)

    lim = 10.0
    s = sub[:, 4] < lim
    bins = np.linspace(0, lim, 61)
    idx = np.digitize(sub[s, 4], bins) - 1
    ctr = 0.5 * (bins[1:] + bins[:-1])
    means = np.array([sub[s, 5][idx == i].mean() if (idx == i).sum() > 30
                      else np.nan for i in range(ctr.size)])
    if save_means is not None:
        np.save(save_means, means)
    ax2.plot(ctr, means, color=TRANSMIT_C, lw=1.8)
    ax2.axhline(np.nanmean(sub[:, 5]), color="k", ls=":", lw=1.2,
                label="beam mean")
    ax2.axvline(GEOM_HALF_ANGLE, color=ACCENT, lw=1.0, ls="--")
    ax2.set_xlabel(r"$\theta$ [deg]")
    ax2.set_ylabel("mean speed [m/s]")
    ax2.set_title("Mean speed vs polar angle\n"
                  "")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def fig_convergence(ns, ws):
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    n = np.asarray(ns, dtype=float)
    w = np.asarray(ws, dtype=float)
    final = w[-1]
    err = np.sqrt(final * (1 - final) / n)
    ax.plot(n, w, color=TRANSMIT_C, lw=1.3, label="running transmission")
    ax.fill_between(n, final - 3 * err, final + 3 * err, color="0.75",
                    alpha=0.55, label=r"$\pm 3\sigma$ binomial band")
    ax.axhline(final, color="k", ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("atoms launched")
    ax.set_ylabel("transmission probability W")
    ax.set_ylim(final - 12 * err[-1], final + 12 * err[-1])
    ax.set_title("Monte Carlo convergence of the 3D transmission probability")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_validation(rng, g, w_mc, n_launched, theta_hist_ref=None):
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8))
    out = {}

    # (1) zero-bounce fraction vs the autocorrelation quadrature
    ax = axes[0]
    ratios = np.array([2.0, 5.0, 10.0, 20.0, 41.0])
    mc, qd, mce = [], [], []
    for r in ratios:
        gg = dict(g)
        gg["L"] = D_HYD * r
        # Zero-bounce atoms are rare at large L/D, so this needs volume; the
        # bounce cap can stay modest because we only count atoms with 0 hits.
        nn = _vn(400_000)
        o = trace_batch_3d(rng, nn, gg,
                           dict(t_gas=T_GAS, t_wall=T_WALL, specular_frac=0.0,
                                stick_prob=0.0, max_bounces=20_000))
        k = int(((o["status"] == ST_TRANSMIT) & (o["hits"] == 0)).sum())
        mc.append(k / nn)
        # The zero-bounce fraction falls as 1/(L/D)^2, so at our aspect ratio
        # this is a handful of atoms out of 400k. Without error bars the last
        # point looks like a discrepancy when it is only Poisson noise.
        mce.append(math.sqrt(max(k, 1)) / nn)
        qd.append(direct_flight_quadrature(rng, gg, _vn(1_000_000, 200_000))[0])
    ax.loglog(ratios, qd, "k-", lw=1.6, label="autocorrelation quadrature")
    ax.errorbar(ratios, mc, yerr=mce, fmt="o", color=TRANSMIT_C, ms=6,
                capsize=3, label="ray tracer (Poisson errors)")
    ax.loglog(ratios, g["area"] / (math.pi * (D_HYD * ratios) ** 2), ":",
              color="0.5", lw=1.4, label=r"long-channel limit $A/\pi L^2$")
    ax.axvline(L / D_HYD, color=ACCENT, ls="--", lw=1.0, label="our channel")
    ax.set_xlabel(r"$L/D_h$")
    ax.set_ylabel("zero-bounce fraction")
    ax.set_title("(1) direct flight vs quadrature")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3, which="both")
    out["direct_quadrature_ratio"] = float(np.mean(np.array(mc) / np.array(qd)))

    # (2) circular tube against the textbook long-tube Clausing limit
    ax = axes[1]
    lr = np.array([10.0, 20.0, 40.0, 80.0, 160.0])
    prod = []
    for v in lr:
        gc = geom_circle(A_HALF, A_HALF * v)
        # Wall-scattered atoms random-walk with step count growing like (L/R)^2,
        # so the bounce budget grows with the tube , but stays bounded, since
        # the vectorised loop iterates while even one atom survives and numpy's
        # ~30 us per-iteration overhead then dominates the runtime.
        o = trace_batch_3d(rng, _vn(80_000), gc,
                           dict(t_gas=T_GAS, t_wall=T_WALL, specular_frac=0.0,
                                stick_prob=0.0,
                                max_bounces=int(max(20_000, 8 * v**2))))
        wv = float((o["status"] == ST_TRANSMIT).mean())
        prod.append(wv * v)
    ax.semilogx(lr, prod, "o-", color=TRANSMIT_C, ms=6, label=r"MC:  $W \cdot L/R$")
    ax.axhline(8.0 / 3.0, color="k", ls="--", lw=1.6, label=r"$8/3$ (Knudsen)")
    ax.set_xlabel("L/R")
    ax.set_ylabel(r"$W \cdot L/R$")
    # W*L/R rises monotonically toward 8/3: at L/R = 10 the exact Clausing
    # factor is 0.1910, i.e. a product of 1.91, and it only reaches ~2.5 by
    # L/R = 80. Approach is from BELOW and is famously slow.
    ax.set_title("(2) circular tube $\\to$ $W = 8R/3L$\n"
                 "(approached from below, slowly)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    out["clausing_product"] = float(prod[-1])

    # (3) a rectangle of growing height must converge onto the 2D slit code.
    # One "wide enough" rectangle is NOT a valid check: at h/w = 60 an atom
    # still hits an end wall about once per traversal, and a diffuse hit there
    # resamples the in-plane direction, which the 2D model never does. That
    # alone costs 6% of the transmission, so the honest test is the limit.
    ax = axes[2]
    slit_w, slit_L = m2d.W, m2d.L
    n3 = _vn(300_000)
    o2 = m2d.trace_batch(rng, _vn(1_000_000), slit_w, slit_L,
                         dict(t_gas=T_GAS, t_wall=T_WALL, specular_frac=0.0,
                              stick_prob=0.0, pair_collisions=None,
                              max_bounces=m2d.MAX_BOUNCES))
    w2 = float((o2["status"] == m2d.ST_TRANSMIT).mean())
    e2 = math.sqrt(w2 * (1 - w2) / _vn(1_000_000))

    ars = np.array([5.0, 20.0, 60.0, 200.0, 600.0, 2000.0])
    ws, es = [], []
    for ar in ars:
        gr = geom_rect(0.5 * slit_w, ar * slit_w, slit_L)
        o = trace_batch_3d(rng, n3, gr,
                           dict(t_gas=T_GAS, t_wall=T_WALL, specular_frac=0.0,
                                stick_prob=0.0, max_bounces=50_000))
        wv = float((o["status"] == ST_TRANSMIT).mean())
        ws.append(wv)
        es.append(math.sqrt(wv * (1 - wv) / n3))
    ws, es = np.array(ws), np.array(es)
    nsig = abs(ws[-1] - w2) / math.sqrt(es[-1] ** 2 + e2**2)

    ax.errorbar(ars, ws, yerr=es, fmt="o-", color=TRANSMIT_C, ms=5,
                label="3D rectangular channel")
    ax.axhline(w2, color="k", ls="--", lw=1.6, label=f"2D slit code ({w2:.4f})")
    ax.fill_between([ars[0], ars[-1]], w2 - 3 * e2, w2 + 3 * e2, color="0.8",
                    alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("rectangle height / width")
    ax.set_ylabel("transmission probability")
    ax.set_title(f"(3) 3D $\\to$ 2D infinite-slit limit\n"
                 f"widest point: $\\Delta W = {nsig:.2f}\\sigma$")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3, which="both")
    out["slit_vs_2d_sigma"] = nsig

    # (4) vectorised vs the independent scalar tracer
    ax = axes[3]
    n_ref = _vn(40000, 4000)
    ref = [trace_single_3d(rng, g, SPECULAR_FRAC) for _ in range(n_ref)]
    wr = sum(1 for r in ref if r["status"] == ST_TRANSMIT) / n_ref
    er = math.sqrt(wr * (1 - wr) / n_ref)
    ev = math.sqrt(w_mc * (1 - w_mc) / n_launched)
    nsig2 = abs(w_mc - wr) / math.sqrt(er**2 + ev**2)
    rth = np.degrees(np.arccos(np.clip(
        [r["d"][0] for r in ref if r["status"] == ST_TRANSMIT], -1.0, 1.0)))
    b4 = np.linspace(0, 60, 61)
    ax.hist(rth, bins=b4, density=True, color=TRANSMIT_C, histtype="step",
            lw=1.8, label=f"scalar  (W={wr:.4f})")
    if theta_hist_ref is not None:
        # Rebin the vectorised theta histogram onto the same grid. Atoms beyond
        # b4[-1] must be DROPPED, not clipped into the last bin: hist discards
        # them for the scalar series, so clipping piles ~40% of the beam into
        # one bin and invents a spike belonging to neither implementation.
        ctr_full = 0.5 * (THETA_BINS[1:] + THETA_BINS[:-1])
        keep = (ctr_full >= b4[0]) & (ctr_full < b4[-1])
        idx = np.searchsorted(b4, ctr_full[keep], side="right") - 1
        idx = np.clip(idx, 0, b4.size - 2)
        reb = np.bincount(idx, weights=np.asarray(theta_hist_ref)[keep],
                          minlength=b4.size - 1)
        reb = reb / reb.sum() / (b4[1] - b4[0])
        ax.stairs(reb, b4, color="0.35", fill=True, alpha=0.6,
                  label=f"vectorised  (W={w_mc:.4f})")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\theta$ [deg]")
    ax.set_title(f"(4) two independent implementations\n"
                 f"$\\Delta W = {nsig2:.2f}\\sigma$")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    out["scalar_vs_vectorised_sigma"] = nsig2
    out["scalar_W"] = wr

    fig.suptitle("Validation")
    fig.tight_layout()
    return fig, out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    g = geom_dshape()

    p_vap = m2d.yb_vapour_pressure_pa(T_GAS)
    lam = m2d.mean_free_path_m(T_GAS)
    vbar = math.sqrt(8 * KB * T_GAS / (math.pi * M_YB))

    # MEASURED_DENSITY wins over the vapour-pressure fit when set; the fit is
    # still reported for comparison. At 419 C the fit gives 2.24e19 m^-3 vs the
    # measured 2.142e19 (~5%), well inside the extrapolation uncertainty.
    n_fit = p_vap / (KB * T_GAS)
    n_dens = MEASURED_DENSITY if MEASURED_DENSITY > 0 else n_fit
    p_eff = n_dens * KB * T_GAS
    # Mean free path follows the density actually in use, not the fit.
    lam_eff = KB * T_GAS / (math.sqrt(2.0) * math.pi * (4.0e-10) ** 2 * p_eff)

    cfg = dict(t_gas=T_GAS, t_wall=T_WALL, specular_frac=SPECULAR_FRAC,
               stick_prob=STICK_PROB, max_bounces=MAX_BOUNCES)

    params = {
        "rect_width_mm": RECT_WIDTH_MM,
        "rect_height_mm": RECT_HEIGHT_MM,
        "cap_radius_mm": CAP_RADIUS_MM,
        "hole_height_mm": HOLE_HEIGHT * 1e3,
        "hole_area_mm2": AREA * 1e6,
        "hydraulic_diameter_mm": D_HYD * 1e3,
        "channel_length_mm": CHANNEL_LENGTH_MM,
        "aspect_L_over_Dh": L / D_HYD,
        "pitch_y_mm": PITCH_Y_MM,
        "pitch_z_mm": PITCH_Z_MM,
        "n_channels": N_CHAN_Y * N_CHAN_Z,
        "geometric_half_angle_deg": GEOM_HALF_ANGLE,
        "reservoir_temp_C": RESERVOIR_TEMP_C,
        "nozzle_wall_temp_C": NOZZLE_TEMP_C,
        "n_atoms": N_ATOMS,
        "seed": SEED,
        "specular_fraction": SPECULAR_FRAC,
        "stick_probability": STICK_PROB,
        "reservoir_density_used_m3": n_dens,
        "reservoir_density_from_vp_fit_m3": n_fit,
        "density_measured_over_fit": n_dens / n_fit,
        "reservoir_pressure_Pa": p_eff,
        "yb_vapour_pressure_fit_Pa": p_vap,
        "mean_free_path_mm": lam_eff * 1e3,
        "knudsen_lambda_over_Dh": lam_eff / D_HYD,
        "knudsen_lambda_over_L": lam_eff / L,
    }

    with start("yb-nozzle-3d", params=params,
               note="3D free-molecular MC, D-shaped apertures, 9x9 array") as run:
        run.say(f"aperture: {RECT_WIDTH_MM} x {RECT_HEIGHT_MM} mm rectangle + "
                f"R={CAP_RADIUS_MM} mm cap, area {AREA*1e6:.5f} mm^2")
        run.say(f"hydraulic diameter {D_HYD*1e3:.4f} mm, L/D_h = {L/D_HYD:.1f}, "
                f"arctan(D_h/L) = {GEOM_HALF_ANGLE:.3f} deg")
        run.say(f"reservoir density {n_dens:.4g} /m^3 "
                f"({'measured' if MEASURED_DENSITY > 0 else 'from vp fit'}); "
                f"vp fit gives {n_fit:.4g} /m^3, ratio {n_dens/n_fit:.3f}")
        run.say(f"Yb at {RESERVOIR_TEMP_C:.0f} C: P = {p_eff:.4g} Pa, "
                f"lambda = {lam_eff*1e3:.1f} mm, lambda/L = {lam_eff/L:.2f} "
                f"-> free-molecular is justified")

        q_est, q_err = direct_flight_quadrature(rng, g, 4_000_000)
        run.say(f"independent quadrature, zero-bounce fraction = "
                f"{q_est:.6f} +/- {q_err:.6f}")

        acc = Accum(g)
        per_chunk = max(1, N_ATOMS // N_CHUNKS)
        keep_per_chunk = max(1, SUBSAMPLE_MAX // N_CHUNKS)
        conv_n, conv_w = [], []

        for c in range(N_CHUNKS):
            out = trace_batch_3d(rng, per_chunk, g, cfg)
            acc.add(out, rng, keep_per_chunk)

            w = acc.n_tx / acc.n_launched
            conv_n.append(acc.n_launched)
            conv_w.append(w)
            cum = np.cumsum(acc.theta)
            i_geo = np.searchsorted(THETA_BINS, GEOM_HALF_ANGLE) - 1
            within = cum[min(i_geo, cum.size - 1)] if cum.size else 0.0

            run.log(
                step=acc.n_launched,
                transmission_probability=w,
                transmission_stderr=math.sqrt(max(w * (1 - w), 0) / acc.n_launched),
                direct_flight_fraction=acc.n_direct / acc.n_launched,
                frac_transmitted_within_geometric_cone=within / max(acc.n_tx, 1),
                frac_launched_within_geometric_cone=within / acc.n_launched,
                mean_wall_hits_transmitted=acc.sum_hits_tx / max(acc.n_tx, 1),
                mean_wall_hits_recirculated=acc.sum_hits_ret / max(acc.n_ret, 1),
                recirculated_fraction=acc.n_ret / acc.n_launched,
            )

            if (c + 1) % PROGRESS_EVERY == 0:
                run.say(f"{acc.n_launched:,}/{per_chunk*N_CHUNKS:,} atoms, "
                        f"W = {w:.5f}")

        # ---------------- derived ----------------
        w = acc.n_tx / acc.n_launched
        w_err = math.sqrt(w * (1 - w) / acc.n_launched)
        direct_mc = acc.n_direct / acc.n_launched
        sub = acc.subsample()

        bp = beam_profile_stats(acc)
        ctr = bp["ctr"]
        cum = bp["cum"]
        frac = bp["frac"]
        n_within = bp["n_within"]
        th50, th90 = bp["th50"], bp["th90"]
        solid = bp["solid"]
        inten = bp["inten"]
        cap_deg = bp["cap_deg"]
        i_cap = bp["i_cap"]
        omega_cap = bp["omega_cap"]
        peak = bp["peak"]
        hwhm = bp["hwhm"]
        fwhm = bp["fwhm"]

        # For ANY prism cross-section the zero-bounce intensity at theta = 0 is
        # 1/pi per sr per launched atom, independent of length and shape. The
        # channel does not raise on-axis intensity per RESERVOIR atom; it
        # removes off-axis ones. The gain per atom CONSUMED is 1/W.
        on_axis_exact = 1.0 / math.pi
        peak_gain = 1.0 / w if w > 0 else float("nan")

        # `peak` is a solid-angle-weighted average over a 0.2 deg cap, not the
        # theta -> 0 limit, so it sits below 1/pi by construction (the
        # covariogram is ~12% down at the cap edge) and comparing the two looks
        # like a 10-15% error that is not real. The honest test is against the
        # quadrature over the identical cap, computed below once dens_ref exists.

        open_area = N_CHAN_Y * N_CHAN_Z * AREA
        flux_in = 0.25 * n_dens * vbar * open_area
        flux_out = flux_in * w
        flux_useful = flux_in * (n_within / acc.n_launched)
        grams_per_day = flux_out * M_YB * 1000.0 * 86400.0

        run.say(f"FLUX at n = {n_dens:.4g} /m^3, {RESERVOIR_TEMP_C:.0f} C, "
                f"vbar = {vbar:.1f} m/s:")
        run.say(f"   into the 81 channels   {flux_in:.4g} atoms/s "
                f"({flux_in/(N_CHAN_Y*N_CHAN_Z):.4g} per hole)")
        run.say(f"   out of the nozzle      {flux_out:.4g} atoms/s "
                f"= {grams_per_day:.4g} g/day consumed")
        run.say(f"   inside arctan(Dh/L)    {flux_useful:.4g} atoms/s")
        run.say(f"transmission W = {w:.5f} +/- {w_err:.5f}")
        run.say(f"zero-bounce: MC {direct_mc:.6f} vs quadrature {q_est:.6f} "
                f"(ratio {direct_mc/q_est:.4f})")
        run.say(f"beam: {100*n_within/acc.n_tx:.1f}% inside arctan(D_h/L), "
                f"median theta {th50:.2f} deg, FWHM {fwhm:.2f} deg "
                f"(peaking factor kappa = 1/W = {1.0/w:.1f})")
        if bp["cap_starved"]:
            run.say(f"  WARNING: only {bp['n_cap']:.0f} atoms inside the "
                    f"{bp['cap_deg']:.2f} deg on-axis cap, so the peak is "
                    f"underestimated and the FWHM is biased WIDE. Needs roughly "
                    f"50M+ atoms for an unbiased width; raise ATOMS.")

        # ---------------- figures ----------------
        prng = np.random.default_rng(SEED + 1)
        dens_ref = direct_flight_intensity(prng, g, THETA_BINS, 8_000_000)

        # Like-for-like on-axis check: zero-bounce atoms only, both sides
        # averaged over the same 0.2 deg cap with the same solid-angle weights.
        w_cap = solid[:i_cap]
        cap_quad = float(np.nansum(dens_ref[:i_cap] * w_cap) / np.sum(w_cap))
        cap_mc = float(acc.theta_direct[:i_cap].sum() / acc.n_launched / omega_cap)
        n_cap = float(acc.theta_direct[:i_cap].sum())
        cap_ratio = cap_mc / cap_quad if cap_quad > 0 else float("nan")
        run.say(f"on-axis (zero-bounce, {cap_deg} deg cap): MC {cap_mc:.4f} vs "
                f"quadrature {cap_quad:.4f} /sr  (ratio {cap_ratio:.3f} from "
                f"{n_cap:.0f} atoms; exact theta->0 limit is 1/pi = "
                f"{on_axis_exact:.4f})")
        run.say(f"peak intensity gain per CONSUMED atom = 1/W = "
                f"{peak_gain:.1f}x a bare orifice")
        zfig, zstats = fig_zero_slab_profile(acc)
        run.figure(zfig, "z0_line_profile", close=True)
        for d_mm, v in zstats.items():
            run.say(f"z=0 profile at {d_mm:.1f} mm: FWHM {v['fwhm_mm']:.1f} mm, "
                    f"central 50% within {v['iqr_mm']:.1f} mm")

        dfig, dprof = fig_doppler_vs_position(
            acc, np.random.default_rng(SEED + 7), FARFIELD_MM[0])
        run.figure(dfig, "doppler_shift_and_broadening_vs_position", close=True)

        tv_rng = np.random.default_rng(SEED + 7)
        run.figure(fig_transverse_velocity(acc, tv_rng, doppler=False),
                   "transverse_velocity_profile", close=True)
        run.figure(fig_transverse_velocity(acc, np.random.default_rng(SEED + 7),
                                           doppler=True),
                   "transverse_doppler_profile", close=True)
        run.figure(fig_beam_quality(acc, bp, w), "beam_quality_fwhm", close=True)
        run.figure(fig_peaking_factor(acc, bp, w, dens_ref), "peaking_factor",
                   close=True)
        run.figure(fig_flux_budget(acc, bp, flux_in, flux_out, grams_per_day),
                   "flux_budget", close=True)
        run.figure(fig_face_geometry(), "nozzle_face_geometry", close=True)
        run.figure(fig_trajectories_column(prng, g),
                   "nozzle_trajectories_9_channels", close=True)
        run.figure(fig_trajectories_3d(prng, g), "trajectories_3d", close=True)
        run.figure(fig_theta_distribution(acc, dens_ref, log_y=False),
                   "exit_angle_vs_expected", close=True)
        run.figure(fig_theta_distribution(acc, dens_ref, log_y=True),
                   "exit_angle_log_shoulders", close=True)
        run.figure(fig_asymmetry(acc), "beam_asymmetry_from_aperture", close=True)
        run.figure(fig_angular_image(acc), "far_field_angular_image", close=True)
        run.figure(fig_exit_position(acc), "exit_position_map", close=True)
        run.figure(fig_farfield_spots(acc), "far_field_spots", close=True)
        run.figure(fig_wall_stats(acc, sub), "wall_collision_statistics",
                   close=True)
        run.figure(fig_phase_space(sub), "exit_phase_space", close=True)
        run.figure(fig_capture(acc), "capture_efficiency_vs_acceptance",
                   close=True)
        run.figure(fig_speeds(acc, sub, save_means=run.path("mean_speed_vs_theta.npy")),
                   "speed_distributions", close=True)
        run.figure(fig_convergence(conv_n, conv_w), "transmission_convergence",
                   close=True)

        vfig, val = fig_validation(np.random.default_rng(SEED + 2), g, w,
                                   acc.n_launched, acc.theta)
        run.figure(vfig, "validation", close=True)

        # ---------------- artefacts ----------------
        np.savetxt(run.path("theta_distribution_3d.csv"),
                   np.column_stack([ctr, acc.theta,
                                    acc.theta / acc.n_launched,
                                    acc.theta_direct / acc.n_launched]),
                   delimiter=",",
                   header="theta_deg,counts,frac_per_bin,frac_per_bin_zero_bounce",
                   comments="")
        run.file("theta_distribution_3d.csv", "polar-angle-distribution")
        np.savez_compressed(run.path("exit_state_3d.npz"), subsample=sub,
                            columns=np.array(["exit_y", "exit_z", "theta_y",
                                              "theta_z", "theta", "speed",
                                              "wall_hits"]))
        run.file("exit_state_3d.npz", "transmitted-atom-subsample")

        # ---- trajectories, tagged free-path vs bounced --------------------
        pm = export_paths(np.random.default_rng(SEED + 11), g,
                          PATH_EXPORT_QUOTAS, stem=run.path("atom_paths"))
        if pm is not None:
            nfree = int(pm["free_path"].sum())
            run.say(f"exported {pm['free_path'].size} trajectories "
                    f"({nfree} free-path, "
                    f"{pm['free_path'].size - nfree} bounced), "
                    f"{pm['verts'].shape[0]:,} vertices total")
            if pm["_shortfall"]:
                run.say(f"  note: quota not met for {pm['_shortfall']} - those "
                        f"categories are genuinely rare, not a failure")
            run.file("atom_paths.npz", "atom-paths-npz")
            run.file("atom_paths.csv", "atom-paths-csv")

        # Transverse-velocity / Doppler scalars. The global width is a property
        # of the nozzle, not of the plane, so it is quoted once; the slice
        # widths are per plane because those DO depend on distance.
        tv = transverse_state(acc, np.random.default_rng(SEED + 7))
        tv_extra = {}

        # ---- save the transverse-velocity data ---------------------------
        # Per-atom arrays inside the +-TRANSVERSE_WINDOW_MM window at each
        # plane, plus the binned profile behind the Doppler figure.
        if tv is not None:
            arrs = {"MHz_per_m_per_s": np.array([MHZ_PER_MPS]),
                    "yb174_f0_MHz": np.array([YB174_F0_MHZ]),
                    "window_mm": np.array([TRANSVERSE_WINDOW_MM]),
                    "reservoir_density_m3": np.array([n_dens]),
                    "flux_out_per_s": np.array([flux_out])}
            for d_mm in FARFIELD_MM:
                fy = (tv["ey"] + d_mm * 1e-3 * tv["ty"]) * 1e3
                fz = (tv["ez"] + d_mm * 1e-3 * tv["tz"]) * 1e3
                k = np.abs(fy) <= TRANSVERSE_WINDOW_MM
                tag = f"{d_mm:.0f}mm"
                arrs[f"y_mm_{tag}"] = fy[k].astype(np.float32)
                arrs[f"z_mm_{tag}"] = fz[k].astype(np.float32)
                arrs[f"vy_mps_{tag}"] = tv["vy"][k].astype(np.float32)
                arrs[f"vz_mps_{tag}"] = tv["vz"][k].astype(np.float32)
                arrs[f"speed_mps_{tag}"] = tv["speed"][k].astype(np.float32)
                arrs[f"doppler_MHz_{tag}"] = (
                    tv["vy"][k] * MHZ_PER_MPS).astype(np.float32)
            np.savez_compressed(run.path("transverse_velocity.npz"), **arrs)
            run.file("transverse_velocity.npz", "transverse-velocity-per-atom")

            if dprof:
                np.savetxt(
                    run.path("doppler_vs_position_76mm.csv"),
                    np.column_stack([dprof["centre_mm"], dprof["mean_MHz"],
                                     dprof["sigma_MHz"],
                                     2.3548 * dprof["sigma_MHz"],
                                     dprof["mean_MHz"] / MHZ_PER_MPS,
                                     dprof["sigma_MHz"] / MHZ_PER_MPS,
                                     dprof["counts"], dprof["linear_MHz"]]),
                    delimiter=",",
                    header="y_mm,mean_doppler_MHz,sigma_doppler_MHz,"
                           "fwhm_doppler_MHz,mean_vy_mps,sigma_vy_mps,"
                           "n_atoms,ballistic_linear_MHz",
                    comments="")
                run.file("doppler_vs_position_76mm.csv",
                         "doppler-shift-and-broadening-vs-position")
        if tv is not None:
            p = adaptive_profile(tv["vy"])
            fw_v = p["fwhm"]
            tv_extra["transverse_vy_fwhm_m_per_s"] = fw_v
            tv_extra["transverse_vy_rms_m_per_s"] = float(np.std(tv["vy"]))
            tv_extra["doppler_fwhm_MHz"] = fw_v * MHZ_PER_MPS
            tv_extra["doppler_rms_MHz"] = float(np.std(tv["vy"])) * MHZ_PER_MPS
            tv_extra["MHz_per_m_per_s"] = MHZ_PER_MPS
            for d_mm in FARFIELD_MM:
                fy = (tv["ey"] + d_mm * 1e-3 * tv["ty"]) * 1e3
                sel = np.abs(fy) < PROBE_SLICE_MM
                key = f"{d_mm:.0f}mm"
                if sel.sum() >= 200:
                    # Own bins again - on the global grid this was pinned at
                    # the bin width and did not measure the slice at all.
                    p2 = adaptive_profile(tv["vy"][sel])
                    tv_extra[f"vy_fwhm_m_per_s_in_slice_at_{key}"] = p2["fwhm"]
                    tv_extra[f"doppler_fwhm_MHz_in_slice_at_{key}"] = \
                        p2["fwhm"] * MHZ_PER_MPS
                    tv_extra[f"frac_of_beam_in_slice_at_{key}"] = \
                        float(sel.mean())
            run.say(f"transverse v_y: FWHM {fw_v:.1f} m/s, rms "
                    f"{np.std(tv['vy']):.1f} m/s  ->  Doppler FWHM "
                    f"{fw_v*MHZ_PER_MPS:.0f} MHz on the 399 nm line "
                    f"({MHZ_PER_MPS:.4f} MHz per m/s)")

        run.finish(summary={
            "transmission_probability_3D": w,
            "transmission_stderr": w_err,
            "direct_flight_fraction_mc": direct_mc,
            "direct_flight_fraction_quadrature": q_est,
            "direct_mc_over_quadrature": direct_mc / q_est,
            "clausing_product_should_be_2p667": val["clausing_product"],
            "wide_slit_vs_2d_code_sigma": val["slit_vs_2d_sigma"],
            "scalar_vs_vectorised_sigma": val["scalar_vs_vectorised_sigma"],
            "fwhm_deg": fwhm,
            "hwhm_deg": hwhm,
            "peaking_factor_kappa": peak_gain,
            "on_axis_cap_deg": bp["cap_deg"],
            "on_axis_cap_atoms": bp["n_cap"],
            "on_axis_cap_starved": int(bp["cap_starved"]),
            "on_axis_intensity_per_sr_all_exits": peak,
            "on_axis_zero_bounce_mc_per_sr": cap_mc,
            "on_axis_zero_bounce_quadrature_per_sr": cap_quad,
            "on_axis_mc_over_quadrature": cap_ratio,
            "on_axis_exact_theta0_limit_1_over_pi": on_axis_exact,
            "peak_intensity_gain_vs_bare_orifice": peak_gain,
            "frac_beam_within_geometric_cone": n_within / acc.n_tx,
            "frac_launched_within_geometric_cone": n_within / acc.n_launched,
            "theta_containing_50pct": th50,
            "theta_containing_90pct": th90,
            "mean_wall_hits_transmitted": acc.sum_hits_tx / max(acc.n_tx, 1),
            "mean_wall_hits_recirculated": acc.sum_hits_ret / max(acc.n_ret, 1),
            "recirculated_fraction": acc.n_ret / acc.n_launched,
            "maxbounce_atoms": acc.n_maxb,
            "flux_into_nozzle_per_s": flux_in,
            "flux_out_per_s": flux_out,
            "flux_within_cone_per_s": flux_useful,
            "yb_consumption_g_per_day": grams_per_day,
            "yb_consumed_per_useful_atom": acc.n_tx / max(n_within, 1),
            **tv_extra,
        })


if __name__ == "__main__":
    main()