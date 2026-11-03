"""Kaiser-group anti-Helmholtz MOT coils (Letellier et al. 2023), magpylib.

Two 16 x 16 windings of 2.2 mm wire, 55 mm bore, 65 mm between the inner
faces, axis along z. Field is tabulated once on an (r, z) grid at 1 A.
"""

import math
import os

import magpylib as magpy
import numpy as np
from scipy.interpolate import RegularGridInterpolator

WIRE_MM = 2.2
N_RADIAL = 16
N_AXIAL = 16
BORE_DIAMETER_MM = 55.0
INNER_GAP_MM = 65.0

LOOP_RADII_MM = BORE_DIAMETER_MM / 2 + WIRE_MM / 2 + WIRE_MM * np.arange(N_RADIAL)
LOOP_HEIGHTS_MM = INNER_GAP_MM / 2 + WIRE_MM / 2 + WIRE_MM * np.arange(N_AXIAL)

R_TABLE_M = np.unique(np.round(np.concatenate([
    np.arange(0.0, 0.150, 0.5e-3), np.arange(0.150, 0.700 + 1e-9, 5e-3)]), 9))
Z_TABLE_M = np.round(np.arange(-0.040, 0.040 + 1e-9, 0.5e-3), 9)

GEOMETRY = dict(wire_mm=WIRE_MM, n_radial=N_RADIAL, n_axial=N_AXIAL,
                bore_mm=BORE_DIAMETER_MM, gap_mm=INNER_GAP_MM,
                r_max=float(R_TABLE_M[-1]), z_max=float(Z_TABLE_M[-1]))

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache",
                     "kaiser_coil_table.npz")

_LEN = 1.0 if int(magpy.__version__.split(".")[0]) >= 5 else 1e3
_TESLA = 1.0 if _LEN == 1.0 else 1e-3
_coils = None
_table = None


def coils():
    """Top coil +1 A, bottom -1 A, so dBz/dz > 0 at the centre."""
    global _coils
    if _coils is None:
        _coils = magpy.Collection()
        for sign in (+1, -1):
            for h in LOOP_HEIGHTS_MM:
                for r in LOOP_RADII_MM:
                    _coils.add(magpy.current.Circle(
                        current=sign, diameter=2 * r * 1e-3 * _LEN,
                        position=(0, 0, sign * h * 1e-3 * _LEN)))
    return _coils


def bfield_exact(xyz_m, current_A=1.0):
    """Direct Biot-Savart sum in tesla, (N, 3) in and out."""
    xyz_m = np.atleast_2d(np.asarray(xyz_m, dtype=float))
    return np.atleast_2d(coils().getB(xyz_m * _LEN)) * _TESLA * current_A


def field_table():
    """(B_r, B_z) at 1 A on R_TABLE_M x Z_TABLE_M, cached to disk."""
    global _table
    if _table is not None:
        return _table
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        if (z["r"].shape == R_TABLE_M.shape and np.allclose(z["r"], R_TABLE_M)
                and z["z"].shape == Z_TABLE_M.shape
                and np.allclose(z["z"], Z_TABLE_M)
                and str(z["geometry"]) == repr(sorted(GEOMETRY.items()))):
            _table = (z["r"], z["z"], z["Br"], z["Bz"])
            return _table
    R, Z = np.meshgrid(R_TABLE_M, Z_TABLE_M, indexing="ij")
    pts = np.column_stack([R.ravel(), np.zeros(R.size), Z.ravel()])
    B = np.concatenate([bfield_exact(pts[i:i + 20000])
                        for i in range(0, len(pts), 20000)])
    Br, Bz = B[:, 0].reshape(R.shape), B[:, 2].reshape(R.shape)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, r=R_TABLE_M, z=Z_TABLE_M, Br=Br, Bz=Bz,
                        geometry=repr(sorted(GEOMETRY.items())))
    _table = (R_TABLE_M, Z_TABLE_M, Br, Bz)
    return _table


def gradient_per_amp_G_cm():
    """dBz/dz at the centre for 1 A, exact."""
    dz = 1e-4
    B = bfield_exact([[0, 0, dz], [0, 0, -dz]])
    return (B[0, 2] - B[1, 2]) / (2 * dz) * 100.0


def current_for_gradient(grad_G_cm):
    return grad_G_cm / gradient_per_amp_G_cm()


def field_function(grad_G_cm):
    """B(xyz) in tesla for xyz of shape (3, ...) in metres, at the current
    giving grad_G_cm along z."""
    r_ax, z_ax, Br, Bz = field_table()
    amps = current_for_gradient(grad_G_cm)
    fr = RegularGridInterpolator((r_ax, z_ax), Br * amps, bounds_error=False,
                                 fill_value=None)
    fz = RegularGridInterpolator((r_ax, z_ax), Bz * amps, bounds_error=False,
                                 fill_value=None)

    def B(xyz):
        xyz = np.asarray(xyz, dtype=float)
        x, y, z = xyz[0], xyz[1], xyz[2]
        r = np.hypot(x, y)
        rz = np.stack([np.ravel(r), np.ravel(z)], axis=-1)
        br = fr(rz).reshape(np.shape(r))
        bz = fz(rz).reshape(np.shape(r))
        safe = np.where(r > 1e-12, r, 1.0)
        return np.array([br * x / safe, br * y / safe, bz])
    return B


if __name__ == "__main__":
    g1 = gradient_per_amp_G_cm()
    print(f"{2 * N_RADIAL * N_AXIAL} turns, loop radii {LOOP_RADII_MM[0]:.1f}-"
          f"{LOOP_RADII_MM[-1]:.1f} mm, heights {LOOP_HEIGHTS_MM[0]:.1f}-"
          f"{LOOP_HEIGHTS_MM[-1]:.1f} mm")
    print(f"dBz/dz = {g1:.3f} G/cm per A")
    for g in (3.1, 18.0, 32.0, 50.0):
        print(f"  {g:5.1f} G/cm  ->  {g / g1:6.2f} A")
    B = field_function(32.0)
    for s in (2, 5, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 520):
        u = np.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0])
        b = B(-s * 1e-3 * u) * 1e4
        print(f"  32 G/cm, {s:4d} mm upstream on the atom beam: "
              f"|B| = {np.linalg.norm(b):7.2f} G")
