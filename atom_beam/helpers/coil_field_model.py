"""magpylib model of the lab's anti-Helmholtz MOT coils, plus the two
helpers that turn any B(metres) into something pylcp
can use.
"""

import os

import numpy as np
import magpylib as magpy
import scipy.constants as sp_const
from scipy.interpolate import RegularGridInterpolator

INNER_RADIUS_MM = 110
OUTER_RADIUS_MM = 158
COIL_STACK_HEIGHT_MM = 48
WIRE_SIDE_MM = 6
COOLING_BORE_MM = 4
GAP_BETWEEN_COILS_MM = 70

N_TOP_COILS = 4
LAYERS_PER_COIL = 2
TURNS_PER_LAYER = int(COIL_STACK_HEIGHT_MM / WIRE_SIDE_MM)
TURNS_PER_COIL = LAYERS_PER_COIL * TURNS_PER_LAYER
TOTAL_COILS = 2 * N_TOP_COILS

TOTAL_TURNS = TOTAL_COILS * TURNS_PER_COIL

N_RADIAL = N_TOP_COILS * LAYERS_PER_COIL

TURNS_PER_SIDE = N_TOP_COILS * TURNS_PER_COIL
N_TURNS_TOP = 0.9 * TURNS_PER_SIDE#float(os.environ.get("FT_COIL_TURNS_TOP", TURNS_PER_SIDE))
N_TURNS_BOTTOM = float(os.environ.get("FT_COIL_TURNS_BOTTOM", TURNS_PER_SIDE))


RADIUS_POSITIONS_MM = INNER_RADIUS_MM + WIRE_SIDE_MM / 2 + np.arange(N_RADIAL) * WIRE_SIDE_MM
HEIGHT_POSITIONS_MM = GAP_BETWEEN_COILS_MM / 2 + WIRE_SIDE_MM / 2 + np.arange(TURNS_PER_LAYER) * WIRE_SIDE_MM

REFERENCE_CURRENT_A = 1.0
REF_CURRENT_TOP_A = float(os.environ.get("FT_COIL_REF_A_TOP", REFERENCE_CURRENT_A))
REF_CURRENT_BOTTOM_A = 0.9176 #float(os.environ.get("FT_COIL_REF_A_BOTTOM", REFERENCE_CURRENT_A))



_MAGPY_MAJOR = int(magpy.__version__.split('.')[0])
_LENGTH_PER_M = 1.0 if _MAGPY_MAJOR >= 5 else 1e3
_TESLA_PER_UNIT = 1.0 if _MAGPY_MAJOR >= 5 else 1e-3

_cached_collection = None


def stack_rows(n_turns):
    n_rows = int(np.ceil(n_turns / N_RADIAL - 1e-9))
    heights = GAP_BETWEEN_COILS_MM / 2 + WIRE_SIDE_MM / 2 + np.arange(n_rows) * WIRE_SIDE_MM
    fill = np.clip(n_turns / N_RADIAL - np.arange(n_rows), 0.0, 1.0)
    return heights, fill


def _get_unit_collection():
    """Anti-Helmholtz Collection at 1 A per turn, built once."""
    global _cached_collection
    if _cached_collection is not None:
        return _cached_collection

    coils = magpy.Collection()
    for sign, n_turns, ref_a in ((+1, N_TURNS_TOP, REF_CURRENT_TOP_A),
                                 (-1, N_TURNS_BOTTOM, REF_CURRENT_BOTTOM_A)):
        for loop_height, fill in zip(*stack_rows(n_turns)):
            height = sign * loop_height * 1e-3 * _LENGTH_PER_M
            for loop_radius in RADIUS_POSITIONS_MM:
                diameter = 2 * loop_radius * 1e-3 * _LENGTH_PER_M
                coils.add(magpy.current.Circle(
                    current=sign * fill * ref_a,
                    diameter=diameter, position=(0, 0, height)))

    _cached_collection = coils
    return _cached_collection


def field_zero_z_mm(search_mm=40.0):
    """Height of the B = 0 point on the coil axis (0 for matched stacks)."""
    z_mm = np.linspace(-search_mm, search_mm, 801)
    xyz_m = np.column_stack([np.zeros_like(z_mm), np.zeros_like(z_mm), z_mm * 1e-3])
    B_z = oswald_coil_bfield(xyz_m, REFERENCE_CURRENT_A)[:, 2]
    i = int(np.argmax(np.diff(np.sign(B_z)) != 0))
    return float(np.interp(0.0, B_z[i:i + 2], z_mm[i:i + 2]))


def oswald_coil_bfield(xyz_meters, current_A):
    """Exact coil field in tesla at positions given in metres"""
    collection = _get_unit_collection()
    xyz_meters = np.atleast_2d(np.asarray(xyz_meters, dtype=float))

    B_unit = collection.getB(xyz_meters * _LENGTH_PER_M)
    B_unit = np.atleast_2d(B_unit)
    B_T = B_unit * _TESLA_PER_UNIT * (current_A / REFERENCE_CURRENT_A)

    return B_T[0] if B_T.shape[0] == 1 else B_T


def axial_gradient_G_per_cm(current_A, half_range_mm=5.0, n_points=41):
    """dBz/dz through the centre, linear-fit over +-half_range_mm.

    Reproduce magfield_test.py
    """
    z_mm = np.linspace(-half_range_mm, half_range_mm, n_points)
    xyz_m = np.column_stack([np.zeros_like(z_mm), np.zeros_like(z_mm), z_mm * 1e-3])
    B_T = oswald_coil_bfield(xyz_m, current_A)
    B_z_G = B_T[:, 2] * 1e4
    slope_G_per_mm, _ = np.polyfit(z_mm, B_z_G, 1)
    return slope_G_per_mm * 10.0


def radial_gradient_G_per_cm(current_A, half_range_mm=5.0, n_points=41):
    """dBx/dx through the centre. 
    
    
    Should come out near -1/2 the axial one, hopefully haha"""
    x_mm = np.linspace(-half_range_mm, half_range_mm, n_points)
    xyz_m = np.column_stack([x_mm * 1e-3, np.zeros_like(x_mm), np.zeros_like(x_mm)])
    B_T = oswald_coil_bfield(xyz_m, current_A)
    B_x_G = B_T[:, 0] * 1e4
    slope_G_per_mm, _ = np.polyfit(x_mm, B_x_G, 1)
    return slope_G_per_mm * 10.0


def grid_field_interpolator(points_m, B_tesla):
    R = np.round(np.asarray(points_m, dtype=float), decimals=9)
    B = np.asarray(B_tesla, dtype=float)

    xs, ys, zs = (np.unique(R[:, i]) for i in range(3))
    if len(xs) * len(ys) * len(zs) != R.shape[0]:
        raise ValueError("field samples are not on a regular Cartesian grid")

    grids = []
    for comp in range(3):
        g = np.full((len(xs), len(ys), len(zs)), np.nan)
        g[np.searchsorted(xs, R[:, 0]),
          np.searchsorted(ys, R[:, 1]),
          np.searchsorted(zs, R[:, 2])] = B[:, comp]
        if np.any(np.isnan(g)):
            raise ValueError("grid detected but some lattice points are missing")
        grids.append(RegularGridInterpolator((xs, ys, zs), g,
                                             bounds_error=False, fill_value=None))

    def B_interp(xyz):
        xyz = np.atleast_2d(xyz)
        out = np.column_stack([g(xyz) for g in grids])
        return out[0] if out.shape[0] == 1 else out

    return B_interp


def make_pylcp_magfield(B_interp_meters_tesla, x0, Gamma_SI, gJ_excited=1.0):
    """Wrap a B(metres) -> tesla function into pylcp's dimensionless form.

    pylcp wants B_bar(R_bar) such that the Zeeman term is m_F * B_bar, so
    B_bar = [gJ * mu_B / (hbar * Gamma)] * B_real(R_bar * x0).

    The returned function's signature must stay exactly (R, t) with no
    defaults , pylcp inspects it to decide how to call it.
    """
    conv = gJ_excited * sp_const.value('Bohr magneton') / (sp_const.hbar * Gamma_SI)

    def B_bar(R, t):
        R = np.asarray(R, dtype=float)
        return conv * np.asarray(B_interp_meters_tesla(R * x0))

    return B_bar

_cached_fast_interp_unit = None


def get_fast_bfield_interpolator(grid_extent_mm=25.0, n_per_axis=31):
    """Fast stand-in for oswald_coil_bfield, for use inside an ODE solver.

    useful for if I wanna do coarse sim
    """
    global _cached_fast_interp_unit
    if _cached_fast_interp_unit is not None:
        return _cached_fast_interp_unit

    axis_mm = np.linspace(-grid_extent_mm, grid_extent_mm, n_per_axis)
    X, Y, Z = np.meshgrid(axis_mm, axis_mm, axis_mm, indexing='ij')
    R_mm = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    R_m = R_mm * 1e-3

    B_T_unit = oswald_coil_bfield(R_m, REFERENCE_CURRENT_A)

    B_interp_unit = grid_field_interpolator(R_m, B_T_unit)

    def B_interp(xyz_meters, current_A):
        return B_interp_unit(xyz_meters) * (current_A / REFERENCE_CURRENT_A)

    _cached_fast_interp_unit = B_interp
    return B_interp

if __name__ == '__main__':
    print("=" * 70)
    print("Confirmation of linear current scaling.")
    print("=" * 70)
    for I in (50.0, 135.0, 300.0):
        grad = axial_gradient_G_per_cm(I)
        grad_r = radial_gradient_G_per_cm(I)
        print(f"  current = {I:6.1f} A  ->  axial gradient = {grad:7.2f} G/cm "
              f"(expect ~50.6 G/cm at 135 A, per magfield_test.py's own printout), "
              f"radial gradient = {grad_r:7.2f} G/cm (expect ~ -{grad/2:.2f} G/cm, "
              "i.e. -1/2 the axial one, for an ideal anti-Helmholtz pair)")

    g50 = axial_gradient_G_per_cm(50.0)
    g135 = axial_gradient_G_per_cm(135.0)
    ratio_expected = 50.0 / 135.0
    ratio_actual = g50 / g135
    print(f"\n  linearity check: gradient(50A)/gradient(135A) = {ratio_actual:.6f} "
          f"(expected {ratio_expected:.6f}) "
          f"{'OK' if abs(ratio_actual - ratio_expected) < 1e-9 else 'MISMATCH , investigate!'}")

    print(f"\n  turns: top {N_TURNS_TOP:g}, bottom {N_TURNS_BOTTOM:g} "
          f"(nominal {TURNS_PER_SIDE} each); current per {REFERENCE_CURRENT_A:g} A: "
          f"top {REF_CURRENT_TOP_A:g}, bottom {REF_CURRENT_BOTTOM_A:g}; field zero at z = "
          f"{field_zero_z_mm():+.2f} mm")
