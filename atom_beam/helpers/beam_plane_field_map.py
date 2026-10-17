"""
more plots, based on coil_field_model.py
"""

import os
import csv
import math

import numpy as np
import matplotlib

SHOW_PLOTS = bool(int(os.environ.get("SHOW_PLOTS", "0")))
if not SHOW_PLOTS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import scipy.constants as sp_const

from coil_field_model import (
    oswald_coil_bfield, REFERENCE_CURRENT_A,
    RADIUS_POSITIONS_MM, HEIGHT_POSITIONS_MM, WIRE_SIDE_MM,
)


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'beam_field_plots')

MAP_CURRENT_A = 98.7
LINE_CURRENTS_A = (100.0, 135.0, 170.0, 205.0)


BEAM_TILT_DEG = 25.5 # atom beam
NOZZLE_MM = 310.0
PAST_TRAP_MM = 20.0        

# Map extents
MAP_S_LO_MM = -PAST_TRAP_MM
MAP_S_HI_MM = NOZZLE_MM + 20.0
MAP_Z_HALF_MM = 110.0      # tall enough to contain both coil stacks (|z| = 38-80 mm)
MAP_DS_MM = 1.0
MAP_DZ_MM = 1.0

#
ZOOM_HALF_MM = 15.0       
ZOOM_D_MM = 0.25
GRAD_FIT_HALF_MM = 5.0     
# path sampling
N_PATH = 1651
TRANSVERSE_OFFSETS_MM = (2.0, 5.0, 10.0)   

# sublevel-projection figure (10)
SUBLEVEL_S_MM = (1.0, 5.0, 20.0)
SUBLEVEL_OFFSET_MAX_MM = 5.0
SUBLEVEL_PATH_OFFSETS_MM = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)

C_SIGMA_P, C_PI, C_SIGMA_M = '#E24A33', '#8C8C8C', '#348ABD'

# Chunking to cap memory so it runs on something reasonable
FIELD_CHUNK = 4000


WINDING_MASK_MM = 0.75 * WIRE_SIDE_MM

CONTOUR_LEVELS_G = (0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000)

# Floor for log scale
MAP_FLOOR_G = 0.5

# YB constants--
YB174_TRANSITION_FREQ_HZ = 751.5265174e12
YB174_LINEWIDTH_HZ = 30.2e6      
YB174_GJ_EXCITED = 1.0           
REF_OVEN_C = 425.0               

WAVELENGTH_M = sp_const.c / YB174_TRANSITION_FREQ_HZ
K_SI = 2 * np.pi / WAVELENGTH_M
GAMMA_SI = 2 * np.pi * YB174_LINEWIDTH_HZ

ZEEMAN_GAMMA_PER_TESLA = (YB174_GJ_EXCITED * sp_const.value('Bohr magneton')
                          / (sp_const.hbar * GAMMA_SI))
V_PER_GAMMA = GAMMA_SI / K_SI    # Doppler velocity worth one Gamma

TILT = math.radians(BEAM_TILT_DEG)
U_HAT = np.array([math.cos(TILT), math.sin(TILT), 0.0])   # atom travel direction
E_H = np.array([-math.sin(TILT), math.cos(TILT), 0.0])    # horizontal, across the beam
E_V = np.array([0.0, 0.0, 1.0])                           # vertical, across the beam
S_HAT = -U_HAT                 # +s direction = upstream = slowing-beam k_hat
K_SLOW = S_HAT

X_HAT = np.array([1.0, 0.0, 0.0])
Y_HAT = np.array([0.0, 1.0, 0.0])
Z_HAT = np.array([0.0, 0.0, 1.0])

CURRENT_COLORS = ('#4C72B0', '#DD8452', '#55A868', '#C44E52')
C_BX, C_BY, C_BZ, C_BMAG = '#E24A33', '#348ABD', '#988ED5', '#111111'
PATH_C = '#00A6FF'
NOZZLE_C = '#FF9F1C'
PAR_LABEL = r'$B_\parallel$ (along beam)'


def lab_xyz_mm(s_mm, z_mm=0.0, h_mm=0.0):
    """
    s_mm : distance still to go to the trap centre, along the beam.
    z_mm : vertical offset (gravity is -z).
    h_mm : horizontal offset across the beam, along E_H.
    """
    s = np.asarray(s_mm, dtype=float)
    z = np.broadcast_to(np.asarray(z_mm, dtype=float), s.shape)
    h = np.broadcast_to(np.asarray(h_mm, dtype=float), s.shape)
    return (s[..., None] * S_HAT[None, :]
            + z[..., None] * E_V[None, :]
            + h[..., None] * E_H[None, :])


def field_G_per_amp(points_mm):
    pts = np.asarray(points_mm, dtype=float).reshape(-1, 3)
    out = np.empty_like(pts)
    for lo in range(0, pts.shape[0], FIELD_CHUNK):
        hi = min(lo + FIELD_CHUNK, pts.shape[0])
        out[lo:hi] = oswald_coil_bfield(pts[lo:hi] * 1e-3, REFERENCE_CURRENT_A)
    return out * 1e4 / REFERENCE_CURRENT_A          # tesla -> gauss, per amp


def winding_mask(S, Z, radius_mm=WINDING_MASK_MM):
    """ Don't draw field in windings"""
    mask = np.zeros(S.shape, dtype=bool)
    for r in RADIUS_POSITIONS_MM:
        for h in HEIGHT_POSITIONS_MM:
            for sr in (+r, -r):
                for sh in (+h, -h):
                    mask |= ((S - sr) ** 2 + (Z - sh) ** 2) < radius_mm ** 2
    return mask


def beam_plane_grid(s_lo=MAP_S_LO_MM, s_hi=MAP_S_HI_MM,
                    z_half=MAP_Z_HALF_MM, ds=MAP_DS_MM, dz=MAP_DZ_MM):
    n_s = int(round((s_hi - s_lo) / ds)) + 1
    n_z = int(round(2 * z_half / dz)) + 1
    s = np.linspace(s_lo, s_hi, n_s)
    z = np.linspace(-z_half, z_half, n_z)
    S, Z = np.meshgrid(s, z, indexing='ij')
    B = field_G_per_amp(lab_xyz_mm(S.ravel(), Z.ravel())).reshape(S.shape + (3,))
    B[winding_mask(S, Z)] = np.nan
    return s, z, S, Z, B


def in_plane_components(B):
    return B @ S_HAT, B[..., 2]


def beam_frame_components(B):
    return B @ U_HAT, B @ E_H, B @ E_V


def polarization_fractions(B, k_hat=None):
    """
    Intensity fractions of a nominally sigma-plus beam, decomposed on the
    LOCAL quantisation axis B_hat. 
    """
    k_hat = K_SLOW if k_hat is None else k_hat
    theta = np.radians(angle_to_deg(B, k_hat))
    c = np.cos(theta)
    return (((1 + c) / 2) ** 2, np.sin(theta) ** 2 / 2, ((1 - c) / 2) ** 2,
            np.degrees(theta))


def unit_vector(B, floor=1e-12):
    """B / |B|"""
    n = np.linalg.norm(B, axis=-1, keepdims=True)
    return B / np.maximum(n, floor)


def angle_to_deg(B, axis):
    c = np.clip(unit_vector(B) @ axis, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def most_probable_flux_speed(temp_c=REF_OVEN_C):
    m = 173.9388621 * sp_const.value('atomic mass constant')
    return math.sqrt(2 * sp_const.k * (temp_c + 273.15) / m)



# Shared plot stuff
def style_path_axis(ax, s_lo, s_hi, show_nozzle=True):
    """
    Common treatment for every axis whose abscissa is s: invert it so the
    nozzle is on the left and the trap on the right, mark the trap centre,
    and mark the nozzle face if it is inside the plotted range.
    """
    ax.set_xlim(s_lo, s_hi)
    ax.invert_xaxis()
    ax.axvline(0.0, color='lime', lw=1.2, ls='-', alpha=0.85, zorder=4)
    if show_nozzle and s_lo <= NOZZLE_MM <= s_hi:
        ax.axvline(NOZZLE_MM, color=NOZZLE_C, lw=1.4, ls=':', alpha=0.95, zorder=4)


def draw_windings(ax, s_lim, z_lim, marker_size=3.0):
    top = bot = None
    for r in RADIUS_POSITIONS_MM:
        for h in HEIGHT_POSITIONS_MM:
            for sr in (+r, -r):
                if not (s_lim[0] <= sr <= s_lim[1]):
                    continue
                if abs(h) <= z_lim:
                    top = ax.plot([sr], [+h], 'o', color=C_BX, ms=marker_size,
                                  mec='k', mew=0.3, alpha=0.95, zorder=6)[0]
                    bot = ax.plot([sr], [-h], 'o', color=C_BY, ms=marker_size,
                                  mec='k', mew=0.3, alpha=0.95, zorder=6)[0]
    return top, bot


def draw_atom_path(ax, s_hi):
    line = ax.plot([min(NOZZLE_MM, s_hi), 0.0], [0.0, 0.0], color=PATH_C,
                   lw=1.2, ls='--', alpha=0.75, zorder=7)[0]
    if NOZZLE_MM <= s_hi:
        ax.plot([NOZZLE_MM], [0.0], 's', color=NOZZLE_C, ms=7, zorder=8)
    
    return line


def signed_map(ax, S, Z, C, label, vmax=None, linthresh=1.0, cmap='RdBu_r',
               colorbar=True):
    finite = C[np.isfinite(C)]
    if vmax is None:
        vmax = np.percentile(np.abs(finite), 99.5)
    vmax = max(vmax, 10 * linthresh)
    norm = mcolors.SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax, base=10)
    pcm = ax.pcolormesh(S, Z, np.ma.masked_invalid(C), cmap=cmap, norm=norm,
                        shading='auto', rasterized=True)
    if colorbar:
        cb = ax.figure.colorbar(pcm, ax=ax, pad=0.02)
        cb.set_label(label)
    return pcm

# Figure 1
def plot_components(S, Z, B_per_A, current_A=MAP_CURRENT_A, out_dir=OUTPUT_DIR):
    B = B_per_A * current_A
    Bmag = np.linalg.norm(B, axis=-1)
    s_lim = (S.min(), S.max())
    z_lim = float(np.abs(Z).max())

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 8.6), constrained_layout=True)
    panels = [
        (axes[0, 0], B[..., 0], '$B_x$ [G]',
         'lab $x$ (along x MOT beams)'),
        (axes[0, 1], B[..., 1], '$B_y$ [G]',
         'lab $y$ ($= B_x\\,\\tan 25.5^\\circ$, )'),
        (axes[1, 0], B[..., 2], '$B_z$ [G]', 'lab $z$ (coil axis)'),
    ]
    
    comp_vmax = float(np.percentile(np.abs(B[np.isfinite(B)]), 99.5))
    for ax, C, cblabel, subtitle in panels:
        signed_map(ax, S, Z, C, cblabel, vmax=comp_vmax)
        ax.set_title(subtitle, fontsize=9)

    ax = axes[1, 1]
    finite = Bmag[np.isfinite(Bmag)]
    vmax = np.percentile(finite, 99.5)
    floor = MAP_FLOOR_G
    pcm = ax.pcolormesh(S, Z, np.ma.masked_invalid(np.clip(Bmag, floor, None)),
                        cmap='plasma', norm=mcolors.LogNorm(vmin=floor, vmax=vmax),
                        shading='auto', rasterized=True)
    cb = fig.colorbar(pcm, ax=ax, pad=0.02,
                      ticks=mticker.LogLocator(base=10, subs=(1.0,)))
    cb.set_label('$|B|$ [G, log]')
    lv = [v for v in CONTOUR_LEVELS_G if floor < v < vmax]
    if lv:
        cs = ax.contour(S, Z, Bmag, levels=lv, colors='white', linewidths=0.8,
                        alpha=0.8)
        ax.clabel(cs, inline=True, fontsize=6, fmt='%g')
    ax.set_title('$|B|$', fontsize=9)

    for ax in axes.ravel():
        draw_windings(ax, s_lim, z_lim)
        draw_atom_path(ax, s_lim[1])
        ax.plot(0, 0, '+', color='lime', ms=12, mew=2, zorder=9)
        ax.set_aspect('equal')
        ax.set_xlabel('$s$ = distance to trap centre along the atom beam [mm]')
        ax.set_ylabel('$z$ [mm]')
        ax.set_xlim(*s_lim)
        ax.invert_xaxis()
        ax.set_ylim(-z_lim, z_lim)

    fig.suptitle(
        f"Current at {current_A:.0f} A",
        fontsize=11)
    return _save(fig, out_dir, '01_components_beam_plane.png')


# fig 2
def plot_magnitude_map(s, z, S, Z, B_per_A, current_A=MAP_CURRENT_A,
                       out_dir=OUTPUT_DIR):
    B = B_per_A * current_A
    Bmag = np.linalg.norm(B, axis=-1)
    Bs, Bz = in_plane_components(B)
    s_lim = (S.min(), S.max())
    z_lim = float(np.abs(Z).max())

    fig, ax = plt.subplots(figsize=(15.0, 7.2), constrained_layout=True)
    finite = Bmag[np.isfinite(Bmag)]
    vmax = np.percentile(finite, 99.5)
    floor = MAP_FLOOR_G
    pcm = ax.pcolormesh(S, Z, np.ma.masked_invalid(np.clip(Bmag, floor, None)),
                        cmap='plasma', norm=mcolors.LogNorm(vmin=floor, vmax=vmax),
                        shading='auto', rasterized=True)
    cb = fig.colorbar(pcm, ax=ax, pad=0.015,
                      ticks=mticker.LogLocator(base=10, subs=(1.0,)))
    cb.ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)))
    cb.set_label('$|B|$ [G, log scale]')

    lv = [v for v in CONTOUR_LEVELS_G if floor < v < vmax]
    if lv:
        cs = ax.contour(S, Z, Bmag, levels=lv, colors='white', linewidths=1.2,
                        alpha=0.85)
        ax.clabel(cs, inline=True, fontsize=7, fmt='%g G')

    Bs_f = np.nan_to_num(Bs)
    Bz_f = np.nan_to_num(Bz)
    ax.streamplot(s, z, Bs_f.T, Bz_f.T, color='#BFE3F2', density=(2.2, 1.0),
                  linewidth=0.55, arrowsize=0.8)

    top, bot = draw_windings(ax, s_lim, z_lim, marker_size=3.5)
    path = draw_atom_path(ax, s_lim[1])
    centre = ax.plot(0, 0, '+', color='lime', ms=16, mew=2.5, zorder=9)[0]
    handles = [h for h in (centre, path, top, bot) if h is not None]
    labels = ['trap centre (field zero)', 'atom path ($z=0$)',
              'windings, top stack ($+I$)', 'windings, bottom stack ($-I$)']
    ax.legend(handles, labels[:len(handles)], loc='upper left', fontsize=8,
              framealpha=0.9)

    ax.set_aspect('equal')
    ax.set_xlabel('$s$ = distance to trap centre along the atom beam [mm]')
    ax.set_ylabel('$z$ [mm]')
    ax.set_xlim(*s_lim)
    ax.invert_xaxis()
    ax.set_ylim(-z_lim, z_lim)
    ax.set_title(f"$|B|$ at {current_A:.0f} A, with field lines ",
                fontsize=11)
    return _save(fig, out_dir, '02_magnitude_map.png')


# Figure 3
def plot_direction_map(S, Z, B_per_A, out_dir=OUTPUT_DIR, quiver_step=14):
    Bs, Bz = in_plane_components(B_per_A)
    phi = np.degrees(np.arctan2(Bz, Bs))
    phi[~np.isfinite(Bs)] = np.nan
    s_lim = (S.min(), S.max())
    z_lim = float(np.abs(Z).max())

    fig, ax = plt.subplots(figsize=(15.0, 7.2), constrained_layout=True)
    pcm = ax.pcolormesh(S, Z, np.ma.masked_invalid(phi), cmap='twilight_shifted',
                        vmin=-180, vmax=180, shading='auto', rasterized=True)
    cb = fig.colorbar(pcm, ax=ax, pad=0.015, ticks=np.arange(-180, 181, 45))
    cb.set_label('direction of $B$ in the beam plane,  '
                 r'$\varphi = \arctan2(B_z,\, B_s)$  [deg]')

    # Unit arrows on top: same information, but readable locally.
    q = quiver_step
    U = unit_vector(np.stack([np.nan_to_num(Bs), np.nan_to_num(Bz)], axis=-1))
    ax.quiver(S[::q, ::q], Z[::q, ::q], U[::q, ::q, 0], U[::q, ::q, 1],
              color='k', alpha=0.55, pivot='mid', scale=38, width=0.0016,
              headwidth=4, headlength=5, zorder=5)

    draw_windings(ax, s_lim, z_lim, marker_size=3.5)
    draw_atom_path(ax, s_lim[1])
    ax.plot(0, 0, '+', color='lime', ms=16, mew=2.5, zorder=9)

    ax.set_aspect('equal')
    ax.set_xlabel('$s$ = distance to trap centre along the atom beam [mm]')
    ax.set_ylabel('$z$ [mm]')
    ax.set_xlim(*s_lim)
    ax.invert_xaxis()
    ax.set_ylim(-z_lim, z_lim)
    ax.set_title("Direction of $B$ at every point", fontsize=11)
    return _save(fig, out_dir, '03_field_direction_map.png')


def sample_path(n=N_PATH, z_mm=0.0, h_mm=0.0):
    s = np.linspace(-PAST_TRAP_MM, NOZZLE_MM, n)
    B = field_G_per_amp(lab_xyz_mm(s, z_mm, h_mm))
    return s, B


# Figure 4
def plot_along_path(s, B_per_A, out_dir=OUTPUT_DIR):
    B = B_per_A * MAP_CURRENT_A
    Bmag = np.linalg.norm(B, axis=-1)
    Bs, Bz = in_plane_components(B)

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 11.0), sharex=True,
                             constrained_layout=True)

    ax = axes[0]
    ax.plot(s, B[:, 0], color=C_BX, lw=1.8, label='$B_x$')
    ax.plot(s, B[:, 1], color=C_BY, lw=1.8, label='$B_y$')
    ax.plot(s, B[:, 2], color=C_BZ, lw=1.8, label='$B_z$')
    ax.plot(s, Bs, color='#666666', lw=1.2, ls='--',
            label=r'$B_\parallel$ (along $+s$, i.e. the beam axis)')
    ax.plot(s, Bmag, color=C_BMAG, lw=1.2, ls=':', label='$|B|$')
    ax.axhline(0, color='k', lw=0.6, alpha=0.4)
    ax.set_yscale('symlog', linthresh=0.1)
    ax.set_ylim(-1.5e3, 1.5e3)
    ax.set_ylabel(f'field at {MAP_CURRENT_A:.0f} A [G]  (symlog)')
    ax.legend(fontsize=8, ncol=3, loc='lower center', framealpha=0.9)
    ax.set_title(f'Field components along  ($z=0$), at {MAP_CURRENT_A:.0f} A.  ',
                 fontsize=10)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for I, c in zip(LINE_CURRENTS_A, CURRENT_COLORS):
        ax.plot(s, Bmag / MAP_CURRENT_A * I, color=c, lw=1.8, label=f'{I:.0f} A')
    ax.set_yscale('log')
    ax.set_ylim(1e-2, None)
    ax.set_ylabel('$|B|$ [G, log]')
    ax.legend(fontsize=8, title='coil current', title_fontsize=8,
              loc='lower left')
    ax.set_title('$|B|$ along the path, four currents',
                 fontsize=10)
    ax.grid(alpha=0.25, which='both')

    ax = axes[2]
    dBds = np.gradient(Bmag, s)
    ax.plot(s, np.abs(dBds), color='#8B2BE2', lw=1.6)
    ax.set_yscale('log')
    ax.set_ylim(1e-3, None)          # same reason as the panel above
    ax.set_ylabel(r'$|\,\mathrm{d}|B|/\mathrm{d}s\,|$ '
                  + f'at {MAP_CURRENT_A:.0f} A [G/mm, log]')
    ax.set_xlabel('$s$ = distance to trap centre along the atom beam [mm]   '
                  '(atoms travel left to right)')
    ax.set_title('How fast the field changes along the flight path', fontsize=10)
    ax.grid(alpha=0.25, which='both')

    for ax in axes:
        style_path_axis(ax, s.min(), s.max())
    # _annotate_landmarks(axes[0])
    return _save(fig, out_dir, '04_field_along_atom_path.png')


def _annotate_landmarks(ax):
    tr = ax.get_xaxis_transform()
    ax.text(NOZZLE_MM, 0.96, ' nozzle face ', transform=tr, color=NOZZLE_C,
            fontsize=8, ha='center', va='top', zorder=10,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=NOZZLE_C, alpha=0.9))
    ax.text(0.0, 0.96, ' trap ', transform=tr, color='green', fontsize=8,
            ha='center', va='top', zorder=10,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='green', alpha=0.9))

# Figure 5
def plot_angles(s, B_per_A, out_dir=OUTPUT_DIR):
    Bs, Bz = in_plane_components(B_per_A)
    phi = np.degrees(np.arctan2(Bz, Bs))

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.4), sharex=True,
                             constrained_layout=True)

    ax = axes[0]
    ax.plot(s, phi, color='#8B2BE2', lw=1.8)
    ax.axhline(0, color='k', lw=0.6, alpha=0.4)
    ax.axhline(180, color='k', lw=0.6, alpha=0.2)
    ax.axhline(-180, color='k', lw=0.6, alpha=0.2)
    ax.set_ylim(-190, 190)
    ax.set_yticks(np.arange(-180, 181, 45))
    ax.set_ylabel(r'$\varphi = \arctan2(B_z, B_s)$ [deg]')
    ax.set_title('Direction of $B$ along the atom path.  $0^\\circ$ = upstream toward '
                 'the nozzle, $\\pm180^\\circ$ = downstream toward the trap, '
                 '$+90^\\circ$ = straight up.', fontsize=10)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for axis_vec, name, c in ((K_SLOW, r'slowing beam $\hat{k}$ (= $+\hat{s}$)', '#8B2BE2'),
                              (X_HAT, r'$x$ MOT pair', C_BX),
                              (Y_HAT, r'$y$ MOT pair', C_BY),
                              (Z_HAT, r'$z$ MOT pair', C_BZ)):
        ax.plot(s, angle_to_deg(B_per_A, axis_vec), color=c, lw=1.8, label=name)
    ax.axhline(90, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.text(0.99, 91, r'90$^\circ$: pure $\pi$ for that beam',
            transform=ax.get_yaxis_transform(),
            ha='right', va='bottom', fontsize=7, color='k', alpha=0.7)
    ax.set_ylim(0, 180)
    ax.set_yticks(np.arange(0, 181, 30))
    ax.set_ylabel(r'angle between $\hat{B}$ and the beam axis [deg]')
    ax.set_xlabel('$s$ = distance to trap centre along the atom beam [mm]   '
                  '(atoms travel left to right)')
    ax.legend(fontsize=8, ncol=2)
    ax.set_title('Angle of the local quantisation axis to each laser axis ', fontsize=10)
    ax.grid(alpha=0.25)

    for ax in axes:
        style_path_axis(ax, s.min(), s.max())
    return _save(fig, out_dir, '05_field_angles_along_path.png')


# Figure 6
def plot_polarization_and_zeeman(s, B_per_A, out_dir=OUTPUT_DIR):
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.8), sharex=True,
                             constrained_layout=True)

    ax = axes[0]
    for lbl, z_off, ls in (('on axis ($z=0$)', 0.0, '-'),
                           ('$z=+5$ mm', 5.0, '--')):
        _, B_off = sample_path(n=len(s), z_mm=z_off)
        th = np.radians(angle_to_deg(B_off, K_SLOW))
        cc = np.cos(th)
        ax.plot(s, ((1 + cc) / 2) ** 2, color='#E24A33', ls=ls, lw=1.8,
                label=rf'$\sigma^+$, {lbl}')
        ax.plot(s, np.sin(th) ** 2 / 2, color='#7F7F7F', ls=ls, lw=1.8,
                label=rf'$\pi$, {lbl}')
        ax.plot(s, ((1 - cc) / 2) ** 2, color='#348ABD', ls=ls, lw=1.8,
                label=rf'$\sigma^-$, {lbl}')
    ax.set_ylim(-0.05, 1.08)
    ax.set_ylabel('intensity fraction')
    ax.legend(fontsize=7, ncol=3, loc='center left', framealpha=0.9)
    ax.grid(alpha=0.25)
    
    ax.set_title('Decomposed on the local quantisation axis '
                 'of a nominally $\\sigma^+$ beam', fontsize=10)

    ax = axes[1]
    Bmag_per_A = np.linalg.norm(B_per_A, axis=-1)
    for I, col in zip(LINE_CURRENTS_A, CURRENT_COLORS):
        shift_gamma = Bmag_per_A * I * 1e-4 * ZEEMAN_GAMMA_PER_TESLA
        ax.plot(s, shift_gamma, color=col, lw=1.8, label=f'{I:.0f} A')
    ax.set_yscale('log')
    # Clamp before the twin axis is built
    ax.set_ylim(1e-2, None)
    ax.set_ylabel(r'$|$Zeeman shift$|$ of $1P_1$, $m_J=\pm1$ [$\Gamma$]')
    ax.set_xlabel('$s$ = distance to trap centre along the atom beam [mm]   '
                  '(atoms travel left $\\rightarrow$ right)')
    ax.legend(fontsize=8, title='coil current', title_fontsize=8)
    ax.grid(alpha=0.25, which='both')

    ax2 = ax.twinx()
    ax2.set_yscale('log')
    lo, hi = ax.get_ylim()
    ax2.set_ylim(lo * V_PER_GAMMA, hi * V_PER_GAMMA)
    ax2.set_ylabel('equivalent Doppler speed [m/s]')
    ax.set_title('Zeeman shift along atom path', fontsize=10)

    for ax_ in axes:
        style_path_axis(ax_, s.min(), s.max())
    return _save(fig, out_dir, '06_polarization_and_zeeman.png')


# Figure 7
def plot_trap_zoom(current_A=MAP_CURRENT_A, out_dir=OUTPUT_DIR):
    n = int(round(2 * ZOOM_HALF_MM / ZOOM_D_MM)) + 1
    ax_mm = np.linspace(-ZOOM_HALF_MM, ZOOM_HALF_MM, n)
    S, Z = np.meshgrid(ax_mm, ax_mm, indexing='ij')
    B = field_G_per_amp(lab_xyz_mm(S.ravel(), Z.ravel())).reshape(S.shape + (3,))
    B = B * current_A
    Bs, Bz = in_plane_components(B)
    Bmag = np.linalg.norm(B, axis=-1)

    fig = plt.figure(figsize=(14.0, 8.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    for k, (C, name) in enumerate(((B[..., 0], '$B_x$'), (B[..., 1], '$B_y$'),
                                   (B[..., 2], '$B_z$'))):
        ax = fig.add_subplot(gs[0, k])
        vmax = float(np.nanmax(np.abs(C)))
        pcm = ax.pcolormesh(S, Z, C, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                            shading='auto', rasterized=True)
        fig.colorbar(pcm, ax=ax, pad=0.02).set_label(f'{name} [G]')
        ax.contour(S, Z, C, levels=9, colors='k', linewidths=0.4, alpha=0.4)
        ax.set_title(f'{name} near the trap', fontsize=9)
        ax.set_aspect('equal')
        ax.set_xlabel('$s$ [mm]')
        ax.set_ylabel('$z$ [mm]')
        ax.invert_xaxis()
        ax.plot(0, 0, '+', color='lime', ms=12, mew=2)

    ax = fig.add_subplot(gs[1, 0])
    pcm = ax.pcolormesh(S, Z, Bmag, cmap='plasma', shading='auto', rasterized=True)
    fig.colorbar(pcm, ax=ax, pad=0.02).set_label('$|B|$ [G]')
    cs = ax.contour(S, Z, Bmag, levels=(1, 2, 5, 10, 20, 50), colors='white',
                    linewidths=0.9)
    ax.clabel(cs, inline=True, fontsize=6, fmt='%g G')
    ax.streamplot(ax_mm, ax_mm, Bs.T, Bz.T, color='#BFE3F2', density=1.0,
                  linewidth=0.5, arrowsize=0.7)
    ax.plot(0, 0, '+', color='lime', ms=14, mew=2.2)
    ax.axhline(0, color=PATH_C, lw=1.6, ls='--')
    ax.set_title('$|B|$ + field lines (atom path dashed)', fontsize=9)
    ax.set_aspect('equal')
    ax.set_xlabel('$s$ [mm]')
    ax.set_ylabel('$z$ [mm]')
    ax.invert_xaxis()



    fit = np.abs(ax_mm) <= GRAD_FIT_HALF_MM
    mid = n // 2

    ax = fig.add_subplot(gs[1, 1])
    grads = {}
    for C, name, c in ((Bs[:, mid], PAR_LABEL, '#666666'),
                       (B[:, mid, 0], '$B_x$', C_BX),
                       (B[:, mid, 1], '$B_y$', C_BY),
                       (B[:, mid, 2], '$B_z$', C_BZ)):
        ax.plot(ax_mm, C, color=c, lw=1.8, label=name)
        grads[name] = float(np.polyfit(ax_mm[fit], C[fit], 1)[0]) * 10.0  # G/mm -> G/cm
    ax.axvline(0, color='lime', lw=1.2)
    ax.set_xlabel('$s$ along the atom path, $z=0$ [mm]')
    ax.set_ylabel(f'field at {current_A:.0f} A [G]')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    ax.invert_xaxis()
    ax.set_title('cut along the beam ($z = 0$)', fontsize=9)

    ax = fig.add_subplot(gs[1, 2])
    grads_z = {}
    for C, name, c in ((Bs[mid, :], PAR_LABEL, '#666666'),
                       (B[mid, :, 0], '$B_x$', C_BX),
                       (B[mid, :, 1], '$B_y$', C_BY),
                       (B[mid, :, 2], '$B_z$', C_BZ)):
        ax.plot(ax_mm, C, color=c, lw=1.8, label=name)
        grads_z[name] = float(np.polyfit(ax_mm[fit], C[fit], 1)[0]) * 10.0
    ax.axvline(0, color='lime', lw=1.2)
    ax.set_xlabel('$z$ through the trap centre [mm]')
    ax.set_ylabel(f'field at {current_A:.0f} A [G]')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    ax.set_title('vertical cut ($s = 0$)', fontsize=9)



    
    g_par = grads[PAR_LABEL]
    g_axial = grads_z['$B_z$']
    txt = (f"fitted over $\\pm${GRAD_FIT_HALF_MM:.0f} mm at {current_A:.0f} A:   "
           f"d$B_\\parallel$/d$s$ = {g_par:+.2f} G/cm  (radial)   |   "
           f"d$B_z$/d$z$ = {g_axial:+.2f} G/cm  (axial)   |   "
           f"ratio = {g_par / g_axial:+.3f}  (ideal anti-Helmholtz: $-1/2$)")
    fig.suptitle('The trap region: $\\pm$10-15 mm' + txt,
                 fontsize=10)
    _save(fig, out_dir, '07_trap_region_zoom.png')
    return grads, grads_z


# Fig 8
def plot_across_beam_width(out_dir=OUTPUT_DIR):
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4), sharex=True,
                             constrained_layout=True)
    offsets = (0.0,) + TRANSVERSE_OFFSETS_MM
    cmap = plt.get_cmap('viridis')
    s = None

    for col, (kind, label) in enumerate((('z', 'vertical offset $\\Delta z$'),
                                         ('h', 'horizontal offset $\\Delta h$'))):
        for k, d in enumerate(offsets):
            kw = {'z_mm': d} if kind == 'z' else {'h_mm': d}
            s, B = sample_path(n=801, **kw)
            Bmag = np.linalg.norm(B, axis=-1) * MAP_CURRENT_A
            col_c = cmap(k / max(1, len(offsets) - 1))
            axes[0, col].plot(s, Bmag, color=col_c, lw=1.6,
                              label=f'{d:+.0f} mm' if d else 'on axis')
            axes[1, col].plot(s, angle_to_deg(B, K_SLOW), color=col_c, lw=1.6)
        axes[0, col].set_yscale('log')
        axes[0, col].set_ylabel(f'$|B|$ at {MAP_CURRENT_A:.0f} A [G, log]')
        axes[0, col].legend(fontsize=8, title=label, title_fontsize=8, ncol=2)
        axes[0, col].set_title(label, fontsize=10)
        axes[1, col].set_ylabel(r'angle $\hat{B}$ to slowing beam [deg]')
        axes[1, col].set_ylim(0, 180)
        axes[1, col].set_yticks(np.arange(0, 181, 30))
        axes[1, col].set_xlabel('$s$ = distance to trap centre [mm]')
        for r in (0, 1):
            axes[r, col].grid(alpha=0.25, which='both')
            style_path_axis(axes[r, col], s.min(), s.max())

    fig.suptitle('Field direction along atom path\n'
                 'Left: displaced vertically.  Right: displaced horizontally, '
                 , fontsize=11)
    return _save(fig, out_dir, '08_across_beam_width.png')


# Fig 9
def plot_beam_frame(S, Z, B_per_A, s_path, B_path, current_A=MAP_CURRENT_A,
                    out_dir=OUTPUT_DIR):
    B = B_per_A * current_A
    Bpar, Bh, Bv = beam_frame_components(B)
    s_lim = (S.min(), S.max())
    z_lim = float(np.abs(Z).max())



   
    fig = plt.figure(figsize=(18.0, 8.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0])

    comp_vmax = float(np.percentile(np.abs(B[np.isfinite(B)]), 99.5))
    panels = ((Bpar, r'$B_\parallel = B\cdot\hat{u}$ [G]',
               'along the beam'),
              (Bh, r'$B_h = B\cdot\hat{e}_h$ [G]',
               'horizontally across the beam'),
              (Bv, r'$B_v = B\cdot\hat{z}$ [G]',
               'vertically across the beam'))
    map_axes = []
    pcm = None
    for k, (C, cblabel, subtitle) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        map_axes.append(ax)
        pcm = signed_map(ax, S, Z, C, cblabel, vmax=comp_vmax, colorbar=False)
        draw_windings(ax, s_lim, z_lim)
        draw_atom_path(ax, s_lim[1])
        ax.plot(0, 0, '+', color='lime', ms=12, mew=2, zorder=9)
        ax.set_title(subtitle, fontsize=9)
        ax.set_aspect('equal')
        ax.set_xlabel('$s$ to trap centre [mm]')
        ax.set_ylabel('$z$ [mm]')
        ax.set_xlim(*s_lim)
        ax.invert_xaxis()
        ax.set_ylim(-z_lim, z_lim)

    cb = fig.colorbar(pcm, ax=map_axes, location='right', shrink=0.72, pad=0.015)
    cb.set_label('field component [G]')

    ax = fig.add_subplot(gs[1, :])
    Bp, Bhz, Bvz = beam_frame_components(B_path * current_A)
    Bmag = np.linalg.norm(B_path, axis=-1) * current_A
    ax.plot(s_path, Bp, color='#8B2BE2', lw=2.0,
            label=r'$B_r$ (along the direction of travel)')
    ax.plot(s_path, Bhz, color=C_BY, lw=2.0,
            label=r'$B_h$ (horizontal)')
    ax.plot(s_path, Bvz, color=C_BZ, lw=2.0,
            label=r'$B_v$ (vertical, across) ')
    ax.plot(s_path, Bmag, color='k', lw=1.1, ls=':', label='$|B|$')
    ax.axhline(0, color='k', lw=0.6, alpha=0.4)
    ax.set_yscale('symlog', linthresh=0.1)
    ax.set_ylim(-1.5e3, 1.5e3)
    ax.set_ylabel(f'field at {current_A:.0f} A [G]  (symlog)')
    ax.set_xlabel('$s$ = distance to trap centre along the atom beam [mm]   '
                  '(atoms travel left to right)')
    ax.legend(fontsize=8, ncol=4, loc='lower center', framealpha=0.9)
    ax.grid(alpha=0.25)
    style_path_axis(ax, s_path.min(), s_path.max())
    # _annotate_landmarks(ax)

    frac = float(np.max(np.abs(Bhz) / np.maximum(Bmag, 1e-30)))
    

    fig.suptitle('Field experienced by atom beam at '
                 f'{current_A:.0f} A\n', fontsize=11)
    return _save(fig, out_dir, '09_beam_frame_components.png')


# Fig 10
def plot_sublevel_projection(out_dir=OUTPUT_DIR, current_A=MAP_CURRENT_A):
    
    dz = np.linspace(0.0, SUBLEVEL_OFFSET_MAX_MM, 241)
    fig = plt.figure(figsize=(15.5, 9.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    for k, s0 in enumerate(SUBLEVEL_S_MM):
        ax = fig.add_subplot(gs[0, k])
        B = field_G_per_amp(lab_xyz_mm(np.full_like(dz, s0), dz))
        f_sp, f_pi, f_sm, _ = polarization_fractions(B)
        ax.stackplot(dz, f_sm, f_pi, f_sp,
                     labels=[r'$\sigma^-$  ($m_J=-1$)', r'$\pi$  ($m_J=0$)',
                             r'$\sigma^+$  ($m_J=+1$)'],
                     colors=[C_SIGMA_M, C_PI, C_SIGMA_P], alpha=0.92)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, SUBLEVEL_OFFSET_MAX_MM)
        ax.set_xlabel(r'vertical offset $\Delta z$ from the beam axis [mm]')
        ax.set_ylabel('fraction of the slowing beam intensity')
        ax.set_title(f'at $s$ = {s0:.0f} mm from the trap', fontsize=10)
        if k == 0:
            ax.legend(fontsize=8, loc='center left', framealpha=0.92)




    ax = fig.add_subplot(gs[1, 0])
    cmap = plt.get_cmap('viridis')
    s_list = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
    for k, s0 in enumerate(s_list):
        B = field_G_per_amp(lab_xyz_mm(np.full_like(dz, s0), dz))
        ax.plot(dz, angle_to_deg(B, K_SLOW), lw=1.8,
                color=cmap(k / (len(s_list) - 1)), label=f'{s0:.0f} mm')
    ax.set_ylim(0, 185)
    ax.set_yticks(np.arange(0, 181, 45))
    ax.set_xlim(0, SUBLEVEL_OFFSET_MAX_MM)
    ax.set_xlabel(r'vertical offset $\Delta z$ [mm]')
    ax.set_ylabel(r'$\theta$ between $\hat{B}$ and the slowing beam [deg]')
    ax.legend(fontsize=7, ncol=2, title='$s$ from trap', title_fontsize=7)
    ax.grid(alpha=0.25)
    ax.set_title(r'angle vs. displacement', fontsize=10)


    ax = fig.add_subplot(gs[1, 1])
    s0 = SUBLEVEL_S_MM[1]
    B = field_G_per_amp(lab_xyz_mm(np.full_like(dz, s0), dz)) * current_A
    shift = np.linalg.norm(B, axis=-1) * 1e-4 * ZEEMAN_GAMMA_PER_TESLA
    f_sp, f_pi, f_sm, _ = polarization_fractions(B)
    for m_j, frac, col, name in ((+1, f_sp, C_SIGMA_P, r'$m_J=+1$'),
                                 (0, f_pi, C_PI, r'$m_J=0$'),
                                 (-1, f_sm, C_SIGMA_M, r'$m_J=-1$')):
        ax.plot(dz, m_j * shift, color=col, lw=1.0, alpha=0.45)
        ax.scatter(dz[::7], (m_j * shift)[::7], s=3 + 130 * frac[::7],
                   color=col, alpha=0.8, edgecolors='none', label=name)
    ax.axhline(0, color='k', lw=0.7, alpha=0.5)
    ax.set_xlim(0, SUBLEVEL_OFFSET_MAX_MM)
    ax.set_xlabel(r'vertical offset $\Delta z$ [mm]')
    ax.set_ylabel(r'$1P_1$ sublevel shift [$\Gamma$] at '
                  f'{current_A:.0f} A')
    ax.legend(fontsize=8, loc='center left')
    ax.grid(alpha=0.25)
    ax.set_title(f'the three sublevels at $s$ = {s0:.0f} mm.\n'
                 'marker AREA = fraction of the beam driving that one',
                 fontsize=10)



    ax = fig.add_subplot(gs[1, 2])
    for k, d in enumerate(SUBLEVEL_PATH_OFFSETS_MM):
        s_p, B_p = sample_path(n=901, z_mm=d)
        _, _, f_sm_p, _ = polarization_fractions(B_p)
        ax.plot(s_p, f_sm_p, lw=1.8,
                color=cmap(k / (len(SUBLEVEL_PATH_OFFSETS_MM) - 1)),
                label=f'{d:.0f} mm')
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel(r'$\sigma^-$ ($m_J=-1$) fraction')
    ax.set_xlabel('$s$ = distance to trap centre [mm]')
    ax.legend(fontsize=7, ncol=2, title=r'$\Delta z$', title_fontsize=7,
              loc='lower left')
    ax.grid(alpha=0.25)
    
    style_path_axis(ax, -PAST_TRAP_MM, 120.0)

    worst = f_sm_p
    lost = np.where(worst < 0.99)[0]
    s_break = s_p[lost].max() if lost.size else 0.0
    ax.set_title(f'Fraction over distance', fontsize=10)

    fig.suptitle('Which $1P_1$ sublevel the 399 nm slowing beam  '
                 'drives '
                 r'(a nominally $\sigma^+$ beam; the labels $\sigma^\pm/\pi$ '
                 'are defined against the local field direction)',
                 fontsize=11)
    return _save(fig, out_dir, '10_sublevel_projection.png')


#
# CSV + summary

def write_csv(s, B_per_A, out_dir=OUTPUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'field_along_path.csv')
    Bs, Bz = in_plane_components(B_per_A)
    Bmag = np.linalg.norm(B_per_A, axis=-1)
    phi = np.degrees(np.arctan2(Bz, Bs))
    ang_k = angle_to_deg(B_per_A, K_SLOW)
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['# field along the atom path, z = 0, in GAUSS PER AMP of coil '
                    'current (linear with current)'])
        w.writerow(['s_mm_to_trap', 'Bx_G_per_A', 'By_G_per_A', 'Bz_G_per_A',
                    'Bpar_along_beam_G_per_A', 'Bmag_G_per_A',
                    'phi_in_plane_deg', 'angle_to_slowing_beam_deg',
                    f'Bmag_G_at_{MAP_CURRENT_A:.0f}A'])
        for i in range(len(s)):
            w.writerow([f'{s[i]:.4f}', f'{B_per_A[i, 0]:.8g}', f'{B_per_A[i, 1]:.8g}',
                        f'{B_per_A[i, 2]:.8g}', f'{Bs[i]:.8g}', f'{Bmag[i]:.8g}',
                        f'{phi[i]:.4f}', f'{ang_k[i]:.4f}',
                        f'{Bmag[i] * MAP_CURRENT_A:.8g}'])
    print(f"[beam_plane_field_map] wrote {path}")
    return path


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150)
    print(f"[beam_plane_field_map] wrote {path}")
    if not SHOW_PLOTS:
        plt.close(fig)
    return path


def self_check(s, B_per_A):
    """"""
    print("\n" + "=" * 70)
    print("SELF CHECK (axisymmetry claims from the module docstring)")
    print("=" * 70)
    out_of_plane = B_per_A @ E_H
    Bmag = np.linalg.norm(B_per_A, axis=-1)
    frac = float(np.max(np.abs(out_of_plane) / np.maximum(Bmag, 1e-30)))
    print(f"  max |B . e_h| / |B| along the path = {frac:.3e}   "
          f"({'OK, B lies in the beam plane' if frac < 1e-9 else 'UNEXPECTED, investigate'})")

    good = np.abs(B_per_A[:, 0]) > 1e-9
    ratio = B_per_A[good, 1] / B_per_A[good, 0]
    print(f"  By/Bx = {ratio.min():.6f} .. {ratio.max():.6f}, "
          f"expected tan({BEAM_TILT_DEG}) = {math.tan(TILT):.6f}   "
          f"({'OK' if np.allclose(ratio, math.tan(TILT), rtol=1e-6) else 'MISMATCH'})")

    i_peak = int(np.argmax(Bmag))
    print(f"\n  |B| at the nozzle face (s = {NOZZLE_MM:.0f} mm): "
          f"{Bmag[-1] * MAP_CURRENT_A:8.3f} G at {MAP_CURRENT_A:.0f} A")
    print(f"  |B| peak along the path:  {Bmag[i_peak] * MAP_CURRENT_A:8.3f} G "
          f"at s = {s[i_peak]:.1f} mm  (the beam passes the winding radii, "
          f"s = {RADIUS_POSITIONS_MM[0]:.0f}-{RADIUS_POSITIONS_MM[-1]:.0f} mm)")
    ang = angle_to_deg(B_per_A, K_SLOW)
    interior = np.abs(s) > 1.0
    print(f"  angle of B to the slowing beam ranges over "
          f"{ang[interior].min():.1f}-{ang[interior].max():.1f} deg along the path")
    shift = Bmag * MAP_CURRENT_A * 1e-4 * ZEEMAN_GAMMA_PER_TESLA
    print(f"  peak 1P1 Zeeman shift: {shift.max():.1f} Gamma "
          f"(= {shift.max() * V_PER_GAMMA:.0f} m/s of Doppler), vs a most probable "
          f"flux speed of {most_probable_flux_speed():.0f} m/s at {REF_OVEN_C:.0f} C")


# ================
def main():
    print("=" * 70)
    print("beam_plane_field_map.py")
    print(f"  plane   : contains the atom beam ({BEAM_TILT_DEG} deg to x) and z")
    print(f"  path    : s = {NOZZLE_MM:.0f} mm (nozzle)  ->  0 mm (trap), "
          f"plus {PAST_TRAP_MM:.0f} mm past")
    print(f"  maps at : {MAP_CURRENT_A:.0f} A;  line plots at "
          f"{', '.join(f'{I:.0f}' for I in LINE_CURRENTS_A)} A")
    print(f"  output  : {OUTPUT_DIR}")
    print("=" * 70)

    print("\n[1/3] evaluating the field on the beam-plane map grid ...")
    s_ax, z_ax, S, Z, B_map = beam_plane_grid()
    print(f"      {S.size:,} points, {S.shape[0]} x {S.shape[1]}, over "
          f"s = {s_ax.min():.0f}..{s_ax.max():.0f} mm, z = +-{MAP_Z_HALF_MM:.0f} mm")

    print("[2/3] evaluating the field along the atom path ...")
    s_path, B_path = sample_path()

    print("[3/3] drawing ...")
    plot_components(S, Z, B_map)
    plot_magnitude_map(s_ax, z_ax, S, Z, B_map)
    plot_direction_map(S, Z, B_map)
    plot_along_path(s_path, B_path)
    plot_angles(s_path, B_path)
    plot_polarization_and_zeeman(s_path, B_path)
    plot_trap_zoom()
    plot_across_beam_width()
    plot_beam_frame(S, Z, B_map, s_path, B_path)
    plot_sublevel_projection()
    write_csv(s_path, B_path)

    self_check(s_path, B_path)
    print(f"\nAll figures are in {OUTPUT_DIR}")

    if SHOW_PLOTS:
        plt.show()


if __name__ == '__main__':
    main()
