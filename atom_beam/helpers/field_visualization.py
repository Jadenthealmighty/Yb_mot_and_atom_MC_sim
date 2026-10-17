"""
field_visualization.py
field+ beam + mot visualization in 3d
XYZ Cross-section of magnetic field
"""

import os
import numpy as np

# Interactive 3D view toggle
INTERACTIVE_3D = True

import matplotlib
if not INTERACTIVE_3D:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import magpylib as magpy

from coil_field_model import (
    oswald_coil_bfield, _get_unit_collection, axial_gradient_G_per_cm,
    radial_gradient_G_per_cm, RADIUS_POSITIONS_MM, HEIGHT_POSITIONS_MM,
)


# CONFIG
CURRENT_A = 135.0
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'field_plots')

# ---- MOT beam overlay (3D plot) 
SHOW_MOT_BEAMS = True

MOT_BEAM_START_MM = 170.0 
MOT_BEAM_LENGTH_MM = 120.0
MOT_BEAM_WAIST_MM = 5.0   
MOT_BEAM_COLOR = '#7B2FBE' 
MOT_BEAM_POL_LABELS = {'x': ('$\\sigma^-$', '$\\sigma^+$'),
                        'y': ('$\\sigma^-$', '$\\sigma^+$'),
                        'z': ('$\\sigma^+$', '$\\sigma^-$')}


SHOW_ATOM_BEAM = True
SHOW_SLOWING_BEAM = True

ATOM_BEAM_TILT_DEG = 25.5
ATOM_BEAM_DIVERGENCE_DEG = 1.39 * 2
ATOM_BEAM_TRUE_DIVERGENCE_DEG = 1.39

ATOM_BEAM_NOZZLE_MM = 344.0      
ATOM_BEAM_DRAW_FROM_MM = 190.0   
ATOM_BEAM_DRAW_PAST_MM = 55.0    
ATOM_BEAM_COLOR = '#FF9F1C'  
ATOM_BEAM_ALPHA = 0.20

SLOWER_WAIST_MM = 5.0            
SLOWER_DIVERGENCE_MRAD = 0.0     
SLOWER_OPTIC_MM = 500.0          
SLOWER_DRAW_FROM_MM = 190.0      
SLOWER_DRAW_TO_MM = 12.0         
SLOWER_DRAW_THROUGH = False
SLOWER_COLOR = '#C05CE8'
SLOWER_ALPHA = 0.16

BEAM_LABEL_OFFSET_MM = 42.0
SHOW_BEAM_LABELS = True


def report_gradient_at_center(current_A=CURRENT_A, verbose=True):
    grad_axial = axial_gradient_G_per_cm(current_A)
    grad_radial = radial_gradient_G_per_cm(current_A)
    if verbose:
        print(f"[field_visualization] Gradient at trap center, {current_A:.1f} A:")
        print(f"    axial  (dBz/dz) = {grad_axial:7.2f} G/cm")
        print(f"    radial (dBx/dx) = {grad_radial:7.2f} G/cm "
              f"(ideal anti-Helmholtz expectation: {-grad_axial/2:.2f} G/cm)")
    return {'axial_G_per_cm': grad_axial, 'radial_G_per_cm': grad_radial,
            'current_A': current_A}


def _draw_mot_beams(ax, start_mm=MOT_BEAM_START_MM,
                     length_mm=MOT_BEAM_LENGTH_MM,
                     waist_mm=MOT_BEAM_WAIST_MM,
                     color=MOT_BEAM_COLOR, label_polarizations=True):
    axes_dirs = {
        'x': np.array([1.0, 0.0, 0.0]),
        'y': np.array([0.0, 1.0, 0.0]),
        'z': np.array([0.0, 0.0, 1.0]),
    }
    theta = np.linspace(0, 2 * np.pi, 24)
    handle = None

    for axis_name, unit in axes_dirs.items():
        
        for sign_idx, sign in enumerate((+1.0, -1.0)):
            start = unit * sign * start_mm
            direction = -unit * sign  # always inward, toward the origin



           
            perp1 = np.roll(unit, 1)
            perp1 = perp1 - np.dot(perp1, unit) * unit
            perp1 /= np.linalg.norm(perp1)
            perp2 = np.cross(unit, perp1)

            s = np.array([0.0, length_mm])
            S, TH = np.meshgrid(s, theta, indexing='ij')
            ring = (np.cos(TH)[..., None] * perp1 + np.sin(TH)[..., None] * perp2) * waist_mm
            centers = start[None, None, :] + S[..., None] * direction[None, None, :]
            tube = centers + ring
            ax.plot_surface(tube[..., 0], tube[..., 1], tube[..., 2],
                            color=color, alpha=0.16, linewidth=0, shade=False)

            
            end = start + direction * length_mm
            line = ax.plot([start[0], end[0]], [start[1], end[1]],
                           [start[2], end[2]],
                           color=color, linewidth=1.6, alpha=0.9)[0]
            handle = handle or line
            # Arrowhead 
            ax.quiver(end[0], end[1], end[2],
                      direction[0], direction[1], direction[2],
                      length=0.22 * length_mm, color=color, linewidth=2.0,
                      arrow_length_ratio=0.55, alpha=0.95)

            if label_polarizations:
                pol = MOT_BEAM_POL_LABELS[axis_name][sign_idx]
                lbl = start + direction * (-0.06 * length_mm) 
                ax.text(lbl[0], lbl[1], lbl[2], pol, color=color,
                        fontsize=9, ha='center', va='center')

    return handle


def _display_collection_mm():



    top, bottom = magpy.Collection(), magpy.Collection()
    for loop_r_mm in RADIUS_POSITIONS_MM:
        for loop_h_mm in HEIGHT_POSITIONS_MM:
            top.add(magpy.current.Circle(current=1.0,
                                         diameter=2 * loop_r_mm,
                                         position=(0, 0, loop_h_mm)))
            bottom.add(magpy.current.Circle(current=-1.0,
                                            diameter=2 * loop_r_mm,
                                            position=(0, 0, -loop_h_mm)))
    return magpy.Collection(top, bottom)


def beam_axis_basis(tilt_deg=ATOM_BEAM_TILT_DEG):
    t = np.radians(tilt_deg)
    u = np.array([np.cos(t), np.sin(t), 0.0])
    e_h = np.array([-np.sin(t), np.cos(t), 0.0])
    e_v = np.array([0.0, 0.0, 1.0])
    return u, e_h, e_v


def _sweep_surface(ax, u, e_h, e_v, s_mm, radius_mm, color, alpha,
                   n_theta=48):
    theta = np.linspace(0, 2 * np.pi, n_theta)
    S, TH = np.meshgrid(s_mm, theta, indexing='ij')
    R = np.broadcast_to(np.asarray(radius_mm)[:, None], S.shape)
    ring = (np.cos(TH)[..., None] * e_h + np.sin(TH)[..., None] * e_v) * R[..., None]
    pts = S[..., None] * u[None, None, :] + ring
    ax.plot_surface(pts[..., 0], pts[..., 1], pts[..., 2],
                    color=color, alpha=alpha, linewidth=0, shade=False)


def _draw_atom_beam(ax, tilt_deg=ATOM_BEAM_TILT_DEG,
                    divergence_deg=None, nozzle_mm=ATOM_BEAM_NOZZLE_MM,
                    draw_from_mm=ATOM_BEAM_DRAW_FROM_MM,
                    draw_past_mm=ATOM_BEAM_DRAW_PAST_MM,
                    color=ATOM_BEAM_COLOR, alpha=ATOM_BEAM_ALPHA,
                    label=True):
    divergence_deg = (ATOM_BEAM_DIVERGENCE_DEG if divergence_deg is None
                      else divergence_deg)
    u, e_h, e_v = beam_axis_basis(tilt_deg)


    s = np.linspace(-draw_from_mm, draw_past_mm, 60)
    r = (s + nozzle_mm) * np.tan(np.radians(divergence_deg))
    r = np.clip(r, 0.0, None)

    _sweep_surface(ax, u, e_h, e_v, s, r, color, alpha)

   
    a, b = -draw_from_mm * u, draw_past_mm * u
    handle = ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                     color=color, linewidth=1.4, alpha=0.85,
                     linestyle=(0, (6, 3)))[0]
    tip = 0.45 * draw_past_mm * u
    ax.quiver(tip[0], tip[1], tip[2], u[0], u[1], u[2],
              length=0.30 * draw_from_mm, color=color, linewidth=2.2,
              arrow_length_ratio=0.45, alpha=0.95)

    if label:
      
        p = -0.90 * draw_from_mm * u + BEAM_LABEL_OFFSET_MM * e_v
        ax.text(p[0], p[1], p[2],
                f"atom beam from oven\n"
                f"({nozzle_mm:.0f} mm, {tilt_deg:.1f}$^\\circ$ from $x$)",
                color=color, fontsize=8, ha='center', va='bottom')
    return handle


def _draw_slowing_beam(ax, tilt_deg=ATOM_BEAM_TILT_DEG,
                       waist_mm=SLOWER_WAIST_MM,
                       divergence_mrad=SLOWER_DIVERGENCE_MRAD,
                       optic_mm=SLOWER_OPTIC_MM,
                       draw_from_mm=SLOWER_DRAW_FROM_MM,
                       draw_to_mm=SLOWER_DRAW_TO_MM,
                       draw_through=SLOWER_DRAW_THROUGH,
                       color=SLOWER_COLOR, alpha=SLOWER_ALPHA, label=True):
    u, e_h, e_v = beam_axis_basis(tilt_deg)

    near = -draw_from_mm if draw_through else draw_to_mm
    s = np.linspace(near, draw_from_mm, 60)
    zeta = optic_mm - s
    w = np.sqrt(waist_mm ** 2 + (divergence_mrad * 1e-3 * zeta) ** 2)

    _sweep_surface(ax, u, e_h, e_v, s, w, color, alpha)

    a, b = draw_from_mm * u, near * u
    handle = ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                     color=color, linewidth=1.6, alpha=0.9)[0]
    tip = (near + 0.10 * (draw_from_mm - near)) * u
    ax.quiver(tip[0], tip[1], tip[2], -u[0], -u[1], -u[2],
              length=0.28 * draw_from_mm, color=color, linewidth=2.2,
              arrow_length_ratio=0.45, alpha=0.95)

    if label:
        p = 0.90 * draw_from_mm * u - BEAM_LABEL_OFFSET_MM * e_v
        ax.text(p[0], p[1], p[2],
                "399 nm slowing beam\n"
                f"($w_0$ = {waist_mm:.0f} mm, counter-propagating)",
                color=color, fontsize=8, ha='center', va='top')
    return handle


def plot_3d_coil_and_field(current_A=CURRENT_A, out_dir=OUTPUT_DIR,
                            arrow_grid_n=8, arrow_plane_extent_mm=80.0,
                            show_mot_beams=None):
    
    show_mot_beams = SHOW_MOT_BEAMS if show_mot_beams is None else show_mot_beams
    os.makedirs(out_dir, exist_ok=True)
    # Display-only geometry in mm
    collection = _display_collection_mm()

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')



    
    collection.show(canvas=ax, backend='matplotlib', style_opacity=0.25)

    axis_mm = np.linspace(-arrow_plane_extent_mm, arrow_plane_extent_mm, arrow_grid_n)

    quiver_points = []
    for X, Y, Z in [np.meshgrid(axis_mm, [0.0], axis_mm, indexing='ij'),
                     np.meshgrid([0.0], axis_mm, axis_mm, indexing='ij')]:
        pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        quiver_points.append(pts)
    quiver_points = np.vstack(quiver_points)
    quiver_points_m = quiver_points * 1e-3

    B_T = oswald_coil_bfield(quiver_points_m, current_A)
    B_mag = np.linalg.norm(B_T, axis=1)
    B_mag_safe = np.where(B_mag < 1e-12, 1e-12, B_mag)

    U, V, W = (B_T[:, 0] / B_mag_safe,
               B_T[:, 1] / B_mag_safe,
               B_T[:, 2] / B_mag_safe)

    norm_ = mcolors.LogNorm(vmin=max(B_mag.min(), 1e-6), vmax=B_mag.max())
    colors = plt.cm.plasma(norm_(B_mag))

    # Arrow length scales with the plotted region...
    ax.quiver(quiver_points[:, 0], quiver_points[:, 1], quiver_points[:, 2],
              U, V, W, length=0.11 * arrow_plane_extent_mm, normalize=False,
              color=colors, linewidth=1.1, alpha=0.9)
    mappable = plt.cm.ScalarMappable(norm=norm_, cmap='plasma')
    mappable.set_array(B_mag)
    cb = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.10)
    cb.set_label('|B| (T, log scale)', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # MOT BEAMS
    beam_handle = None
    if show_mot_beams:
        beam_handle = _draw_mot_beams(ax)
    atom_handle = (_draw_atom_beam(ax, label=SHOW_BEAM_LABELS)
                   if SHOW_ATOM_BEAM else None)
    slower_handle = (_draw_slowing_beam(ax, label=SHOW_BEAM_LABELS)
                     if SHOW_SLOWING_BEAM else None)
    center_handle = ax.scatter([0], [0], [0], color='lime', s=70,
                               marker='+', linewidth=2.5, depthshade=False)

    grad_info = report_gradient_at_center(current_A, verbose=False)
    grad = grad_info['axial_G_per_cm']



    # cosmetics 
    
    coil_extent_mm = max(np.max(RADIUS_POSITIONS_MM), np.max(HEIGHT_POSITIONS_MM))
    view_mm = max(arrow_plane_extent_mm, coil_extent_mm,
                  (MOT_BEAM_START_MM if show_mot_beams else 0.0),
                  (ATOM_BEAM_DRAW_FROM_MM if SHOW_ATOM_BEAM else 0.0),
                  (SLOWER_DRAW_FROM_MM if SHOW_SLOWING_BEAM else 0.0)) * 1.08
    ax.set_xlim(-view_mm, view_mm)
    ax.set_ylim(-view_mm, view_mm)
    ax.set_zlim(-view_mm, view_mm)
    try:
        ax.set_box_aspect((1, 1, 1))  # matplotlib >= 3.3
    except AttributeError:
        pass
    ax.view_init(elev=22, azim=35)
    ax.set_xlabel('x (mm)', labelpad=8)
    ax.set_ylabel('y (mm)', labelpad=8)
    ax.set_zlabel('z (mm)', labelpad=8)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25)

    title = (f"$^{{174}}$Yb MOT 3D Render")
    
    ax.set_title(title, fontsize=11, pad=14)

    handles, labels = [center_handle], ['trap center (field zero)']
    if beam_handle is not None:
        handles.insert(0, beam_handle)
        labels.insert(0, f'399 nm MOT beams ($w_0$ = {MOT_BEAM_WAIST_MM:.0f} mm)')
    if slower_handle is not None:
        handles.insert(0, slower_handle)
        labels.insert(0, f'399 nm slowing beam ($w_0$ = {SLOWER_WAIST_MM:.0f} mm)')
    if atom_handle is not None:
        exaggerated = (abs(ATOM_BEAM_DIVERGENCE_DEG
                           - ATOM_BEAM_TRUE_DIVERGENCE_DEG) > 1e-6)
        note = (f'{ATOM_BEAM_DIVERGENCE_DEG:.2f}$^\\circ$ shown, '
                f'{ATOM_BEAM_TRUE_DIVERGENCE_DEG:.2f}$^\\circ$ real'
                if exaggerated else
                f'{ATOM_BEAM_DIVERGENCE_DEG:.2f}$^\\circ$ half-angle')
        handles.insert(0, atom_handle)
        labels.insert(0, f'Yb atom beam ({note})')
    ax.legend(handles, labels, loc='upper left', fontsize=8, framealpha=0.85)

    
    fig.text(0.02, 0.02,
              f"dBz/dz = {grad_info['axial_G_per_cm']:.2f} G/cm\n"
              f"dBx/dx = {grad_info['radial_G_per_cm']:.2f} G/cm\n"
              f"arrows: B direction, color ~ log|B|",
              fontsize=8, family='monospace',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    fig.tight_layout()
    outpath = os.path.join(out_dir, 'coil_3d_field.png')
    fig.savefig(outpath, dpi=150)
    print(f"[field_visualization] wrote {outpath} (default angle)")

    if INTERACTIVE_3D:
        print("[field_visualization] INTERACTIVE_3D=True")
        plt.show()
    plt.close(fig)
    return grad_info


def plot_cross_section(current_A=CURRENT_A, out_dir=OUTPUT_DIR, plane='xz',
                        extent_mm=40.0, n_grid=150, show_coil_windings=True):
    os.makedirs(out_dir, exist_ok=True)
    axis = np.linspace(-extent_mm, extent_mm, n_grid)
    A, B_ = np.meshgrid(axis, axis, indexing='ij')

    if plane == 'xz':
        X, Y, Z = A, np.zeros_like(A), B_
        horiz_label, vert_label = 'x (mm)', 'z (mm)'
        horiz_field_idx, vert_field_idx = 0, 2  
    elif plane == 'xy':
        X, Y, Z = A, B_, np.zeros_like(A)
        horiz_label, vert_label = 'x (mm)', 'y (mm)'
        horiz_field_idx, vert_field_idx = 0, 1  # Bx, By
    else:
        raise ValueError("plane must be 'xz' or 'xy'")

    pts_mm = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    B_T = oswald_coil_bfield(pts_mm * 1e-3, current_A)
    B_G = B_T * 1e4 
    B_mag_G = np.linalg.norm(B_G, axis=1).reshape(A.shape)
    Bh = B_G[:, horiz_field_idx].reshape(A.shape)
    Bv = B_G[:, vert_field_idx].reshape(A.shape)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    floor_G = max(B_mag_G[B_mag_G > 0].min(), 1e-4)
    B_plot = np.clip(B_mag_G, floor_G, None)

    levels = np.logspace(np.log10(floor_G), np.log10(B_plot.max()), 60)
    cf = ax.contourf(A, B_, B_plot, levels=levels,
                      cmap='plasma', norm=mcolors.LogNorm())

    cb = fig.colorbar(cf, ax=ax, ticks=mticker.LogLocator(base=10, subs=(1.0,)))
    cb.ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)))
    cb.set_label('|B| (G, log scale)')
    ax.streamplot(axis, axis, Bh.T, Bv.T, color='white', density=1.1,
                  linewidth=0.7, arrowsize=0.9)

    if show_coil_windings and plane == 'xz':
        for z_center, marker in [(+1, '^'), (-1, 'v')]:
            for loop_r in RADIUS_POSITIONS_MM:
                for loop_h in HEIGHT_POSITIONS_MM:
                    ax.plot([loop_r, -loop_r], [z_center * loop_h] * 2,
                            marker, color='cyan', markersize=3, alpha=0.6)

    ax.plot(0, 0, '+', color='lime', markersize=14, markeredgewidth=2,
             label='trap center (field zero)')
    ax.set_xlabel(horiz_label)
    ax.set_ylabel(vert_label)
    ax.set_aspect('equal')
    ax.set_xlim(-extent_mm, extent_mm)
    ax.set_ylim(-extent_mm, extent_mm)
    ax.legend(loc='upper right', fontsize=8)
    grad_info = report_gradient_at_center(current_A, verbose=False)
    grad = grad_info['axial_G_per_cm']
    ax.set_title(f"{plane.upper()} cross-section at {current_A:.0f} A "
                 f"(axial gradient {grad:.1f} G/cm)")
    ax.text(0.02, 0.02,
            f"dBz/dz = {grad_info['axial_G_per_cm']:.2f} G/cm\n"
            f"dBx/dx = {grad_info['radial_G_per_cm']:.2f} G/cm",
            transform=ax.transAxes, fontsize=8, family='monospace',
            va='bottom', ha='left', color='black',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    fig.tight_layout()
    tag = '' if extent_mm == 40.0 else f'_{extent_mm:.0f}mm'
    outpath = os.path.join(out_dir, f'cross_section_{plane}{tag}.png')
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[field_visualization] wrote {outpath}")
    return grad_info


if __name__ == '__main__':
    print(f"Generating field visualizations at CURRENT_A = {CURRENT_A} A "
          f"(edit the CURRENT_A constant at the top of this file, or pass "
          "current_A= to any of these functions directly, to look at a "
          "different operating current.")
    plot_3d_coil_and_field(CURRENT_A)
    plot_cross_section(CURRENT_A, plane='xz')                         
    plot_cross_section(CURRENT_A, plane='xz', extent_mm=180.0) 
    plot_cross_section(CURRENT_A, plane='xy')
    print()
    report_gradient_at_center(CURRENT_A) # just print
    print("Done.")
