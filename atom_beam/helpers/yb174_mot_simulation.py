"""pylcp simulation of a Yb-174 MOT on the 399 nm 6s^2 1S0 -> 6s6p 1P1
"blue" transition, with


UPDATE: USED MOSTLY AS A REFERENCE FOR FULL_TRAP_SWEEP.PY, USE THAT, NOT THIS since this
does not include slower beam
"""

import os
import io
import csv
import json
import time
import traceback
import contextlib
import multiprocessing as mp
import numpy as np
import scipy.constants as sp_const
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d

import pylcp

import runlog

import pylcp.integration_tools as _pylcp_integration_tools
_orig_prepare_events = _pylcp_integration_tools.prepare_events


def _prepare_events_compat(events):
    if events is None:
        return None, None, None
    return _orig_prepare_events(events)

_pylcp_integration_tools.prepare_events = _prepare_events_compat

from .coil_field_model import (axial_gradient_G_per_cm,
                               get_fast_bfield_interpolator,
                               make_pylcp_magfield)

RUN_NAME = 'yb174-mot-sim'

COIL_CURRENT_A = 135.0

YB174_MASS_U = 173.9388621
YB174_TRANSITION_FREQ_HZ = 751.5265174e12
YB174_LINEWIDTH_HZ = 30.2e6
YB174_GJ_EXCITED = 1.0
YB174_WAVELENGTH_M = sp_const.c / YB174_TRANSITION_FREQ_HZ

S_PER_BEAM = 0.1
NOMINAL_DETUNING_HZ = -0.5 * YB174_LINEWIDTH_HZ

LASER_BEAM_MODE = 'plane_wave'
GAUSSIAN_WAIST_M = 0.01
GAUSSIAN_POWER_W = 60.0e-3

JITTER_MODE = 'synthetic'
JITTER_PP_HZ = 10.0e6
JITTER_CORR_TIME_S = 5e-4
EXPERIMENTAL_JITTER_CSV = 'my_laser_frequency_log.csv'
EXPERIMENTAL_TIME_COLUMN = 0
EXPERIMENTAL_FREQ_COLUMN = 1
EXPERIMENTAL_TIME_UNIT = 's'
EXPERIMENTAL_FREQ_UNIT = 'MHz'

N_DETUNING_SCAN_POINTS = 9
N_MONTE_CARLO_ATOMS = 6
MC_INITIAL_SPEED_BAR = 2.0
MC_TMAX_BAR = 20000.0

MC_PARALLEL = True
MC_N_WORKERS = 6

MC_SAVE_DATA = True

RANDOM_SEED = 12345

EPS_POSITION_UM_LADDER = (5.0, 20.0, 80.0)
EPS_VELOCITY_BAR_LADDER = (0.01, 0.1, 1.0)
FALLBACK_R_NUDGE_UM = 5.0

MAGFIELD_GRADIENT_EPS_UM = 10.0


def print_environment_info():
    """IMPORTANT CUZ MAGPYLIB CHANGED UNIT CONVENTIONS"""
    import sys
    import platform
    print("=" * 70)
    print("Environment")
    print("=" * 70)
    print(f"  Python   {sys.version.split()[0]}  ({platform.platform()})")
    import importlib.metadata
    for _mod_name in ('numpy', 'scipy', 'pylcp', 'numba', 'magpylib'):
        try:
            _mod = __import__(_mod_name)
            _ver = getattr(_mod, '__version__', None)
            if _ver is None:
                try:
                    _ver = importlib.metadata.version(_mod_name)
                except Exception:
                    _ver = '?'
            print(f"  {_mod_name:8s} {_ver}")
        except ImportError:
            print(f"  {_mod_name:8s} (not installed)")
    try:
        import numpy as _np
        _blas = _np.__config__.show(mode='dicts')['Build Dependencies']['blas']['name']
        print(f"  numpy BLAS backend: {_blas}")
    except Exception:
        pass
    print("=" * 70 + "\n")



def build_normalization():
    """Build pylcp's unit system (x0, t0, mass_bar) for Yb-174 at 399 nm."""
    print_environment_info()

    mass_SI = YB174_MASS_U * sp_const.value('atomic mass constant')
    k_SI = 2 * np.pi / YB174_WAVELENGTH_M
    Gamma_SI = 2 * np.pi * YB174_LINEWIDTH_HZ

    x0 = 1.0 / k_SI
    t0 = 1.0 / Gamma_SI
    v0 = x0 / t0
    mass_bar = mass_SI * x0 ** 2 / (sp_const.hbar * t0)

    Isat_SI = (np.pi * sp_const.h * sp_const.c * Gamma_SI) / (3 * YB174_WAVELENGTH_M ** 3)

    norm = dict(mass_SI=mass_SI, k_SI=k_SI, Gamma_SI=Gamma_SI, x0=x0, t0=t0,
                v0=v0, mass_bar=mass_bar, Isat_SI=Isat_SI,
                Isat_mW_cm2=Isat_SI * 1e-3 / 1e-4 * 1e3 / 1e3)
    norm['Isat_mW_cm2'] = Isat_SI * 0.1

    print("=" * 70)
    print("Normalization / unit system (Yb-174, 399 nm transition)")
    print("=" * 70)
    print(f"  mass            = {mass_SI:.6e} kg  ({YB174_MASS_U} u)")
    print(f"  wavelength      = {YB174_WAVELENGTH_M*1e9:.4f} nm "
          f"(from measured transition freq {YB174_TRANSITION_FREQ_HZ/1e12:.6f} THz)")
    print(f"  Gamma / 2pi     = {YB174_LINEWIDTH_HZ/1e6:.3f} MHz")
    print(f"  Isat            = {norm['Isat_mW_cm2']:.2f} mW/cm^2 "
          "(compare to ~60 mW/cm^2 typically quoted in the literature)")
    print(f"  length unit x0  = {x0*1e6:.4f} um")
    print(f"  time unit t0    = {t0*1e9:.4f} ns")
    print(f"  velocity unit   = Gamma/k = {v0:.4f} m/s")
    print(f"  dimensionless mass_bar = {mass_bar:.4f}")
    return norm


def build_hamiltonian(norm):
    """J=0 -> J=1 Hamiltonian for the 399 nm transition"""
    Hg, mugq = pylcp.hamiltonians.singleF(F=0, muB=1)
    He, mueq = pylcp.hamiltonians.singleF(F=1, muB=1)
    dq = pylcp.hamiltonians.dqij_two_bare_hyperfine(0, 1)

    hamiltonian = pylcp.hamiltonian(Hg, He, mugq, mueq, dq,
                                     mass=norm['mass_bar'], gamma=1.0, k=1.0)
    return hamiltonian


def build_magfield(norm):
    """Build the pylcp.magField for the 8-coil pair at COIL_CURRENT_A.

    Returns (magField, source_label, axial_gradient_G_per_cm).
    """
    print(f"\n[MOT sim] 8-coil anti-Helmholtz model (coil_field_model.py) "
          f"at COIL_CURRENT_A = {COIL_CURRENT_A} A.")
    est_grad = axial_gradient_G_per_cm(COIL_CURRENT_A)
    print(f"[MOT sim] (Predicted axial gradient {est_grad:.2f} G/cm at "
          f"{COIL_CURRENT_A} A,should match magfield_test.py exactly.)")
    print("[MOT sim] Tabulating the field for fast lookup (~2 s)...")
    fast_B_interp = get_fast_bfield_interpolator()

    def B_interp(xyz_meters):
        return fast_B_interp(xyz_meters, COIL_CURRENT_A)
    source = f'coil_model_{COIL_CURRENT_A:.0f}A'

    B_bar_func = make_pylcp_magfield(B_interp, norm['x0'], norm['Gamma_SI'],
                                      gJ_excited=YB174_GJ_EXCITED)
    magfield_eps_bar = (MAGFIELD_GRADIENT_EPS_UM * 1e-6) / norm['x0']
    magField = pylcp.magField(B_bar_func, eps=magfield_eps_bar)

    dz = 1e-4
    Bz_plus = B_interp(np.array([0., 0., dz]))[2]
    Bz_minus = B_interp(np.array([0., 0., -dz]))[2]
    grad_z_G_per_cm = (Bz_plus - Bz_minus) / (2 * dz) * 1e4 * 1e-2
    print(f"[MOT sim] Field source: {source}. "
          f"Axial gradient dBz/dz at origin ~ {grad_z_G_per_cm:.2f} G/cm.")

    return magField, source, grad_z_G_per_cm


def make_synthetic_jitter(nominal_detuning_bar, Gamma_SI, t0,
                           jitter_pp_hz=JITTER_PP_HZ,
                           corr_time_s=JITTER_CORR_TIME_S,
                           t_max_bar=MC_TMAX_BAR, seed=RANDOM_SEED):
    """Useful from my own results for 399 nm frequency stuff"""
    rng = np.random.default_rng(seed)
    corr_time_bar = corr_time_s / t0
    n_control = max(8, int(np.ceil(t_max_bar / corr_time_bar)) + 4)
    t_control = np.linspace(-corr_time_bar, t_max_bar + corr_time_bar, n_control)

    jitter_amp_bar = (jitter_pp_hz / 2) * 2 * np.pi / Gamma_SI
    raw = np.zeros(n_control)
    theta = 0.3
    for i in range(1, n_control):
        raw[i] = raw[i - 1] * (1 - theta) + rng.normal(0, 1) * np.sqrt(theta)
    raw = raw / (np.std(raw) + 1e-12) * (jitter_amp_bar / 1.5)
    raw = np.clip(raw, -jitter_amp_bar, jitter_amp_bar)

    spline = CubicSpline(t_control, raw, extrapolate=True)

    def delta_bar(t):
        return nominal_detuning_bar + float(spline(t))

    return delta_bar, jitter_amp_bar


def load_experimental_jitter(csv_path, nominal_detuning_bar, Gamma_SI, t0,
                              time_col=EXPERIMENTAL_TIME_COLUMN,
                              freq_col=EXPERIMENTAL_FREQ_COLUMN,
                              time_unit=EXPERIMENTAL_TIME_UNIT,
                              freq_unit=EXPERIMENTAL_FREQ_UNIT):
    time_scale = {'s': 1.0, 'ms': 1e-3, 'us': 1e-6, 'min': 60.0}[time_unit]
    freq_scale = {'Hz': 1.0, 'MHz': 1e6, 'kHz': 1e3, 'GHz': 1e9}[freq_unit]

    good_lines = []
    with open(csv_path, 'r') as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped[0] in '$#%':
                continue
            tokens = stripped.replace(',', ' ').split()
            try:
                [float(tok) for tok in tokens]
            except ValueError:
                continue
            good_lines.append(tokens)

    if not good_lines:
        raise ValueError(f"No numeric rows found in {csv_path}.")

    arr = np.array(good_lines, dtype=float)
    t_real = arr[:, time_col] * time_scale
    f_dev_real_Hz = arr[:, freq_col] * freq_scale

    order = np.argsort(t_real)
    t_real, f_dev_real_Hz = t_real[order], f_dev_real_Hz[order]

    interp = interp1d(t_real, f_dev_real_Hz, kind='linear',
                       bounds_error=False,
                       fill_value=(f_dev_real_Hz[0], f_dev_real_Hz[-1]))

    duration_s = t_real[-1] - t_real[0]
    print(f"[MOT sim] Loaded experimental jitter trace: {len(t_real)} points, "
          f"spanning {duration_s*1e3:.3f} ms, "
          f"p2p = {(f_dev_real_Hz.max()-f_dev_real_Hz.min())/1e6:.3f} MHz.")

    def delta_bar(t):
        t_real_now = (t * t0) % duration_s if duration_s > 0 else 0.0
        dev_Hz = interp(t_real_now + t_real[0])
        return nominal_detuning_bar + 2 * np.pi * dev_Hz / Gamma_SI

    return delta_bar


def build_laser_beams(norm, delta_bar_or_const, beam_mode=None):
    """USE DIFFERENT NOW"""
    mode = beam_mode if beam_mode is not None else LASER_BEAM_MODE

    if mode == 'gaussian':
        I_peak_SI = 2.0 * GAUSSIAN_POWER_W / (np.pi * GAUSSIAN_WAIST_M ** 2)
        s_peak = I_peak_SI / norm['Isat_SI']
        wb_bar = GAUSSIAN_WAIST_M / norm['x0']
        laserBeams = pylcp.conventional3DMOTBeams(
            k=1.0, pol=+1, s=s_peak, delta=delta_bar_or_const,
            beam_type=pylcp.gaussianBeam, wb=wb_bar,
        )
    elif mode == 'plane_wave':
        laserBeams = pylcp.conventional3DMOTBeams(
            k=1.0, pol=+1, s=S_PER_BEAM, delta=delta_bar_or_const,
            beam_type=pylcp.infinitePlaneWaveBeam,
        )
    else:
        raise ValueError(f"LASER_BEAM_MODE / beam_mode must be 'plane_wave' "
                          f"or 'gaussian', got {mode!r}")
    return laserBeams


def report_gaussian_beam_parameters(norm):
    """Translate mW and mm radius into s0"""
    I_peak_SI = 2.0 * GAUSSIAN_POWER_W / (np.pi * GAUSSIAN_WAIST_M ** 2)
    s_peak = I_peak_SI / norm['Isat_SI']
    print(f"[MOT sim] Gaussian beam mode: {GAUSSIAN_POWER_W*1e3:.1f} mW, "
          f"{GAUSSIAN_WAIST_M*1e3:.2f} mm 1/e^2 waist per beam "
          f"-> peak intensity {I_peak_SI/10:.2f} mW/cm^2 "
          f"-> peak saturation parameter s_0 = {s_peak:.3f} "
          f"(compare to S_PER_BEAM = {S_PER_BEAM} used in 'plane_wave' mode; "
          "note the Gaussian beam's saturation FALLS OFF away from the "
          "beam axis, unlike the plane-wave case, so an atom off-center "
          "sees noticeably less than this peak value).")
    return s_peak


def doppler_temperature_estimate(Gamma_Hz, detuning_Hz, s_per_beam):
    """Low-saturation 1D Doppler-cooling temperature (Foot, ch. 9):

        k_B T = (hbar Gamma / 4) (1 + s0 + (2 delta/Gamma)^2) / (|delta|/Gamma)

    An analytic cross-check only: 1D, two-beam, low saturation, no magnetic
    field. Prefer the Monte Carlo ensemble's velocity spread, especially
    since this transition is broad enough that Zeeman shifts are a
    non-negligible fraction of Gamma near the trap centre.
    """
    delta_over_gamma = detuning_Hz / Gamma_Hz
    kT = (sp_const.hbar * 2 * np.pi * Gamma_Hz / 4) * \
        (1 + s_per_beam + (2 * delta_over_gamma) ** 2) / (2 * abs(delta_over_gamma))
    return kT / sp_const.k


def _equilibrium_force_bypass(eqn, r, v, t=0.):
    """Equilibrium optical + magnetic force at (r, v, t)"""
    Rev, Rijl = eqn.construct_evolution_matrix(r, v, t)
    _, _, VH = np.linalg.svd(Rev)
    Neq = np.real(VH[-1, :])
    Neq = Neq / np.sum(Neq)
    F, _f_laser, _f_mag = eqn.force(r, t, Neq, return_details=True)
    return F


def robust_axial_trap_properties(eqn, norm, axis=2):
    x0 = norm['x0']
    last_exc = None
    last_tb = None

    def _attempt(r_base, eps_pos_um, eps_v_bar):
        eps_pos_bar = (eps_pos_um * 1e-6) / x0
        Fp = _equilibrium_force_bypass(eqn, r_base + np.eye(3)[axis] * eps_pos_bar,
                                        np.zeros(3))[axis]
        Fm = _equilibrium_force_bypass(eqn, r_base - np.eye(3)[axis] * eps_pos_bar,
                                        np.zeros(3))[axis]
        dF = Fp - Fm
        om_val = np.sqrt(-dF / (2 * eps_pos_bar * norm['mass_bar'])) if dF < 0 else 0.0

        v_p = np.eye(3)[axis] * eps_v_bar
        Fvp = _equilibrium_force_bypass(eqn, r_base, v_p)[axis]
        Fvm = _equilibrium_force_bypass(eqn, r_base, -v_p)[axis]
        beta_val = -(Fvp - Fvm) / (2 * eps_v_bar)
        return om_val, beta_val

    for r_base, note_suffix in ((np.array([0., 0., (FALLBACK_R_NUDGE_UM * 1e-6) / x0]), ''),
                                 (np.array([0., 0., 0.]),
                                  'converged only evaluating exactly at r=(0,0,0) , '
                                  'unusual, worth double-checking this result')):
        for eps_pos_um in EPS_POSITION_UM_LADDER:
            for eps_v_bar in EPS_VELOCITY_BAR_LADDER:
                try:
                    om_val, beta_val = _attempt(r_base, eps_pos_um, eps_v_bar)
                    if om_val != 0 and not np.isnan(om_val) and not np.isnan(beta_val):
                        return om_val, beta_val, True, note_suffix
                except Exception as e:
                    last_exc = e
                    last_tb = traceback.format_exc()

    if last_exc is not None:
        note = f'{type(last_exc).__name__}: {last_exc}\n{last_tb}'
    else:
        note = 'all attempts returned zero/NaN without raising'
    return np.nan, np.nan, False, note


def run_detuning_sensitivity_scan(norm, hamiltonian, magField, nominal_detuning_bar,
                                   jitter_amp_bar, out_dir):

    print("\n" + "=" * 70)
    print("Detuning sensitivity scan (fixed-detuning, across the measured "
          "~4 MHz jitter range)")
    print("=" * 70)

    detunings_bar = np.linspace(nominal_detuning_bar - jitter_amp_bar,
                                 nominal_detuning_bar + jitter_amp_bar,
                                 N_DETUNING_SCAN_POINTS)
    detunings_Hz = detunings_bar * norm['Gamma_SI'] / (2 * np.pi)

    omega_z, beta_z, T_doppler = [], [], []
    for det_bar, det_Hz in zip(detunings_bar, detunings_Hz):
        laserBeams = build_laser_beams(norm, det_bar)
        eqn = pylcp.rateeq(laserBeams, magField, hamiltonian,
                            a=np.array([0., 0., -sp_const.g * norm['t0'] ** 2 / norm['x0']]),
                            include_mag_forces=True)
        eqn.set_initial_position(np.array([0., 0., 0.]))
        om_val, beta_val, converged, note = robust_axial_trap_properties(eqn, norm, axis=2)
        if not converged:
            print(f"  [warn] detuning {det_Hz/1e6:.2f} MHz: could not converge "
                  f"even after the full eps ladder ({note}); recording NaN "
                  "for this point only , the rest of the scan is unaffected.")
        elif note:
            print(f"  [note] detuning {det_Hz/1e6:.2f} MHz: {note}.")
        omega_z.append(om_val)
        beta_z.append(beta_val)
        T_doppler.append(doppler_temperature_estimate(norm['Gamma_SI'] / 2 / np.pi,
                                                        det_Hz, S_PER_BEAM))
        print(f"  detuning = {det_Hz/1e6:+7.2f} MHz "
              f"({det_bar:+.3f} Gamma): omega_z_bar={omega_z[-1]:.4g}, "
              f"beta_z_bar={beta_z[-1]:.4g}, T_doppler={T_doppler[-1]*1e6:.2f} uK")

    omega_z_Hz = np.array(omega_z) * norm['Gamma_SI'] / 2 / np.pi
    T_doppler = np.array(T_doppler)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    axes[0].plot(detunings_Hz / 1e6, omega_z_Hz, 'o-')
    axes[0].axvline(nominal_detuning_bar * norm['Gamma_SI'] / 2 / np.pi / 1e6,
                     color='gray', ls='--', lw=0.8, label='nominal lock point')
    axes[0].set_xlabel('detuning (MHz)')
    axes[0].set_ylabel('axial trap frequency (Hz)')
    axes[0].legend(fontsize=8)

    axes[1].plot(detunings_Hz / 1e6, beta_z, 'o-', color='C1')
    axes[1].set_xlabel('detuning (MHz)')
    axes[1].set_ylabel(r'axial damping coeff. $\beta/(\hbar k^2)$ (dimensionless)')

    axes[2].plot(detunings_Hz / 1e6, T_doppler * 1e6, 'o-', color='C2')
    axes[2].set_xlabel('detuning (MHz)')
    axes[2].set_ylabel('analytic Doppler T estimate (uK)')

    fig.suptitle(f'Sensitivity to the measured ~{JITTER_PP_HZ/1e6:.1f} MHz laser '
                 'frequency jitter')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'detuning_sensitivity_scan.png'), dpi=150)
    plt.close(fig)

    return {
        'detunings_Hz': detunings_Hz.tolist(),
        'omega_z_Hz': omega_z_Hz.tolist(),
        'beta_z_bar': list(np.asarray(beta_z, dtype=float)),
        'T_doppler_K': T_doppler.tolist(),
    }


def run_force_profile(norm, hamiltonian, magField, delta_bar, out_dir):
    """Force-vs-position and force-vs-velocity cuts through the trap centre
    along z."""
    laserBeams = build_laser_beams(norm, delta_bar)
    eqn = pylcp.rateeq(laserBeams, magField, hamiltonian,
                        a=np.array([0., 0., -sp_const.g * norm['t0'] ** 2 / norm['x0']]),
                        include_mag_forces=True)

    z_mm = np.linspace(-20.0, 20.0, 161)
    z = z_mm * 1e-3 / norm['x0']
    R = np.array([np.zeros_like(z), np.zeros_like(z), z])
    V = np.zeros((3,) + z.shape)
    eqn.generate_force_profile(R, V, name='Fz_vs_z', progress_bar=False)

    v = np.linspace(-15, 15, 121)
    Rv = np.zeros((3,) + v.shape)
    Vv = np.array([np.zeros_like(v), np.zeros_like(v), v])
    eqn.generate_force_profile(Rv, Vv, name='Fz_vs_v', progress_bar=False)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(z * norm['x0'] * 1e3, eqn.profile['Fz_vs_z'].F[2])
    axes[0].set_xlabel('z (mm)')
    axes[0].set_ylabel(r'$F_z / (\hbar k \Gamma)$')
    axes[0].set_title('Spatial restoring force (v=0)')
    axes[0].axhline(0, color='gray', lw=0.5)

    axes[1].plot(v * norm['v0'], eqn.profile['Fz_vs_v'].F[2], color='C1')
    axes[1].set_xlabel('v (m/s)')
    axes[1].set_ylabel(r'$F_z / (\hbar k \Gamma)$')
    axes[1].set_title('Velocity damping force (r=0)')
    axes[1].axhline(0, color='gray', lw=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'force_profiles.png'), dpi=150)
    plt.close(fig)

    CAPTURE_FORCE_THRESHOLD_FRAC = 0.05
    F_v = eqn.profile['Fz_vs_v'].F[2]
    pos_mask = v > 0
    F_pos = -F_v[pos_mask]
    v_pos = v[pos_mask]
    F_peak = np.max(F_pos)
    below_threshold = F_pos < CAPTURE_FORCE_THRESHOLD_FRAC * F_peak
    peak_idx = np.argmax(F_pos)
    idx_candidates = np.where(below_threshold[peak_idx:])[0]
    if len(idx_candidates) > 0:
        idx_break = peak_idx + idx_candidates[0]
        hit_edge = False
    else:
        idx_break = len(v_pos) - 1
        hit_edge = True
    v_capture_bar = v_pos[idx_break]
    v_capture_ms = v_capture_bar * norm['v0']
    print(f"\n[MOT sim] Approximate capture velocity (force drops below "
          f"{CAPTURE_FORCE_THRESHOLD_FRAC:.0%} of peak): "
          f"{v_capture_bar:.2f} Gamma/k = {v_capture_ms:.2f} m/s"
          + (" *** hit the edge of the scanned v-range , widen the `v = "
             "np.linspace(...)` range above and re-run if you need this "
             "number to be reliable ***" if hit_edge else ""))
    print("[MOT sim] (For reference, your group's own OOT paper reports "
          "MOT capture velocities of order 10 m/s for a similar Yb blue "
          "MOT with a ~11 G/cm gradient , a reasonable sanity-check scale "
          "for this number, though methods differ; see the paper's Fig. 2.)")

    return {'v_capture_ms': float(v_capture_ms), 'v_capture_hit_edge': hit_edge}

_mc_worker_state = {}


def _mc_detuning_spec(delta_bar_or_const, nominal_detuning_bar):
    """Convert delta_bar_or_const into something safe to send to workers for
    multithreading."""
    if not callable(delta_bar_or_const):
        return delta_bar_or_const
    if JITTER_MODE == 'synthetic':
        return {'kind': 'synthetic', 'nominal_detuning_bar': nominal_detuning_bar}
    elif JITTER_MODE == 'experimental':
        return {'kind': 'experimental', 'nominal_detuning_bar': nominal_detuning_bar}
    else:
        raise ValueError(
            "Parallel Monte Carlo with a time-varying (callable) detuning "
            f"needs JITTER_MODE to be 'synthetic' or 'experimental' so each "
            f"worker can rebuild an equivalent callable , got JITTER_MODE="
            f"{JITTER_MODE!r}. Set MC_PARALLEL=False to fall back to the "
            "single-process path instead, which has no such restriction.")


def _mc_worker_init(detuning_spec):
    """for mc sim    """
    global _mc_worker_state
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        norm = build_normalization()
        hamiltonian = build_hamiltonian(norm)
        magField, _, _ = build_magfield(norm)
        if isinstance(detuning_spec, dict):
            if detuning_spec['kind'] == 'synthetic':
                delta, _ = make_synthetic_jitter(detuning_spec['nominal_detuning_bar'],
                                                  norm['Gamma_SI'], norm['t0'])
            elif detuning_spec['kind'] == 'experimental':
                delta = load_experimental_jitter(EXPERIMENTAL_JITTER_CSV,
                                                  detuning_spec['nominal_detuning_bar'],
                                                  norm['Gamma_SI'], norm['t0'])
            else:
                raise ValueError(f"Unrecognized detuning_spec: {detuning_spec!r}")
        else:
            delta = detuning_spec
        laserBeams = build_laser_beams(norm, delta)
        eqn = pylcp.rateeq(laserBeams, magField, hamiltonian,
                            a=np.array([0., 0., -sp_const.g * norm['t0'] ** 2 / norm['x0']]),
                            include_mag_forces=True)
    _mc_worker_state['norm'] = norm
    _mc_worker_state['eqn'] = eqn


def _mc_worker_run_one(args):
    """"""
    atom_index, v0_bar, tmax_bar, seed = args
    eqn = _mc_worker_state['eqn']
    rng = np.random.default_rng(seed)
    eqn.set_initial_position(np.array([0., 0., 0.]))
    eqn.set_initial_velocity(v0_bar)
    eqn.set_initial_pop(np.array([1., 0., 0., 0.]))
    t_start = time.time()
    sol = eqn.evolve_motion([0, tmax_bar], random_recoil=True,
                             progress_bar=False, max_step=5.0, rng=rng)
    elapsed = time.time() - t_start
    return {
        'atom_index': atom_index,
        't_bar': np.asarray(sol.t),
        'r_bar': np.asarray(sol.r),
        'v_bar': np.asarray(sol.v),
        'success': bool(sol.success),
        'status': int(sol.status),
        'message': str(sol.message),
        'elapsed_s': elapsed,
        'v0_bar': v0_bar,
    }


def run_monte_carlo_ensemble(norm, hamiltonian, magField, delta_bar_or_const,
                              label, out_dir, n_atoms=N_MONTE_CARLO_ATOMS,
                              tmax_bar=MC_TMAX_BAR, seed=RANDOM_SEED,
                              nominal_detuning_bar=None,
                              parallel=None, n_workers=None):
    """Stochastic trajectory simulation (rate-equation Monte Carlo,
    random_recoil=True)    """
    parallel = MC_PARALLEL if parallel is None else parallel
    n_workers = MC_N_WORKERS if n_workers is None else n_workers
    safe_label = label.replace(' ', '_')

    print(f"\n[MOT sim] Running Monte Carlo ensemble '{label}': "
          f"{n_atoms} atoms, tmax = {tmax_bar:.1f}/Gamma "
          f"= {tmax_bar*norm['t0']*1e3:.3f} ms each"
          f"{f', across {n_workers} parallel worker processes' if parallel and n_atoms > 1 else ''}...")

    rng = np.random.default_rng(seed)
    v0_list = []
    for i in range(n_atoms):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        v0_list.append(MC_INITIAL_SPEED_BAR * direction)
    atom_seeds = rng.integers(0, 2**63 - 1, size=n_atoms)
    tasks = [(i, v0_list[i], tmax_bar, int(atom_seeds[i])) for i in range(n_atoms)]

    t_wall_start = time.time()
    results_by_atom = {}

    if parallel and n_atoms > 1 and n_workers > 1:
        detuning_spec = _mc_detuning_spec(delta_bar_or_const, nominal_detuning_bar)
        ctx = mp.get_context('spawn')
        n_workers_eff = min(n_workers, n_atoms)
        with ctx.Pool(processes=n_workers_eff, initializer=_mc_worker_init,
                      initargs=(detuning_spec,)) as pool:
            n_done = 0
            for res in pool.imap_unordered(_mc_worker_run_one, tasks):
                results_by_atom[res['atom_index']] = res
                n_done += 1
                elapsed = time.time() - t_wall_start
                print(f"    ...{n_done}/{n_atoms} trajectories done "
                      f"({elapsed/n_done:.1f} s/atom average across "
                      f"{n_workers_eff} workers, ~{elapsed/n_done*(n_atoms-n_done)/n_workers_eff:.0f} s remaining)")
    else:
        _mc_worker_state['eqn'] = pylcp.rateeq(
            build_laser_beams(norm, delta_bar_or_const), magField, hamiltonian,
            a=np.array([0., 0., -sp_const.g * norm['t0'] ** 2 / norm['x0']]),
            include_mag_forces=True)
        for task in tasks:
            res = _mc_worker_run_one(task)
            results_by_atom[res['atom_index']] = res
            i = task[0]
            elapsed = time.time() - t_wall_start
            print(f"    ...{i+1}/{n_atoms} trajectories done "
                  f"({elapsed/(i+1):.1f} s/atom, ~{elapsed/(i+1)*(n_atoms-i-1):.0f} s remaining)")
        _mc_worker_state.pop('eqn', None)

    results = [results_by_atom[i] for i in range(n_atoms)]
    n_converged = sum(1 for r in results if r['success'])
    if n_converged < n_atoms:
        print(f"[MOT sim] '{label}': {n_atoms - n_converged}/{n_atoms} trajectories did NOT converge")

    v_ss_bar = []
    for r in results:
        if not r['success']:
            continue
        n = len(r['t_bar'])
        v_ss_bar.append(r['v_bar'][:, n // 2:])
    v_ss_bar = np.concatenate(v_ss_bar, axis=1)
    v_ss_ms = v_ss_bar * norm['v0']
    mean_v2 = np.mean(v_ss_ms ** 2)
    T_mc = norm['mass_SI'] * mean_v2 / sp_const.k

    npz_path = None
    csv_path = None
    if MC_SAVE_DATA:
        npz_path = os.path.join(out_dir, f'mc_trajectories_{safe_label}.npz')
        npz_payload = {
            'n_atoms': n_atoms,
            'label': label,
            'success': np.array([r['success'] for r in results]),
            'status': np.array([r['status'] for r in results]),
            'v0_bar': np.array([r['v0_bar'] for r in results]),
            'elapsed_s': np.array([r['elapsed_s'] for r in results]),
            'x0_m': norm['x0'], 't0_s': norm['t0'], 'v0_ms': norm['v0'],
            'mass_SI': norm['mass_SI'], 'T_mc_K': T_mc,
        }
        for i, r in enumerate(results):
            npz_payload[f't_bar_{i}'] = r['t_bar']
            npz_payload[f'r_bar_{i}'] = r['r_bar']
            npz_payload[f'v_bar_{i}'] = r['v_bar']
        np.savez(npz_path, **npz_payload)
        print(f"[MOT sim] wrote {npz_path} (full trajectories, all {n_atoms} atoms)")

        csv_path = os.path.join(out_dir, f'mc_trajectories_{safe_label}_summary.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['atom_index', 'converged', 'status', 'message',
                              'v0_x_ms', 'v0_y_ms', 'v0_z_ms', 'initial_speed_ms',
                              'final_z_mm', 'final_vz_ms', 'n_timesteps', 'wall_time_s'])
            for i, r in enumerate(results):
                v0_ms = r['v0_bar'] * norm['v0']
                writer.writerow([
                    i, r['success'], r['status'], r['message'],
                    f"{v0_ms[0]:.4f}", f"{v0_ms[1]:.4f}", f"{v0_ms[2]:.4f}",
                    f"{np.linalg.norm(v0_ms):.4f}",
                    f"{r['r_bar'][2, -1]*norm['x0']*1e3:.4f}",
                    f"{r['v_bar'][2, -1]*norm['v0']:.4f}",
                    len(r['t_bar']), f"{r['elapsed_s']:.2f}",
                ])
        print(f"[MOT sim] wrote {csv_path} (per-atom summary)")

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for r in results[:min(8, len(results))]:
        t_ms = r['t_bar'] * norm['t0'] * 1e3
        style = '-' if r['success'] else '--'
        color = None if r['success'] else 'crimson'
        axes[0].plot(t_ms, r['r_bar'][2] * norm['x0'] * 1e3, style, color=color, lw=0.7)
        axes[1].plot(t_ms, r['v_bar'][2] * norm['v0'], style, color=color, lw=0.7)
    axes[0].set_ylabel('z (mm)')
    axes[1].set_ylabel(r'$v_z$ (m/s)')
    axes[1].set_xlabel('time (ms)')
    title = f"Sample trajectories , {label}"
    if n_converged < n_atoms:
        title += ' (dashed red = did not converge)'
    axes[0].set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'trajectories_{safe_label}.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(v_ss_ms.flatten(), bins=40, density=True, alpha=0.6,
            label=f'simulated ({n_converged}/{n_atoms} converged trajectories)')
    v_axis = np.linspace(v_ss_ms.min(), v_ss_ms.max(), 200)
    mb = np.sqrt(norm['mass_SI'] / (2 * np.pi * sp_const.k * T_mc)) * \
        np.exp(-norm['mass_SI'] * v_axis ** 2 / (2 * sp_const.k * T_mc))
    ax.plot(v_axis, mb, 'r-', label=f'1D Maxwell-Boltzmann fit\nT={T_mc*1e6:.1f} uK')
    ax.set_xlabel('velocity component (m/s)')
    ax.set_ylabel('probability density')
    ax.legend(fontsize=8)
    ax.set_title(f"Steady-state velocity distribution , {label}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'velocity_distribution_{safe_label}.png'), dpi=150)
    plt.close(fig)

    return {'label': label, 'T_mc_K': float(T_mc), 'n_atoms': n_atoms,
            'n_converged': n_converged,
            'tmax_ms': float(tmax_bar * norm['t0'] * 1e3),
            'npz_path': npz_path, 'csv_path': csv_path}


def main():
    OUTPUT_DIR = runlog.new_run_dir(RUN_NAME)
    print(f"[MOT sim] Writing this run to: {OUTPUT_DIR}")
    np.random.seed(RANDOM_SEED)

    norm = build_normalization()
    hamiltonian = build_hamiltonian(norm)
    magField, field_source, grad_z_G_per_cm = build_magfield(norm)

    nominal_detuning_bar = 2 * np.pi * NOMINAL_DETUNING_HZ / norm['Gamma_SI']
    print(f"\n[MOT sim] Nominal detuning: {NOMINAL_DETUNING_HZ/1e6:.2f} MHz "
          f"= {nominal_detuning_bar:.3f} Gamma")
    print(f"[MOT sim] Laser beam mode: {LASER_BEAM_MODE}")
    gaussian_s_peak = None
    if LASER_BEAM_MODE == 'gaussian':
        gaussian_s_peak = report_gaussian_beam_parameters(norm)

    results = {
        'config_snapshot': {
            'field_source_used': field_source,
            'axial_gradient_G_per_cm': grad_z_G_per_cm,
            'LASER_BEAM_MODE': LASER_BEAM_MODE,
            'S_PER_BEAM': S_PER_BEAM,
            'GAUSSIAN_WAIST_M': GAUSSIAN_WAIST_M,
            'GAUSSIAN_POWER_W': GAUSSIAN_POWER_W,
            'gaussian_s_peak': gaussian_s_peak,
            'NOMINAL_DETUNING_HZ': NOMINAL_DETUNING_HZ,
            'JITTER_MODE': JITTER_MODE,
            'JITTER_PP_HZ': JITTER_PP_HZ,
            'Isat_mW_cm2': norm['Isat_mW_cm2'],
            'wavelength_nm': YB174_WAVELENGTH_M * 1e9,
            'linewidth_MHz': YB174_LINEWIDTH_HZ / 1e6,
        }
    }

    results['force_profile'] = run_force_profile(
        norm, hamiltonian, magField, nominal_detuning_bar, OUTPUT_DIR)

    jitter_amp_bar = (JITTER_PP_HZ / 2) * 2 * np.pi / norm['Gamma_SI']
    results['detuning_scan'] = run_detuning_sensitivity_scan(
        norm, hamiltonian, magField, nominal_detuning_bar, jitter_amp_bar, OUTPUT_DIR)

    beta_nom = results['detuning_scan']['beta_z_bar'][N_DETUNING_SCAN_POINTS // 2]
    if beta_nom and beta_nom > 0:
        tau_damping_bar = norm['mass_bar'] / beta_nom
        print(f"\n[MOT sim] Estimated velocity-damping time constant at nominal "
              f"detuning: tau ~ {tau_damping_bar:.0f} / Gamma "
              f"= {tau_damping_bar*norm['t0']*1e6:.1f} us.")
        print(f"[MOT sim] Your configured MC_TMAX_BAR = {MC_TMAX_BAR:.0f} "
              f"({MC_TMAX_BAR*norm['t0']*1e6:.1f} us) is "
              f"{MC_TMAX_BAR/tau_damping_bar:.2f}x this damping time "
              f"({'looks reasonable, >=3x' if MC_TMAX_BAR/tau_damping_bar >= 3 else 'may be too SHORT , consider increasing MC_TMAX_BAR, see CONFIG comment'}).")

    if JITTER_MODE == 'scan_only':
        print("\n[MOT sim] JITTER_MODE='scan_only': skipping Monte Carlo "
              "trajectory runs. Set JITTER_MODE to 'synthetic' or "
              "'experimental' to also get full stochastic trajectories.")
    else:
        results['mc_no_jitter'] = run_monte_carlo_ensemble(
            norm, hamiltonian, magField, nominal_detuning_bar,
            'no_jitter', OUTPUT_DIR)

        if JITTER_MODE == 'synthetic':
            delta_bar_jittered, _ = make_synthetic_jitter(
                nominal_detuning_bar, norm['Gamma_SI'], norm['t0'])
        elif JITTER_MODE == 'experimental':
            if not os.path.exists(EXPERIMENTAL_JITTER_CSV):
                print(f"\n[MOT sim] *** WARNING: '{EXPERIMENTAL_JITTER_CSV}' not "
                      "found; falling back to the synthetic jitter model for "
                      "this run. ***")
                delta_bar_jittered, _ = make_synthetic_jitter(
                    nominal_detuning_bar, norm['Gamma_SI'], norm['t0'])
            else:
                delta_bar_jittered = load_experimental_jitter(
                    EXPERIMENTAL_JITTER_CSV, nominal_detuning_bar,
                    norm['Gamma_SI'], norm['t0'])
        else:
            raise ValueError(f"Unknown JITTER_MODE: {JITTER_MODE!r}")

        results['mc_with_jitter'] = run_monte_carlo_ensemble(
            norm, hamiltonian, magField, delta_bar_jittered,
            f'with_{JITTER_MODE}_jitter', OUTPUT_DIR,
            nominal_detuning_bar=nominal_detuning_bar)

        dT = (results['mc_with_jitter']['T_mc_K'] - results['mc_no_jitter']['T_mc_K']) * 1e6
        print(f"\n[MOT sim] Temperature change from laser jitter: {dT:+.2f} uK "
              f"({results['mc_no_jitter']['T_mc_K']*1e6:.2f} -> "
              f"{results['mc_with_jitter']['T_mc_K']*1e6:.2f} uK)")

    results_path = os.path.join(OUTPUT_DIR, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[MOT sim] All done. Plots + results.json written to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
