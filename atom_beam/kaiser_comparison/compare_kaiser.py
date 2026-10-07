"""Compare our oven -> nozzle -> MOT pipeline against Letellier et al.,
RSI 94, 123203 (2023), slowing beam OFF data only."""

import csv
import contextlib
import io
import math
import os

import kaiser_mot as km
import kaiser_nozzle as kn

import matplotlib.pyplot as plt
import numpy as np

import full_trap_sweep as fts
import helpers.yb174_mot_simulation as sim
import runlog

PAPER_DIR = os.path.join(km.HERE, "paper")
BLUE, ORANGE, GREEN, GREY = "#1f77b4", "#ff7f0e", "#2ca02c", "0.4"
OVEN_TEMP_OFFSET_C = 50.0


def paper_rows(figure, series):
    with open(os.path.join(PAPER_DIR, "kaiser_paper_data.csv")) as fh:
        rows = [r for r in csv.DictReader(fh)
                if r["figure"] == figure and r["series"] == series]
    x = np.array([float(r["x"]) if r["x"] else np.nan for r in rows])
    y = np.array([float(r["y"]) for r in rows])
    e = np.array([float(r["y_err"]) if r["y_err"] else np.nan for r in rows])
    return x, y, e


def log_fit_intercept(T_c, y, slope):
    return float(np.mean(np.log10(y) + slope / (np.asarray(T_c) + 273.15)))


T12, A12, _ = paper_rows("fig12", "alpha")
B2 = log_fit_intercept(T12[-9:], A12[-9:], 5371.0)
T5, ABS5, _ = paper_rows("fig5b", "peak_abs_174_cell")
B1 = log_fit_intercept(T5[T5 <= 450], ABS5[T5 <= 450], 5461.0)


def alpha_hot(T_c):
    return 10 ** (B2 - 5371.0 / (np.asarray(T_c) + 273.15))


def alpha_paper(T_c, s):
    return alpha_hot(T_c) + 6.5 * s / 2


def abs_paper_fit(T_c):
    return 10 ** (B1 - 5461.0 / (np.asarray(T_c) + 273.15))


def fig8_configs():
    s, _, _ = paper_rows("fig8", "L_mot_only")
    return [(si, km.mot_config(f"fig8_s{si * 1e3:.2f}", 32.0, -2.0,
                                19.0 * si / 0.012, 12.0 * si / 0.012)) for si in s]


def fig10_configs():
    s, _, _ = paper_rows("fig10", "N_inf")
    w = math.sqrt(2 * 0.587 / (math.pi * 2.8 * km.ISAT_PAPER_W_M2))
    return [(si, km.mot_config(f"fig10_s{si:.2f}", 3.1, -1.4,
                                587.0 / 6 * si / 0.39, 587.0 / 6 * si / 0.39,
                                waist_m=w)) for si in s]


def t_fig10():
    return 5371.0 / B2 - 273.15


class Table:
    def __init__(self):
        self.rows = []

    def add(self, figure, quantity, condition, paper, simv, unit, note="",
            sim_cal=None):
        r = simv / paper if paper and np.isfinite(simv) and unit != "C" else np.nan
        rc = sim_cal / paper if sim_cal is not None and paper else np.nan
        self.rows.append([figure, quantity, condition, f"{paper:.4g}",
                          f"{simv:.4g}", "" if np.isnan(r) else f"{r:.3g}",
                          "" if sim_cal is None else f"{sim_cal:.4g}",
                          "" if sim_cal is None else f"{rc:.3g}", unit, note])

    def write(self, path):
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["figure", "quantity", "condition", "paper", "sim",
                        "sim_over_paper", "sim_flux_calibrated",
                        "sim_flux_calibrated_over_paper", "unit", "note"])
            w.writerows(self.rows)


def beam_checks(run, tr, tab):
    """Nozzle transmission, flux, divergence, and both absorption probes."""
    W = tr["dx"].size / float(tr["n_launched"])
    f400 = kn.flux_into_tubes(400.0) * W
    tab.add("text", "Clausing transmission", "l=13mm d=250um", 0.026, W, "")
    tab.add("text", "total flux out of nozzle", "T=400C all isotopes", 7e14,
            f400, "atoms/s", f"{len(kn.tube_centres())} tubes vs ~700")
    tab.add("text", "flux per tube", "T=400C all isotopes", 1e12,
            f400 / len(kn.tube_centres()), "atoms/s")

    edges = np.linspace(0, 0.3, 151)
    th = 0.5 * (edges[1:] + edges[:-1])
    jd = kn.angular_density(tr, edges)
    h_sim = kn.hwhm(th, jd) * 1e3
    tb4, th4, _ = paper_rows("fig4b", "theta_half")
    low = tb4 <= 410
    tab.add("fig4b", "angular HWHM theta_1/2", "T<=410C (transparent)",
            float(th4[low].mean()), h_sim, "mrad",
            "sim is molecular-flow only, T independent")
    tab.add("text", "angular HWHM theta_1/2", "geometric 1.68 d/2l", 16.0,
            h_sim, "mrad")

    nu = np.arange(-160, 161, 2.0)
    spec_p = np.genfromtxt(os.path.join(PAPER_DIR, "fig4a_spectra_digitized.csv"),
                           delimiter=",", names=True)
    spectra = {}
    for T in (392, 409, 425, 452, 505):
        a_sim = kn.oven_spectrum(tr, T, nu)
        a_pap = spec_p[f"absorption_pct_T{T}C"]
        spectra[T] = (a_sim, a_pap)
        note = "" if T < 425 else "paper in opaque regime (lambda < 2 l_c)"
        tab.add("fig4a", "556nm peak absorption, oven cross", f"T={T}C",
                np.nanmax(a_pap), a_sim.max(), "percent", note)
        tab.add("fig4a", "556nm absorption FWHM", f"T={T}C",
                2 * kn.hwhm(nu[nu >= 0], np.nan_to_num(a_pap[nu >= 0])),
                2 * kn.hwhm(nu[nu >= 0], a_sim[nu >= 0]), "MHz", note)

    vrms = kn.cell_vperp_rms(tr, 400.0)
    tab.add("fig5", "v_perp rms at cell centre", "T=400C", 6.0, vrms, "m/s",
            "paper: Voigt Gaussian part")

    nu_c = np.linspace(-60, 60, 61)
    T_grid = np.arange(270, 521, 10.0)
    a_cell = np.array([kn.cell_spectrum(tr, T, nu_c).max() for T in T_grid])
    for T, a in zip(T5, ABS5):
        a_s = float(np.interp(T, T_grid, a_cell))
        tab.add("fig5b", "399nm peak absorption 174Yb, cell", f"T={T:.0f}C",
                a, a_s, "percent", "" if T <= 450 else "paper opaque regime")
    fit = T_grid <= 450
    A1_sim = -np.polyfit(1 / (T_grid[fit] + 273.15), np.log10(a_cell[fit]), 1)[0]
    tab.add("fig5b", "log10 slope A1 of cell absorption", "300-450C", 5461.0,
            A1_sim, "K", "sim follows vapour pressure")

    fig, ax = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    a = ax[0, 0]
    a.plot(th * 1e3, jd, color=BLUE, lw=2, label="sim, per solid angle")
    a.axvline(h_sim, color=BLUE, ls="--", lw=1)
    a.axvline(th4[low].mean(), color=ORANGE, ls="--", lw=1.5,
              label=f"paper $\\theta_{{1/2}}$ ({th4[low].mean():.0f} mrad)")
    a.axvline(16, color=GREY, ls=":", lw=1.5, label="paper geometric (16 mrad)")
    a.set_xlim(0, 150)
    a.set_xlabel("polar angle [mrad]")
    a.set_ylabel("normalised density")
    a.set_title(f"Nozzle angular distribution, sim HWHM {h_sim:.1f} mrad")
    a.legend(fontsize=9)
    a = ax[0, 1]
    for T, col in ((392, BLUE), (425, GREEN), (452, "#d62728")):
        a.plot(nu, spectra[T][1], color=col, lw=1.2, alpha=0.6,
               label=f"paper {T} C")
        a.plot(nu, spectra[T][0], color=col, lw=2, ls="--", label=f"sim {T} C")
    a.set_xlabel("556 nm detuning [MHz]")
    a.set_ylabel("absorption [%]")
    a.set_yscale("log")
    a.set_ylim(5e-3, 10)
    a.set_title("Oven cross spectroscopy, 11 cm after nozzle (Fig. 4a)")
    a.legend(fontsize=8, ncol=2)
    a = ax[1, 0]
    a.semilogy(T5, ABS5, "o", color=ORANGE, label="paper Fig. 5b")
    a.semilogy(T_grid, abs_paper_fit(T_grid), ":", color=ORANGE,
               label="paper fit, A1 = 5461 K")
    a.semilogy(T_grid, a_cell, "-", color=BLUE, lw=2,
               label=f"sim (A1 = {A1_sim:.0f} K)")
    a.set_xlabel("oven temperature [C]")
    a.set_ylabel("peak 174Yb absorption at cell [%]")
    a.set_title("Cell spectroscopy at 399 nm")
    a.legend(fontsize=9)
    a = ax[1, 1]
    nu2 = np.linspace(-120, 120, 121)
    for T, col in ((350, BLUE), (400, GREEN), (450, "#d62728")):
        a.plot(nu2, kn.cell_spectrum(tr, T, nu2), color=col, lw=2,
               label=f"sim {T} C")
    a.set_xlabel("399 nm detuning [MHz]")
    a.set_ylabel("absorption [%]")
    a.set_title(f"Sim cell line shape, v_perp rms {vrms:.1f} m/s (paper 6.0)")
    a.legend(fontsize=9)
    run.figure(fig, "beam_checks")
    return dict(T_grid=T_grid, a_cell=a_cell)


def calib(T_c, beam):
    """Measured / simulated cell absorption at T_c."""
    return float(abs_paper_fit(T_c) / np.interp(T_c, beam["T_grid"], beam["a_cell"]))


def run_mot(run, norm, data, cfg, T_c):
    T_c = T_c - OVEN_TEMP_OFFSET_C
    offs, s_mm, v_ms, a, vc, ts = km.capture_map(run, norm, cfg)
    L = km.load_rate(data, offs, vc, ts, T_c, kn.flux_into_tubes(T_c),
                     kn.YB174_ABUNDANCE)
    run.say(f"{cfg['name']}: s_delta={km.s_delta(cfg):.4f}, v_c centre "
            f"{vc[vc.shape[0] // 2, vc.shape[1] // 2]:.1f} m/s, "
            f"L({T_c:.0f} C) = {L:.3e}/s")
    return dict(offs=offs, s_mm=s_mm, v_ms=v_ms, a=a, vc=vc, ts=ts, L=L)


def rate_curve(data, m, temps):
    return np.array([km.load_rate(data, m["offs"], m["vc"], m["ts"], T,
                                  kn.flux_into_tubes(T), kn.YB174_ABUNDANCE)
                     for T in temps])


def t_match(data, m, L_target, lo=150.0, hi=550.0):
    """Oven temperature at which the sim loads at L_target."""
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if rate_curve(data, m, [mid])[0] < L_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def mot_checks(run, norm, data, tab, beam):
    T8 = 395.0
    c8 = calib(T8, beam)
    sP, LP, _ = paper_rows("fig8", "L_mot_only")
    _, NP, _ = paper_rows("fig8", "N_mot_only")
    L8 = []
    for (s, cfg), Lp, Np in zip(fig8_configs(), LP, NP):
        m = run_mot(run, norm, data, cfg, T8)
        L8.append(m["L"])
        cond = f"T=395C s_delta={s:.4f} ({cfg['P_h_mW'] * 4 + cfg['P_v_mW'] * 2:.0f} mW)"
        tab.add("fig8", "load rate L", cond, Lp, m["L"], "atoms/s", "",
                m["L"] * c8)
        al = alpha_paper(T8, s)
        tab.add("fig8", "N_st = L_sim / alpha_paper", cond, Np, m["L"] / al,
                "atoms", f"alpha={al:.2f}/s from Fig 12 fit", m["L"] * c8 / al)
        tab.add("fig8", "oven T where sim L = paper L", cond, T8,
                t_match(data, m, Lp), "C")
    L8 = np.array(L8)

    cfg6 = km.mot_config("fig6_100mW", 32.0, -2.0, 19.0, 12.0)
    m6 = run_mot(run, norm, data, cfg6, 387.0)
    T6, N6, e6 = paper_rows("fig6", "N_pmo_on")
    temps = np.arange(280, 481, 5.0)
    L6 = rate_curve(data, m6, temps)
    s6 = km.s_delta(cfg6)
    for T, Np in zip(T6, N6):
        Ls = float(np.interp(T, temps, L6))
        al = alpha_paper(T, 0.012)
        tab.add("fig6", "N_st = L_sim / alpha_paper", f"T={T:.0f}C s_delta=0.012",
                Np, Ls / al, "atoms", f"alpha={al:.2f}/s", Ls * calib(T, beam) / al)
        tab.add("fig6", "L inferred = alpha_paper N_st", f"T={T:.0f}C s_delta=0.012",
                Np * al, Ls, "atoms/s", "", Ls * calib(T, beam))
        tab.add("fig6", "oven T where sim L = paper L", f"T={T:.0f}C s_delta=0.012",
                T, t_match(data, m6, Np * al), "C")

    T10 = t_fig10()
    c10 = calib(T10, beam)
    s10, N10, _ = paper_rows("fig10", "N_inf")
    _, al10, _ = paper_rows("fig10", "alpha")
    m10s = []
    for (s, cfg), Np, al in zip(fig10_configs(), N10, al10):
        m = run_mot(run, norm, data, cfg, T10)
        m10s.append(m)
        cond = f"T={T10:.0f}C (inferred) s_delta={s:.2f} w={cfg['waist_m'] * 1e3:.1f}mm"
        tab.add("fig10", "load rate L = alpha N_inf", cond, al * Np, m["L"],
                "atoms/s", "oven T inferred from alpha_hot = 1/s", m["L"] * c10)
        tab.add("fig10", "oven T where sim L = paper L", cond, T10,
                t_match(data, m, al * Np), "C")
    L10 = np.array([m["L"] for m in m10s])

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    a = ax[0]
    a.semilogy(sP * 1e3, LP, "^", color=BLUE, ms=9, label="paper L (MOT only)")
    a.semilogy(sP * 1e3, L8, "o-", color=BLUE, mfc="none", label="sim L")
    # a.semilogy(sP * 1e3, L8 * c8, "s--", color=BLUE, mfc="none", alpha=0.6,
    #            label="sim L, flux calibrated")
    a.semilogy(sP * 1e3, NP, "^", color=ORANGE, ms=9, label="paper N_st")
    a.semilogy(sP * 1e3, L8 / alpha_paper(T8, sP), "o-", color=ORANGE,
               mfc="none", label="sim L / alpha_paper")
    a.set_xlabel(r"$s_\Delta \times 10^3$")
    a.set_ylabel("L [atoms/s], N [atoms]")
    a.set_title("Fig. 8, slowing beam off: 395 C, -2$\\Gamma$, 32 G/cm")
    a.legend(fontsize=8)
    a = ax[1]
    a.semilogy(temps, L6 / alpha_paper(temps, 0.012), "-", color=ORANGE, lw=2,
               label="sim L / alpha_paper")
    cal = np.array([calib(T, beam) for T in temps])
    # a.semilogy(temps, L6 * cal / alpha_paper(temps, 0.012), "--", color=ORANGE,
    #            lw=1.5, label="sim, flux calibrated") # TODO: Get flux calibrated to work
    a.errorbar(T6, N6, e6, fmt="^", color=ORANGE, ms=9, label="paper N_st (Fig. 6)")
    a.semilogy(temps, L6, "-", color=BLUE, lw=2, label="sim L")
    a.semilogy(T6, N6 * alpha_paper(T6, 0.012), "^", color=BLUE, ms=9,
               label="paper alpha N_st")
    a.set_xlabel("oven temperature [C]")
    a.set_ylabel("L [atoms/s], N [atoms]")
    a.set_ylim(1e5, 5e9)
    a.set_title("Fig. 6 plateau before slower on, 100 mW")
    a.legend(fontsize=8)
    run.figure(fig, "mot_fig6_fig8")

    fig, a = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    a.plot(s10, al10 * N10, "^", color=BLUE, ms=9, label=r"paper $\alpha N_\infty$")
    a.plot(s10, L10, "o-", color=BLUE, mfc="none", label=f"sim at {T10:.0f} C")
    a.plot(s10, L10 * c10, "s--", color=BLUE, mfc="none", alpha=0.6,
           label="sim, flux calibrated")
    for dT, ls in ((-15, ":"), (15, "-.")):
        Ls = [km.load_rate(data, m["offs"], m["vc"], m["ts"], T10 + dT,
                           kn.flux_into_tubes(T10 + dT), kn.YB174_ABUNDANCE)
              for m in m10s]
        a.plot(s10, Ls, ls, color=GREY, label=f"sim at {T10 + dT:.0f} C")
    a.set_yscale("log")
    a.set_xlabel(r"$s_\Delta$")
    a.set_ylabel("L [atoms/s]")
    a.set_title("Fig. 10: 3.1 G/cm, -1.4$\\Gamma$, oven T not given")
    a.legend(fontsize=8)
    run.figure(fig, "mot_fig10")

    T0 = T6[0]
    at0 = lambda y: y / np.interp(T0, temps, y)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    a = ax[0]
    a.semilogy(sP * 1e3, LP / LP[0], "^", color=BLUE, ms=9, label="paper L")
    a.semilogy(sP * 1e3, L8 / L8[0], "o-", color=BLUE, mfc="none", label="sim L")
    NS = L8 / alpha_paper(T8, sP)
    a.semilogy(sP * 1e3, NP / NP[0], "^", color=ORANGE, ms=9, label="paper N_st")
    a.semilogy(sP * 1e3, NS / NS[0], "o-", color=ORANGE, mfc="none",
               label="sim L / alpha_paper")
    a.axhline(1, color=GREY, lw=0.8)
    a.set_xlabel(r"$s_\Delta \times 10^3$")
    a.set_ylabel(f"gain over s_delta = {sP[0]:.4f}")
    a.set_title("Fig. 8 gain, slowing beam off: 395 C, -2$\\Gamma$, 32 G/cm")
    a.legend(fontsize=8)
    a = ax[1]
    N6s = L6 / alpha_paper(temps, 0.012)
    # a.semilogy(temps, at0(N6s), "-", color=ORANGE, lw=2, label="sim L / alpha_paper")
    # a.semilogy(temps, at0(N6s * cal), "--", color=ORANGE, lw=1.5,
    #            label="Simulate N_st (trapped atoms)")
    # a.semilogy(T6, N6 / N6[0], e6 / N6[0], marker="^", color=ORANGE, ms=9,
    #            label="paper N_st")
    # a.semilogy(temps, at0(L6), "-", color=BLUE, lw=2, label="sim L")
    a.semilogy(temps, at0(L6 * cal), "--", color=BLUE, lw=1.5,
               label="simulated load rate")
    LP6 = N6 * alpha_paper(T6, 0.012)
    a.semilogy(T6, LP6 / LP6[0], "^", color=BLUE, ms=9, label="paper alpha N_st")
    a.axhline(1, color=GREY, lw=0.8)
    a.set_xlabel("oven temperature [C]")
    a.set_ylabel(f"gain over {T0:.0f} C")
    a.set_title("Fig. 6 gain, 100 mW")
    a.legend(fontsize=8)
    run.figure(fig, "mot_fig6_fig8_gain")

    fig, a = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    LP10 = al10 * N10
    a.plot(s10, LP10 / LP10[0], "^", color=BLUE, ms=9, label=r"paper $\alpha N_\infty$")
    a.plot(s10, L10 / L10[0], "o-", color=BLUE, mfc="none", label=f"sim at {T10:.0f} C")
    for dT, ls in ((-15, ":"), (15, "-.")):
        Ls = np.array([km.load_rate(data, m["offs"], m["vc"], m["ts"], T10 + dT,
                                    kn.flux_into_tubes(T10 + dT), kn.YB174_ABUNDANCE)
                       for m in m10s])
        a.plot(s10, Ls / Ls[0], ls, color=GREY, label=f"sim at {T10 + dT:.0f} C")
    a.axhline(1, color=GREY, lw=0.8)
    a.set_xlabel(r"$s_\Delta$")
    a.set_ylabel(f"gain over s_delta = {s10[0]:.2f}")
    a.set_title("Fig. 10 gain: 3.1 G/cm, -1.4$\\Gamma$")
    a.legend(fontsize=8)
    run.figure(fig, "mot_fig10_gain")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
    for a, m, title in ((ax[0], m6, "32 G/cm, -2$\\Gamma$, 100 mW"),
                        (ax[1], m10s[-1], "3.1 G/cm, -1.4$\\Gamma$, 587 mW")):
        o = m["offs"]
        d = o[1] - o[0]
        im = a.imshow(m["vc"].T, origin="lower", cmap="magma",
                      extent=[o[0] - d / 2, o[-1] + d / 2, o[0] - d / 2, o[-1] + d / 2])
        fig.colorbar(im, ax=a, label="capture velocity [m/s]")
        a.set_xlabel("horizontal offset at trap [mm]")
        a.set_ylabel("vertical offset at trap [mm]")
        a.set_title(title)
    run.figure(fig, "capture_velocity_maps")
    run.figure(fts.fig_phase_portrait(norm, m6["offs"], m6["s_mm"], m6["v_ms"],
                                      m6["a"]), "phase_portrait_fig6_config")
    return dict(L8=L8, L6_387=m6["L"], L10=L10, T10=T10)


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        norm = sim.build_normalization()
    params = dict(nozzle_to_trap_mm=fts.NOZZLE_TO_TRAP_MM,
                  atom_beam_to_mot_deg=fts.BEAM_TILT_DEG,
                  tubes=len(kn.tube_centres()), apertures_m=kn.APERTURES_M,
                  trap_half_mm=fts.TRAP_HALF_MM, offset_step_mm=fts.OFFSET_STEP_MM,
                  fig12_B2=B2, fig5b_B1=B1, fig10_T_inferred_C=t_fig10())
    with runlog.start("kaiser-comparison", params=params,
                      note="Letellier et al. 2023 MOT, slowing beam off") as run:
        tr = kn.trace()
        data = kn.trap_data(tr, fts.TRAP_HALF_MM)
        run.say(f"nozzle: {tr['dx'].size:,} transmitted of "
                f"{int(tr['n_launched']):,}, {data['n_aperture']:,} through "
                f"apertures, {data['n_accept']:,} in the trap window")
        run.figure(fts.fig_geometry(), "beam_and_laser_geometry")
        run.figure(fts.fig_beam_at_chamber(data), "atom_beam_at_trap")
        c = kn.tube_centres() * 1e3
        fig, a = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
        a.scatter(c[:, 0], c[:, 1], s=4, color=BLUE)
        a.add_patch(plt.Circle((0, 0), 9.5, fill=False, color=GREY))
        a.set_aspect("equal")
        a.set_title(f"Nozzle face: {len(c)} tubes, d = 250 um, pitch 340 um")
        a.set_xlabel("y [mm]")
        a.set_ylabel("z [mm]")
        run.figure(fig, "nozzle_face")

        tab = Table()
        beam = beam_checks(run, tr, tab)
        res = mot_checks(run, norm, data, tab, beam)
        tab.write(run.out("kaiser_vs_sim.csv"))
        run.file("kaiser_vs_sim.csv", "comparison table")
        run.finish(summary=dict(
            clausing_W=tr["dx"].size / float(tr["n_launched"]),
            fig8_L_sim=res["L8"], fig6_L_sim_387C=res["L6_387"],
            fig10_L_sim=res["L10"], fig10_T_inferred_C=res["T10"]))


if __name__ == "__main__":
    main()
