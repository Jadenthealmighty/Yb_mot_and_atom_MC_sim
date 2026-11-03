"""Load-rate figures of Letellier et al., RSI 94, 123203 (2023) with their real
coils and the slowing beam: Figs. 7a, 8, 10, 11."""

import csv
import contextlib
import io
import os

import compare_kaiser as ck
import kaiser_coils as kc
import kaiser_mot as km
import kaiser_nozzle as kn

import matplotlib.pyplot as plt
import numpy as np

import full_trap_sweep as fts
import helpers.yb174_mot_simulation as sim
import runlog

BLUE, ORANGE, GREY = ck.BLUE, ck.ORANGE, ck.GREY
T7, T8 = 400.0, 395.0
WAIST_SCAN_MM = [float(x) for x in os.environ.get("KS_SLOWER_WAISTS", "").split(",") if x]
FIG7_SKIP = bool(int(os.environ.get("KS_SKIP_FIG7", 0)))


def mot(name, grad, det_gamma, s_delta, P_slow_mW, **kw):
    """Paper s_delta -> 19/12 mW split, as in Fig. 6 (s_delta 0.012 <-> 100 mW)."""
    return km.mot_config(name, grad, det_gamma, 19.0 * s_delta / 0.012,
                         12.0 * s_delta / 0.012,
                         P_slow_mW=P_slow_mW, **kw)


def L_at(data, m, T_c):
    T = np.atleast_1d(np.asarray(T_c, float)) - ck.OVEN_TEMP_OFFSET_C
    out = np.array([km.load_rate(data, m["offs"], m["vc"], m["ts"], t,
                                 kn.flux_into_tubes(t), kn.YB174_ABUNDANCE)
                    for t in T])
    return out if np.ndim(T_c) else float(out[0])


def solve(run, norm, cfg):
    offs, s_mm, v_ms, a, vc, ts = km.capture_map(run, norm, cfg)
    i = vc.shape[0] // 2
    run.say(f"{cfg['name']}: v_c centre {vc[i, i]:.1f} m/s, max {vc.max():.1f} m/s")
    return dict(offs=offs, s_mm=s_mm, v_ms=v_ms, a=a, vc=vc, ts=ts, cfg=cfg)


def fig7_paper():
    with open(os.path.join(ck.PAPER_DIR, "fig7_maps_digitized.csv")) as fh:
        rows = list(csv.DictReader(fh))
    d = np.array(sorted({float(r["delta_MHz"]) for r in rows}))
    g = np.array(sorted({float(r["gradB_G_cm"]) for r in rows}))
    L = np.zeros((d.size, g.size))
    for r in rows:
        L[np.searchsorted(d, float(r["delta_MHz"])),
          np.searchsorted(g, float(r["gradB_G_cm"]))] = float(r["L_atoms_s"])
    return d, g, L


def run_fig7(run, norm, data, tab):
    d, g, LP = fig7_paper()
    LS = np.zeros_like(LP)
    for i, di in enumerate(d):
        for j, gj in enumerate(g):
            cfg = mot(f"fig7_d{di:.0f}_g{gj:.1f}", gj, -di / 29.0, 0.012 * 0.62, 200.0)
            LS[i, j] = L_at(data, solve(run, norm, cfg), T7)
            tab.add("fig7a", "load rate L", f"T={T7:.0f}C Delta={di:.0f}MHz "
                    f"gradB={gj:.1f}G/cm 62mW Pslow=200mW", LP[i, j], LS[i, j],
                    "atoms/s")

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    dg, dd = g[1] - g[0], d[1] - d[0]
    ext = [g[0] - dg / 2, g[-1] + dg / 2, d[0] - dd / 2, d[-1] + dd / 2]
    for a, M, title in ((ax[0], LP, "paper Fig. 7a"), (ax[1], LS, "sim")):
        im = a.imshow(M / 1e8, origin="lower", aspect="auto", extent=ext,
                      cmap="Blues", vmin=0, vmax=max(LP.max(), LS.max()) / 1e8)
        fig.colorbar(im, ax=a, label=r"$L / 10^8$ [atoms/s]")
        a.set_title(title)
    a = ax[2]
    norm_p, norm_s = LP / LP.max(), LS / max(LS.max(), 1e-30)
    im = a.imshow(norm_s - norm_p, origin="lower", aspect="auto", extent=ext,
                  cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=a, label="sim/max - paper/max")
    ip, is_ = np.unravel_index(LP.argmax(), LP.shape), np.unravel_index(LS.argmax(), LS.shape)
    a.plot(g[ip[1]], d[ip[0]], "k^", ms=10, label="paper optimum")
    a.plot(g[is_[1]], d[is_[0]], "ko", mfc="none", ms=12, label="sim optimum")
    a.legend(fontsize=8)
    a.set_title("shape: each map over its own max")
    for a in ax:
        a.set_xlabel(r"$\nabla B$ [G/cm]")
        a.set_ylabel(r"$\Delta$ [MHz]")
    fig.suptitle(f"Fig. 7a: T = {T7:.0f} C, P_MOT = 62 mW, P_slow = 200 mW, "
                 f"slower waist {km.SLOWER_WAIST_M * 1e3:.1f} mm")
    run.figure(fig, "fig7a_load_rate_map")
    return dict(L7=LS)


def run_fig8(run, norm, data, tab):
    sOn, LOn, _ = ck.paper_rows("fig8", "L_mot_slower")
    sOff, LOff, _ = ck.paper_rows("fig8", "L_mot_only")
    on = [L_at(data, solve(run, norm, mot(f"fig8_slow_s{s * 1e3:.2f}", 32.0, -2.0,
                                          s, 300.0)), T8) for s in sOn]
    off = [L_at(data, solve(run, norm, mot(f"fig8_s{s * 1e3:.2f}", 32.0, -2.0,
                                           s, 0.0)), T8) for s in sOff]
    for s, Lp, Ls in zip(sOn, LOn, on):
        tab.add("fig8", "load rate L, slower on", f"T={T8:.0f}C s_delta={s:.4f}",
                Lp, Ls, "atoms/s")
    for s, Lp, Ls in zip(sOff, LOff, off):
        tab.add("fig8", "load rate L, slower off", f"T={T8:.0f}C s_delta={s:.4f}",
                Lp, Ls, "atoms/s")

    fig, a = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    a.semilogy(sOn * 1e3, LOn, "s", color=BLUE, ms=8, label="paper, MOT + slower")
    a.semilogy(sOn * 1e3, on, "s-", color=BLUE, mfc="none", label="sim, MOT + slower")
    a.semilogy(sOff * 1e3, LOff, "^", color=ORANGE, ms=9, label="paper, MOT")
    a.semilogy(sOff * 1e3, off, "^-", color=ORANGE, mfc="none", label="sim, MOT")
    a.set_xlabel(r"$s_\Delta \times 10^3$")
    a.set_ylabel("L [atoms/s]")
    a.set_title(f"Fig. 8: {T8:.0f} C, -2$\\Gamma$, 32 G/cm, P_slow = 300 mW")
    a.legend(fontsize=8)
    run.figure(fig, "fig8_load_rate_vs_power")
    return dict(L8_on=np.array(on), L8_off=np.array(off))


def run_fig10(run, norm, data, tab):
    T10 = ck.t_fig10()
    s10, N10, _ = ck.paper_rows("fig10", "N_inf")
    _, al10, _ = ck.paper_rows("fig10", "alpha")
    L = np.array([L_at(data, solve(run, norm, cfg), T10)
                  for _, cfg in ck.fig10_configs()])
    for s, Lp, Ls in zip(s10, al10 * N10, L):
        tab.add("fig10", "load rate L = alpha N_inf", f"T={T10:.0f}C (inferred) "
                f"s_delta={s:.2f}", Lp, Ls, "atoms/s", "3.1 G/cm, no slower")

    fig, a = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    a.plot(s10, al10 * N10, "^", color=BLUE, ms=9, label=r"paper $\alpha N_\infty$")
    a.plot(s10, L, "o-", color=BLUE, mfc="none", label=f"sim at {T10:.0f} C")
    a.set_yscale("log")
    a.set_xlabel(r"$s_\Delta$")
    a.set_ylabel("L [atoms/s]")
    a.set_title("Fig. 10: 3.1 G/cm, -1.4$\\Gamma$, no slower, oven T inferred")
    a.legend(fontsize=8)
    run.figure(fig, "fig10_load_rate_low_gradient")
    return dict(L10=L)


def run_fig11(run, norm, data, tab):
    T11, LP, _ = ck.paper_rows("fig11", "L_slower")
    temps = np.arange(280, 481, 5.0)
    m_on = solve(run, norm, mot("fig11_slow", 32.0, -2.0, 0.012, 300.0))
    m_off = solve(run, norm, km.mot_config("fig6_100mW", 32.0, -2.0, 19.0, 12.0))
    on, off = L_at(data, m_on, temps), L_at(data, m_off, temps)
    for T, Lp in zip(T11, LP):
        tab.add("fig11", "load rate L, slower on", f"T={T:.0f}C 100mW",
                Lp, float(np.interp(T, temps, on)), "atoms/s")

    scan = {}
    for w in WAIST_SCAN_MM:
        cfg = mot(f"fig11_slow_w{w:g}", 32.0, -2.0, 0.012, 300.0,
                  slow_waist_m=w * 1e-3)
        scan[w] = L_at(data, solve(run, norm, cfg), temps)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    a = ax[0]
    a.semilogy(T11, LP, "o", color=BLUE, ms=8, label="paper L, MOT + slower")
    a.semilogy(temps, on, "-", color=BLUE, lw=2,
               label=f"sim, slower w = {km.SLOWER_WAIST_M * 1e3:.1f} mm")
    a.semilogy(temps, off, "--", color=GREY, lw=1.5, label="sim, slower off")
    for w, Ls in scan.items():
        a.semilogy(temps, Ls, ":", lw=1.2, label=f"sim, slower w = {w:g} mm")
    a.set_xlabel("nozzle temperature [C]")
    a.set_ylabel("L [atoms/s]")
    a.set_title("Fig. 11: 100 mW, -2$\\Gamma$, 32 G/cm, P_slow = 300 mW")
    a.legend(fontsize=8)
    a = ax[1]
    a.plot(temps, on / off, "-", color=BLUE, lw=2, label="sim L_on / L_off")
    a.axhline(1, color=GREY, lw=0.8)
    a.set_xlabel("nozzle temperature [C]")
    a.set_ylabel("slower gain")
    a.set_title("Slower gain at the Fig. 11 setting")
    a.legend(fontsize=8)
    run.figure(fig, "fig11_load_rate_vs_temperature")
    return dict(L11_on=on, L11_off=off, temps=temps)


def fig_coil_field(run):
    s = np.linspace(-fts.NOZZLE_TO_TRAP_MM, fts.S_PAST_TRAP_MM, 1200)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    top = 0.0
    for g, col in ((32.0, BLUE), (3.1, ORANGE)):
        B = kc.field_function(g)(np.outer(fts.U_HAT, s * 1e-3))
        Bm = np.linalg.norm(B, axis=0) * 1e4
        top = max(top, Bm.max())
        ax[0].plot(s, Bm, color=col, lw=2,
                   label=f"{g:g} G/cm ({kc.current_for_gradient(g):.2f} A)")
        ax[0].plot(s, np.abs(s) * g / 20, ":", color=col, lw=1)
    ax[0].set_xlim(-250, fts.S_PAST_TRAP_MM)
    ax[0].set_ylim(0, 1.6 * top)
    ax[0].set_xlabel("position along the atom beam [mm]")
    ax[0].set_ylabel("|B| [G]")
    ax[0].set_title("Coil field on the atom beam (dotted: linear quadrupole)")
    ax[0].legend(fontsize=9)
    cfg = km.mot_config("f", 32.0, -2.0, 19.0, 12.0, P_slow_mW=300.0)
    v_res = km.slower_v_res(cfg)
    B = np.linalg.norm(kc.field_function(32.0)(np.outer(fts.U_HAT, s * 1e-3)), axis=0)
    shift = 1.4e10 * B * sim.YB174_WAVELENGTH_M
    ax[1].plot(s, v_res + km.SLOWER_POL * np.sign(s) * shift, color=BLUE, lw=2)
    ax[1].axhline(v_res, color=GREY, lw=0.8)
    ax[1].set_xlim(-250, fts.S_PAST_TRAP_MM)
    ax[1].set_xlabel("position along the atom beam [mm]")
    ax[1].set_ylabel("slower resonant velocity [m/s]")
    ax[1].set_title(f"Slower at -2$\\Gamma$ - 220 MHz, pol {km.SLOWER_POL:+d}, 32 G/cm")
    run.figure(fig, "coil_field_on_atom_beam")


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        norm = sim.build_normalization()
    params = dict(field=km.FIELD_MODEL, coils=kc.GEOMETRY,
                  grad_per_amp_G_cm=kc.gradient_per_amp_G_cm(),
                  slower_waist_mm=km.SLOWER_WAIST_M * 1e3, slower_pol=km.SLOWER_POL,
                  slower_angle_mrad=km.SLOWER_ANGLE_MRAD,
                  slower_offset_MHz=km.SLOWER_OFFSET_HZ / 1e6,
                  oven_temp_offset_C=ck.OVEN_TEMP_OFFSET_C,
                  waist_scan_mm=WAIST_SCAN_MM, skip_fig7=FIG7_SKIP)
    with runlog.start("kaiser-load-rates", params=params,
                      note="Letellier et al. 2023 load rates, real coils + slower") as run:
        tr = kn.trace()
        data = kn.trap_data(tr, fts.TRAP_HALF_MM)
        fig_coil_field(run)
        tab = ck.Table()
        res = {}
        res.update(run_fig11(run, norm, data, tab))
        res.update(run_fig8(run, norm, data, tab))
        res.update(run_fig10(run, norm, data, tab))
        if not FIG7_SKIP:
            res.update(run_fig7(run, norm, data, tab))
        tab.write(run.out("kaiser_load_rates.csv"))
        run.file("kaiser_load_rates.csv", "paper vs sim load rates")
        run.finish(summary=res)


if __name__ == "__main__":
    main()
