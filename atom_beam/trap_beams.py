"""I had to make this cuz pylcp doesn't have our unconventional MOT beam orientations"""


import math

import numpy as np
from scipy.spatial.transform import Rotation

import pylcp

__all__ = ["OffsetGaussianBeam", "DivergingGaussianBeam", "BeamAlignment",
           "tilt_unit_vector", "build_trap_beams", "MOT_KVECS", "MOT_POLS"]

MOT_KVECS = [np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]),
             np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0]),
             np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])]
MOT_NAMES = ["+x", "-x", "+y", "-y", "+z", "-z"]


def MOT_POLS(pol=+1):
    """Polarization signs, in the same order as MOT_KVECS."""
    return [-pol, -pol, -pol, -pol, +pol, +pol]


def _perp_basis(khat):
    khat = np.asarray(khat, dtype=float)
    khat = khat / np.linalg.norm(khat)
    seed = np.zeros(3)


    seed[int(np.argmin(np.abs(khat)))] = 1.0
    e1 = np.cross(khat, seed)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(khat, e1)
    return e1, e2


def tilt_unit_vector(khat, tilt_rad_1, tilt_rad_2):
    """Tilt a unit vector by two angles about the axes perpendicular to it.    """
    khat = np.asarray(khat, dtype=float)
    khat = khat / np.linalg.norm(khat)
    if tilt_rad_1 == 0.0 and tilt_rad_2 == 0.0:
        return khat
    e1, e2 = _perp_basis(khat)
    rot = Rotation.from_rotvec(tilt_rad_1 * e1) * Rotation.from_rotvec(tilt_rad_2 * e2)
    out = rot.apply(khat)
    return out / np.linalg.norm(out)


def _sub_offset(R, offset):
    """R, offset in pylcp convention"""
    R = np.asarray(R, dtype=float)
    if R.ndim == 1:
        return R - offset
    return R - offset.reshape((3,) + (1,) * (R.ndim - 1))


class OffsetGaussianBeam(pylcp.gaussianBeam):
    """Collimated Gaussian beam whose axis need not pass through the origin.
used for slowing beam tolerance sweeps"""

    intensity_sig = None

    def __init__(self, kvec, pol, s, delta, wb, offset=None, **kwargs):
        super().__init__(kvec=kvec, pol=pol, s=s, delta=delta, wb=wb, **kwargs)
        self.offset = (np.zeros(3) if offset is None
                       else np.asarray(offset, dtype=float))

    def intensity(self, R=np.array([0., 0., 0.]), t=0.):
        Rp = np.einsum('ij,j...->i...', self.rmat, _sub_offset(R, self.offset))
        rho_sq = np.sum(Rp[:2] ** 2, axis=0)
        return self.s_max * np.exp(-2 * rho_sq / self.wb ** 2)


class DivergingGaussianBeam(pylcp.laserBeam):
    """For slowing beam sweeps"""

    intensity_sig = None

    def __init__(self, kvec, pol, s, delta, w_optic, optic_position,
                 divergence, focus_distance=0.0, **kwargs):
        if callable(kvec):
            raise TypeError("kvec cannot be a function for a Gaussian beam.")
        super().__init__(kvec=kvec, pol=pol, delta=delta, **kwargs)

        self.con_kvec = np.asarray(kvec, dtype=float)
        self.con_khat = self.con_kvec / np.linalg.norm(self.con_kvec)
        self.con_pol = self.pol(np.array([0., 0., 0.]), 0.)

        self.s_max = s
        self.w_optic = float(w_optic)
        self.divergence = float(divergence)
        self.focus_distance = float(focus_distance)
        self.optic_position = np.asarray(optic_position, dtype=float)

        self.zeta_optic = float(np.dot(self.optic_position, self.con_khat))

        self.define_rotation_matrix()

    def define_rotation_matrix(self):
        th = np.arccos(self.con_khat[2])
        phi = np.arctan2(self.con_khat[1], self.con_khat[0])
        self.rmat = Rotation.from_euler('ZY', [phi, th]).inv().as_matrix()

    def radius(self, zeta):
        """1/e^2 intensity radius at distance `zeta` beyond the optic."""
        return np.sqrt(self.w_optic ** 2
                       + (self.divergence * (zeta - self.focus_distance)) ** 2)

    def intensity(self, R=np.array([0., 0., 0.]), t=0.):
        Rp = np.einsum('ij,j...->i...', self.rmat, np.asarray(R, dtype=float))
        rho_sq = np.sum(Rp[:2] ** 2, axis=0)
        zeta = Rp[2] - self.zeta_optic
        w = self.radius(zeta)
        return self.s_max * (self.w_optic / w) ** 2 * np.exp(-2 * rho_sq / w ** 2)


def diffraction_limited_divergence(w0_m, wavelength_m):
    """Far-field half-angle of an ideal Gaussian beam of waist w0."""
    return wavelength_m / (math.pi * w0_m)


class BeamAlignment:

    def __init__(self, mot_offsets_m=None, mot_tilts_rad=None,
                 slower_tilt_rad=(0.0, 0.0), slower_offset_m=None):
        self.mot_offsets_m = dict(mot_offsets_m or {})
        self.mot_tilts_rad = dict(mot_tilts_rad or {})
        self.slower_tilt_rad = tuple(slower_tilt_rad)
        self.slower_offset_m = (np.zeros(3) if slower_offset_m is None
                                else np.asarray(slower_offset_m, dtype=float))

    @property
    def is_nominal(self):
        return (not self.mot_offsets_m and not self.mot_tilts_rad
                and self.slower_tilt_rad == (0.0, 0.0)
                and not np.any(self.slower_offset_m))

    def describe(self):
        if self.is_nominal:
            return "nominal (all beams perfectly aligned)"
        bits = []
        for i, off in sorted(self.mot_offsets_m.items()):
            bits.append(f"{MOT_NAMES[i]} offset "
                        f"{np.linalg.norm(off) * 1e3:.3f} mm")
        for i, tl in sorted(self.mot_tilts_rad.items()):
            bits.append(f"{MOT_NAMES[i]} tilt "
                        f"{math.degrees(math.hypot(*tl)):.4f} deg")
        if self.slower_tilt_rad != (0.0, 0.0):
            bits.append("slower tilt "
                        f"{math.degrees(math.hypot(*self.slower_tilt_rad)):.4f} deg")
        if np.any(self.slower_offset_m):
            bits.append("slower offset "
                        f"{np.linalg.norm(self.slower_offset_m) * 1e3:.3f} mm")
        return ", ".join(bits)


def build_trap_beams(norm, delta_bar, *, waist_m, power_w, pol=+1,
                     alignment=None,
                     slower=None,
                     wavelength_m=None):
    """Assemble six MOT beams and optionally, the slowing beam.

    norm : dict
        From yb174_mot_simulation.build_normalization()
    delta_bar : float or callable in Gamma units
    waist_m, power_w : float
        1/e^2 radius and power of EACH MOT arm
    alignment : BeamAlignment or None
        None means perfectly aligned.
    slower : dict or None
        Keys: khat (propagation direction),
        delta_bar, power_w, w_optic_m, optic_distance_m (optic sits at
        -optic_distance_m * khat), divergence_rad, focus_distance_m
        (measured from the optic), pol.
    wavelength_m : float, optional
        Only used to report the diffraction-limited divergence floor.

    Returns (pylcp.laserBeams, info dict).
    """
    alignment = alignment or BeamAlignment()
    x0 = norm["x0"]
    isat = norm["Isat_SI"]

    s_peak = 2.0 * power_w / (math.pi * waist_m ** 2) / isat
    wb_bar = waist_m / x0

    beams = []
    pols = MOT_POLS(pol)
    for i, (kvec, p) in enumerate(zip(MOT_KVECS, pols)):
        khat = kvec
        tilt = alignment.mot_tilts_rad.get(i)
        if tilt is not None:
            khat = tilt_unit_vector(khat, tilt[0], tilt[1])
        offset_m = alignment.mot_offsets_m.get(i)
        offset_bar = (np.zeros(3) if offset_m is None
                      else np.asarray(offset_m, dtype=float) / x0)
        beams.append(OffsetGaussianBeam(
            kvec=khat, pol=p, s=s_peak, delta=delta_bar, wb=wb_bar,
            offset=offset_bar))

    info = dict(mot_s_peak=s_peak, mot_waist_m=waist_m,
                mot_power_w=power_w, n_beams=len(beams),
                alignment=alignment.describe())

    if slower is not None:
        khat = np.asarray(slower["khat"], dtype=float)
        khat = khat / np.linalg.norm(khat)
        if alignment.slower_tilt_rad != (0.0, 0.0):
            khat = tilt_unit_vector(khat, *alignment.slower_tilt_rad)

        w_optic = slower["w_optic_m"]
        s_slow = 2.0 * slower["power_w"] / (math.pi * w_optic ** 2) / isat
        optic_pos_m = -slower["optic_distance_m"] * khat + alignment.slower_offset_m

        theta = slower.get("divergence_rad", 0.0)
        if wavelength_m is not None and theta > 0:
            theta_min = diffraction_limited_divergence(w_optic, wavelength_m)
            if theta < theta_min:
                print(f"[trap_beams] WARNING: requested slowing-beam "
                      f"divergence {theta * 1e3:.4f} mrad is below the "
                      f"diffraction limit {theta_min * 1e3:.4f} mrad for a "
                      f"{w_optic * 1e3:.2f} mm waist. Physically "
                      "unrealisable; results will be optimistic.")
            info["slower_diffraction_limit_mrad"] = theta_min * 1e3

        beams.append(DivergingGaussianBeam(
            kvec=khat, pol=slower.get("pol", +1), s=s_slow,
            delta=slower["delta_bar"],
            w_optic=w_optic / x0,
            optic_position=optic_pos_m / x0,
            divergence=theta,
            focus_distance=slower.get("focus_distance_m", 0.0) / x0))

        info.update(slower_s_peak_at_optic=s_slow,
                    slower_w_optic_m=w_optic,
                    slower_divergence_mrad=theta * 1e3,
                    slower_khat=khat.tolist(),
                    n_beams=len(beams))

    return pylcp.laserBeams(beams), info

if __name__ == "__main__":
    import scipy.constants as sp_const

    print("=" * 70)
    print("SELF TEST")
    print("=" * 70)

    lam = 398.9114e-9
    Gamma = 2 * np.pi * 30.2e6
    x0 = lam / (2 * np.pi)
    isat = (np.pi * sp_const.h * sp_const.c * Gamma) / (3 * lam ** 3)
    norm = dict(x0=x0, Isat_SI=isat)

    waist, power = 5e-3, 4.8837e-3
    delta_bar = -0.5

    mine, info = build_trap_beams(norm, delta_bar, waist_m=waist,
                                  power_w=power)
    s_peak = info["mot_s_peak"]
    theirs = pylcp.conventional3DMOTBeams(
        k=1.0, pol=+1, s=s_peak, delta=delta_bar,
        beam_type=pylcp.gaussianBeam, wb=waist / x0)

    print(f"  peak saturation s_0 = {s_peak:.6f}")
    print(f"  beam count: mine {mine.num_of_beams}, pylcp {theirs.num_of_beams}")

    probes = [np.array([0., 0., 0.]),
              np.array([1e-3, -2e-3, 0.5e-3]) / x0,
              np.array([-4e-3, 3e-3, -1e-3]) / x0]

    worst_i, worst_k, worst_p = 0.0, 0.0, 0.0
    for bm, bt in zip(mine.beam_vector, theirs.beam_vector):
        worst_k = max(worst_k, float(np.abs(bm.kvec() - bt.kvec()).max()))
        worst_p = max(worst_p, float(np.abs(bm.con_pol - bt.con_pol).max()))
        for R in probes:
            worst_i = max(worst_i, abs(float(bm.intensity(R, 0.))
                                       - float(bt.intensity(R, 0.))))
    print(f"  max |dk|         = {worst_k:.3e}")
    print(f"  max |dpol|       = {worst_p:.3e}")
    print(f"  max |dintensity| = {worst_i:.3e}")
    assert worst_k < 1e-12 and worst_p < 1e-12 and worst_i < 1e-12, \
        "nominal beam set does NOT reproduce conventional3DMOTBeams"
    print("  -> nominal set reproduces conventional3DMOTBeams exactly. OK")

    al = BeamAlignment(mot_offsets_m={0: np.array([0.0, 1e-3, 0.0])})
    off, _ = build_trap_beams(norm, delta_bar, waist_m=waist, power_w=power,
                              alignment=al)
    R = np.array([0., 0., 0.])
    print(f"\n  +x beam at origin, nominal      : "
          f"{float(mine.beam_vector[0].intensity(R, 0.)):.6f}")
    print(f"  +x beam at origin, 1 mm sideways: "
          f"{float(off.beam_vector[0].intensity(R, 0.)):.6f}  "
          f"(expect {s_peak * math.exp(-2 * (1e-3 / waist) ** 2):.6f})")

    al2 = BeamAlignment(mot_offsets_m={0: np.array([5e-3, 0.0, 0.0])})
    along, _ = build_trap_beams(norm, delta_bar, waist_m=waist,
                                power_w=power, alignment=al2)
    d = abs(float(along.beam_vector[0].intensity(R, 0.))
            - float(mine.beam_vector[0].intensity(R, 0.)))
    print(f"  displacing +x beam ALONG its own k by 5 mm changes intensity "
          f"by {d:.3e} (must be 0)")
    assert d < 1e-12

    k0 = np.array([1.0, 0.0, 0.0])
    k1 = tilt_unit_vector(k0, math.radians(0.1), 0.0)
    ang = math.degrees(math.acos(np.clip(np.dot(k0, k1), -1, 1)))
    print(f"\n  tilt of 0.1 deg -> measured {ang:.6f} deg, "
          f"|k| = {np.linalg.norm(k1):.12f}")
    assert abs(ang - 0.1) < 1e-9 and abs(np.linalg.norm(k1) - 1) < 1e-12

    slow = dict(khat=np.array([-1.0, 0.0, 0.0]), delta_bar=-0.5,
                power_w=power, w_optic_m=5e-3, optic_distance_m=0.5,
                divergence_rad=5e-3)
    bs, sinfo = build_trap_beams(norm, delta_bar, waist_m=waist,
                                 power_w=power, slower=slow,
                                 wavelength_m=lam)
    sb = bs.beam_vector[-1]
    print(f"\n  slowing beam: {bs.num_of_beams} beams total, "
          f"s at optic = {sinfo['slower_s_peak_at_optic']:.4f}")
    for d_mm in (0.0, 500.0, 810.0):
        w = sb.radius(d_mm * 1e-3 / x0) * x0
        r_axis = (sb.optic_position * x0 + (d_mm * 1e-3) * sb.con_khat) / x0
        s_here = float(sb.intensity(r_axis, 0.))
        print(f"    {d_mm:5.0f} mm from optic: w = {w * 1e3:.3f} mm, "
              f"s_peak = {s_here:.4f}, s*w^2 = {s_here * (w * 1e3) ** 2:.4f}")
    print("    (s*w^2 constant along the beam = power conserved)")
    print("\nAll checks passed.")
