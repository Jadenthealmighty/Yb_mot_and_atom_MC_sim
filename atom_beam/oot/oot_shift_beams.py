"""1077 nm shift beams for the opto-optical trap
 SI units, Cartesian |x>,|y>,|z> basis for 1P1.
"""

import numpy as np
import scipy.constants as sp_const

from oot_rate_eq import SPIN1

TAU = 2 * np.pi
HBAR = sp_const.hbar
C = sp_const.c
MU_B = sp_const.physical_constants["Bohr magneton"][0]

E_1P1_CM = 25068.222
E_6S7S_CM = 34350.65
GAMMA_S = 3.0e7
GF_1P1 = 1.035

K_S = 100 * TAU * (E_6S7S_CM - E_1P1_CM)
LAMBDA_S = TAU / K_S
ISAT_S = HBAR * C * K_S ** 3 * GAMMA_S / (12 * np.pi)
SHIFT_COEFF = np.pi * GAMMA_S / (2 * HBAR * C * K_S ** 3)


def polarization_state(eps, phi):
    cph = np.cos(phi)
    sph = np.sin(phi)
    return np.array([cph - 1j * eps * sph, sph + 1j * eps * cph, 0]) / np.sqrt(1 + eps ** 2)


SIGMA_PLUS = polarization_state(+1, 0)
SIGMA_MINUS = polarization_state(-1, 0)


def rotation(theta, phi):
    """Laser axes to lab axes."""
    cth = np.cos(theta)
    sth = np.sin(theta)
    cph = np.cos(phi)
    sph = np.sin(phi)
    return (np.array([[cph, sph, 0], [-sph, cph, 0], [0, 0, 1]])
            @ np.array([[cth, 0, -sth], [0, 1, 0], [sth, 0, cth]]))


def displaced_gaussian(I, w, theta, phi, r0=np.zeros(3), pol=SIGMA_PLUS, delta=0.0):
    """One circular component, peak intensity I. r0, pol in the laser frame."""
    Rmat = rotation(theta, phi)
    pol_lab = Rmat @ pol
    return {
        'I': I,
        'w': w,
        'theta': theta,
        'phi': phi,
        'Rmat': Rmat,
        'k_hat': Rmat @ np.array([0, 0, 1]),
        'r0': np.asarray(r0, dtype=float),
        'pol': pol_lab,
        'proj': np.outer(pol_lab.conj(), pol_lab),
        'delta': delta,
    }


def dual_beam(I, w, theta, phi, delta, r0=np.zeros(3)):
    return [
        displaced_gaussian(I, w, theta, phi, r0, SIGMA_PLUS, delta),
        displaced_gaussian(I, w, theta, phi, r0, SIGMA_MINUS, -delta),
    ]


def quad_beam(I, w, theta, phi, delta, retro_reflected=True):
    r0 = np.array([w / 2, 0, 0])
    if retro_reflected:
        theta0 = np.pi
        sign = +1
    else:
        theta0 = 0
        sign = -1
    lasers = []
    lasers.extend(dual_beam(I, w, theta, phi, delta, r0))
    lasers.extend(dual_beam(I, w, theta + theta0, phi, sign * delta, sign * r0))
    return lasers


def pyramid_beams(I, w, theta, delta, n_quad=3, phi0=0.0, retro_reflected=True):
    """n_quad = 3 is the triad, 4 the paper's tetrad. phi0 turns the set about z."""
    lasers = []
    for i in range(n_quad):
        lasers.extend(quad_beam(I, w, theta, phi0 + i * TAU / n_quad, delta,
                                retro_reflected))
    return lasers


def intensity_for_power(P, w, n_lasers):
    """Peak intensity of each component when P is split over every pass."""
    return 2 * P / (n_lasers * np.pi * w ** 2)


def total_power(lasers, retro_reflected=False):
    P = sum(np.pi / 2 * b['I'] * b['w'] ** 2 for b in lasers)
    return P / 2 if retro_reflected else P


def intensity(b, R):
    R = np.asarray(R, dtype=float)
    x, y, _ = b['Rmat'].T @ R - b['r0'].reshape((3,) + (1,) * (R.ndim - 1))
    return b['I'] * np.exp(-2 * (x ** 2 + y ** 2) / b['w'] ** 2)


def shift_hamiltonian(lasers, R, V=None):
    """1P1 light shift (N, 3, 3) in rad/s. R (3, N) m, V (3, N) m/s or None.

    1/delta is regularised with the 6s7s linewidth.
    """
    R = np.asarray(R, dtype=float)
    H = np.zeros((R.shape[1], 3, 3), dtype=complex)
    for b in lasers:
        d = b['delta'] if V is None else b['delta'] - K_S * (b['k_hat'] @ V)
        amp = SHIFT_COEFF * intensity(b, R) * d / (d ** 2 + GAMMA_S ** 2 / 4)
        H += np.asarray(amp)[..., None, None] * b['proj']
    return H


def vector_part(H):
    """a in H = a.F + scalar + tensor, shape (3, N), same units as H."""
    return 0.5 * np.real(np.einsum('nij,kji->kn', H, SPIN1))


def effective_field(lasers, R, gF=GF_1P1):
    """Equivalent B (3, N) in tesla at v = 0."""
    return HBAR * vector_part(shift_hamiltonian(lasers, R)) / (gF * MU_B)


def gradient_tensor(lasers, gF=GF_1P1, h=1e-5):
    """dB_j/dr_i at the origin, T/m."""
    R = np.concatenate([h * np.eye(3), -h * np.eye(3)], axis=1)
    B = effective_field(lasers, R, gF)
    return ((B[:, :3] - B[:, 3:]) / (2 * h)).T


def paper_gradient(P, w, theta, delta, gF=GF_1P1):
    """Paper Eq. 3, T/m, P summed over every pass."""
    beta = P * np.sin(2 * theta) * GAMMA_S / (
        4 * np.exp(0.5) * gF * MU_B * C * (K_S * w) ** 3 * delta)
    return beta * np.diag([1.0, 1.0, -2.0])


if __name__ == "__main__":
    print("=" * 70)
    print("SELF TEST")
    print("=" * 70)
    P, w, delta = 2.0, 0.01, 10 * GAMMA_S
    reg = delta ** 2 / (delta ** 2 + GAMMA_S ** 2 / 4)
    print(f"  lambda_s = {LAMBDA_S * 1e9:.2f} nm, Isat_s = {ISAT_S / 10:.3f} mW/cm^2")

    b = dual_beam(ISAT_S, w, 0.0, 0.0, delta)
    B = effective_field(b, np.zeros((3, 1)), gF=1.0)[:, 0]
    ref = -reg * np.pi * ISAT_S * GAMMA_S / (2 * MU_B * C * K_S ** 3 * delta)
    print(f"  dual beam along z: B = {B * 1e4} G, Eq. 2 gives {ref * 1e4:.6f} G on z")
    assert np.allclose(B, [0, 0, ref], rtol=1e-9, atol=1e-15)

    for n_quad in (3, 4):
        lasers = pyramid_beams(intensity_for_power(P, w, 4 * n_quad), w,
                               TAU / 8, delta, n_quad, phi0=0.3)
        beta = gradient_tensor(lasers, gF=1.0)
        ref = reg * paper_gradient(P, w, TAU / 8, delta, gF=1.0)
        print(f"  {n_quad} quad beams, {P} W: diag(beta) = "
              f"{np.diag(beta) * 100} G/cm, Eq. 3 gives {np.diag(ref) * 100}")
        assert np.allclose(beta, ref, rtol=1e-4, atol=1e-6)

        R = np.random.default_rng(0).normal(size=(3, 50)) * w
        H = shift_hamiltonian(lasers, R)
        a = vector_part(H)
        rest = H - np.einsum('kn,kij->nij', a, SPIN1)
        print(f"    v = 0: largest scalar + tensor part {np.abs(rest).max():.2e} rad/s "
              f"against vector part {np.abs(a).max():.2e} rad/s")
        assert np.abs(rest).max() < 1e-9 * np.abs(a).max()

    print("\nAll checks passed.")
