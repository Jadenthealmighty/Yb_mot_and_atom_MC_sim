"""Steady-state J=0 -> J=1 rate equations.
 Cartesian |x>,|y>,|z> basis
and pylcp's units (k = Gamma = 1). pylcp only diagonalises along B, which a
Doppler-shifted light shift does not have.
"""

import numpy as np

SPIN1 = np.array([[[0, 0, 0], [0, 0, -1j], [0, 1j, 0]],
                  [[0, 0, 1j], [0, 0, 0], [-1j, 0, 0]],
                  [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]]])


def spherical_to_cartesian(pol_sph):
    """pylcp's spherical polarization to an exp(-i w t) Cartesian vector."""
    p = np.asarray(pol_sph, dtype=complex)
    return np.conj(np.array([(p[0] - p[2]) / np.sqrt(2),
                             1j * (p[0] + p[2]) / np.sqrt(2), p[1]]))


class CoolingBeams:
    """The 399 nm beams of a pylcp.laserBeams as flat arrays."""

    def __init__(self, laser_beams):
        self.beams = list(laser_beams.beam_vector)
        k = np.array([b.kvec() for b in self.beams], dtype=float)
        self.khat = k / np.linalg.norm(k, axis=1)[:, None]
        self.pol = np.array([spherical_to_cartesian(b.pol()) for b in self.beams])
        self.delta = np.array([float(b.delta(0.)) for b in self.beams])

    def intensity(self, R):
        R = np.asarray(R, dtype=float)
        return np.array([np.broadcast_to(b.intensity(R, 0.), R.shape[1:])
                         for b in self.beams])


def zeeman(B):
    """B (3, N), pylcp units, to the (N, 3, 3) Zeeman Hamiltonian."""
    return np.einsum('kn,kij->nij', np.asarray(B, dtype=float), SPIN1)


def steady_force(H, s, khat, pol, delta, V):
    """Equilibrium force (3, N) in hbar k Gamma.

    H (N, 3, 3), s (n_beam, N) saturation, V (3, N) velocity.
    """
    E, U = np.linalg.eigh(H)
    f = np.abs(np.einsum('nij,bi->nbj', U.conj(), pol)) ** 2
    det = delta[None, :, None] - (V.T @ khat.T)[:, :, None] - E[:, None, :]
    R = 0.5 * s.T[:, :, None] * f / (1.0 + 4.0 * det ** 2)
    Rn = R.sum(axis=1)
    Ng = 1.0 / (1.0 + np.sum(Rn / (1.0 + Rn), axis=1))
    net = np.sum(R * (Ng[:, None] / (1.0 + Rn))[:, None, :], axis=2)
    return (net @ khat).T


def beam_force(beams, H, R, V):
    """steady_force with the saturation looked up from `beams` at R."""
    return steady_force(H, beams.intensity(R), beams.khat, beams.pol,
                        beams.delta, V)
