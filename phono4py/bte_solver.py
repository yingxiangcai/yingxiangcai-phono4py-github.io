"""
Iterative solver for the linearized phonon Boltzmann transport equation.
Implements the Omini-Sparavigna iterative scheme.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from .utils import (
    mode_heat_capacity, get_mpi_comm, get_mpi_rank, get_mpi_size,
    mpi_allreduce, parallel_split
)

class BTEIterativeSolver:
    """Iterative solver for linearized phonon BTE."""

    def __init__(self, primitive, mesh, temperatures,
                 max_iter: int = 50, tol: float = 1e-5,
                 include_4ph_in_iteration: bool = False):
        self.primitive = primitive
        self.mesh = mesh
        self.temperatures = temperatures
        self.max_iter = max_iter
        self.tol = tol
        self.include_4ph_in_iteration = include_4ph_in_iteration
        self.volume = primitive.volume
        self.nq = np.prod(mesh)

        self.comm = get_mpi_comm()
        self.rank = get_mpi_rank()
        self.size = get_mpi_size()

    def solve(self, freqs, group_velocities, gamma_total,
              interaction, qpoints, eigenvectors,
              gamma_3ph=None, gamma_4ph=None) -> Dict[float, np.ndarray]:
        nq, nband = freqs.shape
        f_dict = {}

        for T in self.temperatures:
            if self.rank == 0:
                print(f"  Solving BTE iteratively at T={T}K...")

            gamma = gamma_total[T]
            Gamma = 2 * np.pi * gamma * 1e12
            cv = mode_heat_capacity(freqs, T)
            gv = group_velocities * 1e3

            tau = np.zeros_like(Gamma)
            mask = Gamma > 1e-10
            tau[mask] = 1.0 / Gamma[mask]

            b = np.zeros((nq, nband, 3))
            for alpha in range(3):
                b[:, :, alpha] = gv[:, :, alpha] * cv

            f = np.zeros((nq, nband, 3))
            for alpha in range(3):
                f[:, :, alpha] = tau * b[:, :, alpha]

            if gamma_3ph is not None and self.max_iter > 0:
                f = self._iterate_3ph(f, b, Gamma, freqs, eigenvectors,
                                       qpoints, interaction, T)

            if self.include_4ph_in_iteration and gamma_4ph is not None:
                if self.rank == 0:
                    print("  WARNING: 4-phonon iteration is very memory intensive!")

            f_dict[T] = f

            kappa = self._calculate_kappa_from_f(f, gv, cv)
            if self.rank == 0:
                print(f"  Converged kappa: {np.trace(kappa)/3:.2f} W/mK")

        return f_dict

    def _iterate_3ph(self, f0, b, Gamma, freqs, eigenvectors,
                      qpoints, interaction, T):
        nq, nband = freqs.shape
        f = f0.copy()
        from .utils import bose_einstein, gaussian_smearing, adaptive_sigma
        n_occ = bose_einstein(freqs, T)

        for iteration in range(self.max_iter):
            delta_f = np.zeros_like(f)
            start, end = parallel_split(nq)

            for iq in range(start, end):
                q = qpoints[iq]
                for iband in range(nband):
                    omega = freqs[iq, iband]
                    if omega < 1e-6 or Gamma[iq, iband] < 1e-10:
                        continue

                    e = eigenvectors[iq, iband]

                    for iq2 in range(nq):
                        q2 = qpoints[iq2]
                        q3 = q + q2
                        q3 = q3 - np.floor(q3)
                        iq3 = self._find_qpoint_index(q3, qpoints)

                        if iq3 is None:
                            continue

                        for ib2 in range(nband):
                            omega2 = freqs[iq2, ib2]
                            e2 = eigenvectors[iq2, ib2]
                            n2 = n_occ[iq2, ib2]

                            for ib3 in range(nband):
                                omega3 = freqs[iq3, ib3]
                                e3 = eigenvectors[iq3, ib3]
                                n3 = n_occ[iq3, ib3]

                                v3 = interaction.compute_v3_vectorized(
                                    q, q2, q3, e, e2, e3)
                                v3_sq = np.abs(v3)**2

                                if v3_sq < 1e-30:
                                    continue

                                delta_e = omega - omega2 - omega3
                                sigma = adaptive_sigma(omega, omega2, omega3,
                                                          scalebroad=1.0, dq=1.0/np.mean(self.mesh))
                                delta = gaussian_smearing(delta_e, sigma)
                                occ = n2 + n3 + 1

                                for alpha in range(3):
                                    delta_f[iq, iband, alpha] += (
                                        v3_sq * occ * delta *
                                        (f[iq2, ib2, alpha] + f[iq3, ib3, alpha])
                                    )

                                delta_e = omega + omega2 - omega3
                                delta = gaussian_smearing(delta_e, sigma)
                                occ = n3 - n2

                                for alpha in range(3):
                                    delta_f[iq, iband, alpha] += (
                                        v3_sq * occ * delta *
                                        (f[iq3, ib3, alpha] - f[iq2, ib2, alpha])
                                    )

            if self.comm is not None:
                delta_f = mpi_allreduce(delta_f, op='sum')

            hbar = 1.054571817e-34
            THz_to_J = 1e12 * 2 * np.pi * hbar
            prefactor = np.pi / (2 * hbar**2 * self.nq) * (THz_to_J)**2
            delta_f *= prefactor

            f_new = f0.copy()
            for alpha in range(3):
                mask = Gamma > 1e-10
                f_new[:, :, alpha] = f0[:, :, alpha] - delta_f[:, :, alpha] / Gamma
                f_new[:, :, alpha] = np.where(mask, f_new[:, :, alpha], 0.0)

            diff = np.abs(f_new - f)
            rel_diff = diff / (np.abs(f) + 1e-20)
            max_rel_diff = np.max(rel_diff)

            f = f_new

            if self.rank == 0:
                print(f"    Iteration {iteration+1}: max_rel_diff = {max_rel_diff:.2e}")

            if max_rel_diff < self.tol:
                if self.rank == 0:
                    print(f"    Converged after {iteration+1} iterations")
                break

        return f

    def _calculate_kappa_from_f(self, f, gv, cv):
        nq, nband, _ = f.shape
        kappa = np.zeros((3, 3))
        for alpha in range(3):
            for beta in range(3):
                kappa[alpha, beta] = np.sum(f[:, :, alpha] * gv[:, :, beta])
        volume_m3 = self.volume * 1e-30
        kappa = kappa / volume_m3
        return kappa

    def _find_qpoint_index(self, q_target, qpoints, tol=1e-5):
        from .utils import find_qpoint_index
        return find_qpoint_index(q_target, qpoints, tol)
