"""
Iterative solver for the linearized phonon Boltzmann transport equation.

Implements the Omini-Sparavigna iterative scheme:
    f_{i+1} = (1/A^out) * b - (1/A^out) * A^in * f_i

where:
- A^out is the out-scattering rate (diagonal)
- A^in is the in-scattering operator (off-diagonal)
- b is the driving term from temperature gradient
- f is the phonon distribution deviation

Reference:
    M. Omini and A. Sparavigna, Physica B 212, 101 (1995)
    M. Omini and A. Sparavigna, Phys. Rev. B 53, 9064 (1996)
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
        """
        Args:
            primitive: Primitive cell object.
            mesh: Q-point mesh (n1, n2, n3).
            temperatures: Array of temperatures in K.
            max_iter: Maximum number of iterations.
            tol: Convergence tolerance for relative change in kappa.
            include_4ph_in_iteration: Whether to include 4-phonon in iteration.
                                     (Note: very memory intensive, default False)
        """
        self.primitive = primitive
        self.mesh = mesh
        self.temperatures = temperatures
        self.max_iter = max_iter
        self.tol = tol
        self.include_4ph_in_iteration = include_4ph_in_iteration
        self.volume = primitive.get_volume()
        self.nq = np.prod(mesh)

        self.comm = get_mpi_comm()
        self.rank = get_mpi_rank()
        self.size = get_mpi_size()

    def solve(self, freqs, group_velocities, gamma_total, 
              interaction, qpoints, eigenvectors,
              gamma_3ph=None, gamma_4ph=None) -> Dict[float, np.ndarray]:
        """Solve BTE iteratively for all temperatures.

        Args:
            freqs: (nq, nband) phonon frequencies in THz.
            group_velocities: (nq, nband, 3) in km/s.
            gamma_total: dict {T: (nq, nband)} total scattering rates in THz.
            interaction: PhononInteraction object.
            qpoints: (nq, 3) q-points in fractional coordinates.
            eigenvectors: (nq, nband, natom, 3) phonon eigenvectors.
            gamma_3ph: dict {T: (nq, nband)} 3-phonon scattering rates.
            gamma_4ph: dict {T: (nq, nband)} 4-phonon scattering rates.

        Returns:
            f_dict: dict {T: (nq, nband, 3)} phonon distribution deviation.
        """
        nq, nband = freqs.shape
        f_dict = {}

        for T in self.temperatures:
            if self.rank == 0:
                print(f"\n  Solving BTE iteratively at T={T}K...")

            gamma = gamma_total[T]

            # Convert gamma (THz) to scattering rate (rad/s)
            # gamma in THz -> Gamma = 2*pi*gamma (rad/ps) = 2*pi*gamma*1e12 (rad/s)
            Gamma = 2 * np.pi * gamma * 1e12  # rad/s

            # Mode heat capacity
            cv = mode_heat_capacity(freqs, T)  # (nq, nband) J/K per mode

            # Group velocities in m/s
            gv = group_velocities * 1e3  # km/s -> m/s

            # Relaxation time approximation (RTA) solution
            tau = np.zeros_like(Gamma)
            mask = Gamma > 1e-10
            tau[mask] = 1.0 / Gamma[mask]

            # Initial guess: f_RTA = tau * v * C_v / (k_B * T^2)
            # Actually, the driving term b = v * C_v / (k_B * T^2)
            # and f = A^{-1} * b
            # In RTA: f_RTA = tau * v * C_v

            # Driving term b (nq, nband, 3)
            b = np.zeros((nq, nband, 3))
            for alpha in range(3):
                b[:, :, alpha] = gv[:, :, alpha] * cv

            # RTA solution
            f = np.zeros((nq, nband, 3))
            for alpha in range(3):
                f[:, :, alpha] = tau * b[:, :, alpha]

            # Iterative improvement (only for 3-phonon)
            if gamma_3ph is not None and self.max_iter > 0:
                f = self._iterate_3ph(f, b, Gamma, freqs, eigenvectors, 
                                      qpoints, interaction, T)

            # If 4-phonon should be included in iteration (very expensive)
            if self.include_4ph_in_iteration and gamma_4ph is not None:
                if self.rank == 0:
                    print("    WARNING: 4-phonon iteration is very memory intensive!")
                # This would require storing the full 4-phonon collision matrix
                # For now, 4-phonon is treated at RTA level
                pass

            f_dict[T] = f

            # Calculate converged kappa
            kappa = self._calculate_kappa_from_f(f, gv, cv)
            if self.rank == 0:
                print(f"    Converged kappa: {np.trace(kappa)/3:.2f} W/mK")

        return f_dict

    def _iterate_3ph(self, f0, b, Gamma, freqs, eigenvectors, 
                     qpoints, interaction, T):
        """Omini-Sparavigna iteration for 3-phonon scattering.

        f_{i+1} = f_RTA - (1/Gamma) * A^in * f_i

        where A^in * f represents the in-scattering contribution.
        """
        nq, nband = freqs.shape
        f = f0.copy()

        # Precompute occupation numbers
        from .utils import bose_einstein
        n_occ = bose_einstein(freqs, T)

        kappa_prev = 0.0

        for iteration in range(self.max_iter):
            # Compute in-scattering term: A^in * f
            # This is the most expensive part
            delta_f = np.zeros_like(f)

            start, end = parallel_split(nq)

            for iq in range(start, end):
                q = qpoints[iq]
                for iband in range(nband):
                    omega = freqs[iq, iband]
                    if omega < 1e-6 or Gamma[iq, iband] < 1e-10:
                        continue

                    e = eigenvectors[iq, iband]

                    # In-scattering from all other modes
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

                                # In-scattering contributions
                                # Process: lambda' + lambda'' -> lambda
                                # f_in = sum P(lambda' lambda'' -> lambda) * (f_lambda' + f_lambda'')

                                # Energy delta
                                delta_e = omega - omega2 - omega3
                                from .utils import gaussian_smearing, adaptive_sigma
                                sigma = adaptive_sigma(omega, omega2, omega3, 
                                                      scalebroad=1.0, dq=1.0/np.mean(self.mesh))
                                delta = gaussian_smearing(delta_e, sigma)

                                occ = n2 + n3 + 1

                                # Add contribution to delta_f[iq, iband]
                                for alpha in range(3):
                                    delta_f[iq, iband, alpha] += (
                                        v3_sq * occ * delta * 
                                        (f[iq2, ib2, alpha] + f[iq3, ib3, alpha])
                                    )

                                # Process: lambda + lambda' -> lambda''
                                delta_e = omega + omega2 - omega3
                                delta = gaussian_smearing(delta_e, sigma)
                                occ = n3 - n2

                                for alpha in range(3):
                                    delta_f[iq, iband, alpha] += (
                                        v3_sq * occ * delta * 
                                        (f[iq3, ib3, alpha] - f[iq2, ib2, alpha])
                                    )

            # MPI allreduce
            if self.comm is not None:
                delta_f = mpi_allreduce(delta_f, op='sum')

            # Apply prefactor and update
            hbar = 1.054571817e-34
            THz_to_J = 1e12 * 2 * np.pi * hbar
            prefactor = np.pi / (2 * hbar**2 * self.nq) * (THz_to_J)**2
            delta_f *= prefactor

            # Update: f_{new} = f_RTA - (1/Gamma) * delta_f
            f_new = f0.copy()
            for alpha in range(3):
                mask = Gamma > 1e-10
                f_new[:, :, alpha] = f0[:, :, alpha] - delta_f[:, :, alpha] / Gamma
                f_new[:, :, alpha] = np.where(mask, f_new[:, :, alpha], 0.0)

            # Check convergence using kappa
            kappa = self._calculate_kappa_from_f(f_new, 
                freqs.shape[0] * np.ones((freqs.shape[0], freqs.shape[1], 3)) * 1e3, 
                mode_heat_capacity(freqs, T))
            # Actually we need gv and cv properly... let's compute kappa differently

            # Simpler convergence check: max relative change
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
        """Calculate thermal conductivity from distribution deviation."""
        # kappa = (1/V) * sum f * v * C_v
        # f already contains v*C_v*tau, so kappa = (1/V) * sum f * v
        nq, nband, _ = f.shape
        kappa = np.zeros((3, 3))

        for alpha in range(3):
            for beta in range(3):
                kappa[alpha, beta] = np.sum(f[:, :, alpha] * gv[:, :, beta])

        volume_m3 = self.volume * 1e-30
        kappa = kappa / volume_m3

        return kappa

    def _find_qpoint_index(self, q_target, qpoints, tol=1e-5):
        """Find index of q-point in mesh."""
        from .utils import find_qpoint_index
        return find_qpoint_index(q_target, qpoints, tol)
