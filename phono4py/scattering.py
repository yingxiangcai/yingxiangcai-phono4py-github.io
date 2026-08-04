"""
Calculation of phonon scattering rates (linewidths) from 3-phonon and 4-phonon processes.

Features:
- MPI parallelization over q-point and band indices
- Full 4-phonon scattering channels:
  * Type I:  q + q2 -> q3 + q4  (two-phonon -> two-phonon)
  * Type II: q -> q2 + q3 + q4  (one-phonon -> three-phonon)
- Adaptive Gaussian smearing
- Support for irreducible q-point meshes
"""

import numpy as np
from typing import Tuple, Optional, Dict
from .utils import (
    bose_einstein, adaptive_sigma, gaussian_smearing,
    get_mpi_comm, get_mpi_rank, get_mpi_size, mpi_allreduce, parallel_split
)
from .interaction import PhononInteraction


class ScatteringCalculator:
    """Calculate phonon scattering rates with MPI parallelization."""

    def __init__(self, 
                 interaction: PhononInteraction,
                 mesh: Tuple[int, int, int],
                 temperatures: np.ndarray,
                 scalebroad: float = 1.0,
                 use_ir_mesh: bool = True,
                 mapping: Optional[np.ndarray] = None,
                 grid_points: Optional[np.ndarray] = None):
        """
        Args:
            interaction: PhononInteraction object.
            mesh: Q-point mesh (n1, n2, n3).
            temperatures: Array of temperatures in K.
            scalebroad: Scaling factor for adaptive broadening.
            use_ir_mesh: Whether to use irreducible mesh.
            mapping: Mapping from full to irreducible indices (if use_ir_mesh).
            grid_points: All grid points in fractional coordinates.
        """
        self.interaction = interaction
        self.mesh = mesh
        self.temperatures = temperatures
        self.scalebroad = scalebroad
        self.nq = np.prod(mesh)
        self.nband = interaction.nbands
        self.use_ir_mesh = use_ir_mesh
        self.mapping = mapping
        self.grid_points = grid_points

        # Grid spacing for adaptive broadening
        self.dq = 1.0 / np.array(mesh)

        # MPI info
        self.comm = get_mpi_comm()
        self.rank = get_mpi_rank()
        self.size = get_mpi_size()

    def calculate_3ph_scattering(self, freqs, eigenvectors, qpoints, T):
        """Calculate 3-phonon scattering rates with MPI parallelization."""
        nq, nband = freqs.shape
        gamma = np.zeros((nq, nband))
        n_occ = bose_einstein(freqs, T)

        hbar = 1.054571817e-34
        THz_to_J = 1e12 * 2 * np.pi * hbar
        prefactor = np.pi / (2 * hbar**2 * self.nq) * (THz_to_J)**2

        # MPI parallelization: split iq loop
        start, end = parallel_split(nq)

        for iq in range(start, end):
            q = qpoints[iq]
            for iband in range(nband):
                omega = freqs[iq, iband]
                if omega < 1e-6:
                    continue

                e = eigenvectors[iq, iband]
                gamma_mode = 0.0

                for iq2 in range(nq):
                    q2 = qpoints[iq2]
                    # Process 1: q + q2 -> q3 (absorption) and q -> q2 + q3 (emission)
                    # Momentum conservation: q3 = q + q2 (mod G)
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

                            v3 = self.interaction.compute_v3_vectorized(
                                q, q2, q3, e, e2, e3)
                            v3_sq = np.abs(v3)**2

                            if v3_sq < 1e-30:
                                continue

                            # Process A: omega = omega2 + omega3 (emission, lambda -> lambda' + lambda'')
                            delta_e = omega - omega2 - omega3
                            sigma = adaptive_sigma(omega, omega2, omega3,
                                                  scalebroad=self.scalebroad,
                                                  dq=np.mean(self.dq))
                            delta = gaussian_smearing(delta_e, sigma)
                            occ = (n2 + n3 + 1)  # emission
                            gamma_mode += v3_sq * occ * delta

                            # Process B: omega + omega2 = omega3 (absorption, lambda + lambda' -> lambda'')
                            delta_e = omega + omega2 - omega3
                            sigma = adaptive_sigma(omega, omega2, omega3,
                                                  scalebroad=self.scalebroad,
                                                  dq=np.mean(self.dq))
                            delta = gaussian_smearing(delta_e, sigma)
                            occ = (n3 - n2)  # absorption
                            gamma_mode += v3_sq * occ * delta

                gamma[iq, iband] = gamma_mode * prefactor

        # MPI allreduce
        if self.comm is not None:
            gamma = mpi_allreduce(gamma, op='sum')

        return gamma

    def calculate_4ph_scattering(self, freqs, eigenvectors, qpoints, T):
        """Calculate 4-phonon scattering rates with full channels and MPI.

        Includes both Type I (2->2) and Type II (1->3) processes.
        """
        nq, nband = freqs.shape
        gamma = np.zeros((nq, nband))
        n_occ = bose_einstein(freqs, T)

        hbar = 1.054571817e-34
        THz_to_J = 1e12 * 2 * np.pi * hbar
        prefactor = np.pi * hbar / (8 * self.nq**2) * (THz_to_J)**3

        # MPI parallelization
        start, end = parallel_split(nq)

        for iq in range(start, end):
            q = qpoints[iq]
            for iband in range(nband):
                omega = freqs[iq, iband]
                if omega < 1e-6:
                    continue

                e = eigenvectors[iq, iband]
                gamma_mode = 0.0

                # ============================================================
                # Type I: q + q2 -> q3 + q4  (two-phonon -> two-phonon)
                # ============================================================
                for iq2 in range(nq):
                    q2 = qpoints[iq2]
                    for iq3 in range(nq):
                        q3 = qpoints[iq3]
                        # Momentum conservation: q4 = q + q2 - q3 (mod G)
                        q4 = q + q2 - q3
                        q4 = q4 - np.floor(q4)
                        iq4 = self._find_qpoint_index(q4, qpoints)

                        if iq4 is None:
                            continue

                        for ib2 in range(nband):
                            omega2 = freqs[iq2, ib2]
                            e2 = eigenvectors[iq2, ib2]
                            n2 = n_occ[iq2, ib2]

                            for ib3 in range(nband):
                                omega3 = freqs[iq3, ib3]
                                e3 = eigenvectors[iq3, ib3]
                                n3 = n_occ[iq3, ib3]

                                for ib4 in range(nband):
                                    omega4 = freqs[iq4, ib4]
                                    e4 = eigenvectors[iq4, ib4]
                                    n4 = n_occ[iq4, ib4]

                                    v4 = self.interaction.compute_v4_vectorized(
                                        q, q2, q3, q4, e, e2, e3, e4)
                                    v4_sq = np.abs(v4)**2

                                    if v4_sq < 1e-30:
                                        continue

                                    # Energy conservation: omega + omega2 = omega3 + omega4
                                    delta_e = omega + omega2 - omega3 - omega4
                                    sigma = adaptive_sigma(omega, omega2, omega3, omega4,
                                                          scalebroad=self.scalebroad,
                                                          dq=np.mean(self.dq))
                                    delta = gaussian_smearing(delta_e, sigma)

                                    # Occupation factor for 2->2 process
                                    # n3*n4*(n1+1)*(n2+1) - n1*n2*(n3+1)*(n4+1)
                                    n1 = n_occ[iq, iband]
                                    occ = (n3 * n4 * (n1 + 1) * (n2 + 1) -
                                           n1 * n2 * (n3 + 1) * (n4 + 1))

                                    gamma_mode += v4_sq * np.abs(occ) * delta

                # ============================================================
                # Type II: q -> q2 + q3 + q4  (one-phonon -> three-phonon)
                # ============================================================
                # Momentum conservation: q = q2 + q3 + q4 (mod G)
                # Energy conservation: omega = omega2 + omega3 + omega4
                for iq2 in range(nq):
                    q2 = qpoints[iq2]
                    for iq3 in range(nq):
                        q3 = qpoints[iq3]
                        q4 = q - q2 - q3
                        q4 = q4 - np.floor(q4)
                        iq4 = self._find_qpoint_index(q4, qpoints)

                        if iq4 is None:
                            continue

                        for ib2 in range(nband):
                            omega2 = freqs[iq2, ib2]
                            e2 = eigenvectors[iq2, ib2]
                            n2 = n_occ[iq2, ib2]

                            for ib3 in range(nband):
                                omega3 = freqs[iq3, ib3]
                                e3 = eigenvectors[iq3, ib3]
                                n3 = n_occ[iq3, ib3]

                                for ib4 in range(nband):
                                    omega4 = freqs[iq4, ib4]
                                    e4 = eigenvectors[iq4, ib4]
                                    n4 = n_occ[iq4, ib4]

                                    v4 = self.interaction.compute_v4_vectorized(
                                        q, q2, q3, q4, e, e2, e3, e4)
                                    v4_sq = np.abs(v4)**2

                                    if v4_sq < 1e-30:
                                        continue

                                    # Energy conservation: omega = omega2 + omega3 + omega4
                                    delta_e = omega - omega2 - omega3 - omega4
                                    sigma = adaptive_sigma(omega, omega2, omega3, omega4,
                                                          scalebroad=self.scalebroad,
                                                          dq=np.mean(self.dq))
                                    delta = gaussian_smearing(delta_e, sigma)

                                    # Occupation factor for 1->3 process
                                    # (n2+1)*(n3+1)*(n4+1)*n1 - n2*n3*n4*(n1+1)
                                    n1 = n_occ[iq, iband]
                                    occ = ((n2 + 1) * (n3 + 1) * (n4 + 1) * n1 -
                                           n2 * n3 * n4 * (n1 + 1))

                                    gamma_mode += v4_sq * np.abs(occ) * delta

                gamma[iq, iband] = gamma_mode * prefactor

        # MPI allreduce
        if self.comm is not None:
            gamma = mpi_allreduce(gamma, op='sum')

        return gamma

    def _find_qpoint_index(self, q_target, qpoints, tol=1e-5):
        """Find index of q-point in mesh."""
        from .utils import find_qpoint_index
        return find_qpoint_index(q_target, qpoints, tol)

    def calculate_total_scattering(self, freqs, eigenvectors, qpoints,
                                    include_3ph=True, include_4ph=True):
        """Calculate total scattering rates at all temperatures with MPI."""
        results = {}

        for T in self.temperatures:
            if self.rank == 0:
                print(f"  T={T}K: Calculating scattering rates...")

            gamma = np.zeros_like(freqs)

            if include_3ph:
                if self.rank == 0:
                    print(f"    -> 3-phonon processes")
                gamma += self.calculate_3ph_scattering(freqs, eigenvectors, qpoints, T)

            if include_4ph:
                if self.rank == 0:
                    print(f"    -> 4-phonon processes (Type I + Type II)")
                gamma += self.calculate_4ph_scattering(freqs, eigenvectors, qpoints, T)

            results[T] = gamma

        return results
