"""
Lattice thermal conductivity calculation.

Supports both RTA and iterative BTE solutions.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from .utils import mode_heat_capacity, get_mpi_rank


class ConductivityCalculator:
    """Calculate lattice thermal conductivity."""

    def __init__(self, primitive, mesh, temperatures):
        self.primitive = primitive
        self.mesh = mesh
        self.temperatures = temperatures
        self.volume = primitive.get_volume()
        self.nq = np.prod(mesh)
        self.rank = get_mpi_rank()

    def calculate_conductivity_rta(self, freqs, group_velocities, 
                                    scattering_rates, qpoints):
        """Calculate thermal conductivity using RTA.

        Args:
            freqs: (nq, nband) in THz
            group_velocities: (nq, nband, 3) in km/s
            scattering_rates: dict {T: (nq, nband)} in THz
            qpoints: (nq, 3)

        Returns:
            kappa_dict: dict {T: (3, 3)} in W/mK
            mode_kappa_dict: dict {T: (nq, nband, 3, 3)}
        """
        kappa_dict = {}
        mode_kappa_dict = {}
        nq, nband = freqs.shape

        for T in self.temperatures:
            gamma = scattering_rates[T]

            # Relaxation time: tau = 1/(2*pi*gamma) [gamma in THz -> tau in ps]
            tau = np.zeros_like(gamma)
            mask = gamma > 1e-10
            tau[mask] = 1.0 / (2 * np.pi * gamma[mask])

            cv = mode_heat_capacity(freqs, T)
            gv = group_velocities * 1e3  # km/s -> m/s

            mode_kappa = np.zeros((nq, nband, 3, 3))

            for iq in range(nq):
                for ib in range(nband):
                    if freqs[iq, ib] < 1e-6 or gamma[iq, ib] < 1e-10:
                        continue

                    for alpha in range(3):
                        for beta in range(3):
                            mode_kappa[iq, ib, alpha, beta] = (
                                cv[iq, ib] * gv[iq, ib, alpha] * gv[iq, ib, beta] * tau[iq, ib]
                            )

            volume_m3 = self.volume * 1e-30
            kappa_total = np.sum(mode_kappa, axis=(0, 1)) / volume_m3

            kappa_dict[T] = kappa_total
            mode_kappa_dict[T] = mode_kappa

        return kappa_dict, mode_kappa_dict

    def calculate_conductivity_iterative(self, freqs, group_velocities,
                                          f_dict, qpoints):
        """Calculate thermal conductivity from iterative BTE solution.

        Args:
            freqs: (nq, nband) in THz
            group_velocities: (nq, nband, 3) in km/s
            f_dict: dict {T: (nq, nband, 3)} phonon distribution deviation
            qpoints: (nq, 3)

        Returns:
            kappa_dict: dict {T: (3, 3)} in W/mK
            mode_kappa_dict: dict {T: (nq, nband, 3, 3)}
        """
        kappa_dict = {}
        mode_kappa_dict = {}
        nq, nband = freqs.shape

        for T in self.temperatures:
            f = f_dict[T]
            cv = mode_heat_capacity(freqs, T)
            gv = group_velocities * 1e3

            mode_kappa = np.zeros((nq, nband, 3, 3))

            for iq in range(nq):
                for ib in range(nband):
                    if freqs[iq, ib] < 1e-6:
                        continue

                    for alpha in range(3):
                        for beta in range(3):
                            # kappa_mode = (1/V) * f_alpha * v_beta
                            mode_kappa[iq, ib, alpha, beta] = (
                                f[iq, ib, alpha] * gv[iq, ib, beta]
                            )

            volume_m3 = self.volume * 1e-30
            kappa_total = np.sum(mode_kappa, axis=(0, 1)) / volume_m3

            kappa_dict[T] = kappa_total
            mode_kappa_dict[T] = mode_kappa

        return kappa_dict, mode_kappa_dict
