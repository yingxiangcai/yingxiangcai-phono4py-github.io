"""
Lattice thermal conductivity calculation.
"""

import numpy as np
from .utils import mode_heat_capacity, get_mpi_rank

class ConductivityCalculator:
    def __init__(self, primitive, mesh, temperatures):
        self.primitive = primitive
        self.mesh = mesh
        self.temperatures = temperatures
        self.volume = primitive.volume
        self.nq = np.prod(mesh)
        self.rank = get_mpi_rank()

    def calculate_conductivity_rta(self, freqs, group_velocities, scattering_rates, qpoints):
        kappa_dict = {}
        mode_kappa_dict = {}
        nq, nband = freqs.shape
        for T in self.temperatures:
            gamma = scattering_rates[T]
            tau = np.zeros_like(gamma)
            mask = gamma > 1e-10
            tau[mask] = 1.0 / (2 * np.pi * gamma[mask] * 1e12)
            cv = mode_heat_capacity(freqs, T)
            gv = group_velocities * 1e3
            mode_kappa = np.zeros((nq, nband, 3, 3))
            for iq in range(nq):
                for ib in range(nband):
                    if freqs[iq, ib] < 1e-6 or gamma[iq, ib] < 1e-10:
                        continue
                    for alpha in range(3):
                        for beta in range(3):
                            mode_kappa[iq, ib, alpha, beta] = cv[iq, ib] * gv[iq, ib, alpha] * gv[iq, ib, beta] * tau[iq, ib]
            volume_m3 = self.volume * 1e-30
            kappa_total = np.sum(mode_kappa, axis=(0, 1)) / volume_m3
            kappa_dict[T] = kappa_total
            mode_kappa_dict[T] = mode_kappa
        return kappa_dict, mode_kappa_dict

    def calculate_conductivity_iterative(self, freqs, group_velocities, f_dict, qpoints):
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
                            mode_kappa[iq, ib, alpha, beta] = f[iq, ib, alpha] * gv[iq, ib, beta]
            volume_m3 = self.volume * 1e-30
            kappa_total = np.sum(mode_kappa, axis=(0, 1)) / volume_m3
            kappa_dict[T] = kappa_total
            mode_kappa_dict[T] = mode_kappa
        return kappa_dict, mode_kappa_dict
