"""
Calculation of 3-phonon and 4-phonon interaction matrix elements.
"""

import numpy as np


class PhononInteraction:
    """Calculate phonon-phonon interaction matrix elements."""

    def __init__(self, primitive, fc2_data, fc3_data, fc4_data, masses):
        self.primitive = primitive
        self.masses = masses
        self.natoms = len(masses)
        self.nbands = 3 * self.natoms

        self.fc2_pairs, self.fc2, self.fc2_cells = fc2_data
        self.fc3_triplets, self.fc3_cells2, self.fc3_cells3, self.fc3 = fc3_data
        self.fc4_quartets, self.fc4_cells2, self.fc4_cells3, self.fc4_cells4, self.fc4 = fc4_data

        self.rec_lat = primitive.get_reciprocal_vectors()

    def _get_phase_factor(self, qpoint, cell_vector):
        q_cart = np.dot(qpoint, self.rec_lat)
        return np.exp(1j * np.dot(q_cart, cell_vector))

    def compute_v3_vectorized(self, q1, q2, q3, e1, e2, e3):
        """Compute 3-phonon interaction matrix element."""
        nfc3 = len(self.fc3_triplets)

        q2_cart = np.dot(q2, self.rec_lat)
        q3_cart = np.dot(q3, self.rec_lat)
        phases2 = np.exp(1j * np.dot(self.fc3_cells2, q2_cart))
        phases3 = np.exp(1j * np.dot(self.fc3_cells3, q3_cart))

        masses_ijk = self.masses[self.fc3_triplets]
        mass_factors = 1.0 / np.sqrt(np.prod(masses_ijk, axis=1))

        e1_sel = e1[self.fc3_triplets[:, 0]]
        e2_sel = e2[self.fc3_triplets[:, 1]]
        e3_sel = e3[self.fc3_triplets[:, 2]]

        v3_total = 0.0 + 0.0j
        for idx in range(nfc3):
            phi = self.fc3[idx]
            contraction = np.einsum('abc,a,b,c->', phi, 
                                   np.conj(e1_sel[idx]), e2_sel[idx], e3_sel[idx])
            v3_total += contraction * mass_factors[idx] * phases2[idx] * phases3[idx]

        return v3_total

    def compute_v4_vectorized(self, q1, q2, q3, q4, e1, e2, e3, e4):
        """Compute 4-phonon interaction matrix element."""
        nfc4 = len(self.fc4_quartets)

        q2_cart = np.dot(q2, self.rec_lat)
        q3_cart = np.dot(q3, self.rec_lat)
        q4_cart = np.dot(q4, self.rec_lat)

        phases2 = np.exp(1j * np.dot(self.fc4_cells2, q2_cart))
        phases3 = np.exp(1j * np.dot(self.fc4_cells3, q3_cart))
        phases4 = np.exp(1j * np.dot(self.fc4_cells4, q4_cart))

        masses_ijkl = self.masses[self.fc4_quartets]
        mass_factors = 1.0 / np.sqrt(np.prod(masses_ijkl, axis=1))

        e1_sel = e1[self.fc4_quartets[:, 0]]
        e2_sel = e2[self.fc4_quartets[:, 1]]
        e3_sel = e3[self.fc4_quartets[:, 2]]
        e4_sel = e4[self.fc4_quartets[:, 3]]

        v4_total = 0.0 + 0.0j
        for idx in range(nfc4):
            phi = self.fc4[idx]
            contraction = np.einsum('abcd,a,b,c,d->', phi,
                                   np.conj(e1_sel[idx]), e2_sel[idx], 
                                   e3_sel[idx], e4_sel[idx])
            v4_total += contraction * mass_factors[idx] * phases2[idx] * phases3[idx] * phases4[idx]

        return v4_total
