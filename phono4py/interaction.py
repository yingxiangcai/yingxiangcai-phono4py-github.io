"""
Calculation of 3-phonon and 4-phonon interaction matrix elements.
Compatible with phonopy 4.x (removed get_reciprocal_vectors API).
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

        # phonopy 4.x removed get_reciprocal_vectors(), compute manually
        self.rec_lat = self._compute_reciprocal_lattice(primitive.cell)

    def _compute_reciprocal_lattice(self, lattice):
        """Compute reciprocal lattice vectors (3, 3) in 1/Angstrom."""
        volume = np.dot(lattice[0], np.cross(lattice[1], lattice[2]))
        rec = np.zeros((3, 3))
        rec[0] = 2 * np.pi * np.cross(lattice[1], lattice[2]) / volume
        rec[1] = 2 * np.pi * np.cross(lattice[2], lattice[0]) / volume
        rec[2] = 2 * np.pi * np.cross(lattice[0], lattice[1]) / volume
        return rec

    def compute_v3_vectorized(self, q1, q2, q3, e1, e2, e3):
        """Compute 3-phonon interaction matrix element - fully vectorized."""
        nfc3 = len(self.fc3_triplets)
        if nfc3 == 0:
            return 0.0 + 0.0j

        q2_cart = np.dot(q2, self.rec_lat)
        q3_cart = np.dot(q3, self.rec_lat)

        phases = np.exp(1j * (
            np.dot(self.fc3_cells2, q2_cart) + 
            np.dot(self.fc3_cells3, q3_cart)
        ))

        masses_ijk = self.masses[self.fc3_triplets]
        mass_factors = 1.0 / np.sqrt(np.prod(masses_ijk, axis=1))

        e1_sel = e1[self.fc3_triplets[:, 0]]
        e2_sel = e2[self.fc3_triplets[:, 1]]
        e3_sel = e3[self.fc3_triplets[:, 2]]

        contractions = np.einsum(
            'nabc,na,nb,nc->n',
            self.fc3,
            np.conj(e1_sel),
            e2_sel,
            e3_sel
        )

        v3_total = np.sum(contractions * mass_factors * phases)
        return v3_total

    def compute_v4_vectorized(self, q1, q2, q3, q4, e1, e2, e3, e4):
        """Compute 4-phonon interaction matrix element - fully vectorized."""
        nfc4 = len(self.fc4_quartets)
        if nfc4 == 0:
            return 0.0 + 0.0j

        q2_cart = np.dot(q2, self.rec_lat)
        q3_cart = np.dot(q3, self.rec_lat)
        q4_cart = np.dot(q4, self.rec_lat)

        phases = np.exp(1j * (
            np.dot(self.fc4_cells2, q2_cart) +
            np.dot(self.fc4_cells3, q3_cart) +
            np.dot(self.fc4_cells4, q4_cart)
        ))

        masses_ijkl = self.masses[self.fc4_quartets]
        mass_factors = 1.0 / np.sqrt(np.prod(masses_ijkl, axis=1))

        e1_sel = e1[self.fc4_quartets[:, 0]]
        e2_sel = e2[self.fc4_quartets[:, 1]]
        e3_sel = e3[self.fc4_quartets[:, 2]]
        e4_sel = e4[self.fc4_quartets[:, 3]]

        contractions = np.einsum(
            'nabcd,na,nb,nc,nd->n',
            self.fc4,
            np.conj(e1_sel),
            e2_sel,
            e3_sel,
            e4_sel
        )

        v4_total = np.sum(contractions * mass_factors * phases)
        return v4_total
