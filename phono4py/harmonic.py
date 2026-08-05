"""
Harmonic phonon calculations using phonopy 4.x.
"""

class HarmonicCalculator:
    def __init__(self, phonopy_obj):
        self.phonopy = phonopy_obj
        self.natom = len(phonopy_obj.supercell)
        self.nband = 3 * len(phonopy_obj.primitive)

    def run_mesh(self, mesh, with_group_velocities=True, with_eigenvectors=True):
        self.phonopy.run_mesh(mesh, with_group_velocities=with_group_velocities, with_eigenvectors=with_eigenvectors)
        return self.phonopy.mesh
