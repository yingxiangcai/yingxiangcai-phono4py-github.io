#!/usr/bin/env python
"""
Example script for running phono4py calculation on BAs.
Compatible with phonopy 4.x + spglib 2.7.0.
"""

import numpy as np
from phonopy.structure.atoms import PhonopyAtoms
from phono4py import Phono4py

# BAs zincblende structure
a = 4.777
lattice = np.array([
    [0.0, a/2, a/2],
    [a/2, 0.0, a/2],
    [a/2, a/2, 0.0]
])

positions = np.array([
    [0.0, 0.0, 0.0],
    [0.25, 0.25, 0.25]
])

unitcell = PhonopyAtoms(
    symbols=['B', 'As'],
    scaled_positions=positions,
    cell=lattice,
)

# IMPORTANT: Use small mesh for testing!
# 4-phonon requires extremely small mesh (e.g., [3,3,3])
ph4 = Phono4py(
    unitcell=unitcell,
    supercell_matrix=[2, 2, 2],
    mesh=[3, 3, 3],              # Small mesh for quick test
    temperatures=np.array([300]),
    scalebroad=0.1,
    use_symmetry=True,
    is_shift=(0, 0, 0),
    symprec=1e-5,
    max_iter=10,                 # Fewer iterations for test
    bte_tol=1e-4,
)

ph4.read_force_constants(
    fc2_file="test/FORCE_CONSTANTS_2ND",
    fc3_file="test/FORCE_CONSTANTS_3RD",
    fc4_file="test/FORCE_CONSTANTS_4TH"
)

# Test 1: 3-phonon only, RTA (fastest)
print("\n" + "="*60)
print("TEST 1: 3-phonon RTA")
print("="*60)
kappa, gamma = ph4.run(
    include_3ph=True,
    include_4ph=False,
    use_iterative_bte=False,
    output_file="test_3ph_rta.hdf5"
)

# Test 2: 3-phonon + 4-phonon, RTA (slow - 4ph is expensive even at [3,3,3])
print("\n" + "="*60)
print("TEST 2: 3+4-phonon RTA")
print("="*60)
kappa, gamma = ph4.run(
    include_3ph=True,
    include_4ph=True,
    use_iterative_bte=False,
    output_file="test_34ph_rta.hdf5"
)
