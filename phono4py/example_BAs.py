#!/usr/bin/env python
"""
Example script for running phono4py calculation on BAs.

This script demonstrates how to use phono4py to calculate
4-phonon thermal conductivity with all advanced features:
- MPI parallelization
- spglib symmetry reduction
- Iterative BTE solver
- Full 4-phonon scattering channels

Usage:
    # Serial run
    python example_BAs.py

    # MPI parallel run (4 processes)
    mpirun -np 4 python example_BAs.py
"""

import numpy as np
from phonopy.structure.atoms import PhonopyAtoms
from phono4py import Phono4py

# ============================================================================
# 1. Define crystal structure (BAs zincblende)
# ============================================================================
# Lattice constant for BAs (Angstrom)
a = 4.777

lattice = np.array([
    [0.0, a/2, a/2],
    [a/2, 0.0, a/2],
    [a/2, a/2, 0.0]
])

# Fractional positions
positions = np.array([
    [0.0, 0.0, 0.0],      # B
    [0.25, 0.25, 0.25]    # As
])

symbols = ['B', 'As']

unitcell = PhonopyAtoms(
    symbols=symbols,
    scaled_positions=positions,
    cell=lattice,
)

# ============================================================================
# 2. Initialize phono4py
# ============================================================================
ph4 = Phono4py(
    unitcell=unitcell,
    supercell_matrix=[2, 2, 2],    # Should match FC calculation supercell
    mesh=[11, 11, 11],              # q-point mesh
    temperatures=np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]),
    scalebroad=0.1,                 # Adaptive broadening factor
    use_symmetry=True,              # Enable spglib symmetry reduction
    is_shift=(0, 0, 0),             # Gamma-centered mesh
    symprec=1e-5,                   # Symmetry precision
    max_iter=50,                    # Max BTE iterations
    bte_tol=1e-5,                   # BTE convergence tolerance
)

# ============================================================================
# 3. Read force constants
# ============================================================================
ph4.read_force_constants(
    fc2_file="FORCE_CONSTANTS_2ND",
    fc3_file="FORCE_CONSTANTS_3RD",
    fc4_file="FORCE_CONSTANTS_4TH"
)

# ============================================================================
# 4. Run calculation
# ============================================================================
kappa, gamma = ph4.run(
    include_3ph=True,               # Include 3-phonon scattering
    include_4ph=True,               # Include 4-phonon scattering
    use_iterative_bte=True,         # Use iterative BTE solver
    output_file="kappa-phono4py.hdf5"
)

# ============================================================================
# 5. Post-process (optional)
# ============================================================================
import h5py

with h5py.File("kappa-phono4py.hdf5", 'r') as f:
    print("\nHDF5 file contents:")
    print("  Datasets:", list(f.keys()))
    print("  Attributes:", dict(f.attrs))

    temps = f['temperature'][:]
    kappa_data = f['kappa'][:]  # (nT, 6) -> [xx, yy, zz, yz, xz, xy]

    print("\nThermal conductivity from HDF5:")
    print(f"{'T (K)':>8} {'kappa_xx':>12} {'kappa_yy':>12} {'kappa_zz':>12} {'kappa_avg':>12}")
    for i, T in enumerate(temps):
        kappa_avg = np.mean([kappa_data[i, 0], kappa_data[i, 1], kappa_data[i, 2]])
        print(f"{T:8.0f} {kappa_data[i,0]:12.2f} {kappa_data[i,1]:12.2f} {kappa_data[i,2]:12.2f} {kappa_avg:12.2f}")
