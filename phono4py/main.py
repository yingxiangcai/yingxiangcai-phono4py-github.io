"""
Main Phono4py class for running 4-phonon thermal conductivity calculations.

Features:
- MPI parallelization
- spglib 2.7.0 symmetry reduction
- Iterative BTE solver
- Full 4-phonon scattering channels
- HDF5 output
"""

import numpy as np
import h5py
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from .force_constants import (
    read_force_constants_2nd,
    read_force_constants_3rd,
    read_force_constants_4th,
)
from .symmetry import SymmetryAnalyzer
from .harmonic import HarmonicCalculator
from .interaction import PhononInteraction
from .scattering import ScatteringCalculator
from .bte_solver import BTEIterativeSolver
from .conductivity import ConductivityCalculator
from .utils import get_mpi_rank, get_mpi_size, mpi_barrier


class Phono4py:
    """Main class for phono4py calculations."""

    def __init__(self, unitcell, supercell_matrix, primitive_matrix="auto",
                 mesh=(11, 11, 11), temperatures=None, scalebroad=1.0,
                 use_symmetry=True, is_shift=(0, 0, 0), symprec=1e-5,
                 max_iter=50, bte_tol=1e-5):
        """
        Args:
            unitcell: ASE Atoms or PhonopyAtoms object.
            supercell_matrix: Supercell matrix for force constants.
            primitive_matrix: Primitive matrix.
            mesh: Q-point mesh for BZ integration.
            temperatures: Array of temperatures in K. Default: [100, 200, ..., 1000].
            scalebroad: Broadening factor.
            use_symmetry: Whether to use spglib symmetry reduction.
            is_shift: Mesh shift (0 or 1 for each axis).
            symprec: Symmetry precision for spglib.
            max_iter: Maximum iterations for BTE solver.
            bte_tol: Convergence tolerance for BTE solver.
        """
        if temperatures is None:
            temperatures = np.arange(100, 1100, 100)
        self.temperatures = np.array(temperatures)
        self.mesh = mesh
        self.scalebroad = scalebroad
        self.use_symmetry = use_symmetry
        self.is_shift = is_shift
        self.symprec = symprec
        self.max_iter = max_iter
        self.bte_tol = bte_tol

        self.rank = get_mpi_rank()
        self.size = get_mpi_size()

        # Initialize phonopy
        if not isinstance(unitcell, PhonopyAtoms):
            try:
                from ase.atoms import Atoms
                if isinstance(unitcell, Atoms):
                    unitcell = PhonopyAtoms(
                        symbols=unitcell.get_chemical_symbols(),
                        scaled_positions=unitcell.get_scaled_positions(),
                        cell=unitcell.cell,
                    )
            except ImportError:
                pass

        self.phonopy = Phonopy(unitcell, supercell_matrix, primitive_matrix=primitive_matrix)
        self.primitive = self.phonopy.primitive
        self.natom = len(self.primitive)
        self.nband = 3 * self.natom

        # Symmetry analyzer
        self.symmetry = None
        if self.use_symmetry:
            lattice = self.primitive.get_cell()
            positions = self.primitive.get_scaled_positions()
            atom_types = [1] * self.natom  # Simplified; should use actual types
            # Try to get unique atom types
            symbols = self.primitive.get_chemical_symbols()
            unique_symbols = list(dict.fromkeys(symbols))
            atom_types = [unique_symbols.index(s) + 1 for s in symbols]

            self.symmetry = SymmetryAnalyzer(lattice, positions, atom_types, symprec=symprec)
            if self.rank == 0:
                print(f"Space group: {self.symmetry.spacegroup}")

        self.fc2_data = None
        self.fc3_data = None
        self.fc4_data = None

    def read_force_constants(self, fc2_file="FORCE_CONSTANTS_2ND",
                            fc3_file="FORCE_CONSTANTS_3RD",
                            fc4_file="FORCE_CONSTANTS_4TH"):
        """Read force constant files."""
        if self.rank == 0:
            print("Reading force constants...")
        self.fc2_data = read_force_constants_2nd(fc2_file)
        self.fc3_data = read_force_constants_3rd(fc3_file)
        self.fc4_data = read_force_constants_4th(fc4_file)
        if self.rank == 0:
            print(f"  FC2: {len(self.fc2_data[0])} pairs")
            print(f"  FC3: {len(self.fc3_data[0])} triplets")
            print(f"  FC4: {len(self.fc4_data[0])} quartets")

    def set_force_constants(self, fc2_data, fc3_data, fc4_data):
        """Set force constant data directly."""
        self.fc2_data = fc2_data
        self.fc3_data = fc3_data
        self.fc4_data = fc4_data

    def run(self, include_3ph=True, include_4ph=True, 
            use_iterative_bte=True, output_file="phono4py_results.hdf5"):
        """Run the full phono4py calculation.

        Args:
            include_3ph: Include 3-phonon scattering.
            include_4ph: Include 4-phonon scattering.
            use_iterative_bte: Use iterative BTE solver instead of RTA.
            output_file: Output HDF5 filename.

        Returns:
            kappa_dict: dict {T: (3, 3)} thermal conductivity in W/mK
            scattering_rates: dict {T: (nq, nband)} scattering rates in THz
        """
        if self.fc2_data is None:
            raise ValueError("Force constants not set. Call read_force_constants() first.")

        # Set up phonopy with 2nd order force constants
        if self.rank == 0:
            print("\nSetting up harmonic calculation...")
        fc2_full = self._build_fc2_full()
        self.phonopy.force_constants = fc2_full

        # Generate q-point mesh
        if self.rank == 0:
            print(f"Generating q-point mesh: {self.mesh}")

        if self.use_symmetry and self.symmetry is not None:
            ir_qpoints, weights, mapping, grid_points = self.symmetry.get_ir_reciprocal_mesh(
                self.mesh, is_shift=self.is_shift
            )
            if self.rank == 0:
                print(f"  Full mesh: {len(grid_points)} q-points")
                print(f"  Irreducible mesh: {len(ir_qpoints)} q-points")
            qpoints = grid_points
            self.mapping = mapping
            self.ir_weights = weights
        else:
            from .utils import get_qpoints_mesh
            qpoints = get_qpoints_mesh(self.mesh, self.is_shift)
            mapping = np.arange(len(qpoints))
            weights = np.ones(len(qpoints))
            self.mapping = mapping
            self.ir_weights = weights

        # Run harmonic mesh
        if self.rank == 0:
            print(f"Running harmonic calculation...")
        harmonic = HarmonicCalculator(self.phonopy)
        mesh_dict = harmonic.run_mesh(self.mesh)

        freqs = mesh_dict['frequencies']
        eigenvectors = mesh_dict['eigenvectors']
        group_velocities = mesh_dict['group_velocities']

        nq = len(qpoints)
        eigenvectors = eigenvectors.reshape(nq, self.nband, self.natom, 3)

        if self.rank == 0:
            print(f"  Number of q-points: {nq}")
            print(f"  Number of bands: {self.nband}")

        # Set up interaction
        if self.rank == 0:
            print("\nSetting up phonon interactions...")
        masses = self.primitive.get_masses()
        interaction = PhononInteraction(
            self.primitive, self.fc2_data, self.fc3_data, self.fc4_data, masses
        )

        # Calculate scattering rates
        if self.rank == 0:
            print("\nCalculating scattering rates...")
        scattering = ScatteringCalculator(
            interaction, self.mesh, self.temperatures, self.scalebroad,
            use_ir_mesh=self.use_symmetry, mapping=mapping, grid_points=qpoints
        )

        gamma_3ph = None
        gamma_4ph = None

        if include_3ph:
            gamma_3ph = {}
            for T in self.temperatures:
                if self.rank == 0:
                    print(f"  T={T}K: 3-phonon scattering")
                gamma_3ph[T] = scattering.calculate_3ph_scattering(
                    freqs, eigenvectors, qpoints, T)

        if include_4ph:
            gamma_4ph = {}
            for T in self.temperatures:
                if self.rank == 0:
                    print(f"  T={T}K: 4-phonon scattering")
                gamma_4ph[T] = scattering.calculate_4ph_scattering(
                    freqs, eigenvectors, qpoints, T)

        # Total scattering rates
        gamma_total = {}
        for T in self.temperatures:
            gamma_total[T] = np.zeros_like(freqs)
            if gamma_3ph is not None:
                gamma_total[T] += gamma_3ph[T]
            if gamma_4ph is not None:
                gamma_total[T] += gamma_4ph[T]

        # Calculate thermal conductivity
        conductivity = ConductivityCalculator(self.primitive, self.mesh, self.temperatures)

        if use_iterative_bte:
            if self.rank == 0:
                print("\nSolving BTE iteratively...")
            bte_solver = BTEIterativeSolver(
                self.primitive, self.mesh, self.temperatures,
                max_iter=self.max_iter, tol=self.bte_tol
            )
            f_dict = bte_solver.solve(
                freqs, group_velocities, gamma_total,
                interaction, qpoints, eigenvectors,
                gamma_3ph=gamma_3ph, gamma_4ph=gamma_4ph
            )
            kappa_dict, mode_kappa_dict = conductivity.calculate_conductivity_iterative(
                freqs, group_velocities, f_dict, qpoints
            )
        else:
            if self.rank == 0:
                print("\nCalculating thermal conductivity (RTA)...")
            kappa_dict, mode_kappa_dict = conductivity.calculate_conductivity_rta(
                freqs, group_velocities, gamma_total, qpoints
            )

        # Print results
        if self.rank == 0:
            print("\n" + "="*60)
            print("Thermal Conductivity Results (W/mK)")
            print("="*60)
            print(f"{'T (K)':>8} {'kappa_xx':>12} {'kappa_yy':>12} {'kappa_zz':>12} {'kappa_avg':>12}")
            print("-"*60)
            for T in self.temperatures:
                kappa = kappa_dict[T]
                kappa_scalar = np.trace(kappa) / 3.0
                print(f"{T:8.0f} {kappa[0,0]:12.2f} {kappa[1,1]:12.2f} {kappa[2,2]:12.2f} {kappa_scalar:12.2f}")
            print("="*60)

        # Save to HDF5
        if self.rank == 0:
            print(f"\nSaving results to {output_file}...")
            self._save_hdf5(output_file, freqs, eigenvectors, group_velocities,
                           qpoints, self.ir_weights, gamma_total, kappa_dict, 
                           mode_kappa_dict, gamma_3ph, gamma_4ph, use_iterative_bte)
            print("Done!")

        mpi_barrier()

        return kappa_dict, gamma_total

    def _build_fc2_full(self):
        """Build full fc2 matrix from sparse data."""
        natoms = self.natom
        fc2_full = np.zeros((natoms, natoms, 3, 3))
        pairs, fcs, _ = self.fc2_data
        for (i, j), fc in zip(pairs, fcs):
            if i < natoms and j < natoms:
                fc2_full[i, j] = fc
        return fc2_full

    def _save_hdf5(self, filename, freqs, eigenvectors, group_velocities,
                  qpoints, weights, scattering_rates, kappa_dict, mode_kappa_dict,
                  gamma_3ph, gamma_4ph, use_iterative_bte):
        """Save all results to HDF5 file (phono3py-compatible format)."""
        with h5py.File(filename, 'w') as f:
            # Mesh info
            f.create_dataset('mesh', data=np.array(self.mesh))
            f.create_dataset('qpoint', data=qpoints)
            f.create_dataset('weight', data=weights)
            f.create_dataset('temperature', data=self.temperatures)

            # Harmonic properties
            f.create_dataset('frequency', data=freqs)
            f.create_dataset('eigenvector', data=eigenvectors)
            f.create_dataset('group_velocity', data=group_velocities)

            # Scattering rates
            gamma_dataset = []
            for T in self.temperatures:
                gamma_dataset.append(scattering_rates[T])
            f.create_dataset('gamma', data=np.array(gamma_dataset))

            # 3-phonon scattering rates
            if gamma_3ph is not None:
                gamma3_dataset = []
                for T in self.temperatures:
                    gamma3_dataset.append(gamma_3ph[T])
                f.create_dataset('gamma_3ph', data=np.array(gamma3_dataset))

            # 4-phonon scattering rates
            if gamma_4ph is not None:
                gamma4_dataset = []
                for T in self.temperatures:
                    gamma4_dataset.append(gamma_4ph[T])
                f.create_dataset('gamma_4ph', data=np.array(gamma4_dataset))

            # Thermal conductivity
            kappa_dataset = []
            mode_kappa_dataset = []
            for T in self.temperatures:
                kappa = kappa_dict[T]
                # Store as [xx, yy, zz, yz, xz, xy]
                kappa_dataset.append([
                    kappa[0,0], kappa[1,1], kappa[2,2],
                    kappa[1,2], kappa[0,2], kappa[0,1]
                ])
                mode_kappa_dataset.append(mode_kappa_dict[T])
            f.create_dataset('kappa', data=np.array(kappa_dataset))
            f.create_dataset('mode_kappa', data=np.array(mode_kappa_dataset))

            # Metadata
            f.attrs['version'] = 'phono4py v0.2.0'
            f.attrs['include_3ph'] = gamma_3ph is not None
            f.attrs['include_4ph'] = gamma_4ph is not None
            f.attrs['use_iterative_bte'] = use_iterative_bte
            f.attrs['scalebroad'] = self.scalebroad
            f.attrs['symmetry'] = self.symmetry.spacegroup if self.symmetry else "P1"
