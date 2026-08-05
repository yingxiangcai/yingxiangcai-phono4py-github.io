"""
Main Phono4py class for running 4-phonon thermal conductivity calculations.
Compatible with phonopy 4.x and spglib 2.7.0.
"""

import numpy as np
import h5py
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from .force_constants import read_force_constants_2nd, read_force_constants_3rd, read_force_constants_4th
from .symmetry import SymmetryAnalyzer
from .harmonic import HarmonicCalculator
from .interaction import PhononInteraction
from .scattering import ScatteringCalculator
from .bte_solver import BTEIterativeSolver
from .conductivity import ConductivityCalculator
from .utils import get_mpi_rank, get_mpi_size, mpi_barrier

class Phono4py:
    def __init__(self, unitcell, supercell_matrix, primitive_matrix="P",
                 mesh=(11, 11, 11), temperatures=None, scalebroad=1.0,
                 use_symmetry=True, is_shift=(0, 0, 0), symprec=1e-5,
                 max_iter=50, bte_tol=1e-5):
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

        # FIX: phonopy 4.x default primitive_matrix='auto' differs from v3
        self.phonopy = Phonopy(unitcell, supercell_matrix, primitive_matrix=primitive_matrix)
        self.primitive = self.phonopy.primitive
        self.natom = len(self.primitive)
        self.nband = 3 * self.natom

        if self.use_symmetry:
            lattice = self.primitive.cell
            positions = self.primitive.scaled_positions
            symbols = self.primitive.symbols
            unique_symbols = list(dict.fromkeys(symbols))
            atom_types = [unique_symbols.index(s) + 1 for s in symbols]
            self.symmetry = SymmetryAnalyzer(lattice, positions, atom_types, symprec=symprec)
            if self.rank == 0:
                print(f"Space group: {self.symmetry.spacegroup}")
        else:
            self.symmetry = None

        self.fc2_data = None
        self.fc3_data = None
        self.fc4_data = None

    def read_force_constants(self, fc2_file="FORCE_CONSTANTS_2ND",
                              fc3_file="FORCE_CONSTANTS_3RD",
                              fc4_file="FORCE_CONSTANTS_4TH"):
        if self.rank == 0:
            print("Reading force constants...")
        self.fc2_data = read_force_constants_2nd(fc2_file)
        self.fc3_data = read_force_constants_3rd(fc3_file)
        self.fc4_data = read_force_constants_4th(fc4_file)
        if self.rank == 0:
            print(f"  FC2: {len(self.fc2_data[0])} pairs")
            print(f"  FC3: {len(self.fc3_data[0])} triplets")
            print(f"  FC4: {len(self.fc4_data[0])} quartets")

    def run(self, include_3ph=True, include_4ph=True,
            use_iterative_bte=True, output_file="phono4py_results.hdf5"):
        if self.fc2_data is None:
            raise ValueError("Force constants not set.")

        if self.rank == 0:
            print("\nSetting up harmonic calculation...")

        # FIX: phonopy 4.x requires SUPERCELL-sized force constants
        fc2_full = self._build_fc2_full()
        self.phonopy.force_constants = fc2_full

        if self.rank == 0:
            print(f"Generating q-point mesh: {self.mesh}")

        # Run harmonic mesh
        if self.rank == 0:
            print("Running harmonic calculation...")
        harmonic = HarmonicCalculator(self.phonopy)
        mesh_data = harmonic.run_mesh(self.mesh)

        # FIX: phonopy 4.x mesh stores data for IRREDUCIBLE q-points only
        # Need to expand to full mesh using grid_mapping_table
        nq_total = np.prod(self.mesh)

        # Build full q-point grid from grid_address
        grid_address = mesh_data.grid_address  # (nq_total, 3) integer
        grid_mapping_table = mesh_data.grid_mapping_table  # (nq_total,)
        ir_grid_points = mesh_data.ir_grid_points  # irreducible indices in grid_address

        # Convert grid_address to fractional q-points
        qpoints = grid_address.astype(float) / np.array(self.mesh)

        # Map irreducible data to full mesh
        freqs = np.zeros((nq_total, self.nband))
        eigenvectors = np.zeros((nq_total, self.nband, self.natom, 3), dtype=complex)
        group_velocities = np.zeros((nq_total, self.nband, 3))

        # Build mapping: grid_point_index -> ir_data_index
        ir_data_indices = {gp: idx for idx, gp in enumerate(ir_grid_points)}

        for i, gp in enumerate(grid_mapping_table):
            ir_idx = ir_data_indices[gp]
            freqs[i] = mesh_data.frequencies[ir_idx]
            group_velocities[i] = mesh_data.group_velocities[ir_idx]
            # eigenvectors: (nq_ir, nband, nband) complex -> (nq, nband, natom, 3)
            for ib in range(self.nband):
                eigenvectors[i, ib] = mesh_data.eigenvectors[ir_idx, ib].reshape(self.natom, 3)

        if self.rank == 0:
            print(f"  Total q-points: {nq_total}")
            print(f"  Irreducible q-points: {len(ir_grid_points)}")
            print(f"  Number of bands: {self.nband}")
            print(f"  Frequencies at Gamma: {freqs[0]}")

        if self.rank == 0:
            print("\nSetting up phonon interactions...")
        masses = self.primitive.masses
        interaction = PhononInteraction(
            self.primitive, self.fc2_data, self.fc3_data, self.fc4_data, masses)

        if self.rank == 0:
            print("\nCalculating scattering rates...")
        scattering = ScatteringCalculator(
            interaction, self.mesh, self.temperatures, self.scalebroad,
            use_ir_mesh=False, mapping=None, grid_points=qpoints)

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

        gamma_total = {}
        for T in self.temperatures:
            gamma_total[T] = np.zeros_like(freqs)
            if gamma_3ph is not None:
                gamma_total[T] += gamma_3ph[T]
            if gamma_4ph is not None:
                gamma_total[T] += gamma_4ph[T]

        conductivity = ConductivityCalculator(self.primitive, self.mesh, self.temperatures)

        if use_iterative_bte:
            if self.rank == 0:
                print("\nSolving BTE iteratively...")
            bte_solver = BTEIterativeSolver(
                self.primitive, self.mesh, self.temperatures,
                max_iter=self.max_iter, tol=self.bte_tol)
            f_dict = bte_solver.solve(
                freqs, group_velocities, gamma_total,
                interaction, qpoints, eigenvectors,
                gamma_3ph=gamma_3ph, gamma_4ph=gamma_4ph)
            kappa_dict, mode_kappa_dict = conductivity.calculate_conductivity_iterative(
                freqs, group_velocities, f_dict, qpoints)
        else:
            if self.rank == 0:
                print("\nCalculating thermal conductivity (RTA)...")
            kappa_dict, mode_kappa_dict = conductivity.calculate_conductivity_rta(
                freqs, group_velocities, gamma_total, qpoints)

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

        if self.rank == 0:
            print(f"\nSaving results to {output_file}...")
            self._save_hdf5(output_file, freqs, eigenvectors, group_velocities,
                            qpoints, np.ones(nq_total), gamma_total, kappa_dict,
                            mode_kappa_dict, gamma_3ph, gamma_4ph, use_iterative_bte)
            print("Done!")

        mpi_barrier()
        return kappa_dict, gamma_total

    def _build_fc2_full(self):
        """Build full fc2 matrix in SUPERCELL shape for phonopy 4.x."""
        natoms_super = len(self.phonopy.supercell)
        fc2_full = np.zeros((natoms_super, natoms_super, 3, 3))
        pairs, fcs, cells = self.fc2_data
        for (i, j), fc, cell in zip(pairs, fcs, cells):
            if i < natoms_super and j < natoms_super:
                fc2_full[i, j] = fc
        return fc2_full

    def _save_hdf5(self, filename, freqs, eigenvectors, group_velocities,
                    qpoints, weights, scattering_rates, kappa_dict, mode_kappa_dict,
                    gamma_3ph, gamma_4ph, use_iterative_bte):
        with h5py.File(filename, 'w') as f:
            f.create_dataset('mesh', data=np.array(self.mesh))
            f.create_dataset('qpoint', data=qpoints)
            f.create_dataset('weight', data=weights)
            f.create_dataset('temperature', data=self.temperatures)
            f.create_dataset('frequency', data=freqs)
            f.create_dataset('eigenvector', data=eigenvectors)
            f.create_dataset('group_velocity', data=group_velocities)
            gamma_dataset = []
            for T in self.temperatures:
                gamma_dataset.append(scattering_rates[T])
            f.create_dataset('gamma', data=np.array(gamma_dataset))
            if gamma_3ph is not None:
                gamma3_dataset = []
                for T in self.temperatures:
                    gamma3_dataset.append(gamma_3ph[T])
                f.create_dataset('gamma_3ph', data=np.array(gamma3_dataset))
            if gamma_4ph is not None:
                gamma4_dataset = []
                for T in self.temperatures:
                    gamma4_dataset.append(gamma_4ph[T])
                f.create_dataset('gamma_4ph', data=np.array(gamma4_dataset))
            kappa_dataset = []
            mode_kappa_dataset = []
            for T in self.temperatures:
                kappa = kappa_dict[T]
                kappa_dataset.append([kappa[0,0], kappa[1,1], kappa[2,2], kappa[1,2], kappa[0,2], kappa[0,1]])
                mode_kappa_dataset.append(mode_kappa_dict[T])
            f.create_dataset('kappa', data=np.array(kappa_dataset))
            f.create_dataset('mode_kappa', data=np.array(mode_kappa_dataset))
            f.attrs['version'] = 'phono4py v0.2.1'
            f.attrs['include_3ph'] = gamma_3ph is not None
            f.attrs['include_4ph'] = gamma_4ph is not None
            f.attrs['use_iterative_bte'] = use_iterative_bte
            f.attrs['scalebroad'] = self.scalebroad
            f.attrs['symmetry'] = self.symmetry.spacegroup if self.symmetry else "P1"
