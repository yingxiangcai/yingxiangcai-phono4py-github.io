"""
Symmetry analysis using spglib 2.7.0 for irreducible q-point reduction.
Compatible with phonopy 4.x property-based API.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict

class SymmetryAnalyzer:
    """Analyze crystal symmetry and generate irreducible q-point meshes."""

    def __init__(self, lattice: np.ndarray, positions: np.ndarray,
                 atom_types: List[int], symprec: float = 1e-5):
        self.lattice = np.array(lattice)
        self.positions = np.array(positions)
        self.atom_types = np.array(atom_types)
        self.symprec = symprec
        self.cell = (self.lattice, self.positions, self.atom_types)

        try:
            import spglib
            self.spglib = spglib
            self.has_spglib = True
        except ImportError:
            self.has_spglib = False
            print("Warning: spglib not found. Running without symmetry reduction.")

        self._spacegroup = None
        self._symmetry_ops = None
        self._mesh_cache = {}

    @property
    def spacegroup(self) -> str:
        if self._spacegroup is None and self.has_spglib:
            self._spacegroup = self.spglib.get_spacegroup(self.cell, symprec=self.symprec)
        return self._spacegroup or "P1"

    @property
    def symmetry_operations(self) -> Dict:
        if self._symmetry_ops is None and self.has_spglib:
            self._symmetry_ops = self.spglib.get_symmetry(self.cell, symprec=self.symprec)
        return self._symmetry_ops

    def get_ir_reciprocal_mesh(self, mesh: Tuple[int, int, int],
                                is_shift: Tuple[int, int, int] = (0, 0, 0),
                                is_time_reversal: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cache_key = (tuple(mesh), tuple(is_shift), is_time_reversal)
        if cache_key in self._mesh_cache:
            return self._mesh_cache[cache_key]

        if not self.has_spglib:
            qpoints = self._generate_full_mesh(mesh, is_shift)
            n_total = len(qpoints)
            mapping = np.arange(n_total)
            weights = np.ones(n_total, dtype=int)
            result = (qpoints, weights, mapping, qpoints)
            self._mesh_cache[cache_key] = result
            return result

        mapping, grid = self.spglib.get_ir_reciprocal_mesh(
            mesh, self.cell, is_shift=list(is_shift)
        )

        if mapping is None:
            raise RuntimeError("spglib failed to generate irreducible mesh")

        grid = np.array(grid)
        mapping = np.array(mapping)

        grid_points = grid.astype(float)
        for i in range(3):
            if is_shift[i]:
                grid_points[:, i] = (grid_points[:, i] + 0.5) / mesh[i]
            else:
                grid_points[:, i] = grid_points[:, i] / mesh[i]

        ir_indices = np.unique(mapping)
        n_ir = len(ir_indices)
        ir_qpoints = grid_points[ir_indices]

        weights = np.zeros(n_ir, dtype=int)
        for i, ir_idx in enumerate(ir_indices):
            weights[i] = np.sum(mapping == ir_idx)

        result = (ir_qpoints, weights, mapping, grid_points)
        self._mesh_cache[cache_key] = result
        return result

    def _generate_full_mesh(self, mesh: Tuple[int, int, int],
                             is_shift: Tuple[int, int, int]) -> np.ndarray:
        n1, n2, n3 = mesh
        qpoints = []
        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    q = [i / n1, j / n2, k / n3]
                    if is_shift[0]:
                        q[0] += 0.5 / n1
                    if is_shift[1]:
                        q[1] += 0.5 / n2
                    if is_shift[2]:
                        q[2] += 0.5 / n3
                    qpoints.append(q)
        return np.array(qpoints)

    def get_star_of_q(self, qpoint: np.ndarray, mesh: Tuple[int, int, int],
                        is_shift: Tuple[int, int, int] = (0, 0, 0)) -> Tuple[np.ndarray, np.ndarray]:
        if not self.has_spglib or self.symmetry_operations is None:
            return np.array([qpoint]), np.array([1.0])

        rotations = self.symmetry_operations['rotations']
        star = []
        for rot in rotations:
            q_rot = np.dot(rot, qpoint)
            q_rot = q_rot - np.floor(q_rot)
            is_new = True
            for q_existing in star:
                diff = q_rot - q_existing
                diff = diff - np.round(diff)
                if np.all(np.abs(diff) < self.symprec):
                    is_new = False
                    break
            if is_new:
                star.append(q_rot)

        star = np.array(star)
        weights = np.ones(len(star)) / len(star)
        return star, weights

    def reduce_by_symmetry(self, data: np.ndarray, mapping: np.ndarray,
                            operation: str = 'sum') -> np.ndarray:
        n_ir = len(np.unique(mapping))
        shape = (n_ir,) + data.shape[1:]
        result = np.zeros(shape)
        counts = np.zeros(n_ir)

        for i, ir_idx in enumerate(mapping):
            if operation == 'sum':
                result[ir_idx] += data[i]
            elif operation == 'mean':
                result[ir_idx] += data[i]
            elif operation == 'max':
                result[ir_idx] = np.maximum(result[ir_idx], data[i])
            counts[ir_idx] += 1

        if operation == 'mean':
            result = result / counts.reshape((-1,) + (1,) * (len(shape) - 1))
        return result

    def expand_to_full_mesh(self, ir_data: np.ndarray, mapping: np.ndarray) -> np.ndarray:
        n_total = len(mapping)
        shape = (n_total,) + ir_data.shape[1:]
        result = np.zeros(shape)
        for i, ir_idx in enumerate(mapping):
            result[i] = ir_data[ir_idx]
        return result
