"""
Symmetry analysis using spglib 2.7.0.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict

class SymmetryAnalyzer:
    def __init__(self, lattice, positions, atom_types, symprec=1e-5):
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
        self._spacegroup = None
        self._symmetry_ops = None
        self._mesh_cache = {}

    @property
    def spacegroup(self):
        if self._spacegroup is None and self.has_spglib:
            self._spacegroup = self.spglib.get_spacegroup(self.cell, symprec=self.symprec)
        return self._spacegroup or "P1"

    @property
    def symmetry_operations(self):
        if self._symmetry_ops is None and self.has_spglib:
            self._symmetry_ops = self.spglib.get_symmetry(self.cell, symprec=self.symprec)
        return self._symmetry_ops

    def get_ir_reciprocal_mesh(self, mesh, is_shift=(0, 0, 0), is_time_reversal=True):
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
        mapping, grid = self.spglib.get_ir_reciprocal_mesh(mesh, self.cell, is_shift=list(is_shift))
        if mapping is None:
            raise RuntimeError("spglib failed")
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

    def _generate_full_mesh(self, mesh, is_shift):
        n1, n2, n3 = mesh
        qpoints = []
        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    q = [i / n1, j / n2, k / n3]
                    if is_shift[0]: q[0] += 0.5 / n1
                    if is_shift[1]: q[1] += 0.5 / n2
                    if is_shift[2]: q[2] += 0.5 / n3
                    qpoints.append(q)
        return np.array(qpoints)
