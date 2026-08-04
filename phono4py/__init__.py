"""
phono4py v0.2.0: Advanced four-phonon scattering and lattice thermal conductivity solver.

Features:
- MPI parallelization for q-point loops
- spglib 2.7.0 integration for symmetry-reduced calculations
- Iterative BTE solver (Omini-Sparavigna method)
- Full 4-phonon scattering channels (Type I & II)
- HDF5 output compatible with phono3py format
"""

__version__ = "0.2.0"

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
from .main import Phono4py

__all__ = [
    "read_force_constants_2nd",
    "read_force_constants_3rd",
    "read_force_constants_4th",
    "SymmetryAnalyzer",
    "HarmonicCalculator",
    "PhononInteraction",
    "ScatteringCalculator",
    "BTEIterativeSolver",
    "ConductivityCalculator",
    "Phono4py",
]
