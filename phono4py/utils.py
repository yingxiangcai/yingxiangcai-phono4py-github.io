"""
Utility functions for phono4py.
"""

import numpy as np
from typing import Tuple, List, Optional

# =============================================================================
# MPI utilities
# =============================================================================

def get_mpi_comm():
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD
    except ImportError:
        return None

def get_mpi_rank() -> int:
    comm = get_mpi_comm()
    if comm is None:
        return 0
    return comm.Get_rank()

def get_mpi_size() -> int:
    comm = get_mpi_comm()
    if comm is None:
        return 1
    return comm.Get_size()

def mpi_barrier():
    comm = get_mpi_comm()
    if comm is not None:
        comm.Barrier()

def mpi_allreduce(data, op='sum'):
    comm = get_mpi_comm()
    if comm is None:
        return data

    if isinstance(data, np.ndarray):
        result = np.empty_like(data)
        comm.Allreduce(data, result, op=get_mpi_op(op))
        return result
    else:
        return comm.allreduce(data, op=get_mpi_op(op))

def get_mpi_op(op_str: str):
    from mpi4py import MPI
    ops = {
        'sum': MPI.SUM,
        'max': MPI.MAX,
        'min': MPI.MIN,
    }
    return ops.get(op_str, MPI.SUM)

def parallel_split(n_total: int) -> Tuple[int, int]:
    rank = get_mpi_rank()
    size = get_mpi_size()

    chunk_size = n_total // size
    remainder = n_total % size

    if rank < remainder:
        start = rank * (chunk_size + 1)
        end = start + chunk_size + 1
    else:
        start = rank * chunk_size + remainder
        end = start + chunk_size

    return start, end

# =============================================================================
# Lattice utilities
# =============================================================================

def reciprocal_lattice_vectors(lattice: np.ndarray) -> np.ndarray:
    volume = np.dot(lattice[0], np.cross(lattice[1], lattice[2]))
    rec = np.zeros((3, 3))
    rec[0] = 2 * np.pi * np.cross(lattice[1], lattice[2]) / volume
    rec[1] = 2 * np.pi * np.cross(lattice[2], lattice[0]) / volume
    rec[2] = 2 * np.pi * np.cross(lattice[0], lattice[1]) / volume
    return rec

def get_qpoints_mesh(mesh: Tuple[int, int, int], is_shift: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
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

def gaussian_smearing(x: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

def adaptive_sigma(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray = None,
                     v4: np.ndarray = None, scalebroad: float = 1.0,
                     dq: float = 1.0) -> float:
    if v3 is None and v4 is None:
        sigma = scalebroad * np.linalg.norm(v1 - v2) * dq
    elif v4 is None:
        sigma = scalebroad * np.linalg.norm(v1 + v2 - v3) * dq
    else:
        sigma = scalebroad * np.linalg.norm(v1 + v2 - v3 - v4) * dq
    return max(sigma, 1e-6)

def bose_einstein(omega: np.ndarray, T: float) -> np.ndarray:
    kB = 1.380649e-23
    hbar = 1.054571817e-34
    THz_to_Hz = 1e12

    if T < 1e-10:
        return np.zeros_like(omega)

    x = hbar * omega * THz_to_Hz / (kB * T)
    x = np.clip(x, -700, 700)
    return 1.0 / (np.exp(x) - 1.0)

def mode_heat_capacity(omega: np.ndarray, T: float) -> np.ndarray:
    kB = 1.380649e-23
    hbar = 1.054571817e-34
    THz_to_Hz = 1e12

    if T < 1e-10:
        return np.zeros_like(omega)

    x = hbar * omega * THz_to_Hz / (kB * T)
    x = np.clip(x, -700, 700)
    return kB * x**2 * np.exp(x) / (np.exp(x) - 1.0)**2

def find_qpoint_index(q_target: np.ndarray, qpoints: np.ndarray,
                       tol: float = 1e-5) -> Optional[int]:
    """Vectorized q-point index search."""
    q_target = q_target - np.floor(q_target)
    diffs = qpoints - q_target
    diffs = diffs - np.round(diffs)
    matches = np.all(np.abs(diffs) < tol, axis=1)
    idx = np.where(matches)[0]
    return int(idx[0]) if len(idx) > 0 else None
