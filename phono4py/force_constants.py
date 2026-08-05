"""
Read second, third, and fourth order force constants from files.
"""

import numpy as np
from typing import Tuple

def read_force_constants_2nd(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(filename, 'r') as f:
        lines = f.readlines()
    natoms = int(lines[0].strip())
    npairs = natoms * natoms
    atom_pairs, force_constants, cell_vectors = [], [], []
    idx = 1
    for _ in range(npairs):
        while idx < len(lines) and lines[idx].strip() == '':
            idx += 1
        pair = list(map(int, lines[idx].strip().split()))
        idx += 1
        mat = []
        for _ in range(3):
            while idx < len(lines) and lines[idx].strip() == '':
                idx += 1
            row = list(map(float, lines[idx].strip().split()))
            mat.append(row)
            idx += 1
        atom_pairs.append([pair[0] - 1, pair[1] - 1])
        force_constants.append(mat)
        cell_vectors.append([0.0, 0.0, 0.0])
    return (np.array(atom_pairs), np.array(force_constants), np.array(cell_vectors))

def read_force_constants_3rd(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with open(filename, 'r') as f:
        lines = f.readlines()
    nb = int(lines[0].strip())
    atom_triplets, cell_vectors_2, cell_vectors_3, force_constants_list = [], [], [], []
    idx = 1
    for _ in range(nb):
        while idx < len(lines) and lines[idx].strip() == '':
            idx += 1
        idx += 1
        while idx < len(lines) and lines[idx].strip() == '':
            idx += 1
        cell2 = list(map(float, lines[idx].strip().split()))
        idx += 1
        cell3 = list(map(float, lines[idx].strip().split()))
        idx += 1
        atoms = list(map(int, lines[idx].strip().split()))
        idx += 1
        fc_tensor = np.zeros((3, 3, 3))
        for _ in range(27):
            while idx < len(lines) and lines[idx].strip() == '':
                idx += 1
            parts = lines[idx].strip().split()
            i, j, k = int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2]) - 1
            val = float(parts[3])
            fc_tensor[i, j, k] = val
            idx += 1
        atom_triplets.append([atoms[0] - 1, atoms[1] - 1, atoms[2] - 1])
        cell_vectors_2.append(cell2)
        cell_vectors_3.append(cell3)
        force_constants_list.append(fc_tensor)
    return (np.array(atom_triplets), np.array(cell_vectors_2), np.array(cell_vectors_3), np.array(force_constants_list))

def read_force_constants_4th(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with open(filename, 'r') as f:
        lines = f.readlines()
    nb = int(lines[0].strip())
    atom_quartets, cell_vectors_2, cell_vectors_3, cell_vectors_4, force_constants_list = [], [], [], [], []
    idx = 1
    for _ in range(nb):
        while idx < len(lines) and lines[idx].strip() == '':
            idx += 1
        idx += 1
        while idx < len(lines) and lines[idx].strip() == '':
            idx += 1
        cell2 = list(map(float, lines[idx].strip().split()))
        idx += 1
        cell3 = list(map(float, lines[idx].strip().split()))
        idx += 1
        cell4 = list(map(float, lines[idx].strip().split()))
        idx += 1
        atoms = list(map(int, lines[idx].strip().split()))
        idx += 1
        fc_tensor = np.zeros((3, 3, 3, 3))
        line_count = 0
        for _ in range(81):
            while idx < len(lines) and lines[idx].strip() == '':
                idx += 1
            parts = lines[idx].strip().split()
            alpha, beta, gamma = int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2]) - 1
            val = float(parts[3])
            delta = line_count // 27
            fc_tensor[alpha, beta, gamma, delta] = val
            line_count += 1
            idx += 1
        atom_quartets.append([atoms[0] - 1, atoms[1] - 1, atoms[2] - 1, atoms[3] - 1])
        cell_vectors_2.append(cell2)
        cell_vectors_3.append(cell3)
        cell_vectors_4.append(cell4)
        force_constants_list.append(fc_tensor)
    return (np.array(atom_quartets), np.array(cell_vectors_2), np.array(cell_vectors_3), np.array(cell_vectors_4), np.array(force_constants_list))
