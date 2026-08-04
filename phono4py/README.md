# phono4py v0.2.0

Advanced four-phonon scattering and lattice thermal conductivity solver, 
extending the phono3py philosophy to 4th-order anharmonicity.

## Features

- **4-phonon scattering rates**: Full calculation of 4-phonon scattering 
  including both Type I (2→2) and Type II (1→3) processes
- **3-phonon scattering**: Standard 3-phonon scattering rates
- **MPI parallelization**: Parallel q-point and band loops using mpi4py
- **spglib 2.7.0 integration**: Symmetry-reduced irreducible q-point meshes
- **Iterative BTE solver**: Omini-Sparavigna iterative scheme for exact 
  solution of linearized phonon Boltzmann transport equation
- **HDF5 output**: Compatible with phono3py format

## Installation

```bash
pip install numpy h5py phonopy spglib mpi4py
```

Or for conda:
```bash
conda install numpy h5py phonopy spglib mpi4py
```

## File Structure

```
phono4py/
├── __init__.py          # Package initialization
├── force_constants.py   # Read 2nd/3rd/4th order FC files
├── symmetry.py          # spglib 2.7.0 symmetry analysis
├── harmonic.py          # Harmonic phonon properties (phonopy wrapper)
├── interaction.py       # 3-phonon/4-phonon interaction matrix elements
├── scattering.py        # Scattering rate calculations (MPI parallelized)
├── bte_solver.py        # Iterative BTE solver
├── conductivity.py      # Thermal conductivity calculation
├── main.py              # Main Phono4py class
└── utils.py             # Utility functions and MPI helpers
```

## Usage

### Basic Usage

```python
from phonopy.structure.atoms import PhonopyAtoms
from phono4py import Phono4py
import numpy as np

# Define crystal structure
unitcell = PhonopyAtoms(
    symbols=['B', 'As'],
    scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
    cell=[[0, 2.38, 2.38], [2.38, 0, 2.38], [2.38, 2.38, 0]],
)

# Initialize phono4py
ph4 = Phono4py(
    unitcell=unitcell,
    supercell_matrix=[2, 2, 2],
    mesh=[11, 11, 11],
    temperatures=np.arange(100, 1100, 100),
    scalebroad=0.1,
    use_symmetry=True,
)

# Read force constants
ph4.read_force_constants(
    fc2_file="FORCE_CONSTANTS_2ND",
    fc3_file="FORCE_CONSTANTS_3RD",
    fc4_file="FORCE_CONSTANTS_4TH"
)

# Run calculation
kappa, gamma = ph4.run(
    include_3ph=True,
    include_4ph=True,
    use_iterative_bte=True,
    output_file="kappa-phono4py.hdf5"
)
```

### MPI Parallel Run

```bash
mpirun -np 8 python example_BAs.py
```

The q-point loops in scattering calculations are automatically parallelized 
across MPI processes.

### Symmetry Reduction

Enable symmetry reduction to significantly reduce computational cost:

```python
ph4 = Phono4py(
    ...,
    use_symmetry=True,      # Enable spglib
    is_shift=(0, 0, 0),     # Gamma-centered mesh
    symprec=1e-5,           # Symmetry precision
)
```

### Iterative BTE vs RTA

```python
# Iterative BTE (more accurate, slower)
kappa, gamma = ph4.run(use_iterative_bte=True, max_iter=50)

# RTA (faster, less accurate for materials with strong N-processes)
kappa, gamma = ph4.run(use_iterative_bte=False)
```

## Input File Formats

### FORCE_CONSTANTS_2ND (Phonopy format)
```
[number of atoms]
[atom1] [atom2]
[3x3 force constant matrix]
...
```

### FORCE_CONSTANTS_3RD (ShengBTE sparse format)
```
[number of blocks]
[index]
[cell vector 2]
[cell vector 3]
[atom1] [atom2] [atom3]
[27 lines: i j k value]
...
```

### FORCE_CONSTANTS_4TH (FourPhonon sparse format)
```
[number of blocks]
[index]
[cell vector 2]
[cell vector 3]
[cell vector 4]
[atom1] [atom2] [atom3] [atom4]
[81 lines: i j k value]
...
```

## Output HDF5 Format

The output file `kappa-phono4py.hdf5` contains:

| Dataset | Shape | Description |
|---------|-------|-------------|
| `mesh` | (3,) | q-point mesh |
| `qpoint` | (nq, 3) | q-points in fractional coords |
| `weight` | (nq,) | q-point weights |
| `temperature` | (nT,) | Temperatures in K |
| `frequency` | (nq, nband) | Phonon frequencies in THz |
| `eigenvector` | (nq, nband, natom, 3) | Phonon eigenvectors |
| `group_velocity` | (nq, nband, 3) | Group velocities in km/s |
| `gamma` | (nT, nq, nband) | Total scattering rates in THz |
| `gamma_3ph` | (nT, nq, nband) | 3-phonon scattering rates |
| `gamma_4ph` | (nT, nq, nband) | 4-phonon scattering rates |
| `kappa` | (nT, 6) | Thermal conductivity [xx,yy,zz,yz,xz,xy] |
| `mode_kappa` | (nT, nq, nband, 3, 3) | Mode-resolved conductivity |

## Testing with BAs Example

The BAs example from FourPhonon can be used to test:

```bash
cd Example-BAs/input
python ../../example_BAs.py
```

Expected room temperature thermal conductivity for BAs:
- 3-phonon only: ~2200 W/mK
- 3+4-phonon (RTA): ~1200 W/mK
- 3+4-phonon (iterative): ~1260 W/mK

## References

1. T. Feng and X. Ruan, Phys. Rev. B **93**, 045202 (2016)
2. T. Feng, L. Lindsay, and X. Ruan, Phys. Rev. B **96**, 161201 (2017)
3. Z. Han et al., Comput. Phys. Commun. **270**, 108179 (2022)
4. M. Omini and A. Sparavigna, Physica B **212**, 101 (1995)
5. M. Omini and A. Sparavigna, Phys. Rev. B **53**, 9064 (1996)

## License

GPL-3.0
