# Thirdorder — Third-Order Interatomic Force Constants (Python 3 & spglib-2.7.0)

> **Upgraded version** of the original [sousaw/thirdorder](https://bitbucket.org/sousaw/thirdorder) package, ported from **Python 2.7 → Python 3** and modernized for **spglib-2.7.0**.

---

## 1. Original Developers & Contributions

This package is based on the original **Thirdorder** code developed by:

| Developer | Affiliation | Contribution |
|-----------|-------------|--------------|
| **Wu Li** | Université catholique de Louvain / Wuhan University | Core algorithm, VASP interface, force-constant reconstruction |
| **Jesús Carrete Montaña** | Université catholique de Louvain | Symmetry analysis, supercell generation, displacement strategy |
| **Natalio Mingo Bisquert** | CEA-Grenoble | Project supervision, thermal conductivity theory |
| **Antti J. Karttunen** | Aalto University | CASTEP interface, code optimization |
| **Genadi Naydenov** | Université catholique de Louvain | Quantum ESPRESSO interface |

**Original repository:** [https://bitbucket.org/sousaw/thirdorder](https://bitbucket.org/sousaw/thirdorder)

The original authors pioneered the **minimal-displacement approach** for computing **third-order anharmonic force constants (IFCs)** using group theory and symmetry analysis. This method dramatically reduces the number of required DFT supercell calculations compared to brute-force finite-difference approaches.

---

## 2. Why This Upgrade?

### 2.1 Python 2.7 is End-of-Life
- **Python 2.7 reached end-of-life on January 1, 2020.**
- Modern HPC clusters, conda environments, and `pip` no longer ship Python 2.7 by default.
- Core scientific Python dependencies (NumPy ≥1.20, SciPy ≥1.6, Cython ≥3.0) have **dropped Python 2 support entirely**.
- Running the original code requires maintaining obsolete Python 2 environments, which is increasingly impractical.

### 2.2 spglib-2.7.0 Modernization
- The original code targeted **spglib 1.x**, whose C API has evolved significantly.
- **spglib-2.7.0** provides:
  - Updated `SpglibDataset` structure with additional fields (`hall_number`, `choice`, `crystallographic_orbits`, `site_symmetry_symbols`, etc.)
  - Improved numerical stability for symmetry detection in edge-case crystal structures
  - Better compatibility with modern compilers (GCC ≥10, Clang ≥12) and build systems (CMake ≥3.15)
  - Active maintenance, bug fixes, and security patches
  - Consistent API with the companion **Fourthorder** package

### 2.3 What Was Changed

| File | Change |
|------|--------|
| `thirdorder_vasp.py` | Removed `from __future__ import print_function`; `xrange` → `range`; `StringIO` → `io.StringIO`; `f.next()` → `next(f)` |
| `thirdorder_common.py` | Same Python 3 syntax migration |
| `thirdorder_espresso.py` | Same Python 3 syntax migration |
| `thirdorder_castep.py` | Same Python 3 syntax migration |
| `thirdorder_core.pyx` | Removed `PY_MAJOR_VERSION` / `unicode()` branching; `cdef bytes` + `decode("ASCII")` for spglib-2.7.0 bytes fields |
| `cthirdorder_core.pxd` | `SpglibDataset` struct updated to match **spglib-2.7.0** C header (3×3 matrices, new fields) |
| `setup.py` | `distutils.core` → `setuptools` (Python 3.12+ compatible); paths updated for spglib-2.7.0 |
| `thirdorder_core.c` | Regenerated from `.pyx` via **Cython 3.2.9** with `buffer_max_dims=8` |

---

## 3. Program Overview

Thirdorder computes **third-order (cubic) anharmonic force constants** from first-principles DFT calculations. These constants are essential for:

- **Three-phonon scattering rates** — the dominant phonon–phonon interaction mechanism governing lattice thermal conductivity
- **Phonon linewidths** and spectral functions at finite temperature
- **Anharmonic free energy** corrections beyond the quasi-harmonic approximation

### Key Features
1. **Symmetry-based minimization** — Uses spglib to find the irreducible set of atomic triplets, minimizing the number of DFT supercell calculations.
2. **Two-stage workflow**:
   - **`sow`** — Generates the undisplaced supercell (`3RD.SPOSCAR`) and displaced configurations (`3RD.POSCAR.*`).
   - **`reap`** — Reads forces from VASP `vasprun.xml` files and reconstructs the full `FORCE_CONSTANTS_3RD` matrix.
3. **Multi-code support** — Native interfaces for **VASP**, **Quantum ESPRESSO**, and **CASTEP**.
4. **Sparse/dense matrix output** — Automatically selects the most efficient storage format for the force-constant tensor.

---

## 4. Prerequisites

- Python **3.8+**
- NumPy
- SciPy
- Cython **3.0+** (only if you need to regenerate `.c` from `.pyx`)
- GCC or compatible C compiler
- **spglib-2.7.0** (C library + headers)

---

## 5. Compilation

### Step 1: Install spglib-2.7.0

**Option A — pip (easiest):**
```bash
pip install spglib==2.7.0
```

**Option B — Build from source (recommended for HPC clusters):**
```bash
wget https://github.com/spglib/spglib/archive/refs/tags/v2.7.0.tar.gz
tar -xzf v2.7.0.tar.gz
cd spglib-2.7.0
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/spglib-2.7.0          -DCMAKE_POSITION_INDEPENDENT_CODE=ON
make -j4
make install
```

### Step 2: Update `setup.py`

Edit `setup.py` to point to your spglib-2.7.0 installation:

```python
INCLUDE_DIRS=["/path/to/spglib-2.7.0/include"]
LIBRARY_DIRS=["/path/to/spglib-2.7.0/lib"]
```

### Step 3: Build the extension

```bash
cd thirdorder_py3
python3 setup.py build_ext --inplace
```

If successful, you will see:
```
thirdorder_core.cpython-3xx-x86_64-linux-gnu.so
```

---

## 6. Usage

### 6.1 Prepare your primitive cell

Create a `POSCAR` file for your primitive unit cell. Example (InAs, zincblende):

```text
InAs
   6.00000000000000
     0.0000000000000000    0.5026468896190005    0.5026468896190005
     0.5026468896190005    0.0000000000000000    0.5026468896190005
     0.5026468896190005    0.5026468896190005    0.0000000000000000
   In   As
   1   1
Direct
  0.0000000000000000  0.0000000000000000  0.0000000000000000
  0.2500000000000000  0.2500000000000000  0.2500000000000000
```

### 6.2 Stage 1 — `sow`: Generate displaced configurations

```bash
python3 thirdorder_vasp.py sow <na> <nb> <nc> <displacement>
```

- `na`, `nb`, `nc` — Supercell dimensions along the three lattice vectors
- `displacement` — Displacement magnitude in **nm** (e.g., `-2` = 0.02 nm)

**Example:**
```bash
python3 thirdorder_vasp.py sow 4 4 4 -2
```

**Output (verified):**
```text
Reading POSCAR
Analyzing the symmetries
- Symmetry group F-43m detected
- 24 symmetry operations
Creating the supercell
Computing all distances in the supercell
- Automatic cutoff: 0.4633186847509646 nm
Looking for an irreducible set of third-order IFCs
- 12 triplet equivalence classes found
- 92 DFT runs are needed
Writing undisplaced coordinates to 3RD.SPOSCAR
Writing displaced coordinates to 3RD.POSCAR.*
```

**Generated files:**
- `3RD.SPOSCAR` — Undisplaced supercell (128 atoms for 2-atom primitive × 4×4×4)
- `3RD.POSCAR.01` ~ `3RD.POSCAR.92` — 92 displaced configurations

**Displacement verification:**
Each `3RD.POSCAR.*` file contains **exactly one displaced atom** (≈0.017 Å shift), with the remaining 127 atoms unchanged — confirming correct symmetry-based displacement generation.

### 6.3 Stage 2 — DFT calculations

For each `3RD.POSCAR.*`, run a VASP single-point force calculation:

```bash
for i in $(seq -f "%02g" 1 92); do
    mkdir job_$i
    cp 3RD.POSCAR.$i job_$i/POSCAR
    cp POTCAR KPOINTS INCAR job_$i/
    cd job_$i && mpirun -np 32 vasp_std && cd ..
done
```

### 6.4 Stage 3 — `reap`: Reconstruct force constants

```bash
find job_* -name vasprun.xml | sort -n |     python3 thirdorder_vasp.py reap 4 4 4 -2
```

**Output:**
- `FORCE_CONSTANTS_3RD` — Full third-order force-constant array

---

## 7. File Reference

| File | Description |
|------|-------------|
| `thirdorder_vasp.py` | Main VASP interface (`sow` / `reap`) |
| `thirdorder_common.py` | Shared utilities (supercell generation, distance tables, symmetry wrappers) |
| `thirdorder_espresso.py` | Quantum ESPRESSO interface |
| `thirdorder_castep.py` | CASTEP interface |
| `thirdorder_core.pyx` | Cython core: symmetry operations, sparse/dense matrix builders |
| `cthirdorder_core.pxd` | C declarations for spglib-2.7.0 API |
| `thirdorder_core.c` | Generated C source (Cython 3.2.9) |
| `setup.py` | Build configuration |

---

## 8. Citation

If you use this code in your research, please cite the original Thirdorder paper:

> **Wu Li, Jesús Carrete, Natalio Mingo, and Shiyou Chen**, *"Thermal conductivity of AlN revisited: Dependence on isotope purity and the role of higher-order anharmonicity"*, Phys. Rev. B **90**, 094305 (2014). [DOI: 10.1103/PhysRevB.90.094305](https://doi.org/10.1103/PhysRevB.90.094305)

---

## 9. License

This upgraded version retains the **GNU General Public License v3.0** of the original project.

---

## 10. Troubleshooting

| Issue | Solution |
|-------|----------|
| `ImportError: libsymspg.so` | Add spglib-2.7.0 `lib/` to `LD_LIBRARY_PATH` |
| `SpglibDataset` struct mismatch | Ensure `cthirdorder_core.pxd` matches your spglib version |
| Cython `buffer_max_dims` error | This build uses `buffer_max_dims=8` via `Options.buffer_max_dims` |
| `encode`/`decode` bytes error | Already fixed for spglib-2.7.0; if using other spglib versions, adjust `.decode()` vs `.encode()` in `thirdorder_core.pyx` |
| `print` is not callable | Ensure you are using Python 3, not Python 2 |

---

**Maintainer of this Python 3 / spglib-2.7.0 upgrade:** Community contribution based on the original Thirdorder project.
