# Fourthorder — Fourth-Order Interatomic Force Constants (Python 3 & spglib-2.7.0)

> **Upgraded version** of the original [FourPhonon/Fourthorder](https://github.com/FourPhonon/Fourthorder) package, ported from **Python 2.7 → Python 3** and modernized for **spglib-2.7.0**.

---

## 1. Original Developers & Contributions

This package is based on the original **Fourthorder** code developed by:

| Developer | Affiliation | Contribution |
|-----------|-------------|--------------|
| **Zherui Han** | Purdue University | Core algorithm & VASP interface |
| **Xiaolong Yang** | Purdue University | Symmetry analysis & displacement generation |
| **Wu Li** | Purdue University | Force-constant reconstruction |
| **Tianli Feng** | Purdue University | Sparse-matrix optimization |
| **Xiulin Ruan** | Purdue University | Project supervision & theory |

**Original repository:** [https://github.com/FourPhonon/Fourthorder](https://github.com/FourPhonon/Fourthorder)

The original authors pioneered the **minimal-displacement approach** for computing **fourth-order anharmonic force constants (IFCs)** using group theory and symmetry analysis, reducing the number of required DFT calculations from thousands to hundreds for typical systems.

---

## 2. Why This Upgrade?

### 2.1 Python 2.7 is End-of-Life
- **Python 2.7 reached end-of-life on January 1, 2020.**
- Modern HPC clusters, conda environments, and pip no longer ship Python 2.7 by default.
- Dependencies (NumPy, SciPy, Cython) have dropped Python 2 support.

### 2.2 spglib-2.7.0 Modernization
- The original code targeted spglib 1.x, whose C API has evolved.
- **spglib-2.7.0** provides:
  - Updated `SpglibDataset` structure with additional fields (`hall_number`, `choice`, `crystallographic_orbits`, etc.)
  - Improved numerical stability for symmetry detection
  - Better compatibility with modern compilers and build systems
  - Active maintenance and bug fixes

### 2.3 What Was Changed
| File | Change |
|------|--------|
| `Fourthorder_vasp.py` | `print` statements → `print()`; `xrange` → `range`; `StringIO` → `io.StringIO`; `f.next()` → `next(f)`; `hashlib` bytes handling |
| `Fourthorder_common.py` | Same Python 3 syntax migration |
| `Fourthorder_espresso.py` | Same Python 3 syntax migration |
| `Fourthorder_core.pyx` | `print`/`xrange` fixes; `decode("ASCII")` for spglib-2.7.0 bytes fields |
| `cfourthorder_core.pxd` | `SpglibDataset` struct updated to match spglib-2.7.0 C header |
| `setup.py` | `distutils` → `setuptools`; paths updated for spglib-2.7.0 |
| `Fourthorder_core.c` | Regenerated from `.pyx` via Cython 3.2.9 with `buffer_max_dims=8` |

---

## 3. Program Overview

Fourthorder computes **fourth-order (quartic) anharmonic force constants** from first-principles DFT calculations. These constants are essential for:

- **Four-phonon scattering rates** in lattice thermal conductivity calculations
- **Anharmonic free energy** at high temperatures
- **Thermal expansion** beyond the quasi-harmonic approximation

### Key Features
1. **Symmetry-based minimization** — Uses spglib to find the irreducible set of atomic quadruplets, minimizing the number of DFT supercell calculations.
2. **Two-stage workflow**:
   - **`sow`** — Generates the undisplaced supercell (`4TH.SPOSCAR`) and displaced configurations (`4TH.POSCAR.*`).
   - **`reap`** — Reads forces from VASP `vasprun.xml` files and reconstructs the full `FORCE_CONSTANTS_4TH` matrix.
3. **VASP interface** — Native support for VASP `POSCAR`/`vasprun.xml` format.
4. **Sparse/dense matrix output** — Automatically selects the most efficient storage format.

---

## 4. Compilation

### Prerequisites
- Python **3.8+**
- NumPy
- SciPy
- Cython **3.0+** (only if you need to regenerate `.c` from `.pyx`)
- GCC or compatible C compiler
- **spglib-2.7.0** (C library + headers)

### Step 1: Install spglib-2.7.0

**Option A — pip (easiest, but may not expose C headers):**
```bash
pip install spglib==2.7.0
```

**Option B — Build from source (recommended for full C API access):**
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
cd fourthorder_py3
python3 setup.py build_ext --inplace
```

If successful, you will see:
```
Fourthorder_core.cpython-3xx-x86_64-linux-gnu.so
```

---

## 5. Usage

### 5.1 Prepare your primitive cell

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

### 5.2 Stage 1 — `sow`: Generate displaced configurations

```bash
python3 Fourthorder_vasp.py sow <na> <nb> <nc> <displacement>
```

- `na`, `nb`, `nc` — Supercell dimensions along the three lattice vectors
- `displacement` — Displacement magnitude in **nm** (e.g., `-2` = 0.02 nm)

**Example:**
```bash
python3 Fourthorder_vasp.py sow 4 4 4 -2
```

**Output:**
- `4TH.SPOSCAR` — Undisplaced supercell (128 atoms for 2-atom primitive × 4×4×4)
- `4TH.POSCAR.001` ~ `4TH.POSCAR.744` — 744 displaced configurations

**Console output (verified):**
```text
Reading POSCAR
Analyzing the symmetries
- Symmetry group F-43m detected
- 24 symmetry operations
Creating the supercell
Computing all distances in the supercell
- Automatic cutoff: 0.4633186847509646 nm
Looking for an irreducible set of fourth-order IFCs
- 23 quartet equivalence classes found
- 744 DFT runs are needed
Writing undisplaced coordinates to 4TH.SPOSCAR
Writing displaced coordinates to 4TH.POSCAR.*
```

### 5.3 Stage 2 — DFT calculations

For each `4TH.POSCAR.*`, run a VASP single-point force calculation:

```bash
for i in $(seq -f "%03g" 1 744); do
    mkdir job_$i
    cp 4TH.POSCAR.$i job_$i/POSCAR
    cp POTCAR KPOINTS INCAR job_$i/
    cd job_$i && mpirun -np 32 vasp_std && cd ..
done
```

### 5.4 Stage 3 — `reap`: Reconstruct force constants

```bash
find job_* -name vasprun.xml | sort -n |     python3 Fourthorder_vasp.py reap 4 4 4 -2
```

**Output:**
- `FORCE_CONSTANTS_4TH` — Full fourth-order force-constant array

---

## 6. File Reference

| File | Description |
|------|-------------|
| `Fourthorder_vasp.py` | Main VASP interface (`sow`/`reap`) |
| `Fourthorder_common.py` | Shared utilities (supercell generation, distance tables) |
| `Fourthorder_espresso.py` | Quantum ESPRESSO interface (unchanged from upstream) |
| `Fourthorder_core.pyx` | Cython core: symmetry operations, sparse/dense matrix builders |
| `cfourthorder_core.pxd` | C declarations for spglib-2.7.0 API |
| `Fourthorder_core.c` | Generated C source (Cython 3.2.9) |
| `setup.py` | Build configuration |

---

## 7. Citation

If you use this code in your research, please cite the original Fourthorder paper:

> **Zherui Han, Xiaolong Yang, Wu Li, Tianli Feng, and Xiulin Ruan**, *"FourPhonon: An extension module to ShengBTE for computing four-phonon scattering rates and thermal conductivity"*, [arXiv:2108.07763](https://arxiv.org/abs/2108.07763) (2021).

---

## 8. License

This upgraded version retains the **GNU General Public License v3.0** of the original project.

---

## 9. Troubleshooting

| Issue | Solution |
|-------|----------|
| `ImportError: libsymspg.so` | Add spglib-2.7.0 `lib/` to `LD_LIBRARY_PATH` |
| `SpglibDataset` struct mismatch | Ensure `cfourthorder_core.pxd` matches your spglib version |
| Cython `buffer_max_dims` error | Cython 3.x limits default dims to 7; this build uses `buffer_max_dims=8` |
| `encode`/`decode` bytes error | Already fixed for spglib-2.7.0; if using other spglib versions, adjust `.decode()` vs `.encode()` in `Fourthorder_core.pyx` |

---

**Maintainer of this Python 3 / spglib-2.7.0 upgrade:** Community contribution based on the original FourPhonon project.
