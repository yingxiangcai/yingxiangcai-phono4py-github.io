# thirdorder, help compute anharmonic IFCs from minimal sets of displacements
# Copyright (C) 2012-2018 Wu Li
# Copyright (C) 2012-2018 Jesús Carrete Montaña
# Copyright (C) 2012-2018 Natalio Mingo Bisquert
# Copyright (C) 2014-2018 Antti J. Karttunen
# Copyright (C) 2016-2018 Genadi Naydenov
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# This file contains declarations needed by the Cython wrapper around
# spglib.
# Updated for spglib-2.7.0 compatibility.

cdef extern from "spglib/spglib.h":
    ctypedef struct SpglibDataset:
        int spacegroup_number
        int hall_number
        char international_symbol[11]
        char hall_symbol[17]
        char choice[6]
        double transformation_matrix[3][3]
        double origin_shift[3]
        int n_operations
        int (*rotations)[3][3]
        double (*translations)[3]
        int n_atoms
        int *wyckoffs
        char (*site_symmetry_symbols)[7]
        int *equivalent_atoms
        int *crystallographic_orbits
        double primitive_lattice[3][3]
        int *mapping_to_primitive
        int n_std_atoms
        double std_lattice[3][3]
        int *std_types
        double (*std_positions)[3]
        double std_rotation_matrix[3][3]
        int *std_mapping_to_primitive
        char pointgroup_symbol[6]

    SpglibDataset *spg_get_dataset(double lattice[3][3],
                                   double position[][3],
                                   int types[],
                                   int num_atom,
                                   double symprec)
    void spg_free_dataset(SpglibDataset *dataset)
