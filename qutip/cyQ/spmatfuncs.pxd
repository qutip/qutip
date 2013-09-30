# This file is part of QuTiP.
#
#    QuTiP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    QuTiP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTiP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2013, Paul D. Nation & Robert J. Johansson
#
###########################################################################

cimport numpy as np

ctypedef np.complex128_t CTYPE_t
ctypedef np.float64_t DTYPE_t

cpdef np.ndarray[CTYPE_t, ndim=1] spmv_csr(np.ndarray[CTYPE_t, ndim=1] data,
                                       np.ndarray[int] idx,
                                       np.ndarray[int] ptr,
                                       np.ndarray[CTYPE_t, ndim=1] vec)
