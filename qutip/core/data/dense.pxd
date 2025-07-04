#cython: language_level=3

cimport numpy as cnp

from . cimport base
from qutip.core.data.csr cimport CSR
from qutip.core.data.dia cimport Dia

cdef class Dense(base.Data):
    cdef double complex *data
    cdef readonly bint fortran
    cdef object _np
    cdef bint _deallocate
    cdef void _fix_flags(Dense self, object array, bint make_owner=*)
    cpdef Dense reorder(Dense self, int fortran=*)
    cpdef Dense copy(Dense self)
    cpdef object as_ndarray(Dense self)
    cpdef object to_array(Dense self)
    cpdef double complex trace(Dense self)
    cpdef Dense adjoint(Dense self)
    cpdef Dense conj(Dense self)
    cpdef Dense transpose(Dense self)

cpdef Dense fast_from_numpy(object array)
cdef Dense wrap(double complex *ptr, base.idxint rows, base.idxint cols, bint fortran=*)
cpdef Dense empty(base.idxint rows, base.idxint cols, bint fortran=*)
cpdef Dense empty_like(Dense other, int fortran=*)
cpdef Dense zeros(base.idxint rows, base.idxint cols, bint fortran=*)
cpdef Dense identity(base.idxint dimension, double complex scale=*,
                     bint fortran=*)
cpdef Dense from_csr(CSR matrix, bint fortran=*)
cpdef Dense from_dia(Dia matrix)
cpdef long nnz(Dense matrix, double tol=*)
