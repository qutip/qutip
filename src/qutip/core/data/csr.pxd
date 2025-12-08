#cython: language_level=3

from cpython cimport mem
from libcpp.algorithm cimport sort
from libc.math cimport fabs

cdef extern from *:
    void *PyMem_Calloc(size_t n, size_t elsize)

import numpy as np
cimport numpy as cnp

from qutip.core.data cimport base
from qutip.core.data.dense cimport Dense
from qutip.core.data.dia cimport Dia

cdef class CSR(base.Data):
    cdef double complex *data
    cdef base.idxint *col_index
    cdef base.idxint *row_index
    cdef size_t size
    cdef object _scipy
    cdef bint _deallocate
    cpdef CSR copy(CSR self)
    cpdef object as_scipy(CSR self, bint full=*)
    cpdef CSR sort_indices(CSR self)
    cpdef double complex trace(CSR self)
    cpdef CSR adjoint(CSR self)
    cpdef CSR conj(CSR self)
    cpdef CSR transpose(CSR self)


cdef struct Accumulator:
    # Provides the scatter/gather accumulator pattern for populating CSR/CSC
    # matrices row-by-row (or column-by-column for CSC) where entries may need
    # to be accumulated (summed) from several locations which may not be
    # sorted.
    #
    # See usage in `csr.from_coo_pointers` and `add_csr`; generally, add values
    # to the accumulator for this row by calling `scatter`, then fill the row
    # in the output by calling `gather`.  Prepare the accumulator to receive
    # the next row by calling `reset`.
    double complex *values
    size_t *modified
    base.idxint *nonzero
    size_t _cur_row, nnz, size
    bint _sorted

cdef inline Accumulator acc_alloc(size_t size):
    """
    Initialise this accumulator.  `size` should be the number of columns in the
    matrix (for CSR) or the number of rows (for CSC).
    """
    cdef Accumulator acc
    acc.values = <double complex *> mem.PyMem_Malloc(size * sizeof(double complex))
    acc.modified = <size_t *> PyMem_Calloc(size, sizeof(size_t))
    acc.nonzero = <base.idxint *> mem.PyMem_Malloc(size * sizeof(base.idxint))
    if acc.values == NULL or acc.modified == NULL or acc.nonzero == NULL:
        raise MemoryError
    acc.size = size
    acc.nnz = 0
    # The value of _cur_row doesn't actually need to match the true row in
    # the output, it just needs to be a unique number so that we can use it
    # as a sentinel in `modified` to tell if there's a value in the current
    # column.
    acc._cur_row = 1
    acc._sorted = True
    return acc

cdef inline void acc_scatter(Accumulator *acc, double complex value, base.idxint position) noexcept nogil:
    """
    Add a value to the accumulator for this row, in column `position`.  The
    value is added on to any value already scattered into this position.
    """
    # We have to branch on modified[position] anyway (to know whether to add an
    # entry in nonzero), so we _actually_ reset `values` here.  This has the
    # potential to save operations too, if the same column is never touched
    # again.
    if acc.modified[position] == acc._cur_row:
        acc.values[position] += value
    else:
        acc.values[position] = value
        acc.modified[position] = acc._cur_row
        acc.nonzero[acc.nnz] = position
        acc._sorted &= acc.nnz == 0 or acc.nonzero[acc.nnz - 1] < position
        acc.nnz += 1

cdef inline base.idxint acc_gather(Accumulator *acc, double complex *values, base.idxint *indices, double tol=0) noexcept nogil:
    """
    Copy all the accumulated values into this row into the output pointers.
    This will always output its values in sorted order.  The pointers should
    point to the first free space for data to be copied into.  This method will
    copy in _at most_ `self.nnz` elements into the pointers, but may copy in
    slightly fewer if some of them are now (explicit) zeros.  `self.nnz` is
    updated after each `self.scatter()` operation, and is reset by
    `self.reset()`.

    Return the actual number of elements copied in.
    """
    cdef size_t i, nnz=0, position
    cdef double complex value
    if not acc._sorted:
        sort(acc.nonzero, acc.nonzero + acc.nnz)
        acc._sorted = True
    for i in range(acc.nnz):
        position = acc.nonzero[i]
        value = acc.values[position]
        if fabs(value.real) < tol:
            value.real = 0
        if fabs(value.imag) < tol:
            value.imag = 0
        if value != 0:
            values[nnz] = value
            indices[nnz] = position
            nnz += 1
    return nnz

cdef inline void acc_reset(Accumulator *acc) noexcept nogil:
    """Prepare the accumulator to accept the next row of input."""
    # We actually don't need to do anything to reset other than to change
    # our sentinel values; the sentinel `_cur_row` makes it easy to detect
    # whether a value was set in this current row (and if not, `scatter`
    # resets it when it's used), while `nnz`
    acc.nnz = 0
    acc._sorted = True
    acc._cur_row += 1

cdef inline void acc_free(Accumulator *acc):
    mem.PyMem_Free(acc.values)
    mem.PyMem_Free(acc.modified)
    mem.PyMem_Free(acc.nonzero)


# Internal structure for sorting pairs of elements.  Not actually meant to be
# used in external code.
cdef struct _data_col:
    double complex data
    base.idxint col

cdef class Sorter:
    cdef size_t size
    cdef base.idxint **argsort
    cdef _data_col *sort

    cdef void inplace(Sorter self, CSR matrix, base.idxint ptr, size_t size) nogil
    cdef void copy(Sorter self,
                   double complex *dest_data, base.idxint *dest_cols,
                   double complex *src_data, base.idxint *src_cols,
                   size_t size) nogil


cpdef CSR fast_from_scipy(object sci)
cpdef CSR copy_structure(CSR matrix)
cpdef CSR sorted(CSR matrix)
cpdef base.idxint nnz(CSR matrix) nogil
cpdef CSR empty(base.idxint rows, base.idxint cols, base.idxint size)
cpdef CSR empty_like(CSR other)
cpdef CSR zeros(base.idxint rows, base.idxint cols)
cpdef CSR identity(base.idxint dimension, double complex scale=*)

cpdef CSR from_dense(Dense matrix)
cdef CSR from_coo_pointers(base.idxint *rows, base.idxint *cols, double complex *data,
                           base.idxint n_rows, base.idxint n_cols, base.idxint nnz,
                           double tol=*)
cpdef CSR from_dia(Dia matrix)

cpdef CSR _from_csr_blocks(base.idxint[:] block_rows, base.idxint[:] block_cols, CSR[:] block_ops,
                          base.idxint n_blocks, base.idxint block_size)
