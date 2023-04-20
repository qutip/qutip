#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from libc.string cimport memset, memcpy

from libcpp cimport bool
from libcpp.algorithm cimport sort

cimport cython

from cpython cimport mem

import numbers
import warnings

import numpy as np
cimport numpy as cnp
import scipy.sparse
from scipy.sparse import dia_array as scipy_dia_array
try:
    from scipy.sparse.data import _data_matrix as scipy_data_matrix
except ImportError:
    # The file data was renamed to _data from scipy 1.8.0
    from scipy.sparse._data import _data_matrix as scipy_data_matrix
from scipy.linalg cimport cython_blas as blas

from qutip.core.data cimport base, Dense
from qutip.core.data.adjoint cimport adjoint_diag, transpose_diag, conj_diag
from qutip.core.data.trace cimport trace_diag
from qutip.core.data.tidyup cimport tidyup_diag
from .base import idxint_dtype
from qutip.settings import settings

cnp.import_array()

cdef extern from *:
    void PyArray_ENABLEFLAGS(cnp.ndarray arr, int flags)
    void *PyDataMem_NEW(size_t size)
    void PyDataMem_FREE(void *ptr)

__all__ = ['Diag']

cdef int _ONE = 1

cdef object _dia_matrix(data, offsets, shape):
    """
    Factory method of scipy diag_matrix: we skip all the index type-checking
    because this takes tens of microseconds, and we already know we're in
    a sensible format.
    """
    cdef object out = scipy_dia_array.__new__(scipy_diag_matrix)
    # `_data_matrix` is the first object in the inheritance chain which
    # doesn't have a really slow __init__.
    scipy_data_matrix.__init__(out)
    out.data = data
    out.offsets = offsets
    out._shape = shape
    return out


cdef class Diag(base.Data):
    def __cinit__(self, *args, **kwargs):
        self._deallocate = True

    def __init__(self, arg=None, shape=None, bint=True, bint=tidyup=False):
        cdef size_t ptr
        cdef base.idxint col
        cdef object data, col_index, row_index

        if isinstance(arg, scipy.sparse.spmatrix):
            arg = arg.todia()
            if shape is not None and shape != arg.shape:
                raise ValueError("".join([
                    "shapes do not match: ", str(shape), " and ", str(arg.shape),
                ]))
            shape = arg.shape
            arg = (arg.data, arg.offsets)
        if not isinstance(arg, tuple):
            raise TypeError("arg must be a scipy matrix or tuple")
        if len(arg) != 2:
            raise ValueError("arg must be a (data, offsets) tuple")
        data = np.array(arg[0], dtype=np.complex128, copy=copy, order='C')
        offsets = np.array(arg[1], dtype=idxint_dtype, copy=copy, order='C')

        self._deallocate = False
        self.data = <double complex *> cnp.PyArray_GETPTR1(data, 0)
        self.offsets = <base.idxint *> cnp.PyArray_GETPTR1(offsets, 0)
        self.num_diag = self.offsets.shape[0]
        self.size = self.data.shape[0]

        if shape is None:
            warnings.warn("instantiating Diag matrix of unknown shape")
            nrows = 0
            ncols = 0
            for i in self.num_diag:
                if self.offsets[i] >= 0:
                    j = 0
                    while self.data[i, j] == 0.0:
                        j += 1
                    nrows = max(nrows, j)
                    ncols = max(ncols, self.offsets[i] + j)
                else:
                    j = 0
                    while self.data[i, self.size -j - 1] == 0.0:
                        j += 1
                    nrows = max(nrows, j - self.offsets[i])
                    ncols = max(ncols, j)
            self.shape = (nrows, ncols)
        else:
            if not isinstance(shape, tuple):
                raise TypeError("shape must be a 2-tuple of positive ints")
            if not (len(shape) == 2
                    and isinstance(shape[0], int)
                    and isinstance(shape[1], int)
                    and shape[0] > 0
                    and shape[1] > 0):
                raise ValueError("shape must be a 2-tuple of positive ints")
            self.shape = shape

        self._scipy = _dia_matrix(data, col_index, row_index, self.shape)
        if tidyup:
            tidyup_diag(self, settings.core['auto_tidyup_atol'], True)

    def __reduce__(self):
        return (fast_from_scipy, (self.as_scipy(),))

    cpdef Diag copy(self):
        """
        Return a complete (deep) copy of this object.

        If the type currently has a scipy backing, such as that produced by
        `as_scipy`, this will not be copied.  The backing is a view onto our
        data, and a straight copy of this view would be incorrect.  We do not
        create a new view at copy time, since the user may only access this
        through a creation method, and creating it ahead of time would incur an
        unnecessary speed penalty for users who do not need it (including
        low-level C code).
        """
        cdef Diag out = empty_like(self)
        memcpy(out.data, self.data, self.num_diag * self.size * sizeof(double complex))
        memcpy(out.col_index, self.offsets, self.num_diag*sizeof(base.idxint))

        return out

    cpdef object to_array(self):
        """
        Get a copy of this data as a full 2D, C-contiguous NumPy array.  This
        is not a view onto the data, and changes to new array will not affect
        the original data structure.
        """
        cdef cnp.npy_intp *dims = [self.shape[0], self.shape[1]]
        cdef object out = cnp.PyArray_ZEROS(2, dims, cnp.NPY_COMPLEX128, 0)
        cdef double complex [:, ::1] buffer = out
        cdef size_t diag, n
        for diag in range(self.num_diag):
            for n in range(max(self.offsets[diag], 0), min(self.size + self.offsets[diag], self.size)):
                buffer[self.offsets[diag] + n, self.offsets[diag] + n] = self.data[diag, n]
        return out

    cpdef object as_scipy(self):
        """
        Get a view onto this object as a `scipy.sparse.dia_matrix`.  The
        underlying data structures are exposed, such that modifications to the
        `data`, `indices` and `indptr` buffers in the resulting object will
        modify this object too.
        """
        # We store a reference to the scipy matrix not only for caching this
        # relatively expensive method, but also because we transferred
        # ownership of our data to the numpy arrays, and we can't allow them to
        # be collected while we're alive.
        if self._scipy is not None:
            return self._scipy
        cdef cnp.npy_intp num_diag = self.num_diag
        cdef cnp.npy_intp size = self.size
        data = cnp.PyArray_SimpleNewFromData(1, [num_diag, size],
                                             cnp.NPY_COMPLEX128,
                                             self.data)
        offsets = cnp.PyArray_SimpleNewFromData(1, [num_diag],
                                                base.idxint_DTYPE,
                                                self.offsets)
        PyArray_ENABLEFLAGS(data, cnp.NPY_ARRAY_OWNDATA)
        PyArray_ENABLEFLAGS(offsets, cnp.NPY_ARRAY_OWNDATA)
        self._deallocate = False
        self._scipy = _dia_matrix(data, offsets, self.shape)
        return self._scipy

    cpdef double complex trace(self):
        return trace_diag(self)

    cpdef Diag adjoint(self):
        return adjoint_diag(self)

    cpdef Diag conj(self):
        return conj_diag(self)

    cpdef Diag transpose(self):
        return transpose_diag(self)

    def __repr__(self):
        return "".join([
            "Diag(shape=", str(self.shape), ", num_diag=", str(self.num_diag), ")",
        ])

    def __str__(self):
        return self.__repr__()

    def __dealloc__(self):
        # If we have a reference to a scipy type, then we've passed ownership
        # of the data to numpy, so we let it handle refcounting and we don't
        # need to deallocate anything ourselves.
        if not self._deallocate:
            return
        if self.data != NULL:
            PyDataMem_FREE(self.data)
        if self.col_index != NULL:
            PyDataMem_FREE(self.offsets)


cpdef Diag fast_from_scipy(object sci):
    """
    Fast path construction from scipy.sparse.csr_matrix.  This does _no_ type
    checking on any of the inputs, and should consequently be considered very
    unsafe.  This is primarily for use in the unpickling operation.
    """
    cdef Diag out = Diag.__new__(Diag)
    out.shape = sci.shape
    out._deallocate = False
    out._scipy = sci
    out.data = <double complex *> cnp.PyArray_GETPTR1(sci.data, 0)
    out.offsets = <base.idxint *> cnp.PyArray_GETPTR1(sci.offsets, 0)
    out.num_diag  = sci.offsets.shape[0]
    out.size = sci.data.shape[1]
    return out
