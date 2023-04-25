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
# from qutip.core.data.adjoint import adjoint_diag, transpose_diag, conj_diag#********************************
# from qutip.core.data.trace import trace_diag#********************************
# from qutip.core.data.tidyup import tidyup_diag #********************************
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
    cdef object out = scipy_dia_array.__new__(scipy_dia_array)
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

    def __init__(self, arg=None, shape=None, bint copy=True, bint tidyup=False):
        cdef size_t ptr
        cdef base.idxint col
        cdef object data, offsets

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
        self.num_diag = offsets.shape[0]

        if shape is None:
            warnings.warn("instantiating Diag matrix of unknown shape")
            nrows = 0
            ncols = 0
            for i in self.num_diag:
                if self.offsets[i] >= 0:
                    j = 0
                    while self.data[i + j * self.shape[1]] == 0.0:
                        j += 1
                    nrows = max(nrows, j)
                    ncols = max(ncols, (self.offsets[i] + j))
                else:
                    j = 0
                    while self.data[i + (self.shape[1] -j - 1) * self.shape[1]] == 0.0:
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

        self._scipy = _dia_matrix(data, offsets, self.shape)
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
        memcpy(out.data, self.data, self.num_diag * self.shape[1] * sizeof(double complex))
        memcpy(out.offsets, self.offsets, self.num_diag * sizeof(base.idxint))

        return out

    cpdef object to_array(self):
        """
        Get a copy of this data as a full 2D, C-contiguous NumPy array.  This
        is not a view onto the data, and changes to new array will not affect
        the original data structure.
        """
        cdef cnp.npy_intp *dims = [self.shape[0], self.shape[1]]
        cdef object out = cnp.PyArray_ZEROS(2, dims, cnp.NPY_COMPLEX128, 0)
        cdef size_t diag, col, nrows = self.shape[0]
        cdef double complex [:, ::1] buffer = out
        for diag in range(self.num_diag):
            for col in range(self.shape[1]):
                if col - diag < 0 or col - diag >= nrows:
                    continue
                out[(col-diag), col] = self.data[diag * self.shape[1] + col]
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
        cdef cnp.npy_intp size = self.shape[1]
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

    """
    cpdef double complex trace(self):
        return trace_diag(self)

    cpdef Diag adjoint(self):
        return adjoint_diag(self)

    cpdef Diag conj(self):
        return conj_diag(self)

    cpdef Diag transpose(self):
        return transpose_diag(self)
    """

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
        if self.offsets != NULL:
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
    return out


cpdef Diag empty(base.idxint rows, base.idxint cols, base.idxint num_diag):
    """
    Allocate an empty Diag matrix of the given shape, ``with num_diag``
    diagonals.

    This does not initialise any of the memory returned.
    """
    if num_diag < 0:
        raise ValueError("num_diag must be a positive integer.")
    cdef Diag out = Diag.__new__(Diag)
    out.shape = (rows, cols)
    out.num_diag = num_diag
    # Python doesn't like allocating nothing.
    if num_diag == 0:
        num_diag += 1
    out.data =\
        <double complex *> PyDataMem_NEW(out.shape[1] * num_diag * sizeof(double complex))
    out.offsets =\
        <base.idxint *> PyDataMem_NEW(num_diag * sizeof(base.idxint))
    return out


cpdef Diag empty_like(Diag other):
    return empty(other.shape[0], other.shape[1], other.num_diag)


cpdef Diag zeros(base.idxint rows, base.idxint cols):
    """
    Allocate the zero matrix with a given shape.  There will not be any room in
    the `data` and `col_index` buffers to add new elements.
    """
    # We always allocate matrices with at least one element to ensure that we
    # actually _are_ asking for memory (Python doesn't like allocating nothing)
    cdef Diag out = empty(rows, cols, 1)
    memset(out.data, 0, out.shape[1] * sizeof(double complex))
    out.offsets[0] = 0
    out.num_diag = 0
    return out


cpdef Diag identity(base.idxint dimension, double complex scale=1):
    """
    Return a square matrix of the specified dimension, with a constant along
    the diagonal.  By default this will be the identity matrix, but if `scale`
    is passed, then the result will be `scale` times the identity.
    """
    cdef Diag out = empty(dimension, dimension, 1)
    memset(out.data, 1, out.shape[1] * sizeof(double complex))
    out.offsets[0] = 0
    return out


cpdef Diag from_dense(Dense matrix):
    # Assume worst-case scenario for non-zero.
    cdef Diag out = empty(matrix.shape[0], matrix.shape[1], matrix.shape[0] + matrix.shape[1] - 1)
    cdef size_t diag_, ptr_in, ptr_out=0, stride
    cdef row, col

    for i in range(matrix.shape[0] + matrix.shape[1] - 1):
        out.offsets[i] = i -matrix.shape[0] + 1
    strideR = matrix.shape[0] if matrix.fortran else 1
    strideC = matrix.shape[1] if not matrix.fortran else 1

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            out.data[(-(-col+row) -matrix.shape[1]) * out.shape[1] + col] = matrix.data[row * strideR + col * strideC]

    return tidyup_diag(out)


cpdef tidyup_diag(mat, atol=1e-5, inplace=False):
    data = mat.data
    offsets = mat.offsets
    new_offsets = []
    diags = []
    for diag, line in zip(offsets, data):
        has_data = False
        for val in line:
            has_data |= abs(val) > atol
            if has_data:
                break
        if not has_data:
            continue
        new_offsets.append(diag)
        diags.append(line)
    data = np.vstack(diags)
    offsets = np.array(new_offsets)
    return data, offsets


cdef inline base.idxint _diagonal_length(
    base.idxint offset, base.idxint n_rows, base.idxint n_cols,
) nogil:
    if offset > 0:
        return n_rows if offset <= n_cols - n_rows else n_cols - offset
    return n_cols if offset > n_cols - n_rows else n_rows + offset


cdef Diag diags_(
    list diagonals, base.idxint[:] offsets,
    base.idxint n_rows, base.idxint n_cols,
):
    """
    Construct a Diag matrix from a list of diagonals and their offsets.  The
    offsets are assumed to be in sorted order.  This is the C-only interface to
    diag.diags, and inputs are not sanity checked (use the Python interface for
    that).

    Parameters
    ----------
    diagonals : list of indexable of double complex
        The entries (including zeros) that should be placed on the diagonals in
        the output matrix.  Each entry must have enough entries in it to fill
        the relevant diagonal (not checked).

    offsets : idxint[:]
        The indices of the diagonals.  These should be sorted and without
        duplicates.  `offsets[i]` is the location of the values `diagonals[i]`.
        An offset of 0 is the main diagonal, positive values are above the main
        diagonal and negative ones are below the main diagonal.

    n_rows, n_cols : idxint
        The shape of the output.  The result does not need to be square, but
        the diagonals must be of the correct length to fit in.
    """
    out = empty(n_rows, n_cols, len(offsets))
    for i in range(len(offsets)):
        out.offsets[i] = offsets[i]
        offset = max(offsets[i], 0)
        for col in range(len(diagonals[i])):
            out.data[i, col + offset] = diagonals[i][col]
    return out

@cython.wraparound(True)
def diags(diagonals, offsets=None, shape=None):
    """
    Construct a Diag matrix from diagonals and their offsets.  Using this
    function in single-argument form produces a square matrix with the given
    values on the main diagonal.

    With lists of diagonals and offsets, the matrix will be the smallest
    possible square matrix if shape is not given, but in all cases the
    diagonals must fit exactly with no extra or missing elements.  Duplicated
    diagonals will be summed together in the output.

    Parameters
    ----------
    diagonals : sequence of array_like of complex or array_like of complex
        The entries (including zeros) that should be placed on the diagonals in
        the output matrix.  Each entry must have enough entries in it to fill
        the relevant diagonal and no more.

    offsets : sequence of integer or integer, optional
        The indices of the diagonals.  `offsets[i]` is the location of the
        values `diagonals[i]`.  An offset of 0 is the main diagonal, positive
        values are above the main diagonal and negative ones are below the main
        diagonal.

    shape : tuple, optional
        The shape of the output as (``rows``, ``columns``).  The result does
        not need to be square, but the diagonals must be of the correct length
        to fit in exactly.
    """
    cdef base.idxint n_rows, n_cols, offset
    try:
        diagonals = list(diagonals)
        if diagonals and np.isscalar(diagonals[0]):
            # Catch the case where we're being called as (for example)
            #   diags([1, 2, 3], 0)
            # with a single diagonal and offset.
            diagonals = [diagonals]
    except TypeError:
        raise TypeError("diagonals must be a list of arrays of complex") from None
    if offsets is None:
        if len(diagonals) == 0:
            offsets = []
        elif len(diagonals) == 1:
            offsets = [0]
        else:
            raise TypeError("offsets must be supplied if passing more than one diagonal")
    offsets = np.atleast_1d(offsets)
    if offsets.ndim > 1:
        raise ValueError("offsets must be a 1D array of integers")
    if len(diagonals) != len(offsets):
        raise ValueError("number of diagonals does not match number of offsets")
    if len(diagonals) == 0:
        if shape is None:
            raise ValueError("cannot construct matrix with no diagonals without a shape")
        else:
            n_rows, n_cols = shape
        return zeros(n_rows, n_cols)
    order = np.argsort(offsets)
    diagonals_ = []
    offsets_ = []
    prev, cur = None, None
    for i in order:
        cur = offsets[i]
        if cur == prev:
            diagonals_[-1] += np.asarray(diagonals[i], dtype=np.complex128)
        else:
            offsets_.append(cur)
            diagonals_.append(np.asarray(diagonals[i], dtype=np.complex128))
        prev = cur
    if shape is None:
        n_rows = n_cols = abs(offsets_[0]) + len(diagonals_[0])
    else:
        try:
            n_rows, n_cols = shape
        except (TypeError, ValueError):
            raise TypeError("shape must be a 2-tuple of positive integers")
        if n_rows < 0 or n_cols < 0:
            raise ValueError("shape must be a 2-tuple of positive integers")
    for i in range(len(diagonals_)):
        offset = offsets_[i]
        if len(diagonals_[i]) != _diagonal_length(offset, n_rows, n_cols):
            raise ValueError("given diagonals do not have the correct lengths")
    if n_rows == 0 and n_cols == 0:
        raise ValueError("can't produce a 0x0 matrix")
    #if len(offsets) == 1:
        # Fast path for a single diagonal.
        #return diag(diagonals_[0], offsets_[0], n_rows, n_cols)
    return diags_(
        diagonals_, np.array(offsets_, dtype=idxint_dtype), n_rows, n_cols,
    )
