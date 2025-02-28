#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from libc.string cimport memcpy
cimport cython

import numbers
import builtins
import numpy as np
cimport numpy as cnp
from scipy.linalg cimport cython_blas as blas

from .base import EfficiencyWarning
from qutip.core.data cimport base, CSR, Dia
from qutip.core.data.adjoint cimport adjoint_dense, transpose_dense, conj_dense
from qutip.core.data.trace cimport trace_dense
from qutip import settings

cnp.import_array()

cdef int _ONE = 1

cdef extern from *:
    void PyArray_ENABLEFLAGS(cnp.ndarray arr, int flags)
    void PyArray_CLEARFLAGS(cnp.ndarray arr, int flags)
    void *PyDataMem_NEW(size_t size)
    void *PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
    void PyDataMem_FREE(void *ptr)


@cython.overflowcheck(True)
cdef size_t _mul_mem_checked(size_t a, size_t b, size_t c=0):
    if c != 0:
        return a * b * c
    return a * b


# Creation functions like 'identity' and 'from_csr' aren't exported in __all__
# to avoid naming clashes with other type modules.
__all__ = [
    'Dense', 'OrderEfficiencyWarning',
]


class OrderEfficiencyWarning(EfficiencyWarning):
    pass

is_numpy1 = np.lib.NumpyVersion(np.__version__) < '2.0.0b1'

cdef class Dense(base.Data):
    def __init__(self, data, shape=None, copy=True):
        if is_numpy1:
            # np2 accept None which act as np1's False
            copy = builtins.bool(copy)
        base = np.array(data, dtype=np.complex128, order='K', copy=copy)
        # Ensure that the array is contiguous.
        # Non contiguous array with copy=False would otherwise slip through
        if not (cnp.PyArray_IS_C_CONTIGUOUS(base) or
                cnp.PyArray_IS_F_CONTIGUOUS(base)):
            base = base.copy()
        if shape is None:
            shape = base.shape
            if len(shape) == 0:
                shape = (1, 1)
            # Promote to a ket by default if passed 1D data.
            if len(shape) == 1:
                shape = (shape[0], 1)
        if not (
            len(shape) == 2
            and isinstance(shape[0], numbers.Integral)
            and isinstance(shape[1], numbers.Integral)
            and shape[0] > 0
            and shape[1] > 0
        ):
            raise ValueError("shape must be a 2-tuple of positive ints, but is " + repr(shape))
        if shape[0] * shape[1] != base.size:
            raise ValueError("".join([
                "invalid shape ",
                str(shape),
                " for input data with size ",
                str(base.size)
            ]))
        self._np = base.reshape(shape, order='A')
        self._deallocate = False
        self.data = <double complex *> cnp.PyArray_GETPTR2(self._np, 0, 0)
        self.fortran = cnp.PyArray_IS_F_CONTIGUOUS(self._np)
        self.shape = (shape[0], shape[1])

    @classmethod
    def sparcity(self):
        return "dense"

    def __reduce__(self):
        return (fast_from_numpy, (self.as_ndarray(),))

    def __repr__(self):
        return "".join([
            "Dense(shape=", str(self.shape), ", fortran=", str(self.fortran), ")",
        ])

    def __str__(self):
        return self.__repr__()

    cpdef Dense reorder(self, int fortran=-1):
        cdef bint fortran_
        if fortran < 0:
            fortran_ = not self.fortran
        else:
            fortran_ = fortran
        if bool(fortran_) == bool(self.fortran):
            return self.copy()
        cdef Dense out = empty_like(self, fortran_)
        cdef size_t idx_self=0, idx_out, idx_out_start, stride, splits
        stride = self.shape[1] if self.fortran else self.shape[0]
        splits = self.shape[0] if self.fortran else self.shape[1]
        for idx_out_start in range(stride):
            idx_out = idx_out_start
            for _ in range(splits):
                out.data[idx_out] = self.data[idx_self]
                idx_self += 1
                idx_out += stride
        return out

    cpdef Dense copy(self):
        """
        Return a complete (deep) copy of this object.

        If the type currently has a numpy backing, such as that produced by
        `as_ndarray`, this will not be copied.  The backing is a view onto our
        data, and a straight copy of this view would be incorrect.  We do not
        create a new view at copy time, since the user may only access this
        through a creation method, and creating it ahead of time would incur an
        unnecessary speed penalty for users who do not need it (including
        low-level C code).
        """
        cdef Dense out = Dense.__new__(Dense)
        cdef size_t size = (
            _mul_mem_checked(self.shape[0], self.shape[1], sizeof(double complex))
        )
        cdef double complex *ptr = <double complex *> PyDataMem_NEW(size)
        if not ptr:
            raise MemoryError(
                "Could not allocate memory to copy a "
                f"({self.shape[0]}, {self.shape[1]}) Dense matrix."
            )
        memcpy(ptr, self.data, size)
        out.shape = self.shape
        out.data = ptr
        out.fortran = self.fortran
        out._deallocate = True
        return out

    cdef void _fix_flags(self, object array, bint make_owner=False):
        cdef int enable = cnp.NPY_ARRAY_OWNDATA if make_owner else 0
        cdef int disable = 0
        cdef cnp.npy_intp *dims = cnp.PyArray_DIMS(array)
        cdef cnp.npy_intp *strides = cnp.PyArray_STRIDES(array)
        # Not necessary when creating a new array because this will already
        # have been done, but needed for as_ndarray() if we have been mutated.
        dims[0] = self.shape[0]
        dims[1] = self.shape[1]
        if self.shape[0] == 1 or self.shape[1] == 1:
            enable |= cnp.NPY_ARRAY_F_CONTIGUOUS | cnp.NPY_ARRAY_C_CONTIGUOUS
            strides[0] = self.shape[1] * sizeof(double complex)
            strides[1] = sizeof(double complex)
        elif self.fortran:
            enable |= cnp.NPY_ARRAY_F_CONTIGUOUS
            disable |= cnp.NPY_ARRAY_C_CONTIGUOUS
            strides[0] = sizeof(double complex)
            strides[1] = self.shape[0] * sizeof(double complex)
        else:
            enable |= cnp.NPY_ARRAY_C_CONTIGUOUS
            disable |= cnp.NPY_ARRAY_F_CONTIGUOUS
            strides[0] = self.shape[1] * sizeof(double complex)
            strides[1] = sizeof(double complex)
        PyArray_ENABLEFLAGS(array, enable)
        PyArray_CLEARFLAGS(array, disable)

    cpdef object to_array(self):
        """
        Get a copy of this data as a full 2D, contiguous NumPy array.  This may
        be Fortran or C-ordered, but will be contiguous in one of the
        dimensions.  This is not a view onto the data, and changes to new array
        will not affect the original data structure.
        """
        cdef size_t size = (
          _mul_mem_checked(self.shape[0], self.shape[1], sizeof(double complex))
        )
        cdef double complex *ptr = <double complex *> PyDataMem_NEW(size)
        if not ptr:
            raise MemoryError(
                "Could not allocate memory to convert to a numpy array a "
                f"({self.shape[0]}, {self.shape[1]}) Dense matrix."
            )
        memcpy(ptr, self.data, size)
        cdef object out =\
            cnp.PyArray_SimpleNewFromData(2, [self.shape[0], self.shape[1]],
                                          cnp.NPY_COMPLEX128, ptr)
        self._fix_flags(out, make_owner=True)
        return out

    cpdef object as_ndarray(self):
        """
        Get a view onto this object as a `numpy.ndarray`.  The underlying data
        structure is exposed, such that modifications to the array will modify
        this object too.

        The array may be uninitialised, depending on how the Dense type was
        created.  The output will be contiguous and of dtype 'complex128', but
        may be C- or Fortran-ordered.
        """
        if self._np is not None:
            # We have to do this every time in case someone has changed our
            # ordering or shape inplace.
            self._fix_flags(self._np, make_owner=False)
            return self._np
        self._np =\
            cnp.PyArray_SimpleNewFromData(
                2, [self.shape[0], self.shape[1]], cnp.NPY_COMPLEX128, self.data
            )
        self._fix_flags(self._np, make_owner=self._deallocate)
        self._deallocate = False
        return self._np

    cpdef double complex trace(self):
        return trace_dense(self)

    cpdef Dense adjoint(self):
        return adjoint_dense(self)

    cpdef Dense conj(self):
        return conj_dense(self)

    cpdef Dense transpose(self):
        return transpose_dense(self)

    def __dealloc__(self):
        if self._deallocate and self.data != NULL:
            PyDataMem_FREE(self.data)


cpdef Dense fast_from_numpy(object array):
    """
    Fast path construction from numpy ndarray.  This does _no_ type checking on
    the input, and should consequently be considered very unsafe.  This is
    primarily for use in the unpickling operation.
    """
    cdef Dense out = Dense.__new__(Dense)
    if array.ndim == 1:
        out.shape = (array.shape[0], 1)
        array = array[:, None]
    else:
        out.shape = (array.shape[0], array.shape[1])
    out._deallocate = False
    out._np = array
    out.data = <double complex *> cnp.PyArray_GETPTR2(array, 0, 0)
    out.fortran = cnp.PyArray_IS_F_CONTIGUOUS(array)
    return out

cdef Dense wrap(double complex *data, base.idxint rows, base.idxint cols, bint fortran=False):
    cdef Dense out = Dense.__new__(Dense)
    out.data = data
    out._deallocate = False
    out.fortran = fortran or cols == 1 or rows == 1
    out.shape = (rows, cols)
    return out


cpdef Dense empty(base.idxint rows, base.idxint cols, bint fortran=True):
    """
    Return a new Dense type of the given shape, with the data allocated but
    uninitialised.
    """
    cdef Dense out = Dense.__new__(Dense)
    out.shape = (rows, cols)
    out.data = <double complex *> PyDataMem_NEW(
        _mul_mem_checked(rows, cols, sizeof(double complex))
    )
    if not out.data:
        raise MemoryError(
            "Could not allocate memory to create an empty "
            f"({rows}, {cols}) Dense matrix."
        )
    out._deallocate = True
    out.fortran = fortran
    return out


cpdef Dense empty_like(Dense other, int fortran=-1):
    cdef bint fortran_
    if fortran < 0:
        fortran_ = other.fortran
    else:
        fortran_ = fortran
    return empty(other.shape[0], other.shape[1], fortran=fortran_)


cpdef Dense zeros(base.idxint rows, base.idxint cols, bint fortran=True):
    """Return the zero matrix with the given shape."""
    cdef Dense out = Dense.__new__(Dense)
    out.shape = (rows, cols)
    out.data =\
        <double complex *> PyDataMem_NEW_ZEROED(
            _mul_mem_checked(rows, cols), sizeof(double complex)
        )
    if not out.data:
        raise MemoryError(
            "Could not allocate memory to create a zero "
            f"({rows}, {cols}) Dense matrix."
        )
    out.fortran = fortran
    out._deallocate = True
    return out


cpdef Dense identity(base.idxint dimension, double complex scale=1,
                     bint fortran=True):
    """
    Return a square matrix of the specified dimension, with a constant along
    the diagonal.  By default this will be the identity matrix, but if `scale`
    is passed, then the result will be `scale` times the identity.
    """
    cdef size_t row
    cdef Dense out = zeros(dimension, dimension, fortran=fortran)
    for row in range(dimension):
        out.data[row*dimension + row] = scale
    return out


cpdef Dense from_csr(CSR matrix, bint fortran=False):
    cdef Dense out = Dense.__new__(Dense)
    out.shape = matrix.shape
    out.data = (
        <double complex *> PyDataMem_NEW_ZEROED(
            _mul_mem_checked(out.shape[0], out.shape[1]), sizeof(double complex)
        )
    )
    if not out.data:
        raise MemoryError(
            "Could not allocate memory to create a "
            f"({out.shape[0]}, {out.shape[1]}) Dense matrix from a CSR."
        )
    out.fortran = fortran
    out._deallocate = True
    cdef size_t row, ptr_in, ptr_out, row_stride, col_stride
    row_stride = 1 if fortran else out.shape[1]
    col_stride = out.shape[0] if fortran else 1
    ptr_out = 0
    for row in range(out.shape[0]):
        for ptr_in in range(matrix.row_index[row], matrix.row_index[row + 1]):
            out.data[ptr_out + matrix.col_index[ptr_in]*col_stride] = matrix.data[ptr_in]
        ptr_out += row_stride
    return out


cpdef Dense from_dia(Dia matrix):
    return Dense(matrix.to_array(), copy=False)


cdef inline base.idxint _diagonal_length(
    base.idxint offset, base.idxint n_rows, base.idxint n_cols,
) nogil:
    if offset > 0:
        return n_rows if offset <= n_cols - n_rows else n_cols - offset
    return n_cols if offset > n_cols - n_rows else n_rows + offset


cpdef long nnz(Dense matrix, double tol=0):
    "Compute the number of element larger than the tolerance"
    cdef long N = 0;
    cdef base.idxint row, col
    cdef double tol2
    cdef double complex val
    if tol:
        tol2 = tol**2
    else:
        tol2 = settings.core["atol"]**2
    for i in range(matrix.shape[0] * matrix.shape[1]):
        val = matrix.data[i]
        if (val.real**2 + val.imag**2) > tol2:
            N += 1
    return N


@cython.wraparound(True)
def diags(diagonals, offsets=None, shape=None):
    """
    Construct a matrix from diagonals and their offsets.  Using this
    function in single-argument form produces a square matrix with the given
    values on the main diagonal.
    With lists of diagonals and offsets, the matrix will be the smallest
    possible square matrix if shape is not given, but in all cases the
    diagonals must fit exactly with no extra or missing elements. Duplicated
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

    out = zeros(n_rows, n_cols, fortran=True)

    cdef size_t diag_idx, idx, n_diagonals = len(diagonals_)

    for diag_idx in range(n_diagonals):
        offset = offsets_[diag_idx]
        if offset <= 0:
            for idx in range(_diagonal_length(offset, n_rows, n_cols)):
                out.data[idx*(n_rows+1) - offset] = diagonals_[diag_idx][idx]
        else:
            for idx in range(_diagonal_length(offset, n_rows, n_cols)):
                out.data[idx*(n_rows+1) + offset*n_rows] = diagonals_[diag_idx][idx]
    return out
