#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from libc.string cimport memcpy
cimport cython

import numbers

import numpy as np
cimport numpy as cnp
from scipy.linalg cimport cython_blas as blas

from .base import EfficiencyWarning
from qutip.core.data cimport base, CSR
from qutip.core.data.add cimport add_dense, sub_dense
from qutip.core.data.adjoint cimport adjoint_dense, transpose_dense, conj_dense
from qutip.core.data.mul cimport mul_dense, neg_dense
from qutip.core.data.matmul cimport matmul_dense
from qutip.core.data.trace cimport trace_dense

cnp.import_array()

cdef int _ONE = 1

cdef extern from *:
    void PyArray_ENABLEFLAGS(cnp.ndarray arr, int flags)
    void PyArray_CLEARFLAGS(cnp.ndarray arr, int flags)
    void *PyDataMem_NEW(size_t size)
    void *PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
    void PyDataMem_FREE(void *ptr)


# Creation functions like 'identity' and 'from_csr' aren't exported in __all__
# to avoid naming clashes with other type modules.
__all__ = [
    'Dense', 'OrderEfficiencyWarning',
]


class OrderEfficiencyWarning(EfficiencyWarning):
    pass


cdef class Dense(base.Data):
    def __init__(self, data, shape=None, copy=True):
        base = np.array(data, dtype=np.complex128, order='K', copy=copy)
        if shape is None:
            shape = base.shape
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
        cdef size_t size = self.shape[0]*self.shape[1]*sizeof(double complex)
        cdef double complex *ptr = <double complex *> PyDataMem_NEW(size)
        memcpy(ptr, self.data, size)
        out.shape = self.shape
        out.data = ptr
        out.fortran = self.fortran
        out._deallocate = True
        return out

    cdef void _fix_flags(self, object array, bint make_owner=False):
        cdef int enable = cnp.NPY_ARRAY_OWNDATA if make_owner else 0
        cdef int disable = 0
        cdef cnp.Py_intptr_t *dims = cnp.PyArray_DIMS(array)
        cdef cnp.Py_intptr_t *strides = cnp.PyArray_STRIDES(array)
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
        cdef size_t size = self.shape[0]*self.shape[1]*sizeof(double complex)
        cdef double complex *ptr = <double complex *> PyDataMem_NEW(size)
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

    def __add__(left, right):
        if not isinstance(left, Dense) or not isinstance(right, Dense):
            return NotImplemented
        return add_dense(left, right)

    def __matmul__(left, right):
        if not isinstance(left, Dense) or not isinstance(right, Dense):
            return NotImplemented
        return matmul_dense(left, right)

    def __mul__(left, right):
        dense, number = (left, right) if isinstance(left, Dense) else (right, left)
        if not isinstance(number, numbers.Number):
            return NotImplemented
        return mul_dense(dense, complex(number))

    def __imul__(self, other):
        if not isinstance(other, numbers.Number):
            return NotImplemented
        cdef int size = self.shape[0]*self.shape[1]
        cdef double complex mul = complex(other)
        blas.zscal(&size, &mul, self.data, &_ONE)
        return self

    def __truediv__(left, right):
        dense, number = (left, right) if isinstance(left, Dense) else (right, left)
        if not isinstance(number, numbers.Number):
            return NotImplemented
        # Technically `(1 / x) * y` doesn't necessarily equal `y / x` in
        # floating point, but multiplication is faster than division, and we
        # don't really care _that_ much anyway.
        return mul_dense(dense, 1 / complex(number))

    def __itruediv__(self, other):
        if not isinstance(other, numbers.Number):
            return NotImplemented
        cdef int size = self.shape[0]*self.shape[1]
        cdef double complex mul = 1 / complex(other)
        blas.zscal(&size, &mul, self.data, &_ONE)
        return self

    def __neg__(self):
        return neg_dense(self)

    def __sub__(left, right):
        if not isinstance(left, Dense) or not isinstance(right, Dense):
            return NotImplemented
        return sub_dense(left, right)

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
    out.data = <double complex *> PyDataMem_NEW(rows * cols * sizeof(double complex))
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
        <double complex *> PyDataMem_NEW_ZEROED(rows * cols, sizeof(double complex))
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
        <double complex *>
        PyDataMem_NEW_ZEROED(out.shape[0]*out.shape[1], sizeof(double complex))
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
