#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from libc.string cimport memcpy
cimport cython

import numpy as np
cimport numpy as cnp

from . cimport base

cnp.import_array()

cdef extern from *:
    void PyArray_ENABLEFLAGS(cnp.ndarray arr, int flags)
    void *PyDataMem_NEW(size_t size)
    void *PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
    void PyDataMem_FREE(void *ptr)


cdef class Dense(base.Data):
    def __init__(self, data, shape=None, copy=True):
        base = np.array(data, dtype=np.complex128, order='C', copy=copy)
        if shape is None:
            shape = base.shape
            # Promote to a ket by default if passed 1D data.
            if len(shape) == 1:
                shape = (shape, 1)
        if not (
            len(shape) == 2
            and isinstance(shape[0], int)
            and isinstance(shape[1], int)
            and shape[0] > 0
            and shape[1] > 0
        ):
            raise ValueError("shape must be a 2-tuple of positive ints")
        if shape[0] * shape[1] != base.size:
            raise ValueError("".join([
                "invalid shape ",
                str(shape),
                " for input data with size ",
                str(base.size)
            ]))
        self._np = base.reshape(shape, order='C')
        self.data = self._np
        self.shape = (base.shape[0], base.shape[1])

    def __repr__(self):
        return "".join([
            "Dense(shape=", str(self.shape), ")",
        ])

    def __str__(self):
        return self.__repr__()

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
        memcpy(ptr, &self.data[0, 0], size)
        out.shape = self.shape
        out.data = <double complex [:self.shape[0], :self.shape[1]:1]> ptr
        return out

    cpdef object to_array(self):
        """
        Get a copy of this data as a full 2D, C-contiguous NumPy array.  This
        is not a view onto the data, and changes to new array will not affect
        the original data structure.
        """
        cdef size_t size = self.shape[0]*self.shape[1]*sizeof(double complex)
        cdef double complex *ptr = <double complex *> PyDataMem_NEW(size)
        memcpy(ptr, &self.data[0, 0], size)
        cdef object out =\
            cnp.PyArray_SimpleNewFromData(2, [self.shape[0], self.shape[1]],
                                          cnp.NPY_COMPLEX128, ptr)
        PyArray_ENABLEFLAGS(out, cnp.NPY_ARRAY_OWNDATA)
        return out

    cpdef object as_ndarray(self):
        """
        Get a view onto this object as a `numpy.ndarray`.  The underlying data
        structure is exposed, such that modifications to the array will modify
        this object too.

        The array may be uninitialised, depending on how the Dense type was
        created.  The output will be C-contiguous and of dtype 'complex128'.
        """
        if self._np is not None:
            return self._np
        self._np =\
            cnp.PyArray_SimpleNewFromData(
                2, [self.shape[0], self.shape[1]],
                cnp.NPY_COMPLEX128, &self.data[0, 0]
            )
        PyArray_ENABLEFLAGS(self._np, cnp.NPY_ARRAY_OWNDATA)
        return self._np

    @cython.initializedcheck(True)
    def __dealloc__(self):
        if self._np is not None:
            return
        try:
            PyDataMem_FREE(&self.data[0, 0])
        except AttributeError:
            pass


cpdef Dense empty(base.idxint rows, base.idxint cols):
    """
    Return a new Dense type of the given shape, with the data allocated but
    uninitialised.
    """
    cdef Dense out = Dense.__new__(Dense)
    out.shape = (rows, cols)
    out.data = (
        <double complex [:rows, :cols:1]>
        PyDataMem_NEW(rows * cols * sizeof(double complex))
    )
    return out


cpdef Dense zeros(base.idxint rows, base.idxint cols):
    """Return the zero matrix with the given shape."""
    cdef Dense out = Dense.__new__(Dense)
    out.shape = (rows, cols)
    out.data = (
        <double complex [:rows, :cols:1]>
        PyDataMem_NEW_ZEROED(rows * cols, sizeof(double complex))
    )
    return out


cpdef Dense identity(base.idxint dimension, double complex scale=1):
    """
    Return a square matrix of the specified dimension, with a constant along
    the diagonal.  By default this will be the identity matrix, but if `scale`
    is passed, then the result will be `scale` times the identity.
    """
    cdef size_t row
    cdef Dense out = zeros(dimension, dimension)
    for row in range(dimension):
        out.data[row, row] = scale
    return out
