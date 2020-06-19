#cython: language_level=3

cimport cython

import numpy as np
cimport numpy as cnp

from . cimport base

cnp.import_array()

cdef extern from *:
    void PyArray_ENABLEFLAGS(cnp.ndarray arr, int flags)
    void *PyDataMem_NEW(size_t size)
    void PyDataMem_FREE(void *ptr)


cdef class Dense(base.Data):
    def __init__(self, data, shape=None, copy=True):
        base = np.array(data, dtype=np.complex128, order='C', copy=copy)
        if not shape:
            shape = base.shape
        if len(shape) > 2:
            raise ValueError(
                "'data' must be 2D or 'shape' must be (n_rows, n_cols)."
            )
        if len(shape) == 1:
            shape = (shape[0], 1)
        self._np = base.reshape(shape)
        self.data = self._np
        self.shape = [base.shape[0], base.shape[1]]

    def as_ndarray(self):
        if not self._np:
            self._np =\
                cnp.PyArray_SimpleNewFromData(
                    2, [self.shape[0], self.shape[1]],
                    cnp.NPY_COMPLEX128, &self.data[0, 0]
                )
            PyArray_ENABLEFLAGS(self._np, cnp.NPY_ARRAY_OWNDATA)
        return self._np

    def __dealloc__(self):
        if self._np is None:
            PyDataMem_FREE(&self.data[0, 0])


cpdef Dense empty((base.idxint, base.idxint) shape):
    cdef Dense out = Dense.__new__(Dense)
    out.shape = shape
    out.data = (
        <double complex [:shape[0], :shape[1]:1]>
        PyDataMem_NEW(shape[0] * shape[1] * sizeof(double complex))
    )
    return out
