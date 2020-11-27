#cython: language_level=3

import numpy as np
from .._solverqevo cimport SolverQEvo
from ...core cimport data as _data
from ...core.data.norm cimport frobenius_dense
from ...core.data.norm import frobenius

cdef class QtOdeFuncWrapper:
    def __init__(self, SolverQEvo evo):
        self.evo = evo

    cpdef void call(self, QtOdeData out, double t, QtOdeData y):
        if out.inplace:
            self.evo.mul_dense(t, y.data(), out.data())
        else:
            out.set_data(self.evo.mul_data(t, y.data()))


cdef class PythonFuncWrapper(QtOdeFuncWrapper):
    cdef object f

    def __init__(self, object f):
        self.f = f

    cpdef void call(self, QtOdeData out, double t, QtOdeData y):
        out.copy(QtOdeData(self.f(t, y.raw())))


cdef class QtOdeData:

    def __init__(self, val):
        self.state = val
        self.inplace = False

    cpdef void inplace_add(self, QtOdeData other, double factor):
        self.state = _data.add(self.state, (<QtOdeData> other).state, factor)

    cpdef void zero(self):
        self.state = _data.mul(self.state, 0.)

    cpdef double norm(self):
        return frobenius(self.state)

    cpdef void copy(self, QtOdeData other):
        self.state = (<QtOdeData> other).state.copy()

    cpdef QtOdeData empty_like(self):
        return QtOdeData(self.state.copy())

    cpdef object raw(self):
        return self.state

    cpdef _data.Data data(self):
        return self.state

    cpdef void set_data(self, _data.Data new):
        self.state = new


cdef class QtOdeDataDense(QtOdeData):
    cdef _data.Dense dense

    def __init__(self, val):
        self.dense = val
        self.inplace = True

    cpdef void inplace_add(self, QtOdeData other, double factor):
        _data.add_dense_eq_order_inplace(self.dense,
                                         (<QtOdeDataDense> other).dense,
                                         factor)

    cpdef void zero(self):
        cdef size_t ptr
        for ptr in range(self.dense.shape[0] * self.dense.shape[1]):
            self.dense.data[ptr] = 0.

    cpdef double norm(self):
        return frobenius_dense(self.dense)

    cpdef void copy(self, QtOdeData other):
        cdef size_t ptr
        for ptr in range(self.dense.shape[0] * self.dense.shape[1]):
            self.dense.data[ptr] = (<QtOdeDataDense> other).dense.data[ptr]

    cpdef QtOdeData empty_like(self):
        return QtOdeDataDense(_data.dense.empty_like(self.dense))

    cpdef object raw(self):
        return self.dense

    cpdef _data.Data data(self):
        return self.dense

    cpdef void set_data(self, _data.Data new):
        self.dense = new


cdef class QtOdeDataArray(QtOdeData):
    cdef double[::1] base
    cdef object _raw

    def __init__(self, val):
        self._raw = np.array(val)
        self.base = self._raw.ravel()
        self.inplace = True

    cpdef void inplace_add(self, QtOdeData other, double factor):
        cdef int i, len_ = self.base.shape[0]
        for i in range(len_):
            self.base[i] += other.base[i] * factor

    cpdef void zero(self):
        cdef int i, len_ = self.base.shape[0]
        for i in range(len_):
            self.base[i] *= 0.

    cpdef double norm(self):
        cdef int i, len_ = self.base.shape[0]
        cdef double sum = 0.
        for i in range(len_):
            sum += self.base[i] * self.base[i]
        return sum

    cpdef void copy(self, QtOdeData other):
        cdef int i, len_ = self.base.shape[0]
        for i in range(len_):
            self.base[i] = other.base[i]

    cpdef QtOdeData empty_like(self):
        return QtOdeData(self.base.copy())

    cpdef object raw(self):
        return self._raw

    cpdef _data.Data data(self):
        raise NotImplementedError

    cpdef void set_data(self, _data.Data new):
        raise NotImplementedError


cdef class QtOdeDataCpxArray(QtOdeDataDense):
    cdef object _raw

    def __init__(self, val):
        self._raw = val
        self.dense = _data.fast_from_numpy(val)
        self.inplace = True

    cpdef object raw(self):
        return self.dense.as_ndarray()


def qtodedata(data):
    if isinstance(data, np.ndarray) and data.dtype == np.double:
        return QtOdeDataArray(data)
    if isinstance(data, np.ndarray) and data.dtype == np.complex:
        return QtOdeDataCpxArray(data)
    if isinstance(data, _data.Dense):
        return QtOdeDataDense(data)
    if isinstance(data, _data.Data):
        return QtOdeData(data)
    raise NotImplementedError
