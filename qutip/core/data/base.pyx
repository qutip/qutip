#cython: language_level=3

import numpy as np
cimport numpy as cnp
import qutip.core.data as _data

__all__ = [
    'idxint_dtype', 'Data', 'EfficiencyWarning',
]

idxint_dtype = np.int32
idxint_DTYPE = cnp.NPY_INT32

# As this is an abstract base class with C entry points, we have to explicitly
# stub out methods since we can't mark them as abstract.
cdef class Data:
    def __init__(self, shape):
        self.shape = shape

    cpdef object to_array(self):
        raise NotImplementedError

    cpdef double complex trace(self):
        return NotImplementedError

    cpdef Data adjoint(self):
        raise NotImplementedError

    cpdef Data conj(self):
        raise NotImplementedError

    cpdef Data transpose(self):
        raise NotImplementedError

    cpdef Data copy(self):
        raise NotImplementedError

    def __add__(left, right):
        if isinstance(left, Data) and isinstance(right, Data):
            return _data.add(left, right)
        return NotImplemented

    def __sub__(left, right):
        if isinstance(left, Data) and isinstance(right, Data):
            return _data.sub(left, right)
        return NotImplemented

    def __matmul__(left, right):
        if isinstance(left, Data) and isinstance(right, Data):
            return _data.matmul(left, right)
        return NotImplemented

    def __mul__(left, right):
        data, number = (left, right) if isinstance(left, Data) else (right, left)
        try:
            return _data.mul(data, number)
        except TypeError:
            return NotImplemented

    def __imul__(self, other):
        try:
            return _data.imul(self, other)
        except TypeError:
            return NotImplemented

    def __truediv__(left, right):
        data, number = (left, right) if isinstance(left, Data) else (right, left)
        try:
            return _data.mul(data, 1/number)
        except TypeError:
            return NotImplemented

    def __itruediv__(self, other):
        try:
            return _data.imul(self, 1/other)
        except TypeError:
            return NotImplemented

    def __neg__(self):
        return _data.neg(self)


class EfficiencyWarning(Warning):
    pass
