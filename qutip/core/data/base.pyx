#cython: language_level=3
#cython: c_api_binop_methods=True

import numpy as np
cimport numpy as cnp
import qutip.core.data as _data
from qutip.settings import settings

__all__ = [
    'idxint_size', 'idxint_dtype', 'Data', 'EfficiencyWarning',
]

if _idxint_size == 32:
    idxint_dtype = np.int32
    idxint_DTYPE = cnp.NPY_INT32
else:
    idxint_dtype = np.int64
    idxint_DTYPE = cnp.NPY_INT64

idxint_size = _idxint_size

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

    def __truediv__(left, right):
        data, number = (left, right) if isinstance(left, Data) else (right, left)
        try:
            return _data.mul(data, 1/number)
        except TypeError:
            return NotImplemented

    def __neg__(self):
        return _data.neg(self)

    def __eq__(left, right):
        if not (isinstance(left, Data) and isinstance(right, Data)):
            return NotImplemented
        if (
            left.shape[0] == right.shape[0]
            and left.shape[1] == right.shape[1]
        ):
            return _data.iszero(_data.sub(left, right), settings.core['atol'])
        return False


class EfficiencyWarning(Warning):
    pass
