#cython: language_level=3

import numpy as np
cimport numpy as cnp

idxint_dtype = np.int32
idxint_DTYPE = cnp.NPY_INT32

# As this is an abstract base class with C entry points, we have to explicitly
# stub out methods since we can't mark them as abstract.
cdef class Data:
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

class EfficiencyWarning(Warning):
    pass
