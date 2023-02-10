#cython: language_level=3

from qutip.core.data cimport Dense, Data
from qutip.core.data.base cimport idxint

cdef class QobjEvo:
    cdef:
        list elements
        readonly list dims
        readonly (idxint, idxint) shape
        readonly str type
        readonly str superrep
        int _issuper
        int _isoper

    cpdef Data _call(QobjEvo self, double t)

    cdef object _prepare(QobjEvo self, object t, Data state=*)

    cpdef object expect_data(QobjEvo self, object t, Data state)

    cdef double complex _expect_dense(QobjEvo self, double t, Dense state) except *

    cpdef Data matmul_data(QobjEvo self, object t, Data state, Data out=*)
