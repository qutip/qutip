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
        double _shift_dt

    cpdef Data _call(QobjEvo self, double t)

    cdef double _prepare(QobjEvo self, double t, Data state=*)

    cpdef double complex expect_data(QobjEvo self, double t, Data state)

    cdef double complex _expect_dense(QobjEvo self, double t, Dense state)

    cpdef Data matmul_data(QobjEvo self, double t, Data state, Data out=*)
