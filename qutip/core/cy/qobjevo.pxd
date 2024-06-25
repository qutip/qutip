#cython: language_level=3

from qutip.core.data cimport Dense, Data
from qutip.core.data.base cimport idxint

cdef class QobjEvo:
    cdef:
        list elements
        readonly object _dims
        readonly (idxint, idxint) shape
        int _issuper
        int _isoper
        readonly dict _feedback_functions
        readonly dict _solver_only_feedback

    cpdef Data _call(QobjEvo self, double t)

    cdef object _prepare(QobjEvo self, object t, Data state=*)

    cpdef object expect_data(QobjEvo self, object t, Data state)

    cdef double complex _expect_dense(QobjEvo self, double t, Dense state) except *

    cpdef Data matmul_data(QobjEvo self, object t, Data state, Data out=*)
