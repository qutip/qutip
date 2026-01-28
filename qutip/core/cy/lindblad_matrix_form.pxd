#cython: language_level=3

from qutip.core.data cimport Data
from qutip.core.cy.qobjevo cimport QobjEvo

cdef class LindbladMatrixForm:
    cdef public QobjEvo H_nh
    cdef public list c_ops
    cdef public int num_collapse
    cdef public object _dims
    cdef public tuple shape
    cdef public bint isconstant
    cdef public str type
    cdef public bint issuper
    
    # Pre-allocated temporary buffer
    cdef public Data _temp_buffer
    cdef public int _buffer_size

    cpdef Data matmul_data(self, object t, Data rho, Data out=*)
