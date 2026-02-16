#cython: language_level=3

from qutip.core.data cimport Data, Dense
from qutip.core.cy.qobjevo cimport QobjEvo

cdef class LindbladMatrixForm(QobjEvo):
    cdef public QobjEvo H_nh
    cdef public list c_ops
    cdef public int num_collapse

    # Pre-allocated temporary buffer
    cdef Dense _temp_buffer
    cdef int _buffer_size
