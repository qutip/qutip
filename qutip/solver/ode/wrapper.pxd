#cython: language_level=3

cimport numpy as cnp
from ...core cimport data as _data
from .._solverqevo cimport SolverQEvo

cdef class QtOdeData:
    cdef _data.Data state
    cdef bint inplace
    cpdef void inplace_add(self, QtOdeData other, double factor)
    cpdef void zero(self)
    cpdef double norm(self)
    cpdef void copy(self, QtOdeData other)
    cpdef QtOdeData empty_like(self)
    cpdef object raw(self)
    cpdef _data.Data data(self)
    cpdef void set_data(self, _data.Data new)

cdef class QtOdeFuncWrapper:
    cdef SolverQEvo evo
    cpdef void call(self, QtOdeData out, double t, QtOdeData y)
