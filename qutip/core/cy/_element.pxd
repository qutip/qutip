#cython: language_level=3

from qutip.core.data cimport CSR, Dense, Data
from qutip.core.cy.coefficient cimport Coefficient
from qutip.core.data.base cimport idxint
from libcpp cimport bool

cdef class _BaseElement:
    cdef:
        Data _data
        double complex _coeff
        readonly object _qobj

    cpdef Data data(self, double t)
    cpdef object qobj(self, double t)
    cpdef double complex coeff(self, double t) except *
    cdef Data matmul(_BaseElement self, double t, Data state, Data out)

cdef class _CteElement(_BaseElement):
    pass

cdef class _EvoElement(_BaseElement):
    cdef readonly Coefficient coefficient

cdef class _FuncElement(_BaseElement):
    cdef:
        object func
        dict args
        tuple _previous

cdef class _MapElement(_BaseElement):
    cdef:
        _FuncElement base
        list transform

cdef class _ProdElement(_BaseElement):
    cdef:
        _BaseElement left
        _BaseElement right
        list transform
        bool conj
