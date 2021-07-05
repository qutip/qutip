#cython: language_level=3

from qutip.core.data cimport CSR, Dense, Data
from qutip.core.cy.coefficient cimport Coefficient
from qutip.core.data.base cimport idxint
from libcpp cimport bool

cdef class _BaseElement:
    cpdef Data data(self, double t)
    cpdef object qobj(self, double t)
    cpdef double complex coeff(self, double t) except *
    cdef Data matmul_data_t(_BaseElement self, double t, Data state, Data out=?)


cdef class _ConstantElement(_BaseElement):
    cdef readonly object _qobj


cdef class _EvoElement(_BaseElement):
    cdef readonly object _qobj
    cdef readonly Coefficient _coefficient


cdef class _FuncElement(_BaseElement):
    cdef object _func
    cdef dict _args
    cdef tuple _previous


cdef class _MapElement(_BaseElement):
    cdef _FuncElement _base
    cdef list _transform
    cdef double complex _coeff


cdef class _ProdElement(_BaseElement):
    cdef _BaseElement _left
    cdef _BaseElement _right
    cdef list _transform
    cdef bool _conj
