#cython: language_level=3

from qutip.core.data cimport CSR, Dense, Data
from qutip.core.cy.coefficient cimport Coefficient
from qutip.core.data.base cimport idxint
from libcpp cimport bool

cdef class _BaseElement:
    cdef Data _data
    cpdef Data data(self, t)
    cpdef object qobj(self, t)
    cpdef object coeff(self, t)
    cdef Data matmul_data_t(_BaseElement self, t, Data state, Data out=?)


cdef class _ConstantElement(_BaseElement):
    cdef readonly object _qobj


cdef class _EvoElement(_BaseElement):
    cdef readonly object _qobj
    cdef readonly Coefficient _coefficient


cdef class _FuncElement(_BaseElement):
    cdef readonly object _func
    cdef readonly dict _args
    cdef readonly tuple _previous
    cdef readonly bint _f_pythonic
    cdef readonly set _f_parameters


cdef class _MapElement(_BaseElement):
    cdef readonly _FuncElement _base
    cdef readonly list _transform
    cdef readonly double complex _coeff


cdef class _ProdElement(_BaseElement):
    cdef readonly _BaseElement _left
    cdef readonly _BaseElement _right
    cdef readonly list _transform
    cdef readonly bool _conj
