#!python
#cython: language_level=3
# This file is generated automatically by QuTiP.

import numpy as np
cimport numpy as np
import scipy.special as spe
cimport cython
np.import_array()
cdef extern from "numpy/arrayobject.h" nogil:
    void PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
from qutip.cy.spmatfuncs cimport spmvpy
from qutip.cy.inter cimport _spline_complex_t_second, _spline_complex_cte_second
from qutip.cy.inter cimport _spline_float_t_second, _spline_float_cte_second
from qutip.cy.inter cimport _step_float_cte, _step_complex_cte
from qutip.cy.inter cimport _step_float_t, _step_complex_t
from qutip.cy.interpolate cimport (interp, zinterp)
from qutip.cy.cqobjevo_factor cimport StrCoeff
from qutip.cy.cqobjevo cimport CQobjEvo
from qutip.cy.math cimport erf, zerf
from qutip.qobj import Qobj
cdef double pi = 3.14159265358979323

include '/home/tarun/anaconda3/envs/qutip-env/lib/python3.7/site-packages/qutip/cy/complex_math.pxi'

cdef class CompiledStrCoeff(StrCoeff):
    cdef double[::1] _td_array_0
    cdef double _t0

    def set_args(self, args):
        self._td_array_0=args['_td_array_0']
        self._t0=args['_t0']

    cdef void _call_core(self, double t, complex * coeff):
        cdef double[::1] _td_array_0 = self._td_array_0
        cdef double _t0 = self._t0

        coeff[0] = ((0 if (t > 6.000000) else _td_array_0[int(round(19 * (t/6.000000)))])) * (conj((0 if (t > 6.000000) else _td_array_0[int(round(19 * (t/6.000000)))])))
        coeff[1] = (conj((0 if (t > 6.000000) else _td_array_0[int(round(19 * (t/6.000000)))]))) * ((0 if (t > 6.000000) else _td_array_0[int(round(19 * (t/6.000000)))]))
        coeff[2] = (conj((0 if (t > 6.000000) else _td_array_0[int(round(19 * (t/6.000000)))]))) * ((0 if (t > 6.000000) else _td_array_0[int(round(19 * (t/6.000000)))]))
