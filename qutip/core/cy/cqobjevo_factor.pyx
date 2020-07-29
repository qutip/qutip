#cython: language_level=3
# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, The QuTiP Project.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

import numpy as np
<<<<<<< HEAD:qutip/core/cy/cqobjevo_factor.pyx
cimport numpy as cnp
cnp.import_array()

from ..qobj import Qobj
from .. import data as _data
from .inter import _prep_cubic_spline
from qutip.core.cy.inter cimport (
    _spline_complex_cte_second, _spline_complex_t_second, _step_complex_t,
    _step_complex_cte,
)
from qutip.core.cy.interpolate cimport zinterp
from qutip.core.cy.cqobjevo cimport CQobjEvo
from qutip.core.data cimport Data

"""
Support CQobjEvo's array and str based coefficient.
By using inheritance, it is possible to 'cimport' coefficient compiled at
runtime. Pure array based, (inter.pyx or interpolate.pyx) are defined here.
str inherite from StrCoeff and just add the _call_core method.
"""


cdef class CoeffFunc:
    def __init__(self, ops, args, tlist):
        self._args = {}

    def __call__(self, double t, args=None):
        cdef object coeff = cnp.PyArray_ZEROS(1, [self._num_ops],
                                              cnp.NPY_COMPLEX128, True)
        self._call_core(t, <double complex *> cnp.PyArray_GETPTR1(coeff, 0))
        return coeff

    def set_args(self, args):
        pass

    cdef void _call_core(self, double t, complex* coeff):
        pass

    cdef void _dyn_args(self, double t, Data state):
        pass

    def __getstate__(self):
        pass

    def __setstate__(self, state):
        pass

    def get_args(self):
        return self._args


cdef class InterpolateCoeff(CoeffFunc):
    cdef double a, b
    cdef complex[:,::1] c

    def __init__(self, ops, args, tlist):
        cdef int i, j, l
        self._args = {}
        self._num_ops = len(ops)
        self.a = ops[0].coeff.a
        self.b = ops[0].coeff.b
        l = len(ops[0].coeff.coeffs)
        self.c = cnp.PyArray_ZEROS(2, [self._num_ops, l], cnp.NPY_COMPLEX128, False)
        for i in range(self._num_ops):
            for j in range(l):
                self.c[i, j] = ops[i].coeff.coeffs[j]

    cdef void _call_core(self, double t, complex* coeff):
        cdef int i
        for i in range(self._num_ops):
            coeff[i] = zinterp(t, self.a, self.b, self.c[i, :])

    def set_args(self, args):
        pass

    def __getstate__(self):
        return (self._num_ops, self.a, self.b, np.array(self.c))

    def __setstate__(self, state):
        self._num_ops = state[0]
        self.a = state[1]
        self.b = state[2]
        self.c = state[3]


cdef class InterCoeffCte(CoeffFunc):
    cdef int n_t
    cdef double dt
    cdef double[::1] tlist
    cdef complex[:,::1] y, M

    def __init__(self, ops, args, tlist):
        cdef int i, j
        self._args = {}
        self._num_ops = len(ops)
        self.tlist = tlist
        self.n_t = len(tlist)
        self.dt = tlist[1]-tlist[0]
        self.y = cnp.PyArray_ZEROS(2, [self._num_ops, self.n_t],
                                   cnp.NPY_COMPLEX128, False)
        self.M = cnp.PyArray_ZEROS(2, [self._num_ops, self.n_t],
                                   cnp.NPY_COMPLEX128, False)

        for i in range(self._num_ops):
            m, cte = _prep_cubic_spline(ops[i].coeff, tlist)
            if not cte:
                raise Exception("tlist not sampled uniformly")
            for j in range(self.n_t):
                self.y[i,j] = ops[i].coeff[j]
                self.M[i,j] = m[j]

    cdef void _call_core(self, double t, complex* coeff):
        cdef int i
        for i in range(self._num_ops):
            coeff[i] = _spline_complex_cte_second(t, self.tlist,
                                    self.y[i,:], self.M[i,:], self.n_t, self.dt)

    def set_args(self, args):
        pass

    def __getstate__(self):
        return (self._num_ops, self.n_t, self.dt, np.array(self.tlist),
                np.array(self.y), np.array(self.M))

    def __setstate__(self, state):
        self._num_ops = state[0]
        self.n_t = state[1]
        self.dt = state[2]
        self.tlist = state[3]
        self.y = state[4]
        self.M = state[5]


cdef class InterCoeffT(CoeffFunc):
    cdef int n_t
    cdef double dt
    cdef double[::1] tlist
    cdef complex[:,::1] y, M

    def __init__(self, ops, args, tlist):
        cdef int i, j
        self._args = {}
        self._num_ops = len(ops)
        self.tlist = tlist
        self.n_t = len(tlist)
        self.y = np.zeros((self._num_ops, self.n_t), dtype=complex)
        self.M = np.zeros((self._num_ops, self.n_t), dtype=complex)
        for i in range(self._num_ops):
            m, cte = _prep_cubic_spline(ops[i].coeff, tlist)
            if cte:
                print("tlist not uniform?")
            for j in range(self.n_t):
                self.y[i,j] = ops[i].coeff[j]
                self.M[i,j] = m[j]

    cdef void _call_core(self, double t, complex* coeff):
        cdef int i
        for i in range(self._num_ops):
            coeff[i] = _spline_complex_t_second(t, self.tlist,
                                    self.y[i,:], self.M[i,:], self.n_t)

    def set_args(self, args):
        pass

    def __getstate__(self):
        return (self._num_ops, self.n_t, None, np.array(self.tlist),
                np.array(self.y), np.array(self.M))

    def __setstate__(self, state):
        self._num_ops = state[0]
        self.n_t = state[1]
        self.tlist = state[3]
        self.y = state[4]
        self.M = state[5]


cdef class StepCoeff(CoeffFunc):
    cdef int n_t
    cdef double[::1] tlist
    cdef complex[:,::1] y

    def __init__(self, ops, args, tlist):
        cdef int i, j
        self._args = {}
        self._num_ops = len(ops)
        self.tlist = tlist
        self.n_t = len(tlist)
        self.y = np.zeros((self._num_ops, self.n_t), dtype=complex)
        for i in range(self._num_ops):
            for j in range(self.n_t):
                self.y[i,j] = ops[i].coeff[j]

    def set_arg(self, args):
        pass

    def __getstate__(self):
        return (self._num_ops, self.n_t, None, np.array(self.tlist),
                np.array(self.y))

    def __setstate__(self, state):
        self._num_ops = state[0]
        self.n_t = state[1]
        self.tlist = state[3]
        self.y = state[4]


cdef class StepCoeffT(StepCoeff):
    cdef void _call_core(self, double t, complex* coeff):
        cdef int i
        for i in range(self._num_ops):
            coeff[i] = _step_complex_t(t, self.tlist, self.y[i, :], self.n_t)


cdef class StepCoeffCte(StepCoeff):
    cdef void _call_core(self, double t, complex* coeff):
        cdef int i
        for i in range(self._num_ops):
            coeff[i] = _step_complex_cte(t, self.tlist, self.y[i, :], self.n_t)


cdef class StrCoeff(CoeffFunc):
    def __init__(self, ops, args, tlist, dyn_args=[]):
        self._num_ops = len(ops)
        self._args = args
        self._dyn_args_list = dyn_args
        self.set_args(args)
        self._set_dyn_args(dyn_args)

    def _set_dyn_args(self, dyn_args):
        self._num_expect = 0
        self._expect_op = []
        expect_def = []
        if dyn_args:
            # This stuff is used by code-generated classes CompiledStrCoeff and
            # the like.
            for name, what, op in dyn_args:
                if what == "expect":
                    self._expect_op.append(op.compiled_qobjevo)
                    expect_def.append(self._args[name])
                    self._num_expect += 1
                elif what == "vec":
                    self._vec = self._args[name]
                elif what == "mat":
                    self._vec = self._args[name].ravel("F")
                    self._mat_shape[0] = self._args[name].shape[0]
                    self._mat_shape[1] = self._args[name].shape[0]
                elif what == "Qobj":
                    self._vec = self._args[name].full().ravel("F")
                    self._mat_shape[0] = self._args[name].shape[0]
                    self._mat_shape[1] = self._args[name].shape[0]
        self._expect_vec = np.array(expect_def, dtype=complex)
=======
cimport numpy as np
import cython
cimport cython
cimport libc.math
from qutip.cy.inter import _prep_cubic_spline
from qutip.cy.inter cimport (_spline_complex_cte_second,
                             _spline_complex_t_second,
                             _step_complex_t, _step_complex_cte)
from qutip.cy.interpolate cimport (interp, zinterp)
from qutip.cy.cqobjevo cimport CQobjEvo


cdef class CoeffFunc:
    def __init__(self, coeff, args, dyn_args=[]):
        self.num_ops = len(coeff)
        self.coeffs = coeff
        self.args = args
        self.dyn_args_list = [obj for obj, dyn_args]

        self._set_dyn_args(dyn_args)

    def _set_dyn_args(self, dyn_args):
        # With datalayer, dyn_args could be a lot better

>>>>>>> Coefficient:qutip/cy/cqobjevo_factor.pyx

    cdef void _dyn_args(self, double t, Data state):
        cdef int ii
        self._vec = state.to_array().reshape((-1,))
        self._mat_shape[0] = state.shape[0]
        self._mat_shape[1] = state.shape[1]
        for ii in range(self._num_expect):
            self._expect_vec[ii] = self._expect_op[ii].expect(t, state)

    cdef void _call_core(t, complex* coeff_ptr):
        cdef int i
        cdef Coefficient coeff
        for i in range(self.num_ops):
            coeff = <Coefficient> self.coeffs[i]
            coeff_ptr[i] = coeff._call(t)

    def __call__(self, double t, args={}, vec=None):
        cdef object coeff = cnp.PyArray_ZEROS(1, [self._num_ops],
                                              cnp.NPY_COMPLEX128, False)
        cdef double complex *coeff_ptr =\
                <double complex *> cnp.PyArray_GETPTR1(coeff, 0)

        if vec is not None:
            if isinstance(vec, Qobj):
                vec = vec.data
            self._dyn_args(t, vec)

        if args:
            now_args = self.args.copy()
            now_args.update(args)
            self.set_args(now_args)
            self._call_core(t, coeff_ptr)
            self.set_args(self._args)
        else:
            self._call_core(t, coeff_ptr)

        return coeff

    def __getstate__(self):
        return (self._num_ops, self._args, self._dyn_args_list)

    def __setstate__(self, state):
        self._num_ops = state[0]
        self._args = state[1]
        self._dyn_args_list = state[2]
        self.set_args(self._args)
        self._set_dyn_args(self._dyn_args_list)
