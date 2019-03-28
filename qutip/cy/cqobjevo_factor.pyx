#!python
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
cimport numpy as np
import cython
cimport cython
cimport libc.math
from qutip.cy.inter import _prep_cubic_spline
from qutip.cy.inter cimport (_spline_complex_cte_second,
                             _spline_complex_t_second)
from qutip.cy.interpolate cimport (interp, zinterp)
from qutip.cy.cqobjevo cimport CQobjEvo

include "complex_math.pxi"

"""
Support cqobjevo's array and str based coefficient.
By using inheritance, it is possible to 'cimport' coefficient compiled at
runtime. Pure array based, (inter.pyx or interpolate.pyx) are defined here.
str inherite from StrCoeff and just add the _call_core method.
"""

cdef np.ndarray[complex, ndim=1] zptr2array1d(complex* ptr, int N):
    cdef np.npy_intp Ns[1]
    Ns[0] = N
    return np.PyArray_SimpleNewFromData(1, Ns, np.NPY_COMPLEX128, ptr)

cdef np.ndarray[complex, ndim=2] zptr2array2d(complex* ptr, int R, int C):
    cdef np.npy_intp Ns[2]
    Ns[0] = R
    Ns[1] = C
    return np.PyArray_SimpleNewFromData(2, Ns, np.NPY_COMPLEX128, ptr)

cdef np.ndarray[int, ndim=1] iprt2array(int* ptr, int N):
    cdef np.npy_intp Ns[1]
    Ns[0] = N
    return np.PyArray_SimpleNewFromData(1, Ns, np.NPY_INT32, ptr)

cdef class CoeffFunc:
    def __init__(self, ops, args, tlist):
        self._args = {}

    def __call__(self, double t, args={}):
        cdef np.ndarray[ndim=1, dtype=complex] coeff = \
                                            np.zeros(self._num_ops, dtype=complex)
        self._call_core(t, &coeff[0])
        return coeff

    def set_args(self, args):
        pass

    cdef void _call_core(self, double t, complex* coeff):
        pass

    cdef void _dyn_args(self, double t, complex* state, int[::1] shape):
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
        self.a = ops[0][2].a
        self.b = ops[0][2].b
        l = len(ops[0][2].coeffs)
        self.c = np.zeros((self._num_ops, l), dtype=complex)
        for i in range(self._num_ops):
            for j in range(l):
                self.c[i,j] = ops[i][2].coeffs[j]

    def __call__(self, t, args={}):
        cdef np.ndarray[ndim=1, dtype=complex] coeff = \
                                            np.zeros(self._num_ops, dtype=complex)
        self._call_core(t, &coeff[0])
        return coeff

    cdef void _call_core(self, double t, complex* coeff):
        cdef int i
        for i in range(self._num_ops):
            coeff[i] = zinterp(t, self.a, self.b, self.c[i,:])

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
        self.y = np.zeros((self._num_ops, self.n_t), dtype=complex)
        self.M = np.zeros((self._num_ops, self.n_t), dtype=complex)

        for i in range(self._num_ops):
            m, cte = _prep_cubic_spline(ops[i][2], tlist)
            if not cte:
                raise Exception("tlist not sampled uniformly")
            for j in range(self.n_t):
                self.y[i,j] = ops[i][2][j]
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
            m, cte = _prep_cubic_spline(ops[i][2], tlist)
            if cte:
                print("tlist not uniform?")
            for j in range(self.n_t):
                self.y[i,j] = ops[i][2][j]
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
        self._mat_shape[0] = 0
        self._mat_shape[1] = 0
        if dyn_args:
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

    cdef void _dyn_args(self, double t, complex* state, int[::1] shape):
        cdef int ii, nn = shape[0] * shape[1]
        self._vec = <complex[:nn]> state
        self._mat_shape[0] = shape[0]
        self._mat_shape[1] = shape[1]
        cdef CQobjEvo cop
        for ii in range(self._num_expect):
            cop = self._expect_op[ii]
            if cop.shape1 != nn:
                self._expect_vec[ii] = cop._overlapse(t, state)
            elif cop.super:
                self._expect_vec[ii] = cop._expect_super(t, state)
            else:
                self._expect_vec[ii] = cop._expect(t, state)

    def __call__(self, double t, args={}, vec=None):
        cdef np.ndarray[ndim=1, dtype=complex] coeff = \
                                    np.zeros(self._num_ops, dtype=complex)
        cdef int[2] shape
        if vec is not None:
            if isinstance(vec, np.ndarray):
                self._vec = vec.ravel("F")
                shape[0] = vec.shape[0]
                shape[1] = vec.shape[1]
            else:
                full = vec.full()
                self._vec = full.ravel("F")
                shape[0] = full.shape[0]
                shape[1] = full.shape[1]
            self._dyn_args(t, &self._vec[0], shape)

        if args:
            now_args = self.args.copy()
            now_args.update(args)
            self.set_args(now_args)
            self._call_core(t, &coeff[0])
            self.set_args(self._args)
        else:
            self._call_core(t, &coeff[0])

        return coeff

    def __getstate__(self):
        return (self._num_ops, self._args, self._dyn_args_list)

    def __setstate__(self, state):
        self._num_ops = state[0]
        self._args = state[1]
        self._dyn_args_list = state[2]
        self.set_args(self._args)
        self._set_dyn_args(self._dyn_args_list)
