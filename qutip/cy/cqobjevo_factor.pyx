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
include "complex_math.pxi"

"""
Support cqobjevo's array and str based coefficient.
By using inheritance, it is possible to 'cimport' coefficient compiled at
runtime. Pure array based, (inter.pyx or interpolate.pyx) are defined here.
str inherite from StrCoeff and just add the _call_core method.
"""
cdef class CoeffFunc:
    def __init__(self, ops, args, tlist):
        pass

    def __call__(self, double t, args={}):
        cdef np.ndarray[ndim=1, dtype=complex] coeff = \
                                            np.zeros(self.num_ops, dtype=complex)
        self._call_core(t, &coeff[0])
        return coeff

    def set_args(self, args):
        pass

    cdef void _call_core(self, double t, complex* coeff):
        pass

    def __getstate__(self):
        pass

    def __setstate__(self, state):
        pass


cdef class InterpolateCoeff(CoeffFunc):
    cdef double a, b
    cdef complex[:,::1] c

    def __init__(self, ops, args, tlist):
        cdef int i, j, l
        self.num_ops = len(ops)
        self.a = ops[0][2].a
        self.b = ops[0][2].b
        l = len(ops[0][2].coeffs)
        self.c = np.zeros((self.num_ops, l), dtype=complex)
        for i in range(self.num_ops):
            for j in range(l):
                self.c[i,j] = ops[i][2].coeffs[j]

    def __call__(self, t, args={}):
        cdef np.ndarray[ndim=1, dtype=complex] coeff = \
                                            np.zeros(self.num_ops, dtype=complex)
        self._call_core(t, &coeff[0])
        return coeff

    cdef void _call_core(self, double t, complex* coeff):
        cdef int i
        for i in range(self.num_ops):
            coeff[i] = zinterp(t, self.a, self.b, self.c[i,:])

    def set_args(self, args):
        pass

    def __getstate__(self):
        return (self.num_ops, self.a, self.b, np.array(self.c))

    def __setstate__(self, state):
        self.num_ops = state[0]
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
        self.num_ops = len(ops)
        self.tlist = tlist
        self.n_t = len(tlist)
        self.dt = tlist[1]-tlist[0]
        self.y = np.zeros((self.num_ops, self.n_t), dtype=complex)
        self.M = np.zeros((self.num_ops, self.n_t), dtype=complex)

        for i in range(self.num_ops):
            m, cte = _prep_cubic_spline(ops[i][2], tlist)
            if not cte:
                raise Exception("tlist not sampled uniformly")
            for j in range(self.n_t):
                self.y[i,j] = ops[i][2][j]
                self.M[i,j] = m[j]

    cdef void _call_core(self, double t, complex* coeff):
        cdef int i
        for i in range(self.num_ops):
            coeff[i] = _spline_complex_cte_second(t, self.tlist,
                                    self.y[i,:], self.M[i,:], self.n_t, self.dt)

    def set_args(self, args):
        pass

    def __getstate__(self):
        return (self.num_ops, self.n_t, self.dt, np.array(self.tlist),
                np.array(self.y), np.array(self.M))

    def __setstate__(self, state):
        self.num_ops = state[0]
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
        self.num_ops = len(ops)
        self.tlist = tlist
        self.n_t = len(tlist)
        self.y = np.zeros((self.num_ops, self.n_t), dtype=complex)
        self.M = np.zeros((self.num_ops, self.n_t), dtype=complex)
        for i in range(self.num_ops):
            m, cte = _prep_cubic_spline(ops[i][2], tlist)
            if cte:
                print("tlist not uniformly?")
            for j in range(self.n_t):
                self.y[i,j] = ops[i][2][j]
                self.M[i,j] = m[j]

    cdef void _call_core(self, double t, complex* coeff):
        cdef int i
        for i in range(self.num_ops):
            coeff[i] = _spline_complex_t_second(t, self.tlist,
                                    self.y[i,:], self.M[i,:], self.n_t)

    def set_args(self, args):
        pass

    def __getstate__(self):
        return (self.num_ops, self.n_t, None, np.array(self.tlist),
                np.array(self.y), np.array(self.M))

    def __setstate__(self, state):
        self.num_ops = state[0]
        self.n_t = state[1]
        self.tlist = state[3]
        self.y = state[4]
        self.M = state[5]


cdef class StrCoeff(CoeffFunc):
    def __init__(self, ops, args, tlist):
        self.num_ops = len(ops)
        self.args = args
        self.set_args(args)

    def __call__(self, double t, args={}):
        cdef np.ndarray[ndim=1, dtype=complex] coeff = \
                                            np.zeros(self.num_ops, dtype=complex)
        if args:
            now_args = self.args.copy()
            now_args.update(args)
            self.set_args(now_args)
            self._call_core(t, &coeff[0])
            self.set_args(self.args)
        else:
            self._call_core(t, &coeff[0])
        return coeff

    def __getstate__(self):
        return (self.num_ops, self.args)

    def __setstate__(self, state):
        self.num_ops = state[0]
        self.args = state[1]
        self.set_args(self.args)
