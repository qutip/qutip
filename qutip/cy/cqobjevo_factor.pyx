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

    cdef void _call_core(t, complex* coeff_ptr):
        cdef int i
        cdef Coefficient coeff
        for i in range(self.num_ops):
            coeff = <Coefficient> self.coeffs[i]
            coeff_ptr[i] = coeff._call(t)

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
