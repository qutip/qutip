# distutils: language = c++
# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
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

from qutip.cy.td_qobj_cy cimport cy_qobj#, cy_td_qobj, cy_cte_qobj

cdef extern from "Python.h":
    object PyLong_FromVoidPtr(void *)
    void* PyLong_AsVoidPtr(object)

cdef cy_qobj compiled_cte_cy

cdef void _rhs(double t, complex* vec, complex* out):
    global compiled_cte_cy
    compiled_cte_cy._rhs_mat(t, vec, out)

cdef extern from "src/dopri5.cpp":
    cdef cppclass ode:
        ode()
        ode(int*, double*, void (*_H)(double, complex *, complex *))
        int step(double, double, complex*, complex*)
        double integrate(double, double, double, complex*, double*)
        int len()
        int debug(complex * , complex *, double* )

cdef class ode_td_dopri:
    cdef ode* cobj

    def __init__(self, int l, H, config):
        global compiled_cte_cy
        _y1 = np.zeros(l, dtype=complex)
        cdef int[::1] int_option = np.zeros(2, dtype=np.intc)
        cdef double[::1] double_option = np.zeros(5, dtype=np.double)
        compiled_cte_cy = H.compiled_Qobj
        cdef void (*rhs)(double, complex*, complex*)
        rhs = <void (*)(double, complex*, complex*)> _rhs

        int_option[0] = l
        int_option[1] = config.norm_steps

        double_option[0] = config.options.atol
        double_option[1] = config.options.rtol
        double_option[2] = config.options.min_step
        double_option[3] = config.options.max_step
        double_option[4] = config.norm_tol

        self.cobj = new ode(<int*>&int_option[0],
                            <double*>&double_option[0],
                            rhs)
        if self.cobj == NULL:
            raise MemoryError('Not enough memory.')

    def __del__(self):
        del self.cobj

    cpdef double integrate(self, double _t_in, double _t_target, double rand,
                           complex[::1] _psi, double[::1] _err):
        return self.cobj.integrate(_t_in, _t_target, rand,
                                   <complex*>&_psi[0],
                                   <double*>&_err[0])

    def debug(self):
        l = self.cobj.len()
        cdef complex[::1] derr_in = np.zeros(l, dtype=complex)
        cdef complex[::1] derr_out = np.zeros(l, dtype=complex)
        cdef double[::1] opt = np.zeros(8)
        print("step_limit", self.cobj.debug(<complex*>&derr_in[0],
                                            <complex*>&derr_out[0],
                                            <double*>&opt[0]))
        for i in range(l):
            print(derr_in[i], derr_out[i])
        for i in range(8):
            print(opt[i])
