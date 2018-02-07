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

from qutip.cy.sparse_structs cimport CSR_Matrix, COO_Matrix


cdef class cy_qobj:
    cdef void _rhs_mat(self, double t, complex* vec, complex* out)
    cdef complex _expect_mat(self, double t, complex* vec, int isherm)
    cdef complex _expect_mat_super(self, double t, complex* vec, int isherm)


cdef class cy_cte_qobj(cy_qobj):
    cdef int total_elem
    cdef int shape0, shape1
    cdef object dims
    cdef int super

    # pointer to data
    cdef CSR_Matrix cte

    cdef void _rhs_mat(self, double t, complex* vec, complex* out)
    cdef complex _expect_mat(self, double t, complex* vec, int isherm)
    cdef complex _expect_mat_super(self, double t, complex* vec, int isherm)


cdef class cy_td_qobj(cy_qobj):
    cdef long total_elem
    cdef int shape0, shape1
    cdef object dims
    cdef int super
    cdef void (*factor_ptr)(double, complex*)
    cdef object factor_func
    cdef int factor_use_ptr

    # pointer to data
    cdef CSR_Matrix cte
    cdef CSR_Matrix ** ops
    cdef long[::1] sum_elem
    cdef int N_ops

    cdef void factor(self, double t, complex* out)
    cdef void _call_core(self, double t, CSR_Matrix * out, complex* coeff)
    cdef void _rhs_mat(self, double t, complex* vec, complex* out)
    cdef complex _expect_psi(self, complex* data, int* idx, int* ptr,
                             complex* vec, int isherm)
    cdef complex _expect_mat(self, double t, complex* vec, int isherm)
    cdef complex _expect_mat_super(self, double t, complex* vec, int isherm)
