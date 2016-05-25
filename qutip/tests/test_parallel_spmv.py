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
from numpy.testing import assert_, assert_almost_equal, run_module_suite
import unittest
from qutip import rand_herm, rand_dm, rand_ket
import qutip.settings as qset
from qutip.cy.spmatfuncs import (spmv_csr, spmvpy, cy_ode_rhs)
if qset.has_parallel:
    from qutip.cy.parallel.parfuncs import (parallel_spmv_csr,
                    parallel_spmvpy, parallel_ode_rhs)


class TestParallelSpMv:
    """
    A test class for the cy.parallel functions.
    """
    @unittest.skipIf(qset.has_parallel == False, 'Parallel extension not found.')
    def test_spmv_csr(self):
        "Parallel: spmv"

        H = rand_herm(10).data
        psi = rand_ket(10).full().ravel()
        threads = qset.num_cpus
        par_ans = parallel_spmv_csr(H.data, H.indices, H.indptr, psi, threads)
        ser_ans = spmv_csr(H.data, H.indices, H.indptr, psi)
        assert_almost_equal(par_ans,ser_ans)

    @unittest.skipIf(qset.has_parallel == False, 'Parallel extension not found.')
    def test_spmvpy(self):
        "Parallel: spmvpy"

        H = rand_herm(10).data
        psi = rand_ket(10).full().ravel()
        threads = qset.num_cpus
        out = np.zeros(H.shape[0], dtype=complex)
        par_ans = parallel_spmvpy(H.data, H.indices, H.indptr, psi, 1.0, out, threads)
        out = np.zeros(H.shape[0], dtype=complex)
        ser_ans = spmvpy(H.data, H.indices, H.indptr, psi, 1.0, out)
        assert_almost_equal(par_ans,ser_ans)
    
    
    @unittest.skipIf(qset.has_parallel == False, 'Parallel extension not found.')
    def test_ode_rhs(self):
        "Parallel: ode_rhs"

        H = rand_herm(10).data
        psi = rand_ket(10).full().ravel()
        threads = qset.num_cpus
        par_ans = parallel_ode_rhs(1.0, psi, H.data, H.indices, H.indptr, threads)
        ser_ans = cy_ode_rhs(1.0, psi, H.data, H.indices, H.indptr)
        assert_almost_equal(par_ans,ser_ans)


if __name__ == "__main__":
    run_module_suite()
