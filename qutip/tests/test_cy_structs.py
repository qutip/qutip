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
import scipy.sparse as sp
from qutip import rand_dm
from qutip.fastsparse import fast_csr_matrix
from qutip.cy.checks import (_test_sorting, _test_coo2csr_inplace_struct,
                            _test_csr2coo_struct, _test_coo2csr_struct)
from qutip.random_objects import rand_jacobi_rotation
from numpy.testing import assert_equal, assert_, run_module_suite

def _unsorted_csr(N, density=0.5):
    N = np.arange(N)
    M = sp.diags(N,0, dtype=complex, format='csr')
    N = N.shape[0]
    nvals = N**2*density
    while M.nnz < 0.95*nvals:
        M = rand_jacobi_rotation(M)
    M = M.tocsr()
    return fast_csr_matrix((M.data,M.indices,M.indptr),
                            shape = M.shape)



def test_coo2csr_struct():
    "Cython structs : COO to CSR"
    for k in range(20):
        A = rand_dm(5,0.5).data
        B = A.tocoo()
        C = _test_coo2csr_struct(B)
        assert_(not (A!=C).data.any())



def test_indices_sort():
    "Cython structs : sort CSR indices inplace"
    for k in range(20):
        A = _unsorted_csr(10,0.25)
        B = A.copy()
        B.sort_indices()
        _test_sorting(A)
        assert_(np.all(A.data == B.data))
        assert_(np.all(A.indices == B.indices))


def test_coo2csr_inplace_nosort():
    "Cython structs : COO to CSR inplace (no sort)"
    for k in range(20):
        A = rand_dm(5,0.5).data
        B = A.tocoo()
        C = _test_coo2csr_inplace_struct(B, sorted = 0)
        assert_(not (A!=C).data.any())


def test_coo2csr_inplace_sort():
    "Cython structs : COO to CSR inplace (sorted)"
    for k in range(20):
        A = rand_dm(5,0.5).data
        B = A.tocoo()
        C = _test_coo2csr_inplace_struct(B, sorted = 1)
        assert_(not (A!=C).data.any())


def test_csr2coo():
    "Cython structs : CSR to COO"
    for k in range(20):
        A = rand_dm(5,0.5).data
        B = A.tocoo()
        C = _test_csr2coo_struct(A)
        assert_(not (B!=C).data.any())



if __name__ == "__main__":
    run_module_suite()
