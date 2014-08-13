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
from numpy.testing import assert_, run_module_suite, assert_equal
import scipy.sparse as sp

from qutip.random_objects import rand_dm
from qutip.operators import create, destroy, qeye
from qutip.states import coherent
from qutip.sparse import sp_bandwidth, sp_permute, sp_reverse_permute
from qutip.graph import reverse_cuthill_mckee


def _permutateIndexes(array, row_perm, col_perm):
    return array[np.ix_(row_perm, col_perm)]


def test_sparse_symmetric_permute():
    "Sparse: Symmetric Permute"
    # CSR version
    A = rand_dm(25, 0.5)
    perm = np.random.permutation(25)
    x = sp_permute(A.data, perm, perm).toarray()
    z = _permutateIndexes(A.full(), perm, perm)
    assert_equal((x - z).all(), 0)
    # CSC version
    B = A.data.tocsc()
    y = sp_permute(B, perm, perm).toarray()
    assert_equal((y - z).all(), 0)


def test_sparse_nonsymmetric_permute():
    "Sparse: Nonsymmetric Permute"
    # CSR version
    A = rand_dm(25, 0.5)
    rperm = np.random.permutation(25)
    cperm = np.random.permutation(25)
    x = sp_permute(A.data, rperm, cperm).toarray()
    z = _permutateIndexes(A.full(), rperm, cperm)
    assert_equal((x - z).all(), 0)
    # CSC version
    B = A.data.tocsc()
    y = sp_permute(B, rperm, cperm).toarray()
    assert_equal((y - z).all(), 0)


def test_sparse_symmetric_reverse_permute():
    "Sparse: Symmetric Reverse Permute"
    # CSR version
    A = rand_dm(25, 0.5)
    perm = np.random.permutation(25)
    x = sp_permute(A.data, perm, perm)
    B = sp_reverse_permute(x, perm, perm)
    assert_equal((A.full() - B.toarray()).all(), 0)
    # CSC version
    B = A.data.tocsc()
    perm = np.random.permutation(25)
    x = sp_permute(B, perm, perm)
    B = sp_reverse_permute(x, perm, perm)
    assert_equal((A.full() - B.toarray()).all(), 0)


def test_sparse_nonsymmetric_reverse_permute():
    "Sparse: Nonsymmetric Reverse Permute"
    # CSR square array check
    A = rand_dm(25, 0.5)
    rperm = np.random.permutation(25)
    cperm = np.random.permutation(25)
    x = sp_permute(A.data, rperm, cperm)
    B = sp_reverse_permute(x, rperm, cperm)
    assert_equal((A.full() - B.toarray()).all(), 0)
    # CSC square array check
    A = rand_dm(25, 0.5)
    rperm = np.random.permutation(25)
    cperm = np.random.permutation(25)
    B = A.data.tocsc()
    x = sp_permute(B, rperm, cperm)
    B = sp_reverse_permute(x, rperm, cperm)
    assert_equal((A.full() - B.toarray()).all(), 0)
    # CSR column vector check
    A = coherent(25, 1)
    rperm = np.random.permutation(25)
    x = sp_permute(A.data, rperm, [])
    B = sp_reverse_permute(x, rperm, [])
    assert_equal((A.full() - B.toarray()).all(), 0)
    # CSC column vector check
    A = coherent(25, 1)
    rperm = np.random.permutation(25)
    B = A.data.tocsc()
    x = sp_permute(B, rperm, [])
    B = sp_reverse_permute(x, rperm, [])
    assert_equal((A.full() - B.toarray()).all(), 0)
    # CSR row vector check
    A = coherent(25, 1).dag()
    cperm = np.random.permutation(25)
    x = sp_permute(A.data, [], cperm)
    B = sp_reverse_permute(x, [], cperm)
    assert_equal((A.full() - B.toarray()).all(), 0)
    # CSC row vector check
    A = coherent(25, 1).dag()
    cperm = np.random.permutation(25)
    B = A.data.tocsc()
    x = sp_permute(B, [], cperm)
    B = sp_reverse_permute(x, [], cperm)
    assert_equal((A.full() - B.toarray()).all(), 0)


def test_sp_bandwidth():
    "Sparse: Bandwidth"
    A = sp.rand(10,10,density=0.3,format='csr')
    ans1 = sp_bandwidth(A)
    A = A.toarray()
    i,j = np.nonzero(A)
    ans2 = ((j-i).max()+(i-j).max()+1,(i-j).max(),(j-i).max())
    assert_equal(ans1, ans2)
    

if __name__ == "__main__":
    run_module_suite()
