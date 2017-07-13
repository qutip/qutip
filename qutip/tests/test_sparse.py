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
from numpy.testing import run_module_suite, assert_equal, assert_almost_equal
import scipy.sparse as sp

from qutip.random_objects import (rand_dm, rand_herm,
                                  rand_ket)
from qutip.states import coherent
from qutip.sparse import (sp_bandwidth, sp_permute, sp_reverse_permute,
                          sp_profile, sp_one_norm, sp_inf_norm)
from qutip.cy.spmath import zcsr_kron


def _permutateIndexes(array, row_perm, col_perm):
    return array[np.ix_(row_perm, col_perm)]


def _dense_profile(B):
    row_pro = 0
    for i in range(B.shape[0]):
        j = np.where(B[i, :] != 0)[0]
        if np.any(j):
            if j[-1] > i:
                row_pro += (j[-1]-i)
    col_pro = 0
    for j in range(B.shape[0]):
        i = np.where(B[:, j] != 0)[0]
        if np.any(i):
            if i[-1] > j:
                col_pro += i[-1]-j
    ans = (row_pro+col_pro, col_pro, row_pro)
    return ans


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
    for kk in range(10):
        A = sp.rand(100, 100, density=0.1, format='csr')
        ans1 = sp_bandwidth(A)
        A = A.toarray()
        i, j = np.nonzero(A)
        ans2 = ((j-i).max()+(i-j).max()+1, (i-j).max(), (j-i).max())
        assert_equal(ans1, ans2)

    for kk in range(10):
        A = sp.rand(100, 100, density=0.1, format='csc')
        ans1 = sp_bandwidth(A)
        A = A.toarray()
        i, j = np.nonzero(A)
        ans2 = ((j-i).max()+(i-j).max()+1, (i-j).max(), (j-i).max())
        assert_equal(ans1, ans2)


def test_sp_profile():
    "Sparse: Profile"
    for kk in range(10):
        A = sp.rand(1000, 1000, 0.1, format='csr')
        pro = sp_profile(A)
        B = A.toarray()
        ans = _dense_profile(B)
        assert_equal(pro, ans)

    for kk in range(10):
        A = sp.rand(1000, 1000, 0.1, format='csc')
        pro = sp_profile(A)
        B = A.toarray()
        ans = _dense_profile(B)
        assert_equal(pro, ans)

def test_sp_one_norm():
    "Sparse: one-norm"
    for kk in range(10):
        H = rand_herm(100,0.1).data
        nrm = sp_one_norm(H)
        ans = max(abs(H).sum(axis=0).flat)
        assert_almost_equal(nrm,ans)
        
def test_sp_inf_norm():
    "Sparse: inf-norm"
    for kk in range(10):
        H = rand_herm(100,0.1).data
        nrm = sp_inf_norm(H)
        ans = max(abs(H).sum(axis=1).flat)
        assert_almost_equal(nrm,ans)


if __name__ == "__main__":
    run_module_suite()
