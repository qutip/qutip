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
import scipy.linalg as la
from numpy.testing import assert_equal, assert_, run_module_suite
import unittest
from qutip import *
import qutip.settings as qset
from qutip.cy.brtools_testing import (_test_zheevr, _test_diag_liou_mult,
                    _test_dense_to_eigbasis, _test_vec_to_eigbasis,
                    _test_eigvec_to_fockbasis)

def test_br_zheevr():
    "BR Tools : zheevr"
    for kk in range(2,100):
        H = rand_herm(kk, 1/kk)
        H2 =np.asfortranarray(H.full())
        eigvals = np.zeros(kk,dtype=float)
        Z = _test_zheevr(H2, eigvals, qset.atol)
        ans_vals, ans_vecs = la.eigh(H.full())
        assert_(np.allclose(ans_vals,eigvals))
        assert_(np.allclose(Z,ans_vecs))

      
def test_br_dense_to_eigbasis():
    "BR Tools : dense operator to eigenbasis"
    N = 10
    for kk in range(50):
        H = rand_herm(N,0.5)
        a = rand_herm(N,0.5)
        evals, evecs = H.eigenstates()
        A = a.transform(evecs).full()
        H2 = H.full('F')
        eigvals = np.zeros(N,dtype=float)
        Z = _test_zheevr(H2, eigvals, qset.atol)
        a2 = a.full('F')
        assert_(np.allclose(A,_test_dense_to_eigbasis(a2,Z)))
        b = destroy(N)
        B = b.transform(evecs).full()
        b2 = b.full('F')
        assert_(np.allclose(B,_test_dense_to_eigbasis(b2,Z)))


def test_vec_to_eigbasis():
    "BR Tools : vector to eigenbasis"
    N = 10
    for kk in range(50):
        H = rand_herm(N,0.5)
        h = H.full('F')
        R = rand_dm(N,0.5)
        r = R.full().ravel()
        ans = R.transform(H.eigenstates()[1]).full().ravel()
        out = _test_vec_to_eigbasis(h, r)
        assert_(np.allclose(ans,out))


def test_eigvec_to_fockbasis():
    "BR Tools : eigvector to fockbasis"
    N = 10
    for kk in range(50):
        H = rand_herm(N,0.5)
        h = H.full('F')
        R = rand_dm(N,0.5)
        r = R.full().ravel()
        eigvals = np.zeros(N,dtype=float)
        Z = _test_zheevr(H.full('F'), eigvals, qset.atol)
        eig_vec = R.transform(H.eigenstates()[1]).full().ravel()
        out = _test_eigvec_to_fockbasis(eig_vec, Z, N)
        assert_(np.allclose(r,out))


def test_diag_liou_mult():
    "BR Tools : Diagonal liouvillian mult"
    for kk in range(2,100):
        H = rand_dm(kk,0.5)
        evals, evecs = H.eigenstates()
        H_eig = H.transform(evecs)
        L = liouvillian(H_eig)
        y = np.ones(kk**2, dtype=complex)
        out = np.zeros(kk**2, dtype=complex)
        ans = L.data.dot(y)
        _test_diag_liou_mult(evals,y,out,H.shape[0])
        assert_(np.allclose(ans,out))


if __name__ == "__main__":
    run_module_suite()
