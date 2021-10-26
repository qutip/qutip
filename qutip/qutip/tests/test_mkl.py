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
import scipy.linalg as la
from numpy.testing import (assert_, run_module_suite, assert_array_almost_equal)
import unittest
from qutip import *
import qutip.settings as qset
if qset.has_mkl:
    from qutip._mkl.spsolve import (mkl_splu, mkl_spsolve)


@unittest.skipIf(qset.has_mkl == False, 'MKL extensions not found.')
def test_mkl_spsolve1():
    """
    MKL spsolve : Single RHS vector (Real)
    """
    Adense = np.array([[0., 1., 1.],
                    [1., 0., 1.],
                    [0., 0., 1.]])
    As = sp.csr_matrix(Adense)
    np.random.seed(1234)
    x = np.random.randn(3)
    b = As * x
    x2 = mkl_spsolve(As, b)
    assert_array_almost_equal(x, x2)


@unittest.skipIf(qset.has_mkl == False, 'MKL extensions not found.')
def test_mklspsolve2():
    """
    MKL spsolve : Single RHS vector (Complex)
    """
    A = rand_herm(10)
    x = rand_ket(10).full()
    b = A * x
    y = mkl_spsolve(A.data,b)
    assert_array_almost_equal(x, y)


@unittest.skipIf(qset.has_mkl == False, 'MKL extensions not found.')
def test_mkl_spsolve3():
    """
    MKL spsolve : Multi RHS vector (Real)
    """
    row = np.array([0,0,1,2,2,2])
    col = np.array([0,2,2,0,1,2])
    data = np.array([1,2,3,-4,5,6])
    sM = sp.csr_matrix((data,(row,col)), shape=(3,3), dtype=float)
    M = sM.toarray()
    row = np.array([0,0,1,1,0,0])
    col = np.array([0,2,1,1,0,0])
    data = np.array([1,1,1,1,1,1])
    sN = sp.csr_matrix((data, (row,col)), shape=(3,3), dtype=float)
    N = sN.toarray()
    sX = mkl_spsolve(sM, N)
    X = la.solve(M, N)
    assert_array_almost_equal(X, sX)


@unittest.skipIf(qset.has_mkl == False, 'MKL extensions not found.')
def test_mkl_spsolve4():
    """
    MKL spsolve : Multi RHS vector (Complex)
    """
    row = np.array([0,0,1,2,2,2])
    col = np.array([0,2,2,0,1,2])
    data = np.array([1,2,3,-4,5,6],dtype=complex)
    sM = sp.csr_matrix((data,(row,col)), shape=(3,3), dtype=complex)
    M = sM.toarray()
    row = np.array([0,0,1,1,0,0])
    col = np.array([0,2,1,1,0,0])
    data = np.array([1,1,1,1,1,1],dtype=complex)
    sN = sp.csr_matrix((data, (row,col)), shape=(3,3), dtype=complex)
    N = sN.toarray()
    sX = mkl_spsolve(sM, N)
    X = la.solve(M, N)
    assert_array_almost_equal(X, sX)


@unittest.skipIf(qset.has_mkl == False, 'MKL extensions not found.')
def test_mkl_spsolve5():
    """
    MKL splu : Repeated RHS solve (Real)
    """
    row = np.array([0,0,1,2,2,2])
    col = np.array([0,2,2,0,1,2])
    data = np.array([1,2,3,-4,5,6])
    sM = sp.csr_matrix((data,(row,col)), shape=(3,3), dtype=float)
    M = sM.toarray()

    row = np.array([0,0,1,1,0,0])
    col = np.array([0,2,1,1,0,0])
    data = np.array([1,1,1,1,1,1])
    sN = sp.csr_matrix((data, (row,col)), shape=(3,3), dtype=float)
    N = sN.toarray()

    sX = np.zeros((3,3),dtype=float)
    lu = mkl_splu(sM)

    for k in range(3):
        sX[:,k] = lu.solve(N[:,k])
    lu.delete()
    
    X = la.solve(M,N)
    assert_array_almost_equal(X,sX)


@unittest.skipIf(qset.has_mkl == False, 'MKL extensions not found.')
def test_mkl_spsolve6():
    """
    MKL splu : Repeated RHS solve (Complex)
    """
    row = np.array([0,0,1,2,2,2])
    col = np.array([0,2,2,0,1,2])
    data = np.array([1,2,3,-4,5,6], dtype=complex)
    sM = sp.csr_matrix((data,(row,col)), shape=(3,3), dtype=complex)
    M = sM.toarray()

    row = np.array([0,0,1,1,0,0])
    col = np.array([0,2,1,1,0,0])
    data = np.array([1,1,1,1,1,1], dtype=complex)
    sN = sp.csr_matrix((data, (row,col)), shape=(3,3), dtype=complex)
    N = sN.toarray()

    sX = np.zeros((3,3),dtype=complex)
    lu = mkl_splu(sM)

    for k in range(3):
        sX[:,k] = lu.solve(N[:,k])
    lu.delete()
    
    X = la.solve(M,N)
    assert_array_almost_equal(X, sX)


@unittest.skipIf(qset.has_mkl == False, 'MKL extensions not found.')
def test_mkl_spsolve7():
    """
    MKL spsolve : Solution shape same as input RHS vec
    """
    row = np.array([0,0,1,2,2,2])
    col = np.array([0,2,2,0,1,2])
    data = np.array([1,2,3,-4,5,6], dtype=complex)
    A = sp.csr_matrix((data,(row,col)), shape=(3,3), dtype=complex)

    b = np.array([0,2,0],dtype=complex)
    out = mkl_spsolve(A,b)
    assert_(b.shape==out.shape)
    
    b = np.array([0,2,0],dtype=complex).reshape((3,1))
    out = mkl_spsolve(A,b)
    assert_(b.shape==out.shape)


@unittest.skipIf(qset.has_mkl == False, 'MKL extensions not found.')
def test_mkl_spsolve8():
    """
    MKL spsolve : Sparse RHS matrix
    """
    A = sp.csr_matrix([
                [1, 2, 0],
                [0, 3, 0],
                [0, 0, 5]])
    b = sp.csr_matrix([
        [0, 1],
        [1, 0],
        [0, 0]])

    x = mkl_spsolve(A, b)
    ans = np.array([[-0.66666667, 1],
                    [0.33333333, 0],
                    [0, 0]])
    assert_array_almost_equal(x.toarray(), ans)


@unittest.skipIf(qset.has_mkl == False, 'MKL extensions not found.')
def test_mkl_spsolve9():
    """
    MKL spsolve : Hermitian (complex) solver
    """
    A = rand_herm(np.arange(1,11)).data
    x = np.ones(10,dtype=complex)
    b = A.dot(x)
    y = mkl_spsolve(A, b, hermitian=1)
    assert_array_almost_equal(x, y)
    

@unittest.skipIf(qset.has_mkl == False, 'MKL extensions not found.')
def test_mkl_spsolve10():
    """
    MKL spsolve : Hermitian (real) solver
    """
    A = rand_herm(np.arange(1,11)).data
    A = sp.csr_matrix((np.real(A.data), A.indices, A.indptr), dtype=float)
    x = np.ones(10, dtype=float)
    b = A.dot(x)
    y = mkl_spsolve(A, b, hermitian=1)
    assert_array_almost_equal(x, y)


if __name__ == "__main__":
    run_module_suite()
