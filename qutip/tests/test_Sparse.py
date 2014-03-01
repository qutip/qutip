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
from qutip import *

def _permutateIndexes(array, row_perm, col_perm):
    return array[np.ix_(row_perm, col_perm)]

def test_sparse_symmetric_permute():
    "Sparse: Symmetric Permute"
    #CSR version
    A=rand_dm(25,0.5)
    perm=np.random.permutation(25)
    x=sparse_permute(A,perm,perm).toarray()
    z=_permutateIndexes(A.full(), perm, perm)
    assert_equal((x - z).all(), 0)
    #CSC version
    B=A.data.tocsc()
    y=sparse_permute(B,perm,perm).toarray()
    assert_equal((y - z).all(), 0)

def test_sparse_nonsymmetric_permute():
    "Sparse: Nonsymmetric Permute"
    #CSR version
    A=rand_dm(25,0.5)
    rperm=np.random.permutation(25)
    cperm=np.random.permutation(25)
    x=sparse_permute(A,rperm,cperm).toarray()
    z=_permutateIndexes(A.full(),rperm, cperm)
    assert_equal((x - z).all(), 0)
    #CSC version
    B=A.data.tocsc()
    y=sparse_permute(B,rperm,cperm).toarray()
    assert_equal((y - z).all(), 0)

def test_sparse_symmetric_reverse_permute():
    "Sparse: Symmetric Reverse Permute"
    #CSR version
    A=rand_dm(25,0.5)
    perm=np.random.permutation(25)
    x=sparse_permute(A,perm,perm)
    B=sparse_reverse_permute(x,perm,perm)
    assert_equal((A.full() - B.toarray()).all(), 0)
    #CSC version
    B=A.data.tocsc()
    perm=np.random.permutation(25)
    x=sparse_permute(B,perm,perm)
    B=sparse_reverse_permute(x,perm,perm)
    assert_equal((A.full() - B.toarray()).all(), 0)

def test_sparse_nonsymmetric_reverse_permute():
    "Sparse: Nonsymmetric Reverse Permute"
    #CSR square array check
    A=rand_dm(25,0.5)
    rperm=np.random.permutation(25)
    cperm=np.random.permutation(25)
    x=sparse_permute(A,rperm,cperm)
    B=sparse_reverse_permute(x,rperm,cperm)
    assert_equal((A.full() - B.toarray()).all(), 0)
    #CSC square array check
    A=rand_dm(25,0.5)
    rperm=np.random.permutation(25)
    cperm=np.random.permutation(25)
    B=A.data.tocsc()
    x=sparse_permute(B,rperm,cperm)
    B=sparse_reverse_permute(x,rperm,cperm)
    assert_equal((A.full() - B.toarray()).all(), 0)
    #CSR column vector check
    A=coherent(25,1)
    rperm=np.random.permutation(25)
    x=sparse_permute(A,rperm,[])
    B=sparse_reverse_permute(x,rperm,[])
    assert_equal((A.full() - B.toarray()).all(), 0)
    #CSC column vector check
    A=coherent(25,1)
    rperm=np.random.permutation(25)
    B=A.data.tocsc()
    x=sparse_permute(B,rperm,[])
    B=sparse_reverse_permute(x,rperm,[])
    assert_equal((A.full() - B.toarray()).all(), 0)
    #CSR row vector check
    A=coherent(25,1).dag()
    cperm=np.random.permutation(25)
    x=sparse_permute(A,[],cperm)
    B=sparse_reverse_permute(x,[],cperm)
    assert_equal((A.full() - B.toarray()).all(), 0)
    #CSC row vector check
    A=coherent(25,1).dag()
    cperm=np.random.permutation(25)
    B=A.data.tocsc()
    x=sparse_permute(B,[],cperm)
    B=sparse_reverse_permute(x,[],cperm)
    assert_equal((A.full() - B.toarray()).all(), 0)

def test_sparse_bandwidth():
    "Sparse: Bandwidth"
    #Bandwidth test 1
    A=create(25)+destroy(25)+qeye(25)
    band=sparse_bandwidth(A)
    assert_equal(band[0], 3)
    assert_equal(band[1]==band[2]==1, 1)
    #Bandwidth test 2
    A=array([[1,0,0,0,1,0,0,0],[0,1,1,0,0,1,0,1],[0,1,1,0,1,0,0,0],
            [0,0,0,1,0,0,1,0],[1,0,1,0,1,0,0,0],[0,1,0,0,0,1,0,1],
            [0,0,0,1,0,0,1,0],[0,1,0,0,0,1,0,1]], dtype=np.int32)
    A=sp.csr_matrix(A)
    out1=sparse_bandwidth(A)
    assert_equal(out1[0], 13)
    assert_equal(out1[1]==out1[2]==6, 1)
    #Bandwidth test 3
    perm=symrcm(A)
    B=sparse_permute(A,perm,perm)
    out2=sparse_bandwidth(B)
    assert_equal(out2[0], 5)
    assert_equal(out2[1]==out2[2]==2, 1)
    #Asymmetric bandwidth test
    A=destroy(25)+qeye(25)
    out1=sparse_bandwidth(A)
    assert_equal(out1[0], 2)
    assert_equal(out1[1], 0)
    assert_equal(out1[2], 1)

if __name__ == "__main__":
    run_module_suite()
