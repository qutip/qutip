# This file is part of QuTiP.
#
#    QuTiP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    QuTiP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTiP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011 and later, Paul D. Nation & Robert J. Johansson
#
###########################################################################
import numpy as np
from numpy.testing import assert_, run_module_suite, assert_equal
from qutip import *

def _permutateIndexes(array, row_perm, col_perm):
    return array[np.ix_(row_perm, col_perm)]

def test_sparse_symmetric_permute():
    "Sparse: Symmetric Permute"
    A=rand_dm(25,0.5)
    perm=np.random.permutation(25)
    x=sparse_permute(A,perm,perm).toarray()
    y=_permutateIndexes(A.full(), perm, perm)
    assert_equal((x - y).all(), 0)

def test_sparse_nonsymmetric_permute():
    "Sparse: Nonsymmetric Permute"
    A=rand_dm(25,0.5)
    row_perm=np.random.permutation(25)
    col_perm=np.random.permutation(25)
    x=sparse_permute(A,row_perm,col_perm).toarray()
    y=_permutateIndexes(A.full(),row_perm, col_perm)
    assert_equal((x - y).all(), 0)

def test_sparse_symmetric_reverse_permute():
    "Sparse: Symmetric Reverse Permute"
    A=rand_dm(25,0.5)
    perm=np.random.permutation(25)
    x=sparse_permute(A,perm,perm)
    B=sparse_reverse_permute(x,perm,perm)
    assert_equal((A.full() - B.toarray()).all(), 0)

def test_sparse_nonsymmetric_reverse_permute():
    "Sparse: Nonsymmetric Reverse Permute"
    #square array check
    A=rand_dm(25,0.5)
    row_perm=np.random.permutation(25)
    col_perm=np.random.permutation(25)
    x=sparse_permute(A,row_perm,col_perm)
    B=sparse_permute(x,row_perm,col_perm)
    assert_equal((A.full() - B.toarray()).all(), 0)
    #column vector check
    A=basis(25)
    x=sparse_permute(A,row_perm,[0])
    B=sparse_permute(x,row_perm,[0])
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
            [0,0,0,1,0,0,1,0],[0,1,0,0,0,1,0,1]])
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
