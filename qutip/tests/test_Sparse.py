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

def _permutateIndexes(array, perm):
    return array[np.ix_(perm, perm)]

def test_sparse_symmetric_permute():
    "Sparse: Symmetric Permute"
    A=rand_dm(5,0.5)
    perm=np.array([3,4,2,0,1])
    x=sparse_permute(A,perm,perm).toarray()
    y=_permutateIndexes(A.full(),perm)
    assert_equal((x - y).all(), 0)

def test_sparse_symmetric_reverse_permute():
    "Sparse: Symmetric Reverse Permute"
    A=rand_dm(5,0.5)
    perm=np.array([3,4,2,0,1])
    x=sparse_permute(A,perm,perm)
    B=sparse_reverse_permute(x,perm,perm)
    assert_equal((A.full() - B.toarray()).all(), 0)


if __name__ == "__main__":
    run_module_suite()
