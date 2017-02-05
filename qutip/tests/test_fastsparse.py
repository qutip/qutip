# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, The QuTiP Project.
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
from numpy.testing import (run_module_suite, assert_,
                        assert_equal, assert_almost_equal)
import scipy.sparse as sp
from qutip.random_objects import rand_herm
from qutip.fastsparse import fast_csr_matrix


def test_fast_sparse_basic():
    "fastsparse: fast_csr_matrix operations"
    H = rand_herm(5).data
    assert_(isinstance(H, fast_csr_matrix))
    Hadd = H + H
    assert_(isinstance(Hadd, fast_csr_matrix))
    Hsub = H + H
    assert_(isinstance(Hsub, fast_csr_matrix))
    Hmult = H * H
    assert_(isinstance(Hmult, fast_csr_matrix))
    Hcopy = H.copy()
    assert_(isinstance(Hcopy, fast_csr_matrix))
    assert_(isinstance(-1*H, fast_csr_matrix))
    assert(isinstance(H/2., fast_csr_matrix))
    
    G = sp.csr_matrix((H.data, H.indices,H.indptr), 
            copy=True, shape=H.shape)
    assert_(not isinstance(G, fast_csr_matrix))
    Hadd = H + G
    assert_(not isinstance(Hadd, fast_csr_matrix))
    Hadd = G + H
    assert_(not isinstance(Hadd, fast_csr_matrix))
    Hsub = H - G
    assert_(not isinstance(Hsub, fast_csr_matrix))
    Hsub = G - H
    assert_(not isinstance(Hsub, fast_csr_matrix))
    Hmult = H*G
    assert_(not isinstance(Hmult, fast_csr_matrix))
    Hmult = G*H
    assert_(not isinstance(Hmult, fast_csr_matrix))

    
def test_fast_sparse_trans():
    "fastsparse: transpose operations"
    H = rand_herm(5).data
    assert_(isinstance(H, fast_csr_matrix))
    assert_(isinstance(H.T, fast_csr_matrix))
    assert_(isinstance(H.trans(), fast_csr_matrix))
    assert_(isinstance(H.transpose(), fast_csr_matrix))

    
def test_fast_sparse_adjoint():
    "fastsparse: adjoint operations"
    H = rand_herm(5).data
    assert_(isinstance(H, fast_csr_matrix))
    assert_(isinstance(H.H, fast_csr_matrix))
    assert_(isinstance(H.getH(), fast_csr_matrix))
    assert_(isinstance(H.adjoint(), fast_csr_matrix))


if __name__ == "__main__":
    run_module_suite()