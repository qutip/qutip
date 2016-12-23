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

from qutip.random_objects import (rand_dm, rand_herm,
                                  rand_ket, rand_unitary)
from qutip.cy.spmath import (zcsr_kron, zcsr_transpose, zcsr_adjoint)


def test_csr_kron():
    "spmath: zcsr_kron"
    for kk in range(10):
        ra = np.random.randint(2,100)
        rb = np.random.randint(2,100)
        A = rand_herm(ra,0.5).data
        B = rand_herm(rb,0.5).data
        As = A.tocsr(1)
        Bs = B.tocsr(1)
        C = sp.kron(As,Bs, format='csr')
        D = zcsr_kron(A, B)
        assert_almost_equal(C.data, D.data)
        assert_equal(C.indices, D.indices)
        assert_equal(C.indptr, D.indptr)
        
    for kk in range(10):
        ra = np.random.randint(2,100)
        rb = np.random.randint(2,100)
        A = rand_ket(ra,0.5).data
        B = rand_herm(rb,0.5).data
        As = A.tocsr(1)
        Bs = B.tocsr(1)
        C = sp.kron(As,Bs, format='csr')
        D = zcsr_kron(A, B)
        assert_almost_equal(C.data, D.data)
        assert_equal(C.indices, D.indices)
        assert_equal(C.indptr, D.indptr)
    
    for kk in range(10):
        ra = np.random.randint(2,100)
        rb = np.random.randint(2,100)
        A = rand_dm(ra,0.5).data
        B = rand_herm(rb,0.5).data
        As = A.tocsr(1)
        Bs = B.tocsr(1)
        C = sp.kron(As,Bs, format='csr')
        D = zcsr_kron(A, B)
        assert_almost_equal(C.data, D.data)
        assert_equal(C.indices, D.indices)
        assert_equal(C.indptr, D.indptr)
        
    for kk in range(10):
        ra = np.random.randint(2,100)
        rb = np.random.randint(2,100)
        A = rand_ket(ra,0.5).data
        B = rand_ket(rb,0.5).data
        As = A.tocsr(1)
        Bs = B.tocsr(1)
        C = sp.kron(As,Bs, format='csr')
        D = zcsr_kron(A, B)
        assert_almost_equal(C.data, D.data)
        assert_equal(C.indices, D.indices)
        assert_equal(C.indptr, D.indptr)
  
def test_zcsr_transpose():
    "spmath: zcsr_transpose"
    for k in range(50):
        ra = np.random.randint(2,100)
        A = rand_ket(ra,0.5).data
        B = A.T.tocsr()
        C = A.trans()
        x = np.all(B.data == C.data)
        y = np.all(B.indices == C.indices)
        z = np.all(B.indptr == C.indptr)
        assert_(x*y*z)
    
    for k in range(50):
        ra = np.random.randint(2,100)
        A = rand_herm(5,1.0/ra).data
        B = A.T.tocsr()
        C = A.trans()
        x = np.all(B.data == C.data)
        y = np.all(B.indices == C.indices)
        z = np.all(B.indptr == C.indptr)
        assert_(x*y*z)
        
    for k in range(50):
        ra = np.random.randint(2,100)
        A = rand_dm(ra,1.0/ra).data
        B = A.T.tocsr()
        C = A.trans()
        x = np.all(B.data == C.data)
        y = np.all(B.indices == C.indices)
        z = np.all(B.indptr == C.indptr)
        assert_(x*y*z)
        
    for k in range(50):
        ra = np.random.randint(2,100)
        A = rand_unitary(ra,1.0/ra).data
        B = A.T.tocsr()
        C = A.trans()
        x = np.all(B.data == C.data)
        y = np.all(B.indices == C.indices)
        z = np.all(B.indptr == C.indptr)
        assert_(x*y*z)


def test_zcsr_adjoint():
    "spmath: zcsr_adjoint"
    for k in range(50):
        ra = np.random.randint(2,100)
        A = rand_ket(ra,0.5).data
        B = A.conj().T.tocsr()
        C = A.adjoint()
        x = np.all(B.data == C.data)
        y = np.all(B.indices == C.indices)
        z = np.all(B.indptr == C.indptr)
        assert_(x*y*z)
    
    for k in range(50):
        ra = np.random.randint(2,100)
        A = rand_herm(5,1.0/ra).data
        B = A.conj().T.tocsr()
        C = A.adjoint()
        x = np.all(B.data == C.data)
        y = np.all(B.indices == C.indices)
        z = np.all(B.indptr == C.indptr)
        assert_(x*y*z)    
        
    for k in range(50):
        ra = np.random.randint(2,100)
        A = rand_dm(ra,1.0/ra).data
        B = A.conj().T.tocsr()
        C = A.adjoint()
        x = np.all(B.data == C.data)
        y = np.all(B.indices == C.indices)
        z = np.all(B.indptr == C.indptr)
        assert_(x*y*z)  
        
    for k in range(50):
        ra = np.random.randint(2,100)
        A = rand_unitary(ra,1.0/ra).data
        B = A.conj().T.tocsr()
        C = A.adjoint()
        x = np.all(B.data == C.data)
        y = np.all(B.indices == C.indices)
        z = np.all(B.indptr == C.indptr)
        assert_(x*y*z)

if __name__ == "__main__":
    run_module_suite()