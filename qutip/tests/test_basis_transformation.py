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
from numpy.testing import assert_equal, assert_, run_module_suite
import scipy
from qutip import sigmax, sigmay, sigmaz, Qobj, rand_ket, rand_dm, ket2dm, rand_herm
from qutip.sparse import sp_eigs


def test_Transformation1():
    "Transform 2-level to eigenbasis and back"
    H1 = scipy.rand() * sigmax() + scipy.rand() * sigmay() + \
        scipy.rand() * sigmaz()
    evals, ekets = H1.eigenstates()
    Heb = H1.transform(ekets)        # eigenbasis (should be diagonal)
    H2 = Heb.transform(ekets, True)  # back to original basis
    assert_equal((H1 - H2).norm() < 1e-6, True)


def test_Transformation2():
    "Transform 10-level real-values to eigenbasis and back"
    N = 10
    H1 = Qobj((0.5 - scipy.rand(N, N)))
    H1 = H1 + H1.dag()
    evals, ekets = H1.eigenstates()
    Heb = H1.transform(ekets)        # eigenbasis (should be diagonal)
    H2 = Heb.transform(ekets, True)  # back to original basis
    assert_equal((H1 - H2).norm() < 1e-6, True)


def test_Transformation3():
    "Transform 10-level to eigenbasis and back"
    N = 10
    H1 = Qobj((0.5 - scipy.rand(N, N)) + 1j * (0.5 - scipy.rand(N, N)))
    H1 = H1 + H1.dag()
    evals, ekets = H1.eigenstates()
    Heb = H1.transform(ekets)        # eigenbasis (should be diagonal)
    H2 = Heb.transform(ekets, True)  # back to original basis
    assert_equal((H1 - H2).norm() < 1e-6, True)


def test_Transformation4():
    "Transform 10-level imag to eigenbasis and back"
    N = 10
    H1 = Qobj(1j * (0.5 - scipy.rand(N, N)))
    H1 = H1 + H1.dag()
    evals, ekets = H1.eigenstates()
    Heb = H1.transform(ekets)        # eigenbasis (should be diagonal)
    H2 = Heb.transform(ekets, True)  # back to original basis
    assert_equal((H1 - H2).norm() < 1e-6, True)


def test_Transformation5():
    "Consistency between transformations of kets and density matrices"

    N = 4
    psi0 = rand_ket(N)

    # generate a random basis
    evals, rand_basis = rand_dm(N, density=1).eigenstates()

    rho1 = ket2dm(psi0).transform(rand_basis, True)
    rho2 = ket2dm(psi0.transform(rand_basis, True))

    assert_((rho1 - rho2).norm() < 1e-6)


def test_Transformation6():
    "Check diagonalization via eigenbasis transformation"

    cx, cy, cz = np.random.rand(), np.random.rand(), np.random.rand()
    H = cx * sigmax() + cy * sigmay() + cz * sigmaz()
    evals, evecs = H.eigenstates()
    Heb = H.transform(evecs).tidyup()  # Heb should be diagonal
    assert_(abs(Heb.full() - np.diag(Heb.full().diagonal())).max() < 1e-6)


def test_Transformation7():
    "Check Qobj eigs and direct eig solver transformations match"

    N = 10
    H = rand_herm(N)

    # generate a random basis
    rand = rand_dm(N, density=1)
    
    evals, rand_basis = rand.eigenstates()
    evals2, rand_basis2 = sp_eigs(rand.data, isherm=1)
    H1 = H.transform(rand_basis)
    H2 = H.transform(rand_basis2)
    assert_((H1 - H2).norm() < 1e-6)
    
    ket = rand_ket(N)
    K1 = ket.transform(rand_basis)
    K2 = ket.transform(rand_basis2)
    assert_((K1 - K2).norm() < 1e-6)
    
    bra = rand_ket(N).dag()
    B1 = bra.transform(rand_basis)
    B2 = bra.transform(rand_basis2)
    assert_((B1 - B2).norm() < 1e-6)


def test_Transformation8():
    "Check Qobj eigs and direct eig solver reverse transformations match"

    N = 10
    H = rand_herm(N)

    # generate a random basis
    rand = rand_dm(N, density=1)
    
    evals, rand_basis = rand.eigenstates()
    evals2, rand_basis2 = sp_eigs(rand.data, isherm=1)
    
    H1 = H.transform(rand_basis, True)
    H2 = H.transform(rand_basis2, True)
    assert_((H1 - H2).norm() < 1e-6)
    
    ket = rand_ket(N)
    K1 = ket.transform(rand_basis,1)
    K2 = ket.transform(rand_basis2,1)
    assert_((K1 - K2).norm() < 1e-6)
    
    bra = rand_ket(N).dag()
    B1 = bra.transform(rand_basis,1)
    B2 = bra.transform(rand_basis2,1)
    assert_((B1 - B2).norm() < 1e-6)



if __name__ == "__main__":
    run_module_suite()
