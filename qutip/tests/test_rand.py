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
from qutip import (
    rand_unitary, qeye, rand_herm, rand_dm, rand_ket, rand_stochastic,
)
from qutip.core.data import eigs_csr


class TestRand:
    """
    A test class for the built-in random quantum object generators.
    """

    def testRandUnitary(self):
        "random Unitary"

        U = [rand_unitary(5) for k in range(5)]
        for u in U:
            assert u * u.dag() == qeye(5)

    def testRandUnitarySeed(self):
        "random Unitary with seed"

        seed = 12345
        U0 = rand_unitary(5, seed=seed)
        U1 = rand_unitary(5, seed=None)
        U2 = rand_unitary(5, seed=seed)
        assert U0 != U1
        assert U0 == U2

    def testRandherm(self):
        "random hermitian"

        H = [rand_herm(5) for k in range(5)]
        for h in H:
            assert h.isherm

    def testRandhermSeed(self):
        "random hermitian with seed"

        seed = 12345
        U0 = rand_herm(5, seed=seed)
        U1 = rand_herm(5, seed=None)
        U2 = rand_herm(5, seed=seed)
        assert U0 != U1
        assert U0 == U2

    def testRandhermPosDef(self):
        "Random: Hermitian - Positive semi-def"

        H = [rand_herm(5,pos_def=1) for k in range(5)]
        for h in H:
            assert not any(eigs_csr(h.data, h.isherm, vecs=False)) < 0

    def testRandhermEigs(self):
        "Random: Hermitian - Eigs given"

        H = [rand_herm([1,2,3,4,5],0.5) for k in range(5)]
        for h in H:
            eigs = eigs_csr(h.data, h.isherm, vecs=False)
            assert np.abs(np.sum(eigs)-15.0) < 1e-12

    def testRanddm(self):
        "random density matrix"

        R = [rand_dm(5) for k in range(5)]
        for r in R:
            assert r.tr() - 1.0 < 1e-15
            # verify all eigvals are >=0
            assert not any(eigs_csr(r.data, r.isherm, vecs=False)) < 0
            # verify hermitian
            assert r.isherm

    def testRandDmSeed(self):
        "random density matrix with seed"

        seed = 12345
        U0 = rand_dm(5, seed=seed)
        U1 = rand_dm(5, seed=None)
        U2 = rand_dm(5, seed=seed)
        assert U0 != U1
        assert U0 == U2

    def testRanddmEigs(self):
        "Random: Density matrix - Eigs given"
        R = []
        for k in range(5):
            eigs = np.random.random(5)
            eigs /= np.sum(eigs)
            R += [rand_dm(eigs)]
        for r in R:
            assert r.tr() - 1.0 < 1e-15
            # verify all eigvals are >=0
            assert not any(eigs_csr(r.data, r.isherm, vecs=False)) < 0
            # verify hermitian
            assert r.isherm

    def testRandket(self):
        "random ket"
        P = [rand_ket(5) for k in range(5)]
        for p in P:
            assert p.type == 'ket'

    def testRandketSeed(self):
        "random ket with seed"

        seed = 12345
        U0 = rand_ket(5, seed=seed)
        U1 = rand_ket(5, seed=None)
        U2 = rand_ket(5, seed=seed)
        assert U0 != U1
        assert U0 == U2

    def testRandStochasticLeft(self):
        'Random: Stochastic - left'
        Q = [rand_stochastic(10) for k in range(5)]
        for i in range(5):
            A = Q[i].full()
            np.testing.assert_allclose(np.sum(A, axis=0), 1, atol=1e-15)

    def testRandStochasticLeftSeed(self):
        "Random: Stochastic - left with seed"

        seed = 12345
        U0 = rand_stochastic(10, seed=seed)
        U1 = rand_stochastic(10, seed=None)
        U2 = rand_stochastic(10, seed=seed)
        assert U0 != U1
        assert U0 == U2

    def testRandStochasticRight(self):
        'Random: Stochastic - right'
        Q = [rand_stochastic(10, kind='right') for k in range(5)]
        for i in range(5):
            A = Q[i].full()
            np.testing.assert_allclose(np.sum(A, axis=1), 1, atol=1e-15)

    def testRandStochasticRightSeed(self):
        "Random: Stochastic - right with seed"

        seed = 12345
        U0 = rand_stochastic(10, kind='right', seed=seed)
        U1 = rand_stochastic(10, kind='right', seed=None)
        U2 = rand_stochastic(10, kind='right', seed=seed)
        assert U0 != U1
        assert U0 == U2
