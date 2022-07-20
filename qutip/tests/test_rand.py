import numpy as np
from qutip import (
    rand_jacobi_rotation,
    rand_herm,
    rand_unitary,
    rand_dm,
    rand_ket,
    rand_stochastic,
    rand_super,
)


def rand_jacobi_rotation():




import qutip.data as _data


def _is_unitary(qobj):
    return qobj * qobj.dag() == qeye(qobj.dims[0])


def _is_herm(qobj):
    return _data.isherm(qobj.data)

def test_rand_herm():


def test_rand_herm_eigen():






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
