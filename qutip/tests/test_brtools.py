import numpy as np
import scipy.linalg
import pytest
import qutip
from qutip.cy.brtools_checks import (
    _test_zheevr, _test_diag_liou_mult, _test_dense_to_eigbasis,
    _test_vec_to_eigbasis, _test_eigvec_to_fockbasis, _test_vector_roundtrip,
    _cop_super_mult, _test_br_term_mult
)


@pytest.mark.parametrize('dimension', list(range(2, 100)))
def test_zheevr(dimension):
    """
    zheevr: store eigenvalues in the passed array, and return the eigenvectors
    of a complex Hermitian matrix.
    """
    H = qutip.rand_herm(dimension, 1/dimension)
    Hf = H.full()
    evals = np.zeros(dimension, dtype=np.float64)
    # This routine modifies its arguments inplace, so we must make a copy.
    evecs = _test_zheevr(Hf.copy(order='F'), evals).T
    # Assert linear independence of all the eigenvectors.
    assert abs(scipy.linalg.det(evecs)) > 1e-12
    for value, vector in zip(evals, evecs):
        # Assert the eigenvector satisfies the eigenvalue equation.
        unit = vector / scipy.linalg.norm(vector)
        test_value = np.conj(unit.T) @ Hf @ unit
        assert abs(test_value.imag) < 1e-12
        assert abs(test_value - value) < 1e-12


@pytest.mark.parametrize("operator", [
        pytest.param(lambda n: qutip.rand_herm(n, 0.5), id='random Hermitian'),
        pytest.param(qutip.destroy, id='annihilation'),
    ])
def test_dense_operator_to_eigbasis(operator):
    "BR Tools : dense operator to eigenbasis"
    dimension = 10
    operator = operator(dimension)
    for _ in range(50):
        H = qutip.rand_herm(dimension, 0.5)
        basis = H.eigenstates()[1]
        target = operator.transform(basis).full()
        _eigenvalues = np.empty((dimension,), dtype=np.float64)
        basis_zheevr = _test_zheevr(H.full('F'), _eigenvalues)
        calculated = _test_dense_to_eigbasis(operator.full('F'), basis_zheevr,
                                             dimension, qutip.settings.atol)
        np.testing.assert_allclose(target, calculated, atol=1e-12)


def test_vec_to_eigbasis():
    "BR Tools : vector to eigenbasis"
    dimension = 10
    for _ in range(50):
        H = qutip.rand_herm(dimension, 0.5)
        basis = H.eigenstates()[1]
        R = qutip.rand_dm(dimension, 0.5)
        target = qutip.mat2vec(R.transform(basis).full()).ravel()
        flat_vector = qutip.mat2vec(R.full()).ravel()
        calculated = _test_vec_to_eigbasis(H.full('F'), flat_vector)
        np.testing.assert_allclose(target, calculated, atol=1e-12)


def test_eigvec_to_fockbasis():
    "BR Tools : eigvector to fockbasis"
    dimension = 10
    for _ in range(50):
        H = qutip.rand_herm(dimension, 0.5)
        basis = H.eigenstates()[1]
        R = qutip.rand_dm(dimension, 0.5)
        target = qutip.mat2vec(R.full()).ravel()
        _eigenvalues = np.empty((dimension,), dtype=np.float64)
        evecs_zheevr = _test_zheevr(H.full('F'), _eigenvalues)
        flat_eigenvectors = qutip.mat2vec(R.transform(basis).full()).ravel()
        calculated = _test_eigvec_to_fockbasis(flat_eigenvectors, evecs_zheevr,
                                               dimension)
        np.testing.assert_allclose(target, calculated, atol=1e-12)


def test_vector_roundtrip():
    "BR Tools : vector roundtrip transform"
    dimension = 10
    for _ in range(50):
        H = qutip.rand_herm(dimension, 0.5).full('F')
        vector = qutip.mat2vec(qutip.rand_dm(dimension, 0.5).full()).ravel()
        np.testing.assert_allclose(vector, _test_vector_roundtrip(H, vector),
                                   atol=1e-12)


@pytest.mark.parametrize('dimension', list(range(2, 100)))
def test_diag_liou_mult(dimension):
    "BR Tools : Diagonal Liouvillian mult"
    H = qutip.rand_dm(dimension, 0.5)
    evals, evecs = H.eigenstates()
    L = qutip.liouvillian(H.transform(evecs))
    coefficients = np.ones((dimension*dimension,), dtype=np.complex128)
    calculated = np.zeros_like(coefficients)
    target = L.data.dot(coefficients)
    _test_diag_liou_mult(evals, coefficients, calculated, dimension)
    np.testing.assert_allclose(target, calculated, atol=1e-9, rtol=1e-6)


def test_cop_super_mult():
    "BR Tools : cop_super_mult"
    dimension = 10
    for _ in range(50):
        H = qutip.rand_herm(dimension, 0.5)
        basis = H.eigenstates()[1]
        a = qutip.destroy(dimension)
        L = qutip.liouvillian(None, [a.transform(basis)])
        vec = np.ones((dimension*dimension,), dtype=np.complex128)
        target = L.data.dot(vec)
        calculated = np.zeros_like(target)
        _eigenvalues = np.empty((dimension,), dtype=np.float64)
        _cop_super_mult(a.full('F'), _test_zheevr(H.full('F'), _eigenvalues),
                        vec, 1, calculated, dimension, qutip.settings.atol)
        np.testing.assert_allclose(target, calculated, atol=1e-12)


@pytest.mark.parametrize("secular",
                         [True, False], ids=["secular", "non-secular"])
def test_br_term_mult(secular):
    "BR Tools : br_term_mult"
    dimension = 10
    time = 1.0
    atol = 1e-12
    for _ in range(10):
        H = qutip.rand_herm(dimension, 0.5)
        basis = H.eigenstates()[1]
        L_diagonal = qutip.liouvillian(H.transform(basis))
        evals = np.empty((dimension,), dtype=np.float64)
        evecs = _test_zheevr(H.full('F'), evals)
        operator = qutip.rand_herm(dimension, 0.5)
        a_ops = [[operator, lambda w: 1.0]]
        vec = np.ones((dimension*dimension,), dtype=np.complex128)
        br_tensor, _ = qutip.bloch_redfield_tensor(H, a_ops,
                                                   use_secular=secular)
        target = (br_tensor - L_diagonal).data.dot(vec)
        calculated = np.zeros_like(target)
        _test_br_term_mult(time, operator.full('F'), evecs, evals, vec,
                           calculated, secular, 0.1, atol)
        np.testing.assert_allclose(target, calculated, atol=atol)
