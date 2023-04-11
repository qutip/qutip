import scipy
import numpy as np
import pytest
import qutip


def is_eigen_set(oper, vals, vecs):
    for val, vec in zip(vals, vecs):
        assert abs(vec.norm() - 1) < 1e-13
        assert abs(qutip.expect(oper, vec) - val) < 1e-13


@pytest.mark.parametrize(["sparse", 'dtype'], [
    pytest.param(True, 'csr', id="sparse"),
    pytest.param(False, 'csr', id="sparse2dense"),
    pytest.param(False, 'dense', id="dense"),
])
def test_eigen_known_oper(sparse, dtype):
    N = qutip.num(10, dtype=dtype)
    spvals, spvecs = N.eigenstates(sparse=sparse)
    expected = np.arange(10)
    is_eigen_set(N, spvals, spvecs)
    np.testing.assert_allclose(spvals, expected, atol=1e-13)


@pytest.mark.parametrize(["sparse", 'dtype'], [
    pytest.param(True, 'csr', id="sparse"),
    pytest.param(False, 'csr', id="sparse2dense"),
    pytest.param(False, 'dense', id="dense"),
])
@pytest.mark.parametrize(["rand"], [
    pytest.param(qutip.rand_herm, id="hermitian"),
    pytest.param(qutip.rand_unitary, id="non-hermitian"),
])
@pytest.mark.parametrize("order", ['low', 'high'])
def test_eigen_rand_oper(rand, sparse, dtype, order):
    H = rand(10, dtype=dtype)
    spvals, spvecs = H.eigenstates(sparse=sparse, sort=order)
    sp_energies = H.eigenenergies(sparse=sparse, sort=order)
    if order == 'low':
        assert np.all(np.diff(spvals).real >= 0)
    else:
        assert np.all(np.diff(spvals).real <= 0)
    is_eigen_set(H, spvals, spvecs)
    np.testing.assert_allclose(spvals, sp_energies, atol=5e-15)


@pytest.mark.parametrize(["sparse", 'dtype'], [
    pytest.param(True, 'csr', id="sparse"),
    pytest.param(False, 'csr', id="sparse2dense"),
    pytest.param(False, 'dense', id="dense"),
])
@pytest.mark.parametrize("rand", [
    pytest.param(qutip.rand_herm, id="hermitian"),
    pytest.param(qutip.rand_unitary, id="non-hermitian"),
])
@pytest.mark.parametrize("order", ['low', 'high'])
@pytest.mark.parametrize("N", [1, 5, 8, 9])
def test_FewState(rand, sparse, dtype, order, N):
    H = rand(10, dtype=dtype)
    all_spvals = H.eigenenergies(sparse=sparse, sort=order)
    spvals, spvecs = H.eigenstates(sparse=sparse, sort=order, eigvals=N)
    assert np.allclose(all_spvals[:N], spvals)
    is_eigen_set(H, spvals, spvecs)
    if order == 'low':
        assert np.all(np.diff(spvals).real >= 0)
    else:
        assert np.all(np.diff(spvals).real <= 0)


@pytest.mark.parametrize(["sparse", 'dtype'], [
    pytest.param(True, 'csr', id="sparse"),
    pytest.param(False, 'csr', id="sparse2dense"),
    pytest.param(False, 'dense', id="dense"),
])
@pytest.mark.parametrize(["rand"], [
    pytest.param(qutip.rand_herm, id="hermitian"),
    pytest.param(qutip.rand_unitary, id="non-hermitian"),
])
@pytest.mark.parametrize("order", ['low', 'high'])
@pytest.mark.parametrize("N", [1, 5, 8, 9])
def test_ValsOnly(rand, sparse, dtype, order, N):
    H = rand(10, dtype=dtype)
    all_spvals = H.eigenenergies(sparse=sparse, sort=order)
    spvals = H.eigenenergies(sparse=sparse, sort=order, eigvals=N)
    assert np.allclose(all_spvals[:N], spvals)
    if order == 'low':
        assert np.all(np.diff(spvals).real >= 0)
    else:
        assert np.all(np.diff(spvals).real <= 0)


@pytest.mark.parametrize(["sparse", 'dtype'], [
    pytest.param(True, 'csr', id="sparse"),
    pytest.param(False, 'csr', id="sparse2dense"),
    pytest.param(False, 'dense', id="dense"),
])
def test_eigen_small(sparse, dtype):
    H = (qutip.sigmax() + qutip.sigmaz()).to(dtype)
    all_spvals = H.eigenenergies(sparse=sparse)
    spvals, spvecs = H.eigenstates(sparse=sparse, eigvals=1)
    assert np.abs(all_spvals[0] - spvals[0]) <= 1e-14
    is_eigen_set(H, spvals, spvecs)


def test_BigDenseValsOnly():
    """
    This checks eigenvalue calculation for large dense matrices, which
    historically have had instabilities with certain OS and BLAS combinations
    (see e.g. #1288 and #1495).
    """
    dimension = 2000
    # Allow an average absolute tolerance for each eigenvalue; we expect
    # uncertainty in the sum to add in quadrature.
    tol = 1e-12 * np.sqrt(dimension)
    H = qutip.rand_herm(dimension, density=1e-2)
    spvals = H.eigenenergies()
    assert abs(H.tr() - spvals.sum()) < tol
