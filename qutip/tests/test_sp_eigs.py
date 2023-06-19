import scipy
import numpy as np
from numpy.testing import assert_equal, assert_
import unittest

from qutip import num, rand_herm, expect, rand_unitary


def test_SparseHermValsVecs():
    """
    Sparse eigs Hermitian
    """

    # check using number operator
    N = num(10)
    spvals, spvecs = N.eigenstates(sparse=True)
    for k in range(10):
        # check that eigvals are in proper order
        assert_equal(abs(spvals[k] - k) <= 1e-13, True)
        # check that eigenvectors are right and in right order
        assert_equal(abs(expect(N, spvecs[k]) - spvals[k]) < 5e-14, True)

    # check ouput of only a few eigenvals/vecs
    spvals, spvecs = N.eigenstates(sparse=True, eigvals=7)
    assert_equal(len(spvals), 7)
    assert_equal(spvals[0] <= spvals[-1], True)
    for k in range(7):
        assert_equal(abs(spvals[k] - k) < 1e-12, True)

    spvals, spvecs = N.eigenstates(sparse=True, sort='high', eigvals=5)
    assert_equal(len(spvals), 5)
    assert_equal(spvals[0] >= spvals[-1], True)
    vals = np.arange(9, 4, -1)
    for k in range(5):
        # check that eigvals are ordered from high to low
        assert_equal(abs(spvals[k] - vals[k]) < 5e-14, True)
        assert_equal(abs(expect(N, spvecs[k]) - vals[k]) < 1e-14, True)
    # check using random Hermitian
    H = rand_herm(10)
    spvals, spvecs = H.eigenstates(sparse=True)
    # check that sorting is lowest eigval first
    assert_equal(spvals[0] <= spvals[-1], True)
    # check that spvals equal expect vals
    for k in range(10):
        assert_equal(abs(expect(H, spvecs[k]) - spvals[k]) < 5e-14, True)
        # check that ouput is real for Hermitian operator
        assert_equal(np.isreal(spvals[k]), True)


def test_SparseValsVecs():
    """
    Sparse eigs non-Hermitian
    """
    U = rand_unitary(10)
    spvals, spvecs = U.eigenstates(sparse=True)
    assert_equal(np.real(spvals[0]) <= np.real(spvals[-1]), True)
    for k in range(10):
        # check that eigenvectors are right and in right order
        assert_equal(abs(expect(U, spvecs[k]) - spvals[k]) < 5e-14, True)
        assert_equal(np.iscomplex(spvals[k]), True)

    # check sorting
    spvals, spvecs = U.eigenstates(sparse=True, sort='high')
    assert_equal(np.real(spvals[0]) >= np.real(spvals[-1]), True)

    # check for N-1 eigenvals
    U = rand_unitary(10)
    spvals, spvecs = U.eigenstates(sparse=True, eigvals=9)
    assert_equal(len(spvals), 9)


def test_SparseValsOnly():
    """
    Sparse eigvals only Hermitian.
    """
    H = rand_herm(10)
    spvals = H.eigenenergies(sparse=True)
    assert_equal(len(spvals), 10)
    # check that sorting is lowest eigval first
    assert_equal(spvals[0] <= spvals[-1], True)
    # check that spvals equal expect vals
    for k in range(10):
        # check that ouput is real for Hermitian operator
        assert_equal(np.isreal(spvals[k]), True)
    spvals = H.eigenenergies(sparse=True, sort='high')
    # check that sorting is lowest eigval first
    assert_equal(spvals[0] >= spvals[-1], True)
    spvals = H.eigenenergies(sparse=True, sort='high', eigvals=4)
    assert_equal(len(spvals), 4)

    U = rand_unitary(10)
    spvals = U.eigenenergies(sparse=True)
    assert_equal(len(spvals), 10)
    # check that sorting is lowest eigval first
    assert_equal(spvals[0] <= spvals[-1], True)
    # check that spvals equal expect vals
    for k in range(10):
        # check that ouput is real for Hermitian operator
        assert_equal(np.iscomplex(spvals[k]), True)
    spvals = U.eigenenergies(sparse=True, sort='high')
    # check that sorting is lowest eigval first
    assert_equal(spvals[0] >= spvals[-1], True)
    spvals = U.eigenenergies(sparse=True, sort='high', eigvals=4)
    assert_equal(len(spvals), 4)


def test_DenseHermValsVecs():
    """
    Dense eigs Hermitian.
    """
    # check using number operator
    N = num(10)
    spvals, spvecs = N.eigenstates(sparse=False)
    for k in range(10):
        # check that eigvals are in proper order
        assert_equal(abs(spvals[k] - k) < 1e-14, True)
        # check that eigenvectors are right and in right order
        assert_equal(abs(expect(N, spvecs[k]) - spvals[k]) < 5e-14, True)

    # check ouput of only a few eigenvals/vecs
    spvals, spvecs = N.eigenstates(sparse=False, eigvals=7)
    assert_equal(len(spvals), 7)
    assert_equal(spvals[0] <= spvals[-1], True)
    for k in range(7):
        assert_equal(abs(spvals[k] - k) < 1e-14, True)

    spvals, spvecs = N.eigenstates(sparse=False, sort='high', eigvals=5)
    assert_equal(len(spvals), 5)
    assert_equal(spvals[0] >= spvals[-1], True)
    vals = np.arange(9, 4, -1)
    for k in range(5):
        # check that eigvals are ordered from high to low
        assert_equal(abs(spvals[k] - vals[k]) < 5e-14, True)
        assert_equal(abs(expect(N, spvecs[k]) - vals[k]) < 5e-14, True)
    # check using random Hermitian
    H = rand_herm(10)
    spvals, spvecs = H.eigenstates(sparse=False)
    # check that sorting is lowest eigval first
    assert_equal(spvals[0] <= spvals[-1], True)
    # check that spvals equal expect vals
    for k in range(10):
        assert_equal(abs(expect(H, spvecs[k]) - spvals[k]) < 5e-14, True)
        # check that ouput is real for Hermitian operator
        assert_equal(np.isreal(spvals[k]), True)


def test_DenseValsVecs():
    """
    Dense eigs non-Hermitian
    """
    W = rand_herm(10,0.5) + 1j*rand_herm(10,0.5)
    spvals, spvecs = W.eigenstates(sparse=False)
    assert_equal(np.real(spvals[0]) <= np.real(spvals[-1]), True)
    for k in range(10):
        # check that eigenvectors are right and in right order
        assert_equal(abs(expect(W, spvecs[k]) - spvals[k]) < 1e-14, True)
        assert_(np.iscomplex(spvals[k]))

    # check sorting
    spvals, spvecs = W.eigenstates(sparse=False, sort='high')
    assert_equal(np.real(spvals[0]) >= np.real(spvals[-1]), True)

    # check for N-1 eigenvals
    W = rand_unitary(10)
    spvals, spvecs = W.eigenstates(sparse=False, eigvals=9)
    assert_equal(len(spvals), 9)


def test_DenseValsOnly():
    """
    Dense eigvals only Hermitian
    """
    H = rand_herm(10)
    spvals = H.eigenenergies(sparse=False)
    assert_equal(len(spvals), 10)
    # check that sorting is lowest eigval first
    assert_equal(spvals[0] <= spvals[-1], True)
    # check that spvals equal expect vals
    for k in range(10):
        # check that ouput is real for Hermitian operator
        assert_equal(np.isreal(spvals[k]), True)
    spvals = H.eigenenergies(sparse=False, sort='high')
    # check that sorting is lowest eigval first
    assert_equal(spvals[0] >= spvals[-1], True)
    spvals = H.eigenenergies(sparse=False, sort='high', eigvals=4)
    assert_equal(len(spvals), 4)

    U = rand_unitary(10)
    spvals = U.eigenenergies(sparse=False)
    assert_equal(len(spvals), 10)
    # check that sorting is lowest eigval first
    assert_equal(spvals[0] <= spvals[-1], True)
    # check that spvals equal expect vals
    for k in range(10):
        # check that ouput is real for Hermitian operator
        assert_equal(np.iscomplex(spvals[k]), True)
    spvals = U.eigenenergies(sparse=False, sort='high')
    # check that sorting is lowest eigval first
    assert_equal(spvals[0] >= spvals[-1], True)
    spvals = U.eigenenergies(sparse=False, sort='high', eigvals=4)
    assert_equal(len(spvals), 4)


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
    H = rand_herm(dimension, density=1e-2)
    spvals = H.eigenenergies()
    assert abs(H.tr() - spvals.sum()) < tol
