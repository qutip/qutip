import pytest
import numpy as np
import qutip


@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize('num_mat', [1, 2, 3, 5])
def test_simdiag(num_mat):
    N = 10

    U = qutip.rand_unitary(N)
    commuting_matrices = [U * qutip.qdiags(np.random.rand(N), 0) * U.dag()
                          for _ in range(num_mat)]
    all_evals, evecs = qutip.simdiag(commuting_matrices)

    for matrix, evals in zip(commuting_matrices, all_evals):
        for eval, evec in zip(evals, evecs):
            assert matrix * evec == evec * eval


@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize('num_mat', [1, 2, 3, 5])
def test_simdiag_no_evals(num_mat):
    N = 10

    U = qutip.rand_unitary(N)
    commuting_matrices = [U * qutip.qdiags(np.random.rand(N), 0) * U.dag()
                          for _ in range(num_mat)]
    evecs = qutip.simdiag(commuting_matrices, evals=False)

    for matrix in commuting_matrices:
        for evec in evecs:
            Mvec = matrix * evec
            eval = Mvec.norm() / evec.norm()
            assert matrix * evec == evec * eval


@pytest.mark.flaky(reruns=2)
def test_simdiag_degen():
    N = 10
    U = qutip.rand_unitary(N)
    commuting_matrices = [
        U * qutip.qdiags([0, 0, 0, 1, 1, 1, 2, 2, 3, 4], 0) * U.dag(),
        U * qutip.qdiags([0, 0, 0, 1, 2, 2, 2, 2, 2, 2], 0) * U.dag(),
        U * qutip.qdiags([0, 0, 2, 1, 1, 2, 2, 3, 3, 4], 0) * U.dag(),
    ]
    all_evals, evecs = qutip.simdiag(commuting_matrices)

    for matrix, evals in zip(commuting_matrices, all_evals):
        for eval, evec in zip(evals, evecs):
            np.testing.assert_allclose(
                (matrix * evec).full(),
                (evec * eval).full(),
                atol=3e-14
            )

@pytest.mark.flaky(reruns=2)
@pytest.mark.repeat(2)
def test_simdiag_degen_large():
    N = 20
    U = qutip.rand_unitary(N)
    commuting_matrices = [
        U * qutip.qdiags(np.random.randint(0, 3, N), 0) * U.dag()
        for _ in range(5)
    ]
    all_evals, evecs = qutip.simdiag(commuting_matrices, tol=1e-12)

    for matrix, evals in zip(commuting_matrices, all_evals):
        for eval, evec in zip(evals, evecs):
            np.testing.assert_allclose(
                (matrix * evec).full(),
                (evec * eval).full(),
                atol=1e-13
            )

def test_simdiag_orthonormal_eigenvectors():

    # Special matrix that used to be problematic (see Issue #2268)
    a = np.array([[1,  0, 1, -1, 0],
                  [0,  4, 0,  0, 1],
                  [1,  0, 4,  1, 0],
                  [-1, 0, 1,  4, 0],
                  [0,  1, 0,  0, 4]])

    _, evecs = qutip.simdiag([qutip.Qobj(a), qutip.qeye(5)])
    evecs = np.array([evec.full() for evec in evecs]).squeeze()

    # Check that eigenvectors form an othonormal basis
    # (<=> matrix of eigenvectors is unitary)
    np.testing.assert_allclose(
        evecs@evecs.conj().T,
        np.eye(len(evecs)),
        atol=1e-13
    )


def test_large_degenerate():
    N = sum(
        qutip.expand_operator((1 + qutip.sigmaz()) / 2, [2] * 6, i)
        for i in range(6)
    )
    vals, vecs = qutip.simdiag([N, N])
    assert len(np.unique(np.round(vals, 14))) == 7


def test_simdiag_no_input():
    with pytest.raises(ValueError):
        qutip.simdiag([])


@pytest.mark.parametrize(['ops', 'error'], [
    pytest.param([qutip.basis(5, 0)], 'square', id="Not square"),
    pytest.param([qutip.qeye(5), qutip.qeye(3)], 'shape', id="shape mismatch"),
    pytest.param([qutip.destroy(5)], 'Hermitian', id="Non Hermitian"),
    pytest.param([qutip.sigmax(), qutip.sigmay()], 'commute',
                 id="Not commuting"),
])
def test_simdiag_errors(ops, error):
    with pytest.raises(TypeError) as err:
        qutip.simdiag(ops)
    assert error in str(err.value)
