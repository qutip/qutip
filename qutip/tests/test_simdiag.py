import pytest
import numpy as np
import qutip


@pytest.mark.parametrize('num_mat', [1, 2, 3, 5])
def test_simdiag(num_mat):
    N = 10

    U = qutip.rand_unitary(N)
    cummuting_matrices = [U * qutip.qdiags(np.random.rand(N), 0) * U.dag()
                          for _ in range(num_mat)]
    all_evals, evecs = qutip.simdiag(cummuting_matrices)

    for matrix, evals in zip(cummuting_matrices, all_evals):
        for eval, evec in zip(evals, evecs):
            assert matrix * evec == evec * eval


@pytest.mark.parametrize('num_mat', [1, 2, 3, 5])
def test_simdiag_no_evals(num_mat):
    N = 10

    U = qutip.rand_unitary(N)
    cummuting_matrices = [U * qutip.qdiags(np.random.rand(N), 0) * U.dag()
                          for _ in range(num_mat)]
    evecs = qutip.simdiag(cummuting_matrices, evals=False)

    for matrix in cummuting_matrices:
        for evec in evecs:
            Mvec = matrix * evec
            eval = Mvec.full()[0,0] / evec.full()[0,0]
            assert matrix * evec == evec * eval


def test_simdiag_degen():
    N = 10
    U = qutip.rand_unitary(N)
    matrices = [
        U * qutip.qdiags([0,0,0,1,1,1,2,2,3,4], 0) * U.dag(),
        U * qutip.qdiags([1,3,4,2,4,0,0,3,0,1], 0) * U.dag(),
    ]
    evals, evecs = qutip.simdiag(matrices)

    for eval, evec in zip(evals[0], evecs):
        assert matrix * evec == evec * eval


def test_simdiag_no_input():
    with pytest.raises(ValueError):
        qutip.simdiag([])


@pytest.mark.parametrize(['ops', 'error'], [
    pytest.param([qutip.basis(5,0)], 'square', id="Not square"),
    pytest.param([qutip.qeye(5), qutip.qeye(3)], 'shape', id="shape mismatch"),
    pytest.param([qutip.destroy(5)], 'Hermitian', id="Non Hermitian"),
    pytest.param([qutip.sigmax(), qutip.sigmay()], 'commut', id="Not commuting"),
])
def test_simdiag_errors(ops, error):
    with pytest.raises(TypeError) as err:
        qutip.simdiag(ops)
    assert error in str(err.value)
