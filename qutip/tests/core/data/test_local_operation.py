from qutip.core.data import (
    one_mode_matmul_data_dense,
    one_mode_matmul_dual_dense_data
)
import qutip
import pytest
import numpy as np

@pytest.mark.parametrize(['hilbert', "mode"], [
    ([2], 0),
    ([2, 2], 0),
    ([2, 2], 1),
    ([2, 3, 4], 0),
    ([2, 3, 4], 1),
    ([2, 3, 4], 2),
])
@pytest.mark.parametrize('type', ["ket", "dm"])
@pytest.mark.parametrize('dtype', ["dense", "dia", "csr"])
def test_local_matmul_left(hilbert, mode, type, dtype):
    if type == "ket":
        state = qutip.rand_ket(hilbert, density=0.8, dtype="Dense")
    else:
        state = qutip.rand_dm(hilbert, density=0.8, dtype="Dense")

    N = hilbert[mode]
    oper = qutip.Qobj(
        np.random.rand(N, N) + 1j * np.random.rand(N, N),
        dtype=dtype
    )

    expected = qutip.expand_operator(oper, hilbert, mode) @ state

    result = qutip.Qobj(
        one_mode_matmul_data_dense(oper.data, state.data, hilbert, mode),
        dims=state.dims,
    )
    assert expected == result


@pytest.mark.parametrize(['hilbert', "mode"], [
    ([2], 0),
    ([2, 2], 0),
    ([2, 2], 1),
    ([2, 3, 4], 0),
    ([2, 3, 4], 1),
    ([2, 3, 4], 2),
])
@pytest.mark.parametrize('type', ["ket", "dm"])
@pytest.mark.parametrize('dtype', ["dense", "dia", "csr"])
def test_local_matmul_right(hilbert, mode, type, dtype):
    if type == "ket":
        state = qutip.rand_ket(hilbert, density=0.8, dtype="Dense").dag()
    else:
        state = qutip.rand_dm(hilbert, density=0.8, dtype="Dense")

    N = hilbert[mode]
    oper = qutip.Qobj(
        np.random.rand(N, N) + 1j * np.random.rand(N, N),
        dtype=dtype
    )

    expected = state @ qutip.expand_operator(oper, hilbert, mode)

    result = qutip.Qobj(
        one_mode_matmul_dual_dense_data(state.data, oper.data, hilbert, mode),
        dims=state.dims,
    )
    assert expected == result
