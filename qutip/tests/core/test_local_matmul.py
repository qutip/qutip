import pytest
import numpy as np

from qutip import (
    rand_ket, rand_dm,
    Qobj, local_matmul, expand_operator,
    tensor, sigmax, destroy, gates, sprepost, qeye,
)


def test_single_qubit_gate_on_ket():
    state = rand_ket([2, 2, 2])
    op = sigmax()

    extended_op = tensor(qeye(2), sigmax(), qeye(2))
    expected_state = extended_op @ state
    result_state = local_matmul(op, state, 1)

    assert result_state == expected_state


def test_super_qubit_gate_on_dm():
    state = rand_dm([2, 2, 2], dtype="dense")
    op = sigmax()

    extended_op = tensor(qeye(2), sigmax(), qeye(2))
    expected_state = extended_op @ state @  extended_op.dag()
    result_state = local_matmul(sprepost(op, op.dag()), state, modes=1)

    assert result_state == expected_state


def test_two_qubit_gate_on_ket():
    state = rand_ket([2, 2, 2])
    op = gates.cnot()

    extended_op = expand_operator(op, [2, 2, 2], [2, 0])
    expected_state = extended_op @ state
    result_state = local_matmul(op, state, [2, 0])

    assert result_state == expected_state


def test_two_qubit_gate_on_dm():
    state = rand_dm([2, 2, 2], dtype="dense")
    op = gates.cnot()

    extended_op = expand_operator(op, [2, 2, 2], [2, 0])
    expected_state = extended_op @ state @ extended_op.dag()
    result_state = local_matmul(sprepost(op, op.dag()), state, [2, 0])

    assert result_state == expected_state


def _rand_single_hilbert_qobj(shape, _dtype):
    data = (np.random.rand(*shape) + 1j * np.random.rand(*shape))
    if _dtype != "dense":
        # Make the data sparse
        data *= (np.random.rand(*shape) < 0.2)
    return Qobj(data, dtype=_dtype)


def _rand_state(hilbert, square, order):
    dN = 0 if square else 1
    state = tensor([
        _rand_single_hilbert_qobj((N, N + dN), "dense")
        for N in hilbert
    ])
    state.data = state.data.reorder(fortran=order == "F")
    return state


def _rand_oper(hilbert, modes, square, dtype):
    dN = 0 if square else 1
    op = [
        _rand_single_hilbert_qobj(
            (hilbert[mode] + dN, hilbert[mode]), dtype
        )
        for mode in modes
    ]
    return op


@pytest.fixture(params=[
    pytest.param([2], id='single space'),
    pytest.param([2] * 2, id='2-qubits'),
    pytest.param([2] * 6, id='6-qubits'),
    pytest.param([2, 3, 5, 2], id='uneven'),
])
def hilbert(request):
    return request.param


@pytest.fixture(params=[
    pytest.param([2] * 4, id='4-qubits'),
    pytest.param([2] * 6, id='6-qubits'),
    pytest.param([2, 3, 5, 2], id='uneven'),
])
def large_hilbert(request):
    return request.param


@pytest.fixture(params=[
    pytest.param(True, id='Square operator'),
    pytest.param(False, id='Rectangular operator'),
])
def square_operator(request):
    return request.param


@pytest.fixture(params=[
    pytest.param(True, id='Square state'),
    pytest.param(False, id='Rectangular state'),
])
def square_state(request):
    return request.param


@pytest.fixture(params=[
    pytest.param("csr"),
    pytest.param("dense"),
    pytest.param("dia"),
])
def dtype(request):
    return request.param


@pytest.fixture(params=[
    pytest.param("F"),
    pytest.param("C"),
])
def order(request):
    return request.param


def _reference_N_mode(opers, state, modes, dual=False):
    if not dual:
        hilbert = state.dims[0]
    else:
        hilbert = state.dims[1]
    if 0 not in modes:
        out_oper = qeye(hilbert[0])
    else:
        for mode, oper in zip(modes, opers):
            if mode == 0:
                out_oper = oper
                break

    for i in range(1, len(hilbert)):
        if i not in modes:
            out_oper = out_oper & qeye(hilbert[i])
        else:
            for mode, oper in zip(modes, opers):
                if mode == i:
                    out_oper = out_oper & oper
                    break

    if not dual:
        return out_oper @ state
    else:
        return state @ out_oper


def _reference_super(pres, posts, state, modes):
    hilbert_left = state.dims[0]
    hilbert_right = state.dims[1]
    if 0 not in modes:
        pre_oper = qeye(hilbert_left[0])
        post_oper = qeye(hilbert_right[0])
    else:
        for mode, pre, post in zip(modes, pres, posts):
            if mode == 0:
                pre_oper = pre
                post_oper = post
                break

    for i in range(1, len(hilbert_left)):
        if i not in modes:
            pre_oper = pre_oper & qeye(hilbert_left[i])
            post_oper = post_oper & qeye(hilbert_right[i])
        else:
            for mode, pre, post in zip(modes, pres, posts):
                if mode == i:
                    pre_oper = pre_oper & pre
                    post_oper = post_oper & post
                    break

    return pre_oper @ state @ post_oper


def test_single_mode_ket(hilbert, square_operator, dtype):
    mode = 6 % len(hilbert)

    op = _rand_oper(hilbert, [mode], square_operator, dtype)[0]
    state = rand_ket(hilbert)

    expected = _reference_N_mode([op], state, [mode])
    result = local_matmul(op, state, mode)

    assert expected == result


def test_single_mode_dm(hilbert, square_operator, square_state, dtype, order):
    mode = 6 % len(hilbert)

    op = _rand_oper(hilbert, [mode], square_operator, dtype)[0]
    state = _rand_state(hilbert, square_state, order)

    expected = _reference_N_mode([op], state, [mode])
    result = local_matmul(op, state, mode)

    assert expected == result


def test_single_mode_dm_dual(hilbert, square_operator, square_state, dtype, order):
    mode = 6 % len(hilbert)

    op = _rand_oper(hilbert, [mode], square_operator, dtype)[0].trans()
    state = _rand_state(hilbert, square_state, order).trans()

    expected = _reference_N_mode([op], state, [mode], True)
    result = local_matmul(op, state, mode, True)

    assert expected == result


def test_multi_mode_ket(large_hilbert, square_operator, dtype):
    modes = [2, 0, 1]

    op = _rand_oper(large_hilbert, modes, square_operator, dtype)
    state = rand_ket(large_hilbert)

    expected = _reference_N_mode(op, state, modes)
    result = local_matmul(tensor(op), state, modes)

    assert expected == result


def test_multi_mode_dm(large_hilbert, square_operator, square_state, dtype, order):
    modes = [1, 2, 0]

    op = _rand_oper(large_hilbert, modes, square_operator, dtype)
    state = _rand_state(large_hilbert, square_state, order)

    expected = _reference_N_mode(op, state, modes)
    result = local_matmul(tensor(op), state, modes)

    assert expected == result


def test_multi_mode_dm_dual(large_hilbert, square_operator, square_state, dtype, order):
    modes = [2, 1, 0]

    op = [
        oper.trans()
        for oper in _rand_oper(large_hilbert, modes, square_operator, dtype)
    ]
    state = _rand_state(large_hilbert, square_state, order).trans()

    expected = _reference_N_mode(op, state, modes, True)
    result = local_matmul(tensor(op), state, modes, True)

    assert expected == result


def test_super_single(hilbert, square_operator, square_state, dtype, order):
    mode = 6 % len(hilbert)
    if square_state:
        hilbert_right = hilbert
    else:
        hilbert_right = [N + 1 for N in hilbert]

    pre = _rand_oper(hilbert, [mode], square_operator, dtype)[0]
    post = _rand_oper(hilbert_right, [mode], square_operator, dtype)[0].trans()
    state = _rand_state(hilbert, square_state, order)

    expected = _reference_super([pre], [post], state, [mode])
    result = local_matmul(sprepost(pre, post), state, mode)

    assert expected == result


def test_super_multi(large_hilbert, square_operator, square_state, dtype, order):
    modes = [3, 0, 1]
    if square_state:
        hilbert_right = large_hilbert
    else:
        hilbert_right = [N + 1 for N in large_hilbert]

    pre = _rand_oper(large_hilbert, modes, square_operator, dtype)
    post = [
        oper.trans()
        for oper in _rand_oper(hilbert_right, modes, square_operator, dtype)
    ]
    state = _rand_state(large_hilbert, square_state, order)

    expected = _reference_super(pre, post, state, modes)
    sop = sprepost(tensor(pre), tensor(post))
    result = local_matmul(sop, state, modes)

    assert expected == result


def test_input_validation_errors():
    """Tests that the function raises appropriate errors for bad input."""
    ket = rand_ket([2] * 3)

    with pytest.raises(ValueError, match="out of bounds"):
        local_matmul(sigmax(), ket, modes=3)

    with pytest.raises(TypeError, match="dimensions do not match"):
        local_matmul(destroy(3), ket, modes=0)

    with pytest.raises(ValueError, match="Number of modes"):
        local_matmul(gates.cnot(), ket, modes=0)

    with pytest.raises(ValueError, match="out of bounds"):
        local_matmul(sprepost(sigmax(), sigmax()), ket.proj(), modes=3)

    with pytest.raises(TypeError, match="dimensions do not match"):
        local_matmul(sprepost(destroy(2), destroy(3)), ket.proj(), modes=0)

    with pytest.raises(ValueError, match="Number of modes"):
        local_matmul(sprepost(gates.cnot(), gates.cnot()), ket.proj(), modes=0)

    with pytest.raises(TypeError, match="same number of subsystems"):
        local_matmul(sprepost(sigmax(), sigmax()), ket @ rand_ket([2]).dag(), modes=0)
