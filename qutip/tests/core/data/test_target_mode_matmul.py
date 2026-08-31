import pytest
import numpy as np

from qutip.core.data import target_mode_matmul

from qutip import (
    rand_ket, rand_dm,
    Qobj, local_matmul, expand_operator,
    tensor, sigmax, destroy, gates, sprepost, qeye,
)


@pytest.mark.parametrize("dtype", ["csr", "dense", "dia"])
@pytest.mark.parametrize("trans", [True, False])
@pytest.mark.parametrize("conj", [True, False])
def test_target_mode_matmul_transformed(dtype, trans, conj):
    state = rand_ket([2] * 3, dtype="dense")
    oper = destroy(2, dtype=dtype) * 1j

    out = target_mode_matmul(
        oper.data, state.data, [1], [2, 2, 2], [2, 2, 2],
        dual=False, transpose=trans, conj=conj
    )
    if trans:
        oper = oper.trans()
    if conj:
        oper = oper.conj()
    expected = (qeye(2) & oper & qeye(2)) @ state
    assert out == expected.data


@pytest.mark.parametrize("dtype", ["csr", "dense", "dia"])
@pytest.mark.parametrize("trans", [True, False])
@pytest.mark.parametrize("conj", [True, False])
def test_target_mode_matmul_transformed_dual(dtype, trans, conj):
    state = rand_dm([2] * 3, dtype="dense")
    oper = destroy(2, dtype=dtype) * 1j

    out = target_mode_matmul(
        oper.data, state.data, [1], [2, 2, 2], [2, 2, 2],
        dual=True, transpose=trans, conj=conj
    )
    if trans:
        oper = oper.trans()
    if conj:
        oper = oper.conj()
    expected =  state @ (qeye(2) & oper & qeye(2))
    assert out == expected.data
