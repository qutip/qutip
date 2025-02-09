"""This file provides tests for expect specialisation. For tests at Qobj level
see `qutip/tests/core/test_expect.py`"""

from .test_mathematics import BinaryOpMixin
import pytest
import numpy as np
from qutip import data
from qutip.core.data import CSR, Dense, Dia
from itertools import product


class TestExpect(BinaryOpMixin):
    def op_numpy(self, op, state):
        is_ket = state.shape[1] == 1
        if is_ket:
            return np.conj(state.T)@op@state
        else:
            return np.trace(op@state)

    _dim = 100
    _ket = pytest.param((_dim, 1), id="ket")
    _dm = pytest.param((_dim, _dim), id="dm")
    _op = pytest.param((_dim, _dim), id="op")
    _bra = pytest.param((1, _dim), id="bra")
    _nonsquare = pytest.param((2, _dim), id="nonsquare")
    _not_op = [_bra, _ket, _nonsquare]

    shapes = [
        (_op, _ket),
        (_op, _dm),
    ]
    bad_shapes = list(product(_not_op, [_ket, _dm]))  # Bad op
    bad_shapes += [
        (_op, _nonsquare),
        (_op, _bra),
    ]  # Bad ket/dm

    specialisations = [
        pytest.param(data.expect_csr, CSR, CSR, complex),
        pytest.param(data.expect_dense, Dense, Dense, complex),
        pytest.param(data.expect_csr_dense, CSR, Dense, complex),
        pytest.param(data.expect_dia, Dia, Dia, complex),
        pytest.param(data.expect_dia_dense, Dia, Dense, complex),
        pytest.param(data.expect_data, Dense, CSR, complex),
    ]


class TestExpectSuper(BinaryOpMixin):
    def op_numpy(self, op, state):
        n = np.sqrt(state.shape[0]).astype(int)
        out_shape = (n, n)
        return np.trace(np.reshape(op@state, out_shape))

    _dim = 100
    _super_ket = pytest.param((_dim, 1), id="super_ket")
    _super_op = pytest.param((_dim, _dim), id="super_op")
    _bra = pytest.param((1, _dim), id="row_stacked")
    _nonsquare = pytest.param((2, _dim), id="nonsquare")
    _not_super_ket = [_super_op, _bra, _nonsquare]
    _not_super_op = [_super_ket, _bra, _nonsquare]

    shapes = [(_super_op, _super_ket), ]
    bad_shapes = list(product(_not_super_op, [_super_ket]))  # Bad super op
    bad_shapes += list(product([_super_op], _not_super_ket))  # Bad super ket

    specialisations = [
        pytest.param(data.expect_super_dense, Dense, Dense, complex),
        pytest.param(data.expect_super_csr, CSR, CSR, complex),
        pytest.param(data.expect_super_csr_dense, CSR, Dense, complex),
        pytest.param(data.expect_super_dia, Dia, Dia, complex),
        pytest.param(data.expect_super_dia_dense, Dia, Dense, complex),
        pytest.param(data.expect_super_data, CSR, Dense, complex),
    ]
