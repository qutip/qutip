"""This file provides tests for expect specialisation. For tests at Qobj level
see `qutip/tests/core/test_expect.py`"""

import pytest
import numpy as np
from qutip import data
from qutip.core.data import CSR, Dense, Dia
from qutip.testing import mixin
from .conftest import CORRECT_CASES, WRONG_CASES


mixin.CORRECT_CASES = CORRECT_CASES
mixin.WRONG_CASES = WRONG_CASES


class TestExpect(mixin.TestExpect):
    specialisations = [
        pytest.param(data.expect_csr, CSR, CSR, complex),
        pytest.param(data.expect_dense, Dense, Dense, complex),
        pytest.param(data.expect_csr_dense, CSR, Dense, complex),
        pytest.param(data.expect_dia, Dia, Dia, complex),
        pytest.param(data.expect_dia_dense, Dia, Dense, complex),
        pytest.param(data.expect_data, Dense, CSR, complex),
    ]


class TestExpectSuper(mixin.TestExpectSuper):
    specialisations = [
        pytest.param(data.expect_super_dense, Dense, Dense, complex),
        pytest.param(data.expect_super_csr, CSR, CSR, complex),
        pytest.param(data.expect_super_csr_dense, CSR, Dense, complex),
        pytest.param(data.expect_super_dia, Dia, Dia, complex),
        pytest.param(data.expect_super_dia_dense, Dia, Dense, complex),
        pytest.param(data.expect_super_data, CSR, Dense, complex),
    ]


@pytest.mark.parametrize("op_type", [CSR, Dense, Dia])
@pytest.mark.parametrize("state_type", [CSR, Dense, Dia])
def test_expect_scalar_ket(op_type, state_type):
    # A 1x1 state is treated as a ket, so it must be conjugated:
    # expect = conj(psi) * a * psi, for every combination of data layers.
    psi, a = 2 - 1j, 3 + 2j
    op = data.to(op_type, data.Dense(np.array([[a]])))
    state = data.to(state_type, data.Dense(np.array([[psi]])))
    np.testing.assert_allclose(data.expect(op, state), np.conj(psi) * a * psi)
