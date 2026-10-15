import pytest
import numpy as np
from qutip import data
from qutip.core.data import CSR, Dense, Dia
from qutip.testing import mixin, reshape_mixin

from .conftest import CORRECT_CASES, WRONG_CASES

mixin.CORRECT_CASES = CORRECT_CASES
mixin.WRONG_CASES = WRONG_CASES


class TestSplitColumns(reshape_mixin.TestSplitColumns):
    specialisations = [
        pytest.param(data.split_columns_csr, CSR, CSR),
        pytest.param(data.split_columns_dia, Dia, Dense),
        pytest.param(data.split_columns_dense, Dense, Dense),
    ]


@pytest.mark.filterwarnings("ignore:Constructing a DIA matrix")
class TestColumnStack(reshape_mixin.TestColumnStack):
    specialisations = [
        pytest.param(data.column_stack_csr, CSR, CSR),
        pytest.param(data.column_stack_dia, Dia, Dia),
        pytest.param(data.column_stack_dense, Dense, Dense),
    ]


class TestColumnUnstack(reshape_mixin.TestColumnUnstack):
    specialisations = [
        pytest.param(data.column_unstack_csr, CSR, CSR),
        pytest.param(data.column_unstack_dia, Dia, Dia),
        pytest.param(data.column_unstack_dense, Dense, Dense),
    ]


class TestReshape(reshape_mixin.TestReshape):
    specialisations = [
        pytest.param(data.reshape_dense, Dense, Dense),
        pytest.param(data.reshape_dia, Dia, Dia),
        pytest.param(data.reshape_csr, CSR, CSR),
    ]
