import pytest
import numbers

from qutip.core.data.mean import (
    mean_csr, mean_dia, mean_dense,
    mean_abs_csr, mean_abs_dia, mean_abs_dense
)
from qutip.core.data import CSR, Dia, Dense
from qutip.testing import stat_mixin


class TestMean(stat_mixin.TestMean):
    specialisations = [
        pytest.param(mean_csr, CSR, numbers.Complex),
        pytest.param(mean_dia, Dia, numbers.Complex),
        pytest.param(mean_dense, Dense, numbers.Complex),
    ]


class TestAbsMean(stat_mixin.TestAbsMean):
    specialisations = [
        pytest.param(mean_abs_csr, CSR, numbers.Real),
        pytest.param(mean_abs_dia, Dia, numbers.Real),
        pytest.param(mean_abs_dense, Dense, numbers.Real),
    ]
