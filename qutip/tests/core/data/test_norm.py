import pytest
import numbers

from .conftest import CORRECT_CASES, WRONG_CASES

from qutip import data
from qutip.core.data import CSR, Dense, Dia
from qutip.testing import mixin, stat_mixin



mixin.CORRECT_CASES = CORRECT_CASES
mixin.WRONG_CASES = WRONG_CASES


class TestOneNorm(stat_mixin.TestOneNorm):
    specialisations = [
        pytest.param(data.norm.one_csr, CSR, numbers.Number),
        pytest.param(data.norm.one_dia, Dia, numbers.Number),
        pytest.param(data.norm.one_dense, Dense, numbers.Number),
    ]


class TestFrobeniusNorm(stat_mixin.TestFrobeniusNorm):
    specialisations = [
        pytest.param(data.norm.frobenius_csr, CSR, numbers.Number),
        pytest.param(data.norm.frobenius_dia, Dia, numbers.Number),
        pytest.param(data.norm.frobenius_dense, Dense, numbers.Number),
    ]


class TestMaxNorm(stat_mixin.TestMaxNorm):
    specialisations = [
        pytest.param(data.norm.max_csr, CSR, numbers.Number),
        pytest.param(data.norm.max_dia, Dia, numbers.Number),
        pytest.param(data.norm.max_dense, Dense, numbers.Number),
    ]


class TestL2Norm(stat_mixin.TestL2Norm):
    specialisations = [
        pytest.param(data.norm.l2_csr, CSR, numbers.Number),
        pytest.param(data.norm.l2_dia, Dia, numbers.Number),
        pytest.param(data.norm.l2_dense, Dense, numbers.Number),
    ]


class TestTraceNorm(stat_mixin.TestTraceNorm):
    specialisations = [
        pytest.param(data.norm.trace_csr, CSR, numbers.Number),
        pytest.param(data.norm.trace_dense, Dense, numbers.Number),
    ]
