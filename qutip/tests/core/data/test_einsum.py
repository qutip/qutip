import pytest

from qutip.core.data import Dense, einsum_dense
from qutip.testing import einsum_mixin


class TestEinsum(einsum_mixin.TestEinsum):
    """
    Test suite for data layer einsum dispatchers.
    """

    specialisations = [
        pytest.param(einsum_dense, Dense, Dense, id="Dense"),
    ]
