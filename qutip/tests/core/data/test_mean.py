import pytest
import numpy as np
import qutip
import numbers

from qutip.core.data.mean import mean_csr, mean_dia, mean_dense
from qutip.core.data.mean import mean_abs_csr, mean_abs_dia, mean_abs_dense
from qutip.core.data import CSR, Dia, Dense
from . import test_mathematics as testing


class TestMean(testing.UnaryOpMixin):

    def op_numpy(self, matrix):
        atol = qutip.settings.core["atol"]

        # Ignore values close to zero
        mask = ~np.isclose(matrix, 0.0, atol=atol)
        nnz = np.count_nonzero(mask)

        if nnz == 0:
            return 0.0

        return matrix[mask].sum() / nnz

    specialisations = [
        pytest.param(mean_csr, CSR, numbers.Complex),
        pytest.param(mean_dia, Dia, numbers.Complex),
        pytest.param(mean_dense, Dense, numbers.Complex),
    ]
    @pytest.mark.parametrize("op, dtype, return_type", specialisations)
    @pytest.mark.parametrize(["scale", "atol"], [(0.999, 1e-12), (1.0, 1e-12), (1.001, 1e-12)])
    def test_atol_boundary(self, op, dtype, return_type, scale, atol):
        """
        Boundary tests around atol value
        """

        data = np.array([[0.0, atol * scale],
                         [1.0, 2.0]], dtype=complex)

        expected = self.op_numpy(data)

        matrix = qutip.Qobj(data, dtype=dtype).data
        result = op(matrix, atol)

        np.testing.assert_allclose(result, expected, atol=self.atol)
        assert isinstance(result, return_type)


class TestAbsMean(testing.UnaryOpMixin):
    def op_numpy(self, matrix):
        atol = qutip.settings.core["atol"]

        # Ignore values close to zero
        mask = ~np.isclose(matrix, 0.0, atol=atol)
        nnz = np.count_nonzero(mask)

        if nnz == 0:
            return 0.0

        return np.abs(matrix[mask]).sum() / nnz

    specialisations = [
        pytest.param(mean_abs_csr, CSR, numbers.Real),
        pytest.param(mean_abs_dia, Dia, numbers.Real),
        pytest.param(mean_abs_dense, Dense, numbers.Real),
    ]
    @pytest.mark.parametrize("op, dtype, result_type", specialisations)
    @pytest.mark.parametrize(["scale", "atol"], [(0.999, 1e-12), (1.0, 1e-12), (1.001, 1e-12)])
    def test_atol_boundary(self, op, dtype, result_type, scale, atol):
        """
        Boundary tests around atol value
        """

        data = np.array([[0.0, atol * scale],
                         [1.0, 2.0]], dtype=complex)
        expected = self.op_numpy(data)

        matrix = qutip.Qobj(data, dtype=dtype).data
        result = op(matrix, atol)

        np.testing.assert_allclose(result, expected)
        assert isinstance(result, result_type)
