from . import test_mathematics as testing
import numpy as np
import scipy.linalg
import pytest
from qutip import data
from qutip.core.data import CSR, Dense, Dia
import numbers


class TestOneNorm(testing.UnaryOpMixin):
    def op_numpy(self, matrix):
        return scipy.linalg.norm(matrix, 1)

    specialisations = [
        pytest.param(data.norm.one_csr, CSR, numbers.Number),
        pytest.param(data.norm.one_dia, Dia, numbers.Number),
        pytest.param(data.norm.one_dense, Dense, numbers.Number),
    ]


class TestFrobeniusNorm(testing.UnaryOpMixin):
    def op_numpy(self, matrix):
        return scipy.linalg.norm(matrix, 'fro')

    specialisations = [
        pytest.param(data.norm.frobenius_csr, CSR, numbers.Number),
        pytest.param(data.norm.frobenius_dia, Dia, numbers.Number),
        pytest.param(data.norm.frobenius_dense, Dense, numbers.Number),
    ]


class TestMaxNorm(testing.UnaryOpMixin):
    def op_numpy(self, matrix):
        # There is no scipy-equvalent as sc.linalg.norm(matrix, np.inf)
        # works differently for matrices.
        return np.max(np.abs(matrix))

    specialisations = [
        pytest.param(data.norm.max_csr, CSR, numbers.Number),
        pytest.param(data.norm.max_dia, Dia, numbers.Number),
        pytest.param(data.norm.max_dense, Dense, numbers.Number),
    ]


class TestL2Norm(testing.UnaryOpMixin):
    def op_numpy(self, matrix):
        return scipy.linalg.norm(matrix, 'fro')

    # These shapes correspond to kets or bras
    shapes = [
        (x,) for x in testing.shapes_unary() if (x.values[0][0] == 1
                                                 or x.values[0][1] == 1)
    ]
    # These shapes are everything except for kets and bras
    bad_shapes = [
        (x,) for x in testing.shapes_unary() if not (x.values[0][0] == 1
                                                     or x.values[0][1] == 1)
    ]
    specialisations = [
        pytest.param(data.norm.l2_csr, CSR, numbers.Number),
        pytest.param(data.norm.l2_dia, Dia, numbers.Number),
        pytest.param(data.norm.l2_dense, Dense, numbers.Number),
    ]


class TestTraceNorm(testing.UnaryOpMixin):
    def op_numpy(self, matrix):
        return np.linalg.svd(matrix)[1].sum()

    specialisations = [
        pytest.param(data.norm.trace_csr, CSR, numbers.Number),
        pytest.param(data.norm.trace_dense, Dense, numbers.Number),
    ]
