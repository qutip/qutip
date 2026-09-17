from .mixin import UnaryOpMixin, shapes_unary

import pytest
import numpy as np
import qutip
import scipy.linalg


class TestMean(UnaryOpMixin):
    def op_numpy(self, matrix):
        atol = qutip.settings.core["atol"]

        # Ignore values close to zero
        mask = ~np.isclose(matrix, 0.0, atol=atol)
        nnz = np.count_nonzero(mask)

        if nnz == 0:
            return 0.0

        return matrix[mask].sum() / nnz

    @pytest.mark.parametrize(["scale", "atol"],
        [(0.999, 1e-12), (1.0, 1e-12), (1.001, 1e-12)]
    )
    def test_atol_boundary(self, op, dtype, return_type, scale, atol):
        """
        Boundary tests around atol value
        """
        data = np.array([[0.0, atol * scale],
                         [1.0, 2.0]], dtype=complex)

        expected = self.op_numpy(data)

        matrix = qutip.data.to(dtype, qutip.data.Dense(data))
        result = op(matrix, atol)

        np.testing.assert_allclose(result, expected, atol=self.atol)
        assert isinstance(result, return_type)

    def generate_atol_boundary(self, metafunc):
        metafunc.parametrize("op, dtype, return_type", self.specialisations)

class TestAbsMean(TestMean):
    def op_numpy(self, matrix):
        atol = qutip.settings.core["atol"]

        # Ignore values close to zero
        mask = ~np.isclose(matrix, 0.0, atol=atol)
        nnz = np.count_nonzero(mask)

        if nnz == 0:
            return 0.0

        return np.abs(matrix[mask]).sum() / nnz


class TestOneNorm(UnaryOpMixin):
    def op_numpy(self, matrix):
        return scipy.linalg.norm(matrix, 1)


class TestFrobeniusNorm(UnaryOpMixin):
    def op_numpy(self, matrix):
        return scipy.linalg.norm(matrix, 'fro')


class TestMaxNorm(UnaryOpMixin):
    def op_numpy(self, matrix):
        # There is no scipy-equvalent as sc.linalg.norm(matrix, np.inf)
        # works differently for matrices.
        return np.max(np.abs(matrix))


class TestL2Norm(UnaryOpMixin):
    def op_numpy(self, matrix):
        return scipy.linalg.norm(matrix, 'fro')

    # These shapes correspond to kets or bras
    shapes = [
        (x,)
        for x in shapes_unary()
        if (x.values[0][0] == 1 or x.values[0][1] == 1)
    ]
    # These shapes are everything except for kets and bras
    bad_shapes = [
        (x,)
        for x in shapes_unary()
        if not (x.values[0][0] == 1 or x.values[0][1] == 1)
    ]


class TestTraceNorm(UnaryOpMixin):
    def op_numpy(self, matrix):
        return np.linalg.svd(matrix)[1].sum()
