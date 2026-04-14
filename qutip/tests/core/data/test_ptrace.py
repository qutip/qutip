from . import test_mathematics as testing
import numpy as np
import scipy.linalg
import pytest
from qutip import data
from qutip.core.data import CSR, Dense, Dia


class TestPtrace(testing.UnaryOpMixin):
    def op_numpy(self, matrix, dims, sel):
        sel.sort()
        ndims = len(dims)
        dkeep = [dims[x] for x in sel]
        qtrace = list(set(range(ndims)) - set(sel))
        dtrace = [dims[x] for x in qtrace]

        matrix = matrix.reshape(dims + dims)
        matrix = matrix.transpose(
            qtrace + [ndims + i for i in qtrace] + sel + [ndims + i for i in sel]
        )
        matrix = matrix.reshape(
            [
                np.prod(dtrace, dtype=int),
                np.prod(dtrace, dtype=int),
                np.prod(dkeep, dtype=int),
                np.prod(dkeep, dtype=int),
            ]
        )
        return np.trace(matrix)

    # Custom shapes to have also custom dims and sel arguments.
    # These values should not be changed.
    dims = [2] * 7
    shapes = [(pytest.param((np.prod(dims), np.prod(dims))),)]
    bad_shapes = testing.shapes_not_square(np.prod(dims))

    specialisations = [
        pytest.param(data.ptrace_csr, CSR, CSR),
        pytest.param(data.ptrace_csr_dense, CSR, Dense),
        pytest.param(data.ptrace_dense, Dense, Dense),
        pytest.param(data.ptrace_dia, Dia, Dia),
    ]

    @pytest.mark.parametrize(
        "sel",
        [[0], [0, 3, 6], [0, 6, 3], list(range(len(dims))), []],
        ids=[
            "keep_one",
            "keep_multiple_sorted",
            "keep_multiple_unsorted",
            "trace_none",
            "trace_all",
        ],
    )
    def test_mathematically_correct(self, op, data_m, out_type, sel):
        """
        Test that the unary operation is mathematically correct for all the
        known type specialisations.
        """
        matrix = data_m()
        expected = self.op_numpy(matrix.to_array(), self.dims, sel)
        test = op(matrix, self.dims, sel)
        assert isinstance(test, out_type)
        assert test.shape == expected.shape
        np.testing.assert_allclose(
            test.to_array(), expected, atol=self.atol, rtol=self.rtol
        )

    def test_incorrect_shape_raises(self, op, data_m):
        """
        Test that the operation produces a suitable error if the shape of the
        operand is not square.
        """
        with pytest.raises(ValueError):
            op(data_m(), self.dims, sel=[0, 1])

    # `out_type` is included but not used so that
    # `generate_mathematically_correct` can be re-used.
    @pytest.mark.parametrize(
        "dims",
        [[2], [0], [-2, -2] + [2] * 5, [1.2, 2.2, 3.3]],
        ids=[
            "dims_different_to_shape",
            "dims_0",
            "dims_prod_is_shape_but_negative",
            "dims_is_not_int",
        ],
    )
    def test_incorrect_dims_raises(self, op, data_m, out_type, dims):
        with pytest.raises(ValueError):
            op(data_m(), dims, sel=[0, 1])

    def generate_incorrect_dims_raises(self, metafunc):
        self.generate_mathematically_correct(metafunc)

    @pytest.mark.parametrize(
        "sel",
        [[2, 10], [-1, 2]],
        ids=[
            "sel_value_larger_than_dims",
            "sel_value_negative",
        ],
    )
    def test_incorrect_sel_raises(self, op, data_m, out_type, sel):
        with pytest.raises(IndexError):
            op(data_m(), dims=self.dims, sel=sel)

    def generate_incorrect_sel_raises(self, metafunc):
        self.generate_mathematically_correct(metafunc)
