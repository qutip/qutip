from .test_mathematics import UnaryOpMixin
import pytest
import numpy as np
from qutip import data
from qutip.core.data import CSR, Dense
from itertools import product


class TestSplitColumns(UnaryOpMixin):
    def op_numpy(self, matrix):
        return [matrix[:, i].reshape((-1, 1)) for i in range(matrix.shape[1])]

    specialisations = [
        pytest.param(data.split_columns_csr, CSR, list),
        pytest.param(data.split_columns_dense, Dense, list),
    ]


class TestColumnStack(UnaryOpMixin):
    # It seem column_stack assumes F ordered array?
    def op_numpy(self, matrix):
        out_shape = (matrix.shape[0]*matrix.shape[1], 1)
        return np.reshape(matrix, newshape=out_shape, order='F')

    specialisations = [
        pytest.param(data.column_stack_csr, CSR, CSR),
        pytest.param(data.column_stack_dense, Dense, Dense),
    ]


class TestColumnUnstack(UnaryOpMixin):
    # It seem column_unstack assumes F ordered array?
    def op_numpy(self, matrix, rows):
        out_shape = (rows, matrix.shape[0]*matrix.shape[1]//rows)
        return np.reshape(matrix, newshape=out_shape, order='F')

    shapes = [
        (pytest.param((10, 1), id="ket"), ),
    ]

    bad_shapes = [
        (pytest.param((1, 10), id="bra"), ),
        (pytest.param((10, 10), id="square"), ),
        (pytest.param((2, 10), id="non_square"), ),
    ]

    specialisations = [
        pytest.param(data.column_unstack_csr, CSR, CSR),
        pytest.param(data.column_unstack_dense, Dense, Dense),
    ]

    # `ColumnUnstack` has an additional scalar parameters. This is why tests
    # are a little bit more involved.
    @pytest.mark.parametrize('rows', [2, 5])
    def test_mathematically_correct(self, op, data_m, rows, out_type):
        """
        Test that the binary operation is correct for all the known type
        specialisations.
        """
        matrix = data_m()
        expected = self.op_numpy(matrix.to_array(), rows)
        test = op(matrix, rows)
        assert isinstance(test, out_type)
        assert test.shape == expected.shape
        np.testing.assert_allclose(test.to_array(), expected, atol=self.tol)

    # These tests will check that ValueError is raised when input matrices have
    # invalid shape.
    def test_incorrect_shape_raises(self, op, data_m):
        with pytest.raises(ValueError):
            op(data_m(), 1)  # We set rows to one so that it is always valid

    # We also test that wrong value for `rows` raises ValueError We will
    # generate tests with valid data_m since we want to make sure that the
    # error is raised due to wrong `rows` value. For this we will make use of
    # the generate_mathematically_correct function. This is why out_type is
    # included although not used at all.
    def test_incorrect_rows_raises(self, op, data_m, out_type):
        with pytest.raises(ValueError):
            op(data_m(), 3)

    # To generate tests we simply call to getenerate_mathematically correct.
    def generate_incorrect_rows_raises(self, metafunc):
        self.generate_mathematically_correct(metafunc)


class TestReshape(UnaryOpMixin):
    def op_numpy(self, matrix, rows, columns):
        # It seem to_array returns F ordered numpy arrays.
        out_shape = (rows, columns)
        return np.reshape(matrix, newshape=out_shape, order='C')

    # I use custom shapes so that they all have 100 elements.
    shapes = [
        (pytest.param((1, 100), id="bra"), ),
        (pytest.param((100, 1), id="ket"), ),
        (pytest.param((2, 50), id="non_square"), ),
        (pytest.param((10, 10), id="square"), ),
    ]

    specialisations = [
        pytest.param(data.reshape_dense, Dense, Dense),
        pytest.param(data.reshape_csr, CSR, CSR),
    ]

    # `reshape` has an additional scalar parameters. This is why tests
    # are a little bit more involved.
    @pytest.mark.parametrize('rows, columns', [(5, 20), (10, 10)])
    def test_mathematically_correct(self, op, data_m, rows, columns, out_type):
        """
        Test that the binary operation is correct for all the known type
        specialisations.
        """
        matrix = data_m()
        expected = self.op_numpy(matrix.to_array(), rows, columns)
        test = op(matrix, rows, columns)
        assert isinstance(test, out_type)
        assert test.shape == expected.shape
        np.testing.assert_allclose(test.to_array(), expected, atol=self.tol)

    # We also test that wrong value for `rows` or `columns` raises ValueError
    # We will generate tests with valid data_m since we want to make sure that
    # the error is raised due to wrong `rows` value. For this we will make use
    # of the generate_mathematically_correct function. This is why out_type is
    # included although not used at all.
    def test_incorrect_rows_raises(self, op, data_m, out_type):
        with pytest.raises(ValueError):
            op(data_m(), 3, 10)

    # To generate tests we simply call to getenerate_mathematically correct.
    def generate_incorrect_rows_raises(self, metafunc):
        self.generate_mathematically_correct(metafunc)
