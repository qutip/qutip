from .test_mathematics import UnaryOpMixin
import pytest
import numpy as np
from qutip import data
from qutip.core.data import CSR, Dense, Dia


class TestSplitColumns(UnaryOpMixin):
    def op_numpy(self, matrix):
        return [column[:, np.newaxis] for column in matrix.T]

    specialisations = [
        pytest.param(data.split_columns_csr, CSR, list),
        pytest.param(data.split_columns_dia, Dia, list),
        pytest.param(data.split_columns_dense, Dense, list),
    ]


@pytest.mark.filterwarnings("ignore:Constructing a DIA matrix")
class TestColumnStack(UnaryOpMixin):
    def op_numpy(self, matrix):
        out_shape = (matrix.shape[0]*matrix.shape[1], 1)
        return np.reshape(matrix, out_shape, order='F')

    specialisations = [
        pytest.param(data.column_stack_csr, CSR, CSR),
        pytest.param(data.column_stack_dia, Dia, Dia),
        pytest.param(data.column_stack_dense, Dense, Dense),
    ]


class TestColumnUnstack(UnaryOpMixin):
    def op_numpy(self, matrix, rows):
        out_shape = (rows, matrix.shape[0]*matrix.shape[1]//rows)
        return np.reshape(matrix, out_shape, order='F')

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
        pytest.param(data.column_unstack_dia, Dia, Dia),
        pytest.param(data.column_unstack_dense, Dense, Dense),
    ]

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
        np.testing.assert_allclose(test.to_array(), expected, atol=self.atol,
                                   rtol=self.rtol)

    def test_incorrect_shape_raises(self, op, data_m):
        with pytest.raises(ValueError):
            op(data_m(), 1)  # We set rows to one so that it is always valid

    # `out_type` is included but not used so that
    # `generate_mathematically_correct` can be re-used.
    @pytest.mark.parametrize('rows', [-1, 0, 3], ids=['negative', 'zero',
                                                      'invalid'])
    def test_incorrect_rows_raises(self, op, data_m, out_type, rows):
        with pytest.raises(ValueError):
            op(data_m(), rows)

    def generate_incorrect_rows_raises(self, metafunc):
        self.generate_mathematically_correct(metafunc)


class TestReshape(UnaryOpMixin):
    def op_numpy(self, matrix, rows, columns):
        out_shape = (rows, columns)
        return np.reshape(matrix, out_shape, order='C')

    # All matrices should have the same number of elements in total, so we can
    # use the same (rows, columns) parametrisation for each input.
    shapes = [
        (pytest.param((1, 100), id="bra"), ),
        (pytest.param((100, 1), id="ket"), ),
        (pytest.param((2, 50), id="non_square"), ),
        (pytest.param((10, 10), id="square"), ),
    ]

    specialisations = [
        pytest.param(data.reshape_dense, Dense, Dense),
        pytest.param(data.reshape_dia, Dia, Dia),
        pytest.param(data.reshape_csr, CSR, CSR),
    ]

    # `out_type` is included but not used so that
    # `generate_mathematically_correct` can be re-used.
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
        np.testing.assert_allclose(test.to_array(), expected, atol=self.atol,
                                   rtol=self.rtol)

    @pytest.mark.parametrize('rows, columns', [(-2, -50), (-50, -2), (3, 10)],
                             ids=["negative1", "negative2", "invalid"])
    def test_incorrect_rows_raises(self, op, data_m, out_type, rows, columns):
        with pytest.raises(ValueError):
            op(data_m(), rows, columns)

    def generate_incorrect_rows_raises(self, metafunc):
        self.generate_mathematically_correct(metafunc)
