from .mixin import UnaryOpMixin

import pytest
import numpy as np


class TestSplitColumns(UnaryOpMixin):
    def op_numpy(self, matrix):
        return [column[:, np.newaxis] for column in matrix.T]

    def check_result(self, test, expected, out_type):
        for test_, expected_ in zip(test, expected):
            assert isinstance(test_, out_type)
            assert test_.shape == expected_.shape
            np.testing.assert_allclose(
                test_.to_array(), expected_,
                atol=self.atol, rtol=self.rtol
            )


class TestColumnStack(UnaryOpMixin):
    def op_numpy(self, matrix):
        out_shape = (matrix.shape[0]*matrix.shape[1], 1)
        return np.reshape(matrix, out_shape, order='F')


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

    extra_param = {"rows": [2, 5]}
    wrong_extra_param = {"rows": [1]}

    # `out_type` is included but not used so that
    # `generate_mathematically_correct` can be re-used.
    @pytest.mark.parametrize(
        'rows', [-1, 0, 3], ids=['negative', 'zero', 'invalid']
    )
    def test_incorrect_rows_raises(
        self, op, data_m, random_generator, rows
    ):
        with pytest.raises(ValueError):
            op(data_m(random_generator), rows)

    def generate_incorrect_rows_raises(self, metafunc):
        self.generate_exception(metafunc)


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

    @pytest.mark.parametrize('rows, columns', [(5, 20), (10, 10)])
    def test_mathematically_correct(
        self, op, data_m, rows, columns, out_type, random_generator, kw
    ):
        """
        Test that the binary operation is correct for all the known type
        specialisations.
        """
        matrix = data_m(random_generator)
        expected = self.op_numpy(matrix.to_array(), rows, columns)
        test = op(matrix, rows, columns)
        self.check_result(test, expected, out_type)

    @pytest.mark.parametrize('rows, columns', [(-2, -50), (-50, -2), (3, 10)],
                             ids=["negative1", "negative2", "invalid"])
    def test_incorrect_rows_raises(
        self, op, data_m, rows, columns, random_generator
    ):
        with pytest.raises(ValueError):
            op(data_m(random_generator), rows, columns)

    def generate_incorrect_rows_raises(self, metafunc):
        self.generate_exception(metafunc)
