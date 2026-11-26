import numpy as np
import pytest

from qutip.core import data


class TestEinsum:
    """
    Test suite for data layer einsum dispatcher.
    """

    specialisations = []

    def make_operand(self, data_type, array):
        """
        Build an operand of ``data_type`` from a dense numpy array.

        The default goes through the registered ``Dense`` conversion, which
        every data type is expected to provide.  Override only for a type
        that cannot be converted from ``Dense``.
        """
        return data.to(data_type, data.Dense(array))

    def pytest_generate_tests(self, metafunc):
        metafunc.parametrize(
            "einsum_func, data_type, out_type", self.specialisations
        )

    @pytest.mark.parametrize(
        [
            "subscripts",
            "shapes",
            "perms",
            "out_perm",
            "out_shape",
            "operands_data",
            "expected_data",
        ],
        [
            pytest.param(
                "ii",
                [(2, 2)], [(0, 1)], (0,), (1, 1),
                [np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)],
                np.array([[2.0]], dtype=complex),
                id="trace_scalar",
            ),
            pytest.param(
                "ij,ji",
                [(2, 2), (2, 2)], [(0, 1), (0, 1)], (0,), (1, 1),
                [
                    np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
                    np.array([[2.0, 0.0], [0.0, 3.0]], dtype=complex),
                ],
                np.array([[5.0]], dtype=complex),
                id="inner_product_scalar",
            ),
            pytest.param(
                "ij,jk->ik",
                [(2, 2), (2, 2)], [(0, 1), (0, 1)], (0, 1), (2, 2),
                [
                    np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
                    np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
                ],
                np.array([[1.0j, 0.0], [0.0, -1.0j]], dtype=complex),
                id="matrix_multiplication",
            ),
            pytest.param(
                "ij->ji",
                [(2, 2)], [(0, 1)], (0, 1), (2, 2),
                [np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)],
                np.array([[0.0, 1.0j], [-1.0j, 0.0]], dtype=complex),
                id="transpose",
            ),
            pytest.param(
                "ijkl->kjil",
                [(3, 2, 2, 2)], [(2, 0, 1, 3)], (1, 2, 0, 3), (4, 6),
                [np.arange(24, dtype=complex).reshape(6, 4)],
                np.array(
                    [
                        [0, 1, 8, 9, 16, 17],
                        [4, 5, 12, 13, 20, 21],
                        [2, 3, 10, 11, 18, 19],
                        [6, 7, 14, 15, 22, 23],
                    ],
                    dtype=complex,
                ),
                id="tensor_permutation",
            ),
        ],
    )
    def test_einsum(
        self,
        einsum_func,
        data_type,
        out_type,
        subscripts,
        shapes,
        perms,
        out_perm,
        out_shape,
        operands_data,
        expected_data,
    ):
        """
        Test the mathematical correctness of the einsum data layer
        specializations.
        """

        operands = [self.make_operand(data_type, op) for op in operands_data]

        result = einsum_func(
            operands[0],
            *operands[1:],
            subscripts=subscripts,
            tensor_shapes=tuple(shapes),
            tensor_perms=tuple(perms),
            out_perm=out_perm,
            out_shape=out_shape,
        )

        assert isinstance(result, out_type)
        assert result.shape == out_shape
        np.testing.assert_allclose(
            result.to_array(), expected_data, atol=1e-12
        )
