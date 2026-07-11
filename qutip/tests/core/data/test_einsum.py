import numpy as np
import pytest

from qutip.core import data
from qutip.core.data import Dense, einsum_dense


class TestEinsum:
    """
    Test suite for data layer einsum dispatchers.
    """

    specialisations = [
        pytest.param(einsum_dense, Dense, Dense, id="Dense"),
    ]

    @pytest.mark.parametrize("einsum_func, data_type, out_type",
                             specialisations)
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
                [(2, 2, 2, 2)], [(0, 1, 2, 3)], (1, 0, 2, 3), (4, 4),
                [np.zeros((16,), dtype=complex).reshape(4, 4)],
                np.zeros((16,), dtype=complex).reshape(4, 4),
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

        operands = [data_type(op) for op in operands_data]

        result = einsum_func(
            operands[0],
            subscripts=subscripts,
            rest_operands=tuple(operands[1:]),
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
