import pytest

from qutip import (
    Qobj, QobjEvo,
    basis, fock_dm, qzero_like, sigmax, sigmay, spre,
    operator_to_vector, vector_to_operator
)
from qutip.core.dimensions import Dimensions

from qutip.core.direct_sum import (
    direct_sum, sparse_direct_sum, component, set_component
)

from numbers import Number
import numpy as np


def _ramp(t, args):
    return t


@pytest.mark.parametrize(
    ["arguments", "result_type", "result_dims"], [
        pytest.param([basis(2, 0)],
                     "ket",
                     Dimensions([([2],), ([1],)]),
                     id="single_ket"),
        pytest.param([QobjEvo([[basis(2, 0), _ramp]]), basis(3, 0)],
                     "ket",
                     Dimensions([([2], [3]), ([1],)]),
                     id="ket_with_evo"),
        pytest.param([1, 2j, 3],
                     "ket",
                     Dimensions([([1], [1], [1]), ([1],)]),
                     id="scalars"),
        pytest.param([basis(2, 0).dag(), basis([3, 4], [0, 0]).dag(), 1],
                     "bra",
                     Dimensions([([1],), ([2], [3, 4], [1])]),
                     id="bras"),
        pytest.param([sigmax(), sigmay()],
                     "operator-ket",
                     Dimensions([([[2], [2]], [[2], [2]]), ([1],)]),
                     id="operators"),
        pytest.param([sigmax(), operator_to_vector(sigmay()).dag()],
                     "operator-bra",
                     Dimensions([([1],), ([[2], [2]], [[2], [2]])]),
                     id="operator_bras"),
])
@pytest.mark.parametrize("dtype", [None, "CSR", "Dense"])
def test_linear(arguments, result_type, result_dims, dtype):
    result = direct_sum(arguments, dtype=dtype)

    assert isinstance(result, (Qobj, QobjEvo))
    assert result.type == result_type
    assert result._dims == result_dims
    assert result.shape == result_dims.shape
    if result_type in ["super", "operator-ket", "operator-bra"]:
        assert result.superrep == "super"
    if dtype is not None:
        if isinstance(result, Qobj):
            assert result.dtype.__name__ == dtype
        else:
            assert result(0).dtype.__name__ == dtype

    if len(arguments) == 1:
        return

    for i in range(len(arguments)):
        cmp = component(result, i)
        _assert_equal(cmp, arguments[i])
        if type in ["ket", "operator-ket"]:
            _assert_equal(component(result, 0, i), arguments[i])
        if type in ["bra", "operator-bra"]:
            _assert_equal(component(result, i, 0), arguments[i])

        replacement = Qobj(np.full(cmp.shape, 1), dims=cmp.dims)
        new_sum = set_component(result, replacement, i)
        if isinstance(result, Qobj) or isinstance(cmp, QobjEvo):
            assert isinstance(new_sum, Qobj)
        else:
            assert isinstance(new_sum, QobjEvo)
        _assert_equal(component(new_sum, i), replacement)

        replacement = QobjEvo([[replacement, _ramp]])
        new_sum = set_component(result, replacement, i)
        assert isinstance(new_sum, QobjEvo)
        _assert_equal(component(new_sum, i), replacement)


@pytest.mark.parametrize(
    ["qobj_dict", "result_dims", "result_type"], [
        pytest.param({(0, 0): basis(2, 0)},
                     Dimensions([([2],), ([1],)]),
                     "ket",
                     id="single_ket"),
        pytest.param({(0, 0): QobjEvo([[basis(2, 0), _ramp]]),
                      (1, 0): basis(3, 0)},
                     Dimensions([([2], [3]), ([1],)]),
                     "ket",
                     id="ket_with_evo"),
        pytest.param({(0, 0): 1,
                      (1, 0): 2j,
                      (2, 0): 3},
                     Dimensions([([1], [1], [1]), ([1],)]),
                     "ket",
                     id="scalars"),
        pytest.param({(0, 0): basis(2, 0).dag(),
                      (0, 1): basis([3, 4], [0, 0]).dag(),
                      (0, 2): 1},
                     Dimensions([([1],), ([2], [3, 4], [1])]),
                     "bra",
                     id="bras"),
        pytest.param({(0, 0): operator_to_vector(sigmax()),
                      (1, 0): operator_to_vector(sigmay())},
                     Dimensions([([[2], [2]], [[2], [2]]), ([1],)]),
                     "operator-ket",
                     id="operators"),
        pytest.param({(0, 0): operator_to_vector(sigmax()).dag(),
                      (0, 1): operator_to_vector(sigmay()).dag()},
                     Dimensions([([1],), ([[2], [2]], [[2], [2]])]),
                     "operator-bra",
                     id="operator_bras"),
])
@pytest.mark.parametrize("dtype", [None, "CSR", "Dense"])
def test_linear_sparse(qobj_dict, result_dims, result_type, dtype):
    result = sparse_direct_sum(qobj_dict, result_dims, dtype=dtype)

    assert isinstance(result, (Qobj, QobjEvo))
    assert result.type == result_type
    assert result._dims == result_dims
    assert result.shape == result_dims.shape
    if result_type in ["super", "operator-ket", "operator-bra"]:
        assert result.superrep == "super"
    if dtype is not None:
        if isinstance(result, Qobj):
            assert result.dtype.__name__ == dtype
        else:
            assert result(0).dtype.__name__ == dtype


@pytest.mark.parametrize(
    ["arguments", "result_type", "result_dims"], [
        pytest.param([[basis(2, 0)], [basis(3, 0)]],
                     "ket",
                     Dimensions([([2], [3]), ([1],)]),
                     id="vector"),
        pytest.param([[sigmax(), None], [None, QobjEvo([[sigmay(), _ramp]])]],
                     "oper",
                     Dimensions([([2], [2]), ([2], [2])]),
                     id="diag_operators"),
        pytest.param([[sigmax(), None], [basis(2, 0).dag(), 10]],
                     "oper",
                     Dimensions([([2], [1]), ([2], [1])]),
                     id="mixed_operators"),
        pytest.param([[spre(sigmax()), operator_to_vector(sigmay())],
                      [None, 1j]],
                     "super",
                     Dimensions([([[2], [2]], [1]), ([[2], [2]], [1])]),
                     id="mixed_super"),
        pytest.param([[direct_sum([basis(2, 0), basis(3, 0)]),
                       direct_sum([basis(2, 1), basis(3, 1)])]],
                     "oper",
                     Dimensions([(([2], [3]),), (([1],), ([1],))]),
                     id="nested1"),
        pytest.param([[direct_sum([[sigmax(), None], [basis(2, 0).dag(), 10]])],
                      [direct_sum([basis(2, 1).dag(), 0])]],
                     "oper",
                     Dimensions([(([2], [1]), ([1],)), (([2], [1]),)]),
                     id="nested2"),
])
@pytest.mark.parametrize("dtype", [None, "CSR", "Dense"])
def test_matrix(arguments, result_type, result_dims, dtype):
    result = direct_sum(arguments, dtype=dtype)

    assert isinstance(result, (Qobj, QobjEvo))
    assert result.type == result_type
    assert result._dims == result_dims
    assert result.shape == result_dims.shape
    if result_type in ["super", "operator-ket", "operator-bra"]:
        assert result.superrep == "super"
    if dtype is not None:
        if isinstance(result, Qobj):
            assert result.dtype.__name__ == dtype
        else:
            assert result(0).dtype.__name__ == dtype

    for i in range(len(arguments)):
        for j in range(len(arguments[0])):
            cmp = component(result, i, j)
            _assert_equal(cmp, arguments[i][j])

            replacement = Qobj(np.full(cmp.shape, 1), dims=cmp.dims)
            new_sum = set_component(result, replacement, i, j)
            if isinstance(result, Qobj) or isinstance(cmp, QobjEvo):
                assert isinstance(new_sum, Qobj)
            else:
                assert isinstance(new_sum, QobjEvo)
            _assert_equal(component(new_sum, i, j), replacement)

            replacement = QobjEvo([[replacement, _ramp]])
            new_sum = set_component(result, replacement, i, j)
            assert isinstance(new_sum, QobjEvo)
            _assert_equal(component(new_sum, i, j), replacement)


@pytest.mark.parametrize(
    ["qobj_dict", "result_dims", "result_type"], [
        pytest.param({(0, 0): basis(2, 0),
                      (1, 0): basis(3, 0)},
                     Dimensions([([2], [3]), ([1],)]),
                     "ket",
                     id="vector"),
        pytest.param({(0, 0): sigmax(),
                      (1, 1): QobjEvo([[sigmay(), _ramp]])},
                     Dimensions([([2], [2]), ([2], [2])]),
                     "oper",
                     id="diag_operators"),
        pytest.param({(0, 0): sigmax(),
                      (1, 0): basis(2, 0).dag(),
                      (1, 1): 10},
                     Dimensions([([2], [1]), ([2], [1])]),
                     "oper",
                     id="mixed_operators"),
        pytest.param({(0, 0): spre(sigmax()),
                      (0, 1): operator_to_vector(sigmay()),
                      (1, 1): 1j},
                     Dimensions([([[2], [2]], [1]), ([[2], [2]], [1])]),
                     "super",
                     id="mixed_super"),
        pytest.param({(0, 0): direct_sum([basis(2, 0), basis(3, 0)]),
                      (0, 1): direct_sum([basis(2, 1), basis(3, 1)])},
                     Dimensions([(([2], [3]),), (([1],), ([1],))]),
                     "oper",
                     id="nested1"),
        pytest.param({(0, 0): direct_sum([[sigmax(), None], [basis(2, 0).dag(), 10]]),
                      (1, 0): direct_sum([basis(2, 1).dag(), 0])},
                     Dimensions([(([2], [1]), ([1],)), (([2], [1]),)]),
                     "oper",
                     id="nested2"),
])
@pytest.mark.parametrize("dtype", [None, "CSR", "Dense"])
def test_matrix_sparse(qobj_dict, result_dims, result_type, dtype):
    result = sparse_direct_sum(qobj_dict, result_dims, dtype=dtype)

    assert isinstance(result, (Qobj, QobjEvo))
    assert result.type == result_type
    assert result._dims == result_dims
    assert result.shape == result_dims.shape
    if result_type in ["super", "operator-ket", "operator-bra"]:
        assert result.superrep == "super"
    if dtype is not None:
        if isinstance(result, Qobj):
            assert result.dtype.__name__ == dtype
        else:
            assert result(0).dtype.__name__ == dtype


def _assert_equal(qobj1, qobj2):
    if isinstance(qobj1, QobjEvo) and isinstance(qobj2, QobjEvo):
        # == for QobjEvo tests identity, not equality
        # this should be enough to check equality
        assert qobj1(0) == qobj2(0)
        assert qobj1(1) == qobj2(1)
    elif isinstance(qobj2, Number):
        # if we put in a plain scalar, we will get out a Qobj
        assert qobj1 == Qobj(qobj2)
    elif qobj2 is None:
        assert qobj1 == Qobj(0) or qobj1 == qzero_like(qobj1)
    elif qobj1.type == "operator-ket":
        # accept getting out vectorized operators
        assert qobj1 == qobj2 or vector_to_operator(qobj1) == qobj2
    elif qobj1.type == "operator-bra":
        assert qobj1 == qobj2 or vector_to_operator(qobj1.dag()) == qobj2
    else:
        assert qobj1 == qobj2


@pytest.mark.parametrize("arguments", [
    pytest.param([], id="empty1"),
    pytest.param([[]], id="empty2"),
    pytest.param([basis(2, 0), None], id="none_in_linear"),
    pytest.param([basis(2, 0), basis(2, 0).dag()], id="mixed_types"),
    pytest.param([[sigmax(), None], [None, spre(sigmay())]], id="mixed_super"),
    pytest.param([[sigmax(), None], [sigmay(), None]], id="empty_col"),
    pytest.param([[sigmax(), sigmay()], [None, None]], id="empty_row"),
    pytest.param([[basis(2, 0), sigmax()], [sigmax(), sigmax()]],
                 id="wrong_dims1"),
    pytest.param([[sigmax(), basis(2, 0).dag()], [sigmax(), sigmax()]],
                 id="wrong_dims2"),
    pytest.param([[sigmax(), basis(2, 0)], [3]], id="not_square"),
])
def test_invalid_args(arguments):
    with pytest.raises(ValueError):
        direct_sum(arguments)


def test_sum_times_sum():
    rho1 = fock_dm(2, 0)
    rho2 = fock_dm(2, 1)

    oper = direct_sum([[spre(sigmax()), operator_to_vector(rho2)],
                       [None, 1]])
    vec = direct_sum([rho1, 1])
    result = oper @ vec

    assert isinstance(result, Qobj)
    assert result.type == "operator-ket"
    assert result._dims == vec._dims
    assert result.shape == result._dims.shape
    assert result.superrep == "super"

    assert component(result, 0) == operator_to_vector(sigmax() @ rho1 + rho2)
    assert component(result, 1) == Qobj(1)
