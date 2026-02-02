import pytest

from qutip import (
    Qobj, QobjEvo,
    basis, fock_dm, qzero, qzero_like, sigmax, sigmay, spre,
    operator_to_vector, vector_to_operator
)
from qutip.core.dimensions import Dimensions, SumSpace

from qutip.core.direct_sum import (
    direct_sum, direct_sum_sparse, direct_component, set_direct_component
)

from numbers import Number
import numpy as np


def _ramp(t, args):
    return t


@pytest.mark.parametrize(
    ["arguments", "result_type", "result_dims"], [
        pytest.param([basis(2, 0)],
                     "ket",
                     Dimensions([([2],), [1]]),
                     id="single_ket"),
        pytest.param([QobjEvo([[basis(2, 0), _ramp]]), basis(3, 0)],
                     "ket",
                     Dimensions([([2], [3]), [1]]),
                     id="ket_with_evo"),
        pytest.param([1, 2j, 3],
                     "ket",
                     Dimensions([([1], [1], [1]), [1]]),
                     id="scalars"),
        pytest.param([direct_sum([1, basis(2, 0)]), 3],
                     "ket",
                     Dimensions([(([1], [2]), [1]), [1]]),
                     id="nested_ket"),
        pytest.param([basis(2, 0).dag(), basis([3, 4], [0, 0]).dag(), 1],
                     "bra",
                     Dimensions([[1], ([2], [3, 4], [1])]),
                     id="bras"),
        pytest.param([sigmax(), sigmay()],
                     "operator-ket",
                     Dimensions([([[2], [2]], [[2], [2]]), [1]]),
                     id="operators"),
        pytest.param([sigmax(), operator_to_vector(sigmay()).dag()],
                     "operator-bra",
                     Dimensions([[1], ([[2], [2]], [[2], [2]])]),
                     id="operator_bras"),
    ])
@pytest.mark.parametrize("dtype", ["CSR", "Dense"])
def test_linear(arguments, result_type, result_dims, dtype):
    result = direct_sum(arguments, dtype=dtype)

    assert isinstance(result, (Qobj, QobjEvo))
    assert result.type == result_type
    assert result._dims == result_dims
    assert result.shape == result_dims.shape
    if result_type in ["operator-ket", "operator-bra"]:
        assert result.superrep == "super"
    if dtype is not None:
        if isinstance(result, Qobj):
            assert result.dtype.__name__ == dtype
        else:
            assert result(0).dtype.__name__ == dtype

    for i in range(len(arguments)):
        cmp = direct_component(result, i)
        _assert_equal(cmp, arguments[i])
        if result_type in ["ket", "operator-ket"]:
            _assert_equal(direct_component(result, i, 0), arguments[i])
        if result_type in ["bra", "operator-bra"]:
            _assert_equal(direct_component(result, 0, i), arguments[i])

        replacement = Qobj(np.full(cmp.shape, 1), dims=cmp._dims)
        new_sum = set_direct_component(result, replacement, i)
        if isinstance(result, Qobj) or isinstance(cmp, QobjEvo):
            assert isinstance(new_sum, Qobj)
        else:
            assert isinstance(new_sum, QobjEvo)
        _assert_equal(direct_component(new_sum, i), replacement)

        replacement = QobjEvo([[replacement, _ramp]])
        if result_type in ["ket", "operator-ket"]:
            new_sum = set_direct_component(result, replacement, i, 0)
        if result_type in ["bra", "operator-bra"]:
            new_sum = set_direct_component(result, replacement, 0, i)
        assert isinstance(new_sum, QobjEvo)
        _assert_equal(direct_component(new_sum, i), replacement)


@pytest.mark.parametrize(
    ["arguments", "result_type", "result_dims"], [
        pytest.param([[basis(2, 0)], [basis(3, 0)]],
                     "ket",
                     Dimensions([([2], [3]), [1]]),
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
                     Dimensions([(([2], [3]),), ([1], [1])]),
                     id="nested1"),
        pytest.param([[direct_sum([[basis(2, 0), basis(2, 1)]])],
                      [direct_sum([[basis(3, 0), basis(3, 1)]])]],
                     "oper",
                     Dimensions([([2], [3]), (([1], [1]),)]),
                     id="nested2"),
        pytest.param([[direct_sum([[sigmax(), None],
                                   [basis(2, 0).dag(), 10]])],
                      [direct_sum([basis(2, 1).dag(), 0])]],
                     "oper",
                     Dimensions([(([2], [1]), [1]), (([2], [1]),)]),
                     id="nested3"),
    ])
@pytest.mark.parametrize("dtype", ["CSR", "Dense"])
def test_matrix(arguments, result_type, result_dims, dtype):
    result = direct_sum(arguments, dtype=dtype)

    assert isinstance(result, (Qobj, QobjEvo))
    assert result.type == result_type
    assert result._dims == result_dims
    assert result.shape == result_dims.shape
    if result_type == "super":
        assert result.superrep == "super"
    if dtype is not None:
        if isinstance(result, Qobj):
            assert result.dtype.__name__ == dtype
        else:
            assert result(0).dtype.__name__ == dtype

    copy = result.copy()

    for i in range(len(arguments)):
        for j in range(len(arguments[0])):
            cmp = direct_component(result, i, j)
            _assert_equal(cmp, arguments[i][j])

            replacement = Qobj(np.full(cmp.shape, 1), dims=cmp._dims)
            new_sum = set_direct_component(result, replacement, i, j)
            if isinstance(result, Qobj) or isinstance(cmp, QobjEvo):
                assert isinstance(new_sum, Qobj)
            else:
                assert isinstance(new_sum, QobjEvo)
            _assert_equal(direct_component(new_sum, i, j), replacement)

            replacement = QobjEvo([[replacement, _ramp]])
            new_sum = set_direct_component(result, replacement, i, j)
            assert isinstance(new_sum, QobjEvo)
            _assert_equal(direct_component(new_sum, i, j), replacement)

            copy = set_direct_component(copy, None, i, j)

    assert copy.norm() == 0


@pytest.mark.parametrize(
    ["qobj_dict", "result_dims", "result_type"], [
        pytest.param({(0, 0): basis(2, 0)},
                     Dimensions([([2],), [1]]),
                     "ket",
                     id="single_ket"),
        pytest.param({(0, 0): QobjEvo([[basis(2, 0), _ramp]]),
                      (1, 0): basis(3, 0)},
                     Dimensions([([2], [3]), [1]]),
                     "ket",
                     id="ket_with_evo"),
        pytest.param({(0, 0): direct_sum([1, basis(2, 0)]),
                      (1, 0): 3},
                     Dimensions([(([1], [2]), [1]), [1]]),
                     "ket",
                     id="nested_ket"),
        pytest.param({(0, 0): operator_to_vector(sigmax()).dag(),
                      (0, 1): operator_to_vector(sigmay()).dag()},
                     Dimensions([[1], ([[2], [2]], [[2], [2]])]),
                     "operator-bra",
                     id="operator_bras"),
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
        pytest.param({(0, 0): direct_sum([[basis(2, 0), basis(2, 1)]]),
                      (1, 0): direct_sum([[basis(3, 0), basis(3, 1)]])},
                     Dimensions([([2], [3]), (([1], [1]),)]),
                     "oper",
                     id="nested2"),
        pytest.param({},
                     Dimensions([([2], [3]), (([1], [1]),)]),
                     "oper",
                     id="empty"),
    ])
@pytest.mark.parametrize("dtype", ["CSR", "Dense"])
def test_sparse(qobj_dict, result_dims, result_type, dtype):
    result = direct_sum_sparse(qobj_dict, result_dims, dtype=dtype)

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

    height = len(result_dims[0].spaces) if result_dims[1] is SumSpace else 1
    width = len(result_dims[1].spaces) if result_dims[1] is SumSpace else 1
    for row in range(height):
        for col in range(width):
            cmp = direct_component(result, row, col)
            if (row, col) in qobj_dict:
                _assert_equal(cmp, qobj_dict[(row, col)])
            else:
                assert cmp.norm() == 0


def test_slicing_linear():
    kets = [basis(2, 0),
            basis(([2], [2]), [0, [0]]),
            Qobj([2]),
            QobjEvo([[basis(2, 0), _ramp]])]
    sum = direct_sum(kets)

    for start in range(len(kets)):
        for stop in range(start + 1, len(kets) + 1):
            slice = np.s_[start:stop]
            _assert_equal(
                direct_component(sum, slice), direct_sum(kets[slice])
            )
            _assert_equal(
                direct_component(sum, slice, 0), direct_sum(kets[slice])
            )

    mod1 = set_direct_component(
        sum, 2 * direct_component(sum, np.s_[2:4]), np.s_[2:4])
    mod2 = set_direct_component(sum, None, np.s_[0:2])
    _assert_equal(mod1 - sum, mod2)

    with pytest.raises(IndexError):
        direct_component(sum, np.s_[0:0])
    with pytest.raises(IndexError):
        direct_component(sum, np.s_[0:2:2])
    with pytest.raises(IndexError):
        direct_component(sum, np.s_[1:10])
    with pytest.raises(IndexError):
        direct_component(sum, np.s_[-1:3])


def test_slicing_matrix():
    qobjs = [[(3*i + j) * sigmax() for j in range(3)] for i in range(3)]
    sum = direct_sum(qobjs)
    _assert_equal(
        direct_component(sum, 0, np.s_[1:3]), direct_sum([qobjs[0][1:3]])
    )
    _assert_equal(
        direct_component(sum, np.s_[0:3], 1),
        direct_sum([[qobjs[i][1]] for i in range(3)])
    )
    _assert_equal(
        direct_component(sum, np.s_[1:3], np.s_[1:3]),
        direct_sum([qobjs[1][1:3], qobjs[2][1:3]])
    )


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
def test_direct_sum_validation(arguments):
    with pytest.raises(ValueError):
        direct_sum(arguments)


@pytest.mark.parametrize("arguments", [
    pytest.param([
        {(0, 0): basis(2, 0), (0, 1): sigmax(), (1, 0): sigmax()},
        Dimensions([([2], [2]), ([2], [2])]),
    ]),
    pytest.param([
        {(0, 0): spre(sigmax()), (1, 1): qzero(4)},
        Dimensions([([[2], [2]],) * 2, ([[2], [2]],) * 2]),
    ]),
])
def test_direct_sum_sparse_validation(arguments):
    with pytest.raises(ValueError):
        direct_sum_sparse(*arguments)


def test_direct_component_validation():
    linear = direct_sum([basis(2, 0), basis(3, 0)])
    for args in ([], [-1], [2], [0, 0, 0], [0, 1], [-1, 0], [2, 0]):
        with pytest.raises(IndexError):
            direct_component(linear, *args)
            set_direct_component(linear, None, *args)

    matrix = direct_sum([[sigmax(), None], [None, sigmay()]])
    for args in (
        [], [0], [-1], [-1, 0], [0, -1], [2, 0], [0, 2], [2, 2], [0, 0, 0]
    ):
        with pytest.raises(IndexError):
            direct_component(matrix, *args)
            set_direct_component(linear, None, *args)


def test_set_direct_component_validation():
    matrix = direct_sum([[1, 2], [None, 3]])
    with pytest.raises(ValueError):
        set_direct_component(matrix, basis(2, 0), 1, 0)
        set_direct_component(matrix, sigmax(), 0, 0)


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

    assert (
        direct_component(result, 0)
        == operator_to_vector(sigmax() @ rho1 + rho2)
    )
    assert direct_component(result, 1) == Qobj(1)
