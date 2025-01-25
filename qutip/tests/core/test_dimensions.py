# -*- coding: utf-8 -*-

import pytest
import collections
import qutip
from qutip.core.dimensions import (
    flatten, unflatten, enumerate_flat, deep_remove, deep_map,
    dims_idxs_to_tensor_idxs, dims_to_tensor_shape, dims_to_tensor_perm,
    collapse_dims_super, collapse_dims_oper, einsum, Dimensions,
    to_tensor_rep, from_tensor_rep
)


@pytest.mark.parametrize(["base", "flat"], [
    pytest.param([[[0], 1], 2], [0, 1, 2], id="standard"),
    pytest.param([1, 2, [3, [4]], [5, 6], [7, [[[[[[[8]]]]]]]]],
                 [1, 2, 3, 4, 5, 6, 7, 8], id="standard"),
    pytest.param([1, 2, 3], [1, 2, 3], id="already flat"),
    pytest.param([], [], id="empty list"),
    pytest.param([[], [], [[[], [], []]]], [], id="nested empty lists"),
])
class TestFlattenUnflatten:
    def test_flatten(self, base, flat):
        assert flatten(base) == flat

    def test_unflatten(self, base, flat):
        labels = enumerate_flat(base)
        assert unflatten(flat, labels) == base


@pytest.mark.parametrize(["base", "expected"], [
    pytest.param([1, 2, 3], [0, 1, 2], id="flat"),
    pytest.param([[1], [2], [3]], [[0], [1], [2]], id="nested"),
    pytest.param([[[1], [2, 3]], 4], [[[0], [1, 2]], 3], id="nested"),
    pytest.param([], [], id="empty"),
    pytest.param([[], [], [[[], [], []]]],
                 [[], [], [[[], [], []]]], id="nested empty lists"),
])
def test_enumerate_flat(base, expected):
    assert enumerate_flat(base) == expected


@pytest.mark.parametrize(["base", "to_remove", "expected"], [
    pytest.param([[[0], 1], 2], (1,), [[[0]], 2], id="simple"),
    pytest.param([[[[0, 1, 2]], [3, 4], [5], [6, 7]]], (0, 5),
                 [[[[1, 2]], [3, 4], [], [6, 7]]], id="harder"),
    pytest.param([1, 2, 3], (), [1, 2, 3], id="no-op"),
    pytest.param([], (), [], id="empty"),
])
def test_deep_remove(base, to_remove, expected):
    assert deep_remove(base, *to_remove) == expected


@pytest.mark.parametrize("mapping", [
    pytest.param(lambda x: x*2, id="(x -> 2x)"),
    pytest.param(lambda x: [x], id="(x -> [x])"),
])
@pytest.mark.parametrize("base", [
    pytest.param([[[0], 1], 2], id="standard"),
    pytest.param([1, 2, [3, [4]], [7, [[[[[[[8]]]]]]]]], id="standard"),
    pytest.param([1, 2, 3], id="flat"),
    pytest.param([], id="empty list"),
    pytest.param([[], [], [[[], [], []]]], id="nested empty lists"),
])
def test_deep_map(base, mapping):
    """
    Test the deep mapping.  To simplify generation of edge-cases, this tests
    against an equivalent (but slower) operation of flattening and unflattening
    the list.  We can get false negatives if the `flatten` or `unflatten`
    functions are broken, but other tests should catch those.
    """
    # This function might not need to be public, and consequently might not
    # need to be tested here.
    labels = enumerate_flat(base)
    expected = unflatten([mapping(x) for x in flatten(base)], labels)
    assert deep_map(mapping, base) == expected


_Indices = collections.namedtuple('_Indices', ['base', 'permutation', 'shape'])


@pytest.mark.parametrize("indices", [
    pytest.param(_Indices([[2], [1]], [0, 1], (2, 1)), id="ket preserved"),
    pytest.param(_Indices([[2, 3], [1]], [0, 1, 2], (2, 3, 1)),
                 id="tensor-ket preserved"),
    pytest.param(_Indices([[1], [2]], [0, 1], (1, 2)), id="bra preserved"),
    pytest.param(_Indices([[1], [2, 2]], [0, 1, 2], (1, 2, 2)),
                 id="tensor-bra preserved"),
    pytest.param(_Indices([[2], [3]], [0, 1], (2, 3)), id="oper preserved"),
    pytest.param(_Indices([[2, 3], [1, 2]], [0, 1, 2, 3], (2, 3, 1, 2)),
                 id="tensor-oper preserved"),
    pytest.param(_Indices([[[2, 4], [6, 8]], [[1, 3], [5, 7]]],
                          [2, 3, 0, 1, 6, 7, 4, 5],
                          (6, 8, 2, 4, 5, 7, 1, 3)),
                 id="super-oper"),
    pytest.param(_Indices([[[1], [2, 3]], [[4, 5], [6]]],
                          [2, 0, 1, 4, 5, 3],
                          (2, 3, 1, 6, 4, 5)),
                 id="super-oper-uneven"),
    pytest.param(_Indices([[[2, 4], [6, 8]], [1]],
                          [2, 3, 0, 1, 4], (6, 8, 2, 4, 1)),
                 id="operator-ket"),
    pytest.param(_Indices([[1], [[2, 4], [6, 8]]],
                          [0, 3, 4, 1, 2], (1, 6, 8, 2, 4)),
                 id="operator-bra"),
])
class TestSuperOperatorDimsModification:
    def test_dims_to_tensor_perm(self, indices):
        # This function might not need to be public, and consequently might not
        # need to be tested here.
        assert dims_to_tensor_perm(indices.base) == indices.permutation

    def test_dims_idxs_to_tensor_idxs(self, indices):
        test_indices = list(range(len(flatten(indices.base))))
        assert (dims_idxs_to_tensor_idxs(indices.base, test_indices)
                == indices.permutation)

    def test_dims_to_tensor_shape(self, indices):
        assert dims_to_tensor_shape(indices.base) == indices.shape


@pytest.mark.parametrize("dims", [
    pytest.param([[2, 3], [1]], id="ket"),
    pytest.param([[1, 1], [2, 3]], id="bra"),
    pytest.param([[2, 3], [5, 7]], id="oper"),
    pytest.param([[[2], [3]], [[5], [7]]], id="super-oper"),
    pytest.param([[[2], [3]], [1]], id="oper-ket"),
    pytest.param([[[1, 1], [1, 1]], [[2, 3], [5, 7]]], id="oper-bra"),
    pytest.param([[[2], [3, 5]], [[7], [11]]], id="super-oper-uneven"),
])
class test_to_tensor_rep:
    class mylist(list):
        def at(self, i, j):
            out = self.copy()
            out[i] = j
            return out

    def _build_qobj(self, dims, idx=-1):
        """
        Create an Qobj with given dims which values only increment for the
        `idx` dims.
        """
        if isinstance(dims[0], int):
            ket = qutip.tensor([
                (
                    qutip.Qobj(np.arange(N))
                    if i == idx else (qutip.Qobj(np.ones(N)))
                ) for i, N in enumerate(dims)
            ])
            ket.dims = [ket.dims[0], [1]]
            return ket

        left = self._build_qobj(dims[0], idx)
        right = self._build_qobj(dims[1], idx - len(flatten(dims[0])))
        if left.isoper and left.type != 'scalar':
            left = qutip.operator_to_vector(left)
            left.dims = [left.dims[0], [1]]
        if right.isoper and right.type != 'scalar':
            right = qutip.operator_to_vector(right)
            left.dims = [left.dims[0], [1]]
        return left @ right.dag()

    def test_tensor_shape(self, dims):
        qobj = self._build_qobj(dims)
        array = to_tensor_rep(qobj)
        assert array.shape == tuple(flatten(dims))

    def test_tensor_rep(self, dims):
        N = len(flatten(dims))
        for i in range(N):
            qobj = self._build_qobj(dims, i)
            array = to_tensor_rep(qobj)
            slices = mylist([slice(j) for j in array.shape])
            assert all(
                np.all(array.__getitem__(*slices.at(i, j)) == j)
                for j in range(array.shape[i])
            )

    def from_tensor_rep(self, dims):
        qobj = self._build_qobj(dims)
        array = to_tensor_rep(qobj)
        back = from_tensor_rep(array, dims)
        assert qobj == back


class TestTypeFromDims:
    @pytest.mark.parametrize(["base", "expected"], [
        pytest.param([[2], [2]], 'oper'),
        pytest.param([[2, 3], [2, 3]], 'oper'),
        pytest.param([[2], [3]], 'oper'),
        pytest.param([[2], [1]], 'ket'),
        pytest.param([[1], [2]], 'bra'),
        pytest.param([[[2, 3], [2, 3]], [1]], 'operator-ket'),
        pytest.param([[1], [[2, 3], [2, 3]]], 'operator-bra'),
        pytest.param([[[3], [3]], [[2, 3], [2, 3]]], 'super'),
    ])
    def test_Dimensions_type(self, base, expected):
        assert Dimensions(base).type == expected


class TestCollapseDims:
    @pytest.mark.parametrize(["base", "expected"], [
        pytest.param([[1], [3]], [[1], [3]], id="ket trivial"),
        pytest.param([[1, 1], [2, 3]], [[1], [6]], id="ket tensor"),
        pytest.param([[2], [1]], [[2], [1]], id="bra trivial"),
        pytest.param([[2, 3], [1, 1]], [[6], [1]], id="bra tensor"),
        pytest.param([[5], [5]], [[5], [5]], id="oper trivial"),
        pytest.param([[2, 3], [2, 3]], [[6], [6]], id="oper tensor"),
    ])
    def test_oper(self, base, expected):
        assert collapse_dims_oper(base) == expected

    @pytest.mark.parametrize(["base", "expected"], [
        pytest.param([[[1]], [[2, 3], [2, 3]]],
                     [[[1]], [[6], [6]]], id="operator-ket"),
        pytest.param([[[2, 3], [2, 3]], [[1]]],
                     [[[6], [6]], [[1]]], id="operator-bra"),
        pytest.param([[[2, 3], [2, 3]], [[2, 3], [2, 3]]],
                     [[[6], [6]], [[6], [6]]], id="super"),
    ])
    def test_super(self, base, expected):
        assert collapse_dims_super(base) == expected


@pytest.mark.parametrize("dims_list", [
    pytest.param([0], id="zero"),
    pytest.param([], id="empty"),
    pytest.param([1, [2]], id="mixed depth"),
    pytest.param([[2], [3], [4]], id="bay type"),
])
def test_bad_dims(dims_list):
    with pytest.raises(ValueError):
        Dimensions([dims_list, [1]])


@pytest.mark.parametrize("space_l", [[1], [2], [2, 3]])
@pytest.mark.parametrize("space_m", [[1], [2], [2, 3]])
@pytest.mark.parametrize("space_r", [[1], [2], [2, 3]])
def test_dims_matmul(space_l, space_m, space_r):
    dims_l = Dimensions([space_l, space_m])
    dims_r = Dimensions([space_m, space_r])
    assert dims_l @ dims_r == Dimensions([space_l, space_r])


def test_dims_matmul_bad():
    dims_l = Dimensions([[1], [3]])
    dims_r = Dimensions([[2], [2]])
    with pytest.raises(TypeError):
        dims_l @ dims_r


def test_dims_comparison():
    assert Dimensions([[1], [2]]) == Dimensions([[1], [2]])
    assert not Dimensions([[1], [2]]) != Dimensions([[1], [2]])
    assert Dimensions([[1], [2]]) != Dimensions([[2], [1]])
    assert not Dimensions([[1], [2]]) == Dimensions([[2], [1]])
    assert Dimensions([[1], [2]])[1] == Dimensions([[1], [2]])[1]
    assert Dimensions([[1], [2]])[0] != Dimensions([[1], [2]])[1]
    assert not Dimensions([[1], [2]])[1] != Dimensions([[1], [2]])[1]
    assert not Dimensions([[1], [2]])[0] != Dimensions([[1], [2]])[0]


@pytest.mark.parametrize(["subscripts", "operands", "expected"], [
    pytest.param("ii", [qutip.sigmaz()], 0),
    pytest.param("ij", [qutip.sigmax()], qutip.sigmax()),
    pytest.param("ij->ji", [qutip.sigmay()], qutip.sigmay().trans()),
    pytest.param("ij,ji", [qutip.sigmaz(), qutip.sigmaz()], 2),
    pytest.param("ijij", [qutip.tensor(qutip.thermal_dm(2,1), qutip.thermal_dm(2,1))], 1),
    pytest.param("ikjl,jm->ikml", [qutip.tensor(qutip.sigmaz(), qutip.sigmaz()),
                             qutip.sigmaz()], qutip.tensor(qutip.qeye(2), qutip.sigmaz())),
    pytest.param("ijkl->kjil", [qutip.tensor(qutip.sigmam(), qutip.sigmaz())], qutip.tensor(qutip.sigmap(), qutip.sigmaz()))
])
def test_einsum(subscripts, operands, expected):
    assert einsum(subscripts, *operands) == expected
