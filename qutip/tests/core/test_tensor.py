import numpy as np

from numpy.testing import assert_equal, assert_
import pytest
import functools
import itertools
import qutip
from qutip import (
    Qobj, identity, sigmax, to_super, to_choi, rand_super_bcsz, basis,
    tensor_contract, tensor_swap, num, QobjEvo, destroy, tensor,
    expand_operator
)


def test_tensor_contract_ident():
    qobj = identity([2, 3, 4])
    ans = 3 * identity([2, 4])

    assert_(ans == tensor_contract(qobj, (1, 4)))

    # Now try for superoperators.
    # For now, we just ensure the dims are correct.
    sqobj = to_super(qobj)
    correct_dims = [[[2, 4], [2, 4]], [[2, 4], [2, 4]]]
    assert_equal(correct_dims, tensor_contract(sqobj, (1, 4), (7, 10)).dims)


def case_tensor_contract_other(left, right, pairs,
                               expected_dims, expected_data):
    dat = np.arange(np.prod(left) * np.prod(right))
    dat = dat.reshape((np.prod(left), np.prod(right)))

    qobj = Qobj(dat, dims=[left, right])
    cqobj = tensor_contract(qobj, *pairs)
    assert_equal(cqobj.dims, expected_dims)
    assert_equal(cqobj.full(), expected_data)


def test_tensor_contract_other():
    case_tensor_contract_other(
        [2, 3], [3, 4], [(1, 2)],
        [[2], [4]],
        np.einsum('abbc', np.arange(2 * 3 * 3 * 4).reshape((2, 3, 3, 4)))
    )

    case_tensor_contract_other(
        [2, 3], [4, 3], [(1, 3)],
        [[2], [4]],
        np.einsum('abcb', np.arange(2 * 3 * 3 * 4).reshape((2, 3, 4, 3)))
    )

    case_tensor_contract_other(
        [2, 3], [4, 3], [(1, 3)],
        [[2], [4]],
        np.einsum('abcb', np.arange(2 * 3 * 3 * 4).reshape((2, 3, 4, 3)))
    )

    # Make non-rectangular outputs in a column-/row-symmetric fashion.
    big_dat = np.arange(2 * 3 * 2 * 3 * 2 * 3 * 2 * 3).reshape((2, 3) * 4)

    case_tensor_contract_other(
        [[2, 3], [2, 3]], [[2, 3], [2, 3]], [(0, 2)],
        [[[3], [3]], [[2, 3], [2, 3]]],
        np.einsum('ibidwxyz', big_dat).reshape((3 * 3, 3 * 2 * 3 * 2)))

    case_tensor_contract_other(
        [[2, 3], [2, 3]], [[2, 3], [2, 3]],
        [(0, 2), (5, 7)], [[[3], [3]], [[2], [2]]],
        # We separate einsum into two calls due to a bug internal to
        # einsum.
        np.einsum('ibidwy', np.einsum('abcdwjyj', big_dat)).reshape(3*3, 2*2))

    # Now we try collapsing in a way that's sensitive to column- and row-
    # stacking conventions.
    big_dat = np.arange(2 * 2 * 3 * 3 * 2 * 3 * 2 * 3)
    big_dat = big_dat.reshape((3, 3, 2, 2, 2, 3, 2, 3))
    # Note that the order of [2, 2] and [3, 3] is swapped on the left!
    big_dims = [[[2, 2], [3, 3]], [[2, 3], [2, 3]]]

    # Let's try a weird tensor contraction; this will likely never come up in
    # practice, but it should serve as a good canary for more reasonable
    # contractions.
    case_tensor_contract_other(
        big_dims[0], big_dims[1], [(0, 4)],
        [[[2], [3, 3]], [[3], [2, 3]]],
        np.einsum('abidwxiz', big_dat).reshape((2 * 3 * 3, 3 * 2 * 3)))


def case_tensor_swap(qobj, pairs, expected_dims, expected_data=None):
    sqobj = tensor_swap(qobj, *pairs)

    assert_equal(sqobj.dims, expected_dims)
    assert_equal(sqobj.full(), expected_data.full())


def test_tensor_swap_other():
    dims = (2, 3, 4, 5, 7)
    for dim in dims:
        S = to_super(rand_super_bcsz(dim))
        # Swapping the inner indices on a superoperator should give a Choi
        # matrix.
        J = to_choi(S)
        case_tensor_swap(S, [(1, 2)], [[[dim], [dim]], [[dim], [dim]]], J)


def test_tensor_qobjevo():
    N = 5
    t = 1.5
    left = QobjEvo([num(N),[destroy(N),"t"]])
    right = QobjEvo([identity(2),[sigmax(),"t"]])
    assert tensor(left, right)(t) == tensor(left(t), right(t))
    assert tensor(left, sigmax())(t) == tensor(left(t), sigmax())
    assert tensor(num(N), right)(t) == tensor(num(N), right(t))


def test_tensor_qobjevo_non_square():
    N = 5
    t = 1.5
    left = QobjEvo([basis(N, 0), [basis(N, 1), "t"]])
    right = QobjEvo([basis(2, 0).dag(), [basis(2, 1).dag(), "t"]])
    assert tensor(left, right)(t) == tensor(left(t), right(t))


def test_tensor_qobjevo_multiple():
    N = 5
    t = 1.5
    left = QobjEvo([basis(N, 0), [basis(N, 1), "t"]])
    center = QobjEvo([basis(2, 0).dag(), [basis(2, 1).dag(), "t"]])
    right = QobjEvo([sigmax()])
    as_QobjEvo = tensor(left, center, right)(t)
    as_Qobj = tensor(left(t), center(t), right(t))
    assert as_QobjEvo == as_Qobj


def test_tensor_and():
    N = 5
    t = 1.5
    sx = sigmax()
    evo = QobjEvo([num(N),[destroy(N),"t"]])
    assert tensor([sx, sx, sx]) == sx & sx & sx
    assert tensor(evo, sx)(t) == (evo & sx)(t)
    assert tensor(sx, evo)(t) == (sx & evo)(t)
    assert tensor(evo, evo)(t) == (evo & evo)(t)


def _permutation_id(permutation):
    return str(len(permutation)) + "-" + "".join(map(str, permutation))


def _tensor_with_entanglement(all_qubits, entangled, entangled_locations):
    """
    Create a full tensor product when a subspace component is already in an
    entangled state.  The locations in `all_qubits` which are the entangled
    points in the output are ignored and can take any value.

    For example,
        _tensor_with_entanglement([|a>, |b>, |c>, |d>], (|00> + |11>), [0, 2])
    should product a tensor product like (|0b0d> + |1b1d>), i.e. qubits 0 and 2
    in the final output are entangled, but the others are still separable.

    Parameters:
        all_qubits: list of kets --
            A list of separable parts to tensor together.  States that are in
            the locations referred to by `entangled_locations` are completely
            ignored.
        entangled: tensor-product ket -- the full entangled subspace
        entangled_locations: list of int --
            The locations that the qubits in the entangled subspace should be
            in in the final tensor-product space.
    """
    n_entangled = len(entangled.dims[0])
    n_separable = len(all_qubits) - n_entangled
    separable = all_qubits.copy()
    # Remove in reverse order so subsequent deletion locations don't change.
    for location in sorted(entangled_locations, reverse=True):
        del separable[location]
    # Can't separate out entangled states to pass to tensor in the right places
    # immediately, so tensor in at one end and then permute into place.
    out = qutip.tensor(*separable, entangled)
    permutation = list(range(n_separable))
    current_locations = range(n_separable, n_separable + n_entangled)
    # Sort to prevert later insertions changing previous locations.
    insertions = sorted(zip(entangled_locations, current_locations),
                        key=lambda x: x[0])
    for out_location, current_location in insertions:
        permutation.insert(out_location, current_location)
    return out.permute(permutation)


def _apply_permutation(permutation):
    """
    Permute the given permutation into the order denoted by its elements, i.e.
        out[0] = permutation[permutation[0]]
        out[1] = permutation[permutation[1]]
        ...

    This function is its own inverse.
    """
    out = [None] * len(permutation)
    for value, location in enumerate(permutation):
        out[location] = value
    return out


class Test_expand_operator:
    @pytest.mark.parametrize(
        'permutation',
        tuple(itertools.chain(*[
            itertools.permutations(range(k)) for k in [2, 3, 4]
        ])),
        ids=_permutation_id)
    def test_permutation_without_expansion(self, permutation):
        base = qutip.tensor([qutip.rand_unitary(2) for _ in permutation])
        test = expand_operator(base, [2] * len(permutation), permutation)
        expected = base.permute(_apply_permutation(permutation))
        np.testing.assert_allclose(test.full(), expected.full(), atol=1e-15)

    @pytest.mark.parametrize('n_targets', range(1, 5))
    def test_general_qubit_expansion(self, n_targets):
        # Test all permutations with the given number of targets.
        n_qubits = 5
        operation = qutip.rand_unitary([2]*n_targets)
        for targets in itertools.permutations(range(n_qubits), n_targets):
            expected = _tensor_with_entanglement([qutip.qeye(2)] * n_qubits,
                                                 operation, targets)
            test = expand_operator(operation, [2] * 5, targets)
            np.testing.assert_allclose(test.full(), expected.full(),
                                       atol=1e-15)

    def test_cnot_explicit(self):
        test = expand_operator(qutip.gates.cnot(), [2]*3, [2, 0]).full()
        expected = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 1, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 0, 0, 0]])
        np.testing.assert_allclose(test, expected, atol=1e-15)

    @pytest.mark.parametrize('dimensions', [
        pytest.param([3, 4, 5], id="standard"),
        pytest.param([3, 3, 4, 4, 2], id="standard"),
        pytest.param([1, 2, 3], id="1D space"),
    ])
    def test_non_qubit_systems(self, dimensions):
        n_qubits = len(dimensions)
        for targets in itertools.permutations(range(n_qubits), 2):
            operators = [qutip.rand_unitary(dimension) if n in targets
                         else qutip.qeye(dimension)
                         for n, dimension in enumerate(dimensions)]
            expected = qutip.tensor(*operators)
            base_test = qutip.tensor(*[operators[x] for x in targets])
            test = expand_operator(base_test, dims=dimensions, targets=targets)
            assert test.dims == expected.dims
            np.testing.assert_allclose(test.full(), expected.full())

    def test_dtype(self):
        expanded_qobj = expand_operator(
            qutip.gates.cnot(), dims=[2, 2, 2], targets=[0, 1], dtype="csr"
        ).data
        assert isinstance(expanded_qobj, qutip.data.CSR)
        expanded_qobj = expand_operator(
            qutip.gates.cnot(), dims=[2, 2, 2], targets=[0, 1], dtype="dense"
        ).data
        assert isinstance(expanded_qobj, qutip.data.Dense)
