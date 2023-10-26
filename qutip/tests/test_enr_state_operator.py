import pytest
import itertools
import random
import numpy as np
import qutip


def _n_enr_states(dimensions, n_excitations):
    """
    Calculate the total number of distinct ENR states for a given set of
    subspaces.  This method is not intended to be fast or efficient, it's
    intended to be obviously correct for testing purposes.
    """
    count = 0
    for excitations in itertools.product(*map(range, dimensions)):
        count += int(sum(excitations) <= n_excitations)
    return count


@pytest.fixture(params=[
    pytest.param([4], id="single"),
    pytest.param([4]*2, id="tensor-equal-2"),
    pytest.param([4]*3, id="tensor-equal-3"),
    pytest.param([4]*4, id="tensor-equal-4"),
    pytest.param([2, 3, 4], id="tensor-not-equal"),
])
def dimensions(request):
    return request.param


@pytest.fixture(params=[1, 2, 3, 1_000_000])
def n_excitations(request):
    return request.param


class TestOperator:
    def test_no_restrictions(self, dimensions):
        """
        Test that the restricted-excitation operators are equal to the standard
        operators when there aren't any restrictions.
        """
        test_operators = qutip.enr_destroy(dimensions, sum(dimensions))
        a = [qutip.destroy(n) for n in dimensions]
        iden = [qutip.qeye(n) for n in dimensions]
        for i, test in enumerate(test_operators):
            expected = qutip.tensor(iden[:i] + [a[i]] + iden[i+1:])
            assert test == expected
            assert test.dims == [dimensions, dimensions]

    def test_space_size_reduction(self, dimensions, n_excitations):
        test_operators = qutip.enr_destroy(dimensions, n_excitations)
        expected_size = _n_enr_states(dimensions, n_excitations)
        expected_shape = (expected_size, expected_size)
        for test in test_operators:
            assert test.shape == expected_shape
            assert test.dims == [dimensions, dimensions]

    def test_identity(self, dimensions, n_excitations):
        iden = qutip.enr_identity(dimensions, n_excitations)
        expected_size = _n_enr_states(dimensions, n_excitations)
        expected_shape = (expected_size, expected_size)
        assert np.all(iden.diag() == 1)
        assert np.all(iden.full() - np.diag(iden.diag()) == 0)
        assert iden.shape == expected_shape
        assert iden.dims == [dimensions, dimensions]


def test_fock_state(dimensions, n_excitations):
    """
    Test Fock state creation agrees with the number operators implied by the
    existence of the ENR annihiliation operators.
    """
    number = [a.dag()*a for a in qutip.enr_destroy(dimensions, n_excitations)]
    bases = list(qutip.state_number_enumerate(dimensions, n_excitations))
    n_samples = min((len(bases), 5))
    for basis in random.sample(bases, n_samples):
        state = qutip.enr_fock(dimensions, n_excitations, basis)
        for n, x in zip(number, basis):
            assert abs(n.matrix_element(state.dag(), state)) - x < 1e-10


def test_fock_state_error():
    with pytest.raises(ValueError) as e:
        state = qutip.enr_fock([2, 2, 2], 1, [1, 1, 1])
    assert str(e.value).startswith("The state tuple ")


def _reference_dm(dimensions, n_excitations, nbars):
    """
    Get the reference density matrix using `Qobj.eliminate_states` explicitly,
    to compare to the direct ENR construction.
    """
    if np.isscalar(nbars):
        nbars = [nbars] * len(dimensions)
    out = qutip.tensor([qutip.thermal_dm(dimension, nbar)
                        for dimension, nbar in zip(dimensions, nbars)])
    eliminate = [
        i for i, state in enumerate(qutip.state_number_enumerate(dimensions))
        if sum(state) > n_excitations]
    out = out.eliminate_states(eliminate)
    return out / out.tr()


@pytest.mark.parametrize("nbar_type", ["scalar", "vector"])
def test_thermal_dm(dimensions, n_excitations, nbar_type):
    # Ensure that the average number of excitations over all the states is
    # much less than the total number of allowed excitations.
    if nbar_type == "scalar":
        nbars = 0.1 * n_excitations / len(dimensions)
    else:
        nbars = np.random.rand(len(dimensions))
        nbars *= (0.1 * n_excitations) / np.sum(nbars)
    test_dm = qutip.enr_thermal_dm(dimensions, n_excitations, nbars)
    expect_dm = _reference_dm(dimensions, n_excitations, nbars)
    np.testing.assert_allclose(test_dm.full(), expect_dm.full(), atol=1e-12)


def test_large_space():
    # Check for #2249
    qutip.enr_destroy([12]*12, 2)
