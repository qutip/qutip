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
            assert test.data == expected.data
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
    assert str(e.value).startswith("state tuple ")


def _reference_dm(dimensions, n_excitations, nbars):
    """
    Get the reference density matrix explicitly, to compare to the direct ENR
    construction.
    """
    if np.isscalar(nbars):
        nbars = [nbars] * len(dimensions)
    keep = np.array([
        i for i, state in enumerate(qutip.state_number_enumerate(dimensions))
        if sum(state) <= n_excitations
    ])
    dm = qutip.tensor([qutip.thermal_dm(dimension, nbar)
                       for dimension, nbar in zip(dimensions, nbars)])
    out = qutip.Qobj(dm.full()[keep[:, None], keep[None, :]])
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


def test_mesolve_ENR():
    # Ensure ENR states work with mesolve
    # We compare the output to an exact truncation of the
    # single-excitation Jaynes-Cummings model
    eps = 2 * np.pi
    omega_c = 2 * np.pi
    g = 0.1 * omega_c
    gam = 0.01 * omega_c
    tlist = np.linspace(0, 20, 100)
    N_cut = 2

    sz = qutip.sigmaz() & qutip.qeye(N_cut)
    sm = qutip.destroy(2).dag() & qutip.qeye(N_cut)
    a = qutip.qeye(2) & qutip.destroy(N_cut)
    H_JC = (0.5 * eps * sz + omega_c * a.dag()*a +
            g * (a * sm.dag() + a.dag() * sm))
    psi0 = qutip.basis(2, 0) & qutip.basis(N_cut, 0)
    c_ops = [np.sqrt(gam) * a]

    result_JC = qutip.mesolve(H_JC, psi0, tlist, c_ops, e_ops=[sz])

    N_exc = 1
    dims = [2, N_cut]
    d = qutip.enr_destroy(dims, N_exc)
    sz = 2*d[0].dag()*d[0]-1
    b = d[0]
    a = d[1]
    psi0 = qutip.enr_fock(dims, N_exc, [1, 0])
    H_enr = (eps * b.dag()*b + omega_c * a.dag() * a +
             g * (b.dag() * a + a.dag() * b))
    c_ops = [np.sqrt(gam) * a]

    result_enr = qutip.mesolve(H_enr, psi0, tlist, c_ops, e_ops=[sz])

    np.testing.assert_allclose(result_JC.expect[0],
                               result_enr.expect[0], atol=1e-2)


def test_steadystate_ENR():
    # Ensure ENR states work with steadystate functions
    # We compare the output to an exact truncation of the
    # single-excitation Jaynes-Cummings model
    eps = 2 * np.pi
    omega_c = 2 * np.pi
    g = 0.1 * omega_c
    gam = 0.01 * omega_c
    N_cut = 2

    sz = qutip.sigmaz() & qutip.qeye(N_cut)
    sm = qutip.destroy(2).dag() & qutip.qeye(N_cut)
    a = qutip.qeye(2) & qutip.destroy(N_cut)
    H_JC = (0.5 * eps * sz + omega_c * a.dag()*a +
            g * (a * sm.dag() + a.dag() * sm))
    c_ops = [np.sqrt(gam) * a]

    result_JC = qutip.steadystate(H_JC, c_ops)
    exp_sz_JC = qutip.expect(sz, result_JC)

    N_exc = 1
    dims = [2, N_cut]
    d = qutip.enr_destroy(dims, N_exc)
    sz = 2*d[0].dag()*d[0]-1
    b = d[0]
    a = d[1]
    H_enr = (eps * b.dag()*b + omega_c * a.dag() * a +
             g * (b.dag() * a + a.dag() * b))
    c_ops = [np.sqrt(gam) * a]

    result_enr = qutip.steadystate(H_enr, c_ops)
    exp_sz_enr = qutip.expect(sz, result_enr)

    np.testing.assert_allclose(exp_sz_JC,
                               exp_sz_enr, atol=1e-2)


def test_eigenstates_ENR():
    a1, a2 = qutip.enr_destroy([2, 2], 1)
    H = a1.dag() * a1 + a2.dag() * a2
    vals, vecs = H.eigenstates()
    for val, vec in zip(vals, vecs):
        assert val * vec == H @ vec
