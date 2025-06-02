import pytest
import itertools
import random
import numpy as np
import qutip
import scipy.sparse


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

    def test_nstates(self, dimensions, n_excitations):
        expected_size = _n_enr_states(dimensions, n_excitations)
        nstates = qutip.enr_nstates(dimensions, n_excitations)
        assert nstates == expected_size


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


# function to convert an ENR Qobj to a regular Qobj
def enr_to_reg(q):
    if q.type == "scalar":
        return q
    dims = q.dims[0]
    n_ex = q._dims[0].n_excitations

    if q.isoper:
        newdims = [list(dims)]*2
        newshape = [np.prod(dims)]*2
    elif q.isket:
        newdims = [list(dims), [1]*len(dims)]
        newshape = [np.prod(dims), 1]
    else:
        raise ValueError("Invalid Qobj type")

    newindices = [i for i, state  # get indices for unrestricted space
                  in enumerate(qutip.state_number_enumerate(dims))
                  if sum(state) <= n_ex]
    dtype = q.dtype
    if dtype is qutip.data.Dense:
        mat = q.data_as()
        newmat = np.zeros(newshape, dtype=mat.dtype)
        if q.isoper:
            newmat[np.ix_(newindices, newindices)] = mat
        else:
            newmat[newindices] = mat
    else:
        mat = q.to("csr").data_as().tocoo()
        newrows = [newindices[i] for i in mat.row]
        newcols = [newindices[i] for i in mat.col]
        newmat = scipy.sparse.coo_matrix(
            (mat.data, (newrows, newcols)), shape=newshape)

    return qutip.Qobj(newmat, dims=newdims, dtype=dtype)


@pytest.fixture(params=["csr", "dense", "dia"])
def dtype(request):
    return request.param


# copied parameters from test_ptrace.py
@pytest.mark.parametrize('dims, sel',
                         [
                             ([5, 2, 3], [2, 1]),
                             ([5, 2, 3], [0, 2]),
                             ([5, 2, 3], [0, 1]),
                             ([2]*6, [3, 2]),
                             ([2]*6, [0, 2]),
                             ([2]*6, [0, 1]),
                             ([4, 4, 4], []),
                             ([4, 4, 4], [0, 1, 2]),
                         ])
@pytest.mark.filterwarnings("ignore:enr_ptrace")  
def test_enr_ptrace(dims, sel, n_excitations, dtype):
    nstates_enr = _n_enr_states(dims, n_excitations)
    # use qutip to make a random sparse Hermitian matrix w/ trace 1
    rho_in_enr = qutip.rand_dm(
        nstates_enr, 0.05, distribution="herm", dtype=dtype)

    rho_in_enr.dims = [qutip.energy_restricted.EnrSpace(dims, n_excitations)]*2
    rho_in_reg = enr_to_reg(rho_in_enr)

    rho_out_enr = qutip.enr_ptrace(rho_in_enr, sel)
    rho_out_toreg = enr_to_reg(rho_out_enr)
    rho_out_check = qutip.ptrace(rho_in_reg, sel)

    assert rho_out_toreg == rho_out_check


@pytest.fixture(params=[
    pytest.param(0, id="ket"),
    pytest.param(1, id="oper"),
])
def isoper(request):
    return request.param


@pytest.mark.parametrize('dims_list, n_ex_list',
                         [
                             ([[3]], [2]),
                             ([[3], [4]], [2, 1]),
                             ([[3, 2], [2, 2]], [2, 1]),
                             ([[3, 2], [4], [2, 2]], [2, 3, 2]),
                         ])
def test_enr_tensor(dims_list, n_ex_list, isoper, dtype):
    # isoper = 1 to test for operators, 0 to test for kets
    # figure out how big the matrices/vectors will be
    nstates_list = [_n_enr_states(dims, n_ex)
                    for dims, n_ex in zip(dims_list, n_ex_list)]
    # make sure sparse matrices are dense enough to not be empty
    dens_list = [max(0.1, 1/(nstates**(isoper+1))) for nstates in nstates_list]
    # number of columns in the random matrices, depending on if operator or ket
    ncol_arr = nstates_list if isoper else [1]*len(nstates_list)

    # generate the random matrices
    rand_mat_list = [scipy.sparse.random(
        nstates, ncol,
        density=dens, dtype="complex128")
        for (nstates, ncol, dens)
        in zip(nstates_list, ncol_arr, dens_list)]

    # figure out the row and column spaces for the ENR Qobj's
    rowspace_list = [qutip.energy_restricted.EnrSpace(dims, n_ex)
                     for dims, n_ex in zip(dims_list, n_ex_list)]
    if isoper:
        colspace_list = rowspace_list
    else:
        colspace_list = [[1]] * len(rowspace_list)

    # turn the random matrices into Qobj's
    q_in_enr_list = [
        qutip.Qobj(rand_mat, dims=[rowspace, colspace], dtype=dtype)
        for rand_mat, rowspace, colspace
        in zip(rand_mat_list, rowspace_list, colspace_list)]
    assert all(q_in_enr.isoper == isoper for q_in_enr in q_in_enr_list)

    q_in_reg_list = [enr_to_reg(q_in_enr) for q_in_enr in q_in_enr_list]

    q_out_enr = qutip.enr_tensor(*q_in_enr_list, newexcitations=sum(n_ex_list))
    q_out_toreg = enr_to_reg(q_out_enr)
    q_out_check = qutip.tensor(*q_in_reg_list)

    assert q_out_toreg == q_out_check


@pytest.mark.parametrize('truncate', [True, False])
def test_enr_tensor_trunc(isoper, dtype, truncate):
    q_in = qutip.enr_fock([2], 1, [1], dtype=dtype)
    if isoper:
        q_in = q_in.proj()

    if not truncate:
        with pytest.raises(ValueError) as e:
            qutip.enr_tensor(q_in, q_in)
        assert "not in the new restricted state space" in str(e.value)
    else:
        q_out = qutip.enr_tensor(q_in, q_in, truncate=truncate, verbose=False)
        q_out_check = qutip.zero_ket(
            qutip.energy_restricted.EnrSpace([2, 2], 1), dtype=dtype)
        if isoper:
            q_out_check = q_out_check.proj()
        assert q_out == q_out_check
