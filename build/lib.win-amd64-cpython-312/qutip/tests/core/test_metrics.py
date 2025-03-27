# -*- coding: utf-8 -*-

import platform

import numpy as np
import pytest

from qutip import (
    Qobj, tensor, fock_dm, basis, destroy, qdiags, sigmax, sigmay, sigmaz,
    qeye, rand_ket, rand_super_bcsz, rand_dm,
    rand_unitary, to_super, to_choi, kraus_to_choi,
    to_chi, to_kraus,
)
from qutip.core.gates import snot, swap

try:
    import cvxpy
    import cvxopt
except ImportError:
    cvxpy, cvxopt = None, None

# These ones are the metrics functions that we actually want to test.
from qutip import (
    fidelity, tracedist, hellinger_dist, dnorm, average_gate_fidelity,
    unitarity, hilbert_dist, bures_dist, process_fidelity,
)
from qutip.core.metrics import _hilbert_space_dims


@pytest.fixture(scope="function", params=[2, 5, 10, 15, 25, 100])
def dimension(request):
    # There are also some cases in the file where this fixture is explicitly
    # overridden by a more local mark.  That is deliberate; this dimension is
    # intended for non-superoperators, and may cause inordinantly long tests if
    # (for example) something uses dimension=100 then makes a superoperator out
    # of it.
    return request.param


@pytest.fixture(scope="function", params=[
    pytest.param(rand_ket, id="pure"),
    pytest.param(rand_dm, id="mixed"),
])
def state(request, dimension):
    return request.param(dimension)


# Also parametrise left, right as if they're the names of two states for tests
# that need to take two states.
left = right = state


# The class names have an unusual naming convention to make them more
# convenient to use with the `pytest -k "expr"` selection syntax.  They start
# with the standard `Test`, but then are the name of the function they are
# testing in the function naming convention, so it's easy to remember the
# selector to choose a particular function.

class Test_fidelity:
    def test_mixed_state_inequality(self, dimension):
        tol = 1e-7
        rho1 = rand_dm(dimension, density=0.25)
        rho2 = rand_dm(dimension, density=0.25)
        F = fidelity(rho1, rho2)
        assert 1 - F <= np.sqrt(1 - F*F) + tol

    @pytest.mark.parametrize('right_dm', [True, False], ids=['mixed', 'pure'])
    @pytest.mark.parametrize('left_dm', [True, False], ids=['mixed', 'pure'])
    def test_orthogonal(self, left_dm, right_dm, dimension):
        left = basis(dimension, 0)
        right = basis(dimension, dimension//2)
        if left_dm:
            left = left.proj()
        if right_dm:
            right = right.proj()
        assert fidelity(left, right) == pytest.approx(0, abs=1e-6)

    def test_invariant_under_unitary_transformation(self, dimension):
        rho1 = rand_dm(dimension, density=0.25)
        rho2 = rand_dm(dimension, density=0.25)
        U = rand_unitary(dimension)
        F = fidelity(rho1, rho2)
        FU = fidelity(U*rho1*U.dag(), U*rho2*U.dag())
        assert F == pytest.approx(FU, rel=1e-5, abs=1e-7)

    def test_state_with_itself(self, state):
        assert fidelity(state, state) == pytest.approx(1, abs=1e-6)

    def test_bounded(self, left, right, dimension):
        """Test that fidelity is bounded on [0, 1]."""
        tol = 1e-7
        assert -tol <= fidelity(left, right) <= 1 + tol

    def test_pure_state_equivalent_to_overlap(self, dimension):
        """Check fidelity against pure-state overlap, see gh-361."""
        psi = rand_ket(dimension)
        phi = rand_ket(dimension)
        overlap = np.abs(psi.overlap(phi))
        assert fidelity(psi, phi) == pytest.approx(overlap, abs=1e-7)

    ket_0 = basis(2, 0)
    ket_1 = basis(2, 1)
    ket_p = (ket_0 + ket_1).unit()
    ket_py = (ket_0 + np.exp(0.25j*np.pi)*ket_1).unit()
    max_mixed = qeye(2).unit()

    @pytest.mark.parametrize(['left', 'right', 'expected'], [
        pytest.param(ket_0, ket_p, np.sqrt(0.5), id="|0>,|+>"),
        pytest.param(ket_0, ket_1, 0, id="|0>,|1>"),
        pytest.param(ket_0, max_mixed, np.sqrt(0.5), id="|0>,id/2"),
        pytest.param(ket_p, ket_py, np.sqrt(0.125 + (0.5+np.sqrt(0.125))**2),
                     id="|+>,|+'>"),
    ])
    def test_known_cases(self, left, right, expected):
        assert fidelity(left, right) == pytest.approx(expected, abs=1e-7)


class Test_tracedist:
    def test_state_with_itself(self, state):
        assert tracedist(state, state) == pytest.approx(0, abs=1e-6)

    @pytest.mark.parametrize('right_dm', [True, False], ids=['mixed', 'pure'])
    @pytest.mark.parametrize('left_dm', [True, False], ids=['mixed', 'pure'])
    def test_orthogonal(self, left_dm, right_dm, dimension):
        left = basis(dimension, 0)
        right = basis(dimension, dimension//2)
        if left_dm:
            left = left.proj()
        if right_dm:
            right = right.proj()
        assert tracedist(left, right) == pytest.approx(1, abs=1e-6)

    def test_invariant_under_unitary_transformation(self, dimension):
        rho1 = rand_dm(dimension, density=0.25)
        rho2 = rand_dm(dimension, density=0.25)
        U = rand_unitary(dimension)
        D = tracedist(rho1, rho2)
        DU = tracedist(U*rho1*U.dag(), U*rho2*U.dag())
        assert D == pytest.approx(DU, rel=1e-5)


class Test_hellinger_dist:
    @pytest.mark.parametrize('right_dm', [True, False], ids=['mixed', 'pure'])
    @pytest.mark.parametrize('left_dm', [True, False], ids=['mixed', 'pure'])
    def test_orthogonal(self, left_dm, right_dm, dimension):
        left = basis(dimension, 0)
        right = basis(dimension, dimension//2)
        if left_dm:
            left = left.proj()
        if right_dm:
            right = right.proj()
        expected = np.sqrt(2)
        assert hellinger_dist(left, right) == pytest.approx(expected, abs=1e-6)

    def test_state_with_itself(self, state):
        assert hellinger_dist(state, state) == pytest.approx(0, abs=1e-6)

    def test_known_cases_pure_states(self, dimension):
        left = rand_ket(dimension)
        right = rand_ket(dimension)
        expected = np.sqrt(2 * (1 - np.abs(left.overlap(right))**2))
        assert hellinger_dist(left, right) == pytest.approx(expected, abs=1e-7)

    @pytest.mark.parametrize('dimension', [2, 5, 10, 25])
    def test_monotonicity(self, dimension):
        """
        Check monotonicity w.r.t. tensor product, see. Eq. (45) in
        arXiv:1611.03449v2:
            hellinger_dist(rhoA & rhoB, sigmaA & sigmaB)
            >= hellinger_dist(rhoA, sigmaA)
        with equality iff sigmaB = rhoB where '&' is the tensor product.
        """
        tol = 1e-5
        rhoA, rhoB, sigmaA, sigmaB = [rand_dm(dimension) for _ in [None]*4]
        rho = tensor(rhoA, rhoB)
        rho_sim = tensor(rhoA, sigmaB)
        sigma = tensor(sigmaA, sigmaB)
        dist = hellinger_dist(rhoA, sigmaA)
        assert hellinger_dist(rho, sigma) + tol > dist
        assert hellinger_dist(rho_sim, sigma) == pytest.approx(dist, abs=tol)


class Test_average_gate_fidelity:
    def test_identity(self, dimension):
        id = qeye(dimension)
        assert average_gate_fidelity(id) == pytest.approx(1, abs=1e-12)

    @pytest.mark.parametrize('dimension', [2, 5, 10, 20])
    def test_bounded(self, dimension):
        tol = 1e-7
        channel = rand_super_bcsz(dimension)
        assert -tol <= average_gate_fidelity(channel) <= 1 + tol

    @pytest.mark.parametrize('dimension', [2, 5, 10, 20])
    def test_unitaries_equal_1(self, dimension):
        """Tests that for random unitaries U, AGF(U, U) = 1."""
        tol = 1e-7
        U = rand_unitary(dimension)
        SU = to_super(U)
        assert average_gate_fidelity(SU, target=U) == pytest.approx(1, abs=tol)

    def test_average_gate_fidelity_against_legacy_implementation(self):
        """
        Metrics: Test that AGF coincides with pre-5.0 implementation
        """
        def agf_pre_50(oper, target=None):
            kraus = to_kraus(oper)
            d = kraus[0].shape[0]

            if kraus[0].shape[1] != d:
                return TypeError(
                    "Average gate fidelity only implemented for square "
                    "superoperators.")

            if target is None:
                return (
                    (d + np.sum([np.abs(A_k.tr()) ** 2 for A_k in kraus])) /
                    (d * d + d)
                )
            return (
                (d + np.sum([
                    np.abs((A_k * target.dag()).tr()) ** 2 for A_k in kraus
                ])) / (d * d + d)
            )

        oper = rand_super_bcsz(16)
        target = rand_unitary(16)
        np.testing.assert_almost_equal(
            average_gate_fidelity(oper, target),
            agf_pre_50(oper, target)
        )


class Test_hilbert_dist:
    def test_known_cases(self):
        r1 = qdiags(np.array([0.5, 0.5, 0, 0]), 0)
        r2 = qdiags(np.array([0, 0, 0.5, 0.5]), 0)
        assert hilbert_dist(r1, r2) == pytest.approx(1, abs=1e-6)


paulis = [qeye(2), sigmax(), sigmay(), sigmaz()]


class Test_unitarity:
    @pytest.mark.parametrize(['operator', 'expected'], [
        pytest.param(to_super(sigmax()), 1, id="sigmax"),
        pytest.param(0.25 * sum(to_super(x) for x in paulis), 0, id="paulis"),
        pytest.param(0.5 * (to_super(qeye(2)) + to_super(sigmax())), 1/3,
                     id="id+sigmax"),
    ])
    def test_known_cases(self, operator, expected):
        assert unitarity(operator) == pytest.approx(expected, abs=1e-7)

    @pytest.mark.parametrize('n_qubits', [1, 2, 3, 4, 5])
    def test_bounded(self, n_qubits):
        tol = 1e-7
        operator = rand_super_bcsz(2**n_qubits)
        assert -tol <= unitarity(operator) <= 1 + tol


class TestComparisons:
    """Test some known inequalities between two different metrics."""

    def test_inequality_tracedist_to_fidelity(self, left, right):
        tol = 1e-7
        assert 1 - fidelity(left, right) <= tracedist(left, right) + tol

    def test_inequality_hellinger_dist_to_bures_dist(self, left, right):
        tol = 1e-7
        hellinger = hellinger_dist(left, right)
        bures = bures_dist(left, right)
        assert bures <= hellinger + tol


def overrotation(x):
    return to_super((1j * np.pi * x * sigmax() / 2).expm())


def had_mixture(x):
    id_chan = to_choi(qeye(2))
    S_eye = to_super(id_chan)
    S_H = to_super(snot())
    return (1 - x) * S_eye + x * S_H


def swap_map(x):
    base = (1j * x * swap()).expm()
    dims = [[[2], [2]], [[2], [2]]]
    return Qobj(base, dims=dims, superrep='super')


def adc_choi(x):
    kraus = [
        np.sqrt(1 - x) * qeye(2),
        np.sqrt(x) * destroy(2),
        np.sqrt(x) * fock_dm(2, 0),
    ]
    return kraus_to_choi(kraus)


# dnorm tests have always been slightly flaky; in some cases, cvxpy will fail
# to solve the problem, and this can cause an entire test-suite failure.  As
# long as we are using random tests (perhaps not ideal), this will happen
# occasionally.  This isn't entirely a bug, it's just a reality of using a
# one-size-fits-all solver; we've historically assumed users who come up
# against this sort of thing will be accepting of the fact that dnorm
# calculation is nontrivial, and isn't always entirely feasible.
#
# To deal with it, we allow each test to be rerun twice, using
# pytest-rerunfailures.  This should forbid pathological cases where the test
# is failing every time, but not penalise one-off failures.  As far as we know,
# the failing tests always involve a random step, so triggering a re-run will
# have them choose new variables as well.
#
# The warning filter is to account for cvxpy < 1.1.10 which uses np.complex,
# which is deprecated as of numpy 1.20.
#
# Skip dnorm tests if we don't have cvxpy or cvxopt available, since dnorm
# depends on them.
@pytest.mark.skipif(cvxpy is None or cvxopt is None,
                    reason="Skipping dnorm tests because dnorm requires cvxpy"
                    " and cvxopt which are not installed.")
@pytest.mark.skipif("Windows" in platform.system(),
                    reason="Skipping dnorm tests -- cvxpy and cvxopt can be"
                    " installed on Windows but fail to correctly solve"
                    " the given optimization problems.")
@pytest.mark.flaky(reruns=2)
@pytest.mark.filterwarnings(
    "ignore:`np.complex` is a deprecated alias:DeprecationWarning:cvxpy"
)
class Test_dnorm:
    @pytest.fixture(params=[2, 3])
    def dimension(self, request):
        return request.param

    @pytest.fixture(params=[True, False], ids=['sparse', 'dense'])
    def sparse(self, request):
        return request.param

    @pytest.mark.parametrize("variable", [0.1, 0.5, 0.9])
    def test_sparse_against_dense_adc(self, variable):
        """
        Test sparse versus dense dnorm calculation for a sample of
        amplitude-damping channels.
        """
        # Choi matrix for identity channel on 1 qubit
        A = kraus_to_choi([qeye(2)])
        B = adc_choi(variable)
        dense = dnorm(A, B, force_solve=True, sparse=False)
        sparse = dnorm(A, B, force_solve=True, sparse=True)
        assert dense == pytest.approx(sparse, abs=1e-7)

    @pytest.mark.repeat(3)
    def test_sparse_against_dense_random(self, dimension):
        """
        Test sparse versus dense dnorm calculation for random superoperators.
        """
        A = rand_super_bcsz(dimension)
        dense_run_result = dnorm(A, force_solve=True, sparse=False)
        sparse_run_result = dnorm(A, force_solve=True, sparse=True)
        assert dense_run_result == pytest.approx(sparse_run_result, abs=1e-7)

    def test_bounded(self, dimension, sparse):
        """dnorm(A - B) in [0, 2] for random superops A, B."""
        tol = 1e-7
        A, B = rand_super_bcsz(dimension), rand_super_bcsz(dimension)
        assert -tol <= dnorm(A, B, sparse=sparse) <= 2 + tol

    def test_qubit_simple_known_cases(self, sparse):
        """Check agreement for known qubit channels."""
        id_chan = to_choi(qeye(2))
        X_chan = to_choi(sigmax())
        depol = to_choi(Qobj(
            qeye(4), dims=[[[2], [2]], [[2], [2]]], superrep='chi',
        ))
        # We need to restrict the number of iterations for things on the
        # boundary, such as perfectly distinguishable channels.
        assert (
            dnorm(id_chan, X_chan, sparse=sparse)
            == pytest.approx(2, abs=1e-7)
        )
        assert (
            dnorm(id_chan, depol, sparse=sparse)
            == pytest.approx(1.5, abs=1e-7)
        )
        # Finally, we add a known case from Johnston's QETLAB documentation,
        #   || Phi - I ||_♢,
        # where Phi(X) = UXU⁺ and U = [[1, 1], [-1, 1]] / sqrt(2).
        U = np.sqrt(0.5) * Qobj([[1, 1], [-1, 1]])
        assert (
            dnorm(U, qeye(2), sparse=sparse)
            == pytest.approx(np.sqrt(2), abs=1e-7)
        )

    @pytest.mark.parametrize(["variable", "expected", "generator"], [
        [1.0e-3, 3.141591e-03, overrotation],
        [3.1e-3, 9.738899e-03, overrotation],
        [1.0e-2, 3.141463e-02, overrotation],
        [3.1e-2, 9.735089e-02, overrotation],
        [1.0e-1, 3.128689e-01, overrotation],
        [3.1e-1, 9.358596e-01, overrotation],
        [1.0e-3, 2.000000e-03, had_mixture],
        [3.1e-3, 6.200000e-03, had_mixture],
        [1.0e-2, 2.000000e-02, had_mixture],
        [3.1e-2, 6.200000e-02, had_mixture],
        [1.0e-1, 2.000000e-01, had_mixture],
        [3.1e-1, 6.200000e-01, had_mixture],
        [1.0e-3, 2.000000e-03, swap_map],
        [3.1e-3, 6.199997e-03, swap_map],
        [1.0e-2, 1.999992e-02, swap_map],
        [3.1e-2, 6.199752e-02, swap_map],
        [1.0e-1, 1.999162e-01, swap_map],
        [3.1e-1, 6.173918e-01, swap_map],
    ])
    def test_qubit_known_cases(self, variable, expected, generator, sparse):
        """
        Test cases based on comparisons to pre-existing dnorm implementations.
        In particular, the targets for the following test cases were generated
        using QuantumUtils for MATLAB (https://goo.gl/oWXhO9).
        """
        id_chan = to_choi(qeye(2))
        channel = generator(variable)
        assert (
            dnorm(channel, id_chan, sparse=sparse)
            == pytest.approx(expected, abs=1e-7)
        )

    def test_qubit_scalar(self, dimension):
        """dnorm(a * A) == a * dnorm(A) for scalar a, qobj A."""
        a = np.random.random()
        A = rand_super_bcsz(dimension)
        B = rand_super_bcsz(dimension)
        assert dnorm(a*A, a*B) == pytest.approx(a*dnorm(A, B), abs=1e-7)

    def test_qubit_triangle(self, dimension):
        """Check that dnorm(A + B) <= dnorm(A) + dnorm(B)."""
        A = rand_super_bcsz(dimension)
        B = rand_super_bcsz(dimension)
        assert dnorm(A + B) <= dnorm(A) + dnorm(B) + 1e-7

    @pytest.mark.repeat(3)
    def test_unitary_case(self, dimension):
        """Check that the diamond norm is one for unitary maps."""
        A, B = rand_unitary(dimension), rand_unitary(dimension)
        assert (
            dnorm(A, B)
            == pytest.approx(dnorm(A, B, force_solve=True), abs=1e-5)
        )

    @pytest.mark.repeat(3)
    def test_cp_case(self, dimension):
        """Check that the diamond norm is one for unitary maps."""
        A = rand_super_bcsz(dimension, enforce_tp=False)
        assert (
            dnorm(A)
            == pytest.approx(dnorm(A, force_solve=True), abs=1e-5)
        )

    @pytest.mark.repeat(3)
    def test_cptp_case(self, dimension, sparse):
        """Check that the diamond norm is one for CPTP maps."""
        A = rand_super_bcsz(dimension)
        assert A.iscptp
        assert dnorm(A, sparse=sparse) == pytest.approx(1, abs=1e-7)


@pytest.mark.parametrize('superrep_conversion',
                         [lambda x: x, to_super, to_choi, to_chi, to_kraus])
def test_process_fidelity_of_identity(superrep_conversion):
    """
    Metrics: process fidelity of identity map is 1
    """
    num_qubits = 3
    oper = qeye(num_qubits*[2])
    f = process_fidelity(superrep_conversion(oper))
    assert np.isrealobj(f)
    assert f == pytest.approx(1)


@pytest.mark.parametrize('superrep_conversion',
                         [to_super, to_choi, to_chi, to_kraus])
def test_process_fidelity_identical_channels(superrep_conversion):
    """
    Metrics: process fidelity of a map to itself is 1
    """
    num_qubits = 2
    for k in range(10):
        oper = rand_super_bcsz(num_qubits*[2])
        oper = superrep_conversion(oper)
        f = process_fidelity(oper, oper)
        assert f == pytest.approx(1)


def test_process_fidelity_identical_unitaries():
    """
    Metrics: process fidelity of a unitary to itself is 1
    """
    num_qubits = 3
    for k in range(10):
        oper = rand_unitary(num_qubits * [2])
        f = process_fidelity(oper, oper)
        assert f == pytest.approx(1)


def test_process_fidelity_consistency():
    """
    Metrics: process fidelity independent of how channels are represented
    """
    num_qubits = 2
    for k in range(10):
        fidelities_u_to_u = []
        fidelities_u_to_id = []
        u1 = rand_unitary(num_qubits * [2])
        u2 = rand_unitary(num_qubits * [2])
        for map1 in [lambda x:x, to_super, to_choi, to_chi, to_kraus]:
            fidelities_u_to_id.append(process_fidelity(map1(u1)))
            for map2 in [lambda x:x, to_super, to_choi, to_chi, to_kraus]:
                fidelities_u_to_u.append(process_fidelity(map1(u1), map2(u2)))
        assert all(abs(fidelities_u_to_id - fidelities_u_to_id[0]) < 1e-6)
        assert all(abs(fidelities_u_to_u - fidelities_u_to_u[0]) < 1e-6)


def test_process_fidelity_unitary_invariance():
    """
    Metrics: process fidelity, invariance under unitary trans.
    """
    for k in range(10):
        op1 = rand_super_bcsz(10)
        op2 = rand_super_bcsz(10)
        u = to_super(rand_unitary(10))
        assert abs(
            process_fidelity(op1, op2)
            - process_fidelity(u*op1, u*op2)
        ) < 1e-8


@pytest.mark.parametrize('superrep_conversion',
                         [lambda x: x, to_super, to_choi, to_kraus])
def test_hilbert_space_dims(superrep_conversion):
    """
    Metrics: check _hilbert_space_dims
    """
    dims = [[2, 3], [4, 5]]
    u = Qobj(np.ones((np.prod(dims[0]), np.prod(dims[1]))), dims=dims)
    assert _hilbert_space_dims(superrep_conversion(u)) == dims


def test_hilbert_space_dims_chi():
    """
    Metrics: check _hilbert_space_dims for a chi channel
    """
    dims = [[2, 2], [2, 2]]
    u = Qobj(np.ones((np.prod(dims[0]), np.prod(dims[1]))), dims=dims)
    assert _hilbert_space_dims(to_chi(u)) == dims
