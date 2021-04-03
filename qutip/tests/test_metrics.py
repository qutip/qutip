# -*- coding: utf-8 -*-
# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

import platform

import numpy as np
import pytest

from qutip import (
    Qobj, tensor, fock_dm, basis, destroy, qdiags, sigmax, sigmay, sigmaz,
    qeye, rand_ket, rand_super_bcsz, rand_ket_haar, rand_dm_ginibre, rand_dm,
    rand_unitary, rand_unitary_haar, to_super, to_choi, kraus_to_choi,
)
from qutip.qip.operations import (
    hadamard_transform, swap,
)
# These ones are the metrics functions that we actually want to test.
from qutip import (
    fidelity, tracedist, hellinger_dist, dnorm, average_gate_fidelity,
    unitarity, hilbert_dist, bures_dist,
)


@pytest.fixture(scope="function", params=[2, 5, 10, 15, 25, 100])
def dimension(request):
    # There are also some cases in the file where this fixture is explicitly
    # overridden by a more local mark.  That is deliberate; this dimension is
    # intended for non-superoperators, and may cause inordinantly long tests if
    # (for example) something uses dimension=100 then makes a superoperator out
    # of it.
    return request.param


@pytest.fixture(scope="function", params=[
    pytest.param(rand_ket_haar, id="pure"),
    pytest.param(rand_dm_ginibre, id="mixed"),
])
def state(request, dimension):
    return request.param(dimension)


# Also parametrise left, right as if they're the names of two states for tests
# that need to take two states.
left = right = state


class Test_fidelity:
    def test_mixed_state_inequality(self, dimension):
        tol = 1e-7
        rho1 = rand_dm(dimension, 0.25)
        rho2 = rand_dm(dimension, 0.25)
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
        rho1 = rand_dm(dimension, 0.25)
        rho2 = rand_dm(dimension, 0.25)
        U = rand_unitary(dimension, 0.25)
        F = fidelity(rho1, rho2)
        FU = fidelity(U*rho1*U.dag(), U*rho2*U.dag())
        assert F == pytest.approx(FU, rel=1e-5)

    def test_state_with_itself(self, state):
        assert fidelity(state, state) == pytest.approx(1, abs=1e-6)

    def test_bounded(self, left, right, dimension):
        """Test that fidelity is bounded on [0, 1]."""
        tol = 1e-7
        assert -tol <= fidelity(left, right) <= 1 + tol

    def test_pure_state_equivalent_to_overlap(self, dimension):
        """Check fidelity against pure-state overlap, see gh-361."""
        psi = rand_ket_haar(dimension)
        phi = rand_ket_haar(dimension)
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
        rho1 = rand_dm(dimension, 0.25)
        rho2 = rand_dm(dimension, 0.25)
        U = rand_unitary(dimension, 0.25)
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


# TODO: resolve the Mac failures.
@pytest.mark.skipif(
    "Darwin" in platform.system(),
    reason="average gate fidelity tests broken on macOS as of July 2019",
)
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
        U = rand_unitary_haar(dimension)
        SU = to_super(U)
        assert average_gate_fidelity(SU, target=U) == pytest.approx(1, abs=tol)


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
    S_H = to_super(hadamard_transform())
    return (1 - x) * S_eye + x * S_H


def swap_map(x):
    base = (1j * x * swap()).expm()
    dims = [[[2], [2]], [[2], [2]]]
    return Qobj(base, dims=dims, type='super', superrep='super')


class Test_dnorm:
    # Skip dnorm tests if we don't have cvxpy or cvxopt available, since it
    # depends on them.
    pytest.importorskip("cvxpy")
    pytest.importorskip("cvxopt")

    @pytest.mark.repeat(10)
    def test_on_sparse_matrix(self):
        """
        Tests sparse versus dense dnorm calculation on sparse matrices.
        """
        force_solve = True

        def AmpDampChoi(p):
            Kraus = [(1-p)**.5*qeye(2), p**.5*destroy(2), p**.5*fock_dm(2, 0)]
            return kraus_to_choi(Kraus)

        # Choi matrix for identity channel on 1 qubit
        A = kraus_to_choi([qeye(2)])
        p = np.random.uniform(0.1, 0.9)
        B = AmpDampChoi(p)
        dense_run_result = dnorm(A, B, force_solve=force_solve, sparse=False)
        sparse_run_result = dnorm(A, B, force_solve=force_solve, sparse=True)
        assert dense_run_result == pytest.approx(sparse_run_result, abs=1e-7)

    @pytest.mark.repeat(10)
    @pytest.mark.parametrize(["dim"], [
        pytest.param(2, id="dim2"),
        pytest.param(3, id="dim3"),
    ])
    def test_on_dense_matrix(self, dim):
        """
        Metrics: Tests sparse versus dense dnorm calculation on dense matrices.
        """
        force_solve = True
        A = rand_super_bcsz(dim)
        dense_run_result = dnorm(A, force_solve=force_solve, sparse=False)
        sparse_run_result = dnorm(A, force_solve=force_solve, sparse=True)
        assert dense_run_result == pytest.approx(sparse_run_result, abs=1e-7)

    @pytest.mark.repeat(10)
    def test_bounded(self):
        """
        Metrics: dnorm(A - B) in [0, 2] for random superops A, B.
        """
        A, B = rand_super_bcsz(3), rand_super_bcsz(3)
        norm_result = dnorm(A, B)
        assert 1 == pytest.approx(norm_result, abs=1 + 1e-7)

    def test_qubit_simple_known_cases(self):
        """
        Metrics: check agreement for known qubit channels.
        """
        id_chan = to_choi(qeye(2))
        X_chan = to_choi(sigmax())
        depol = to_choi(Qobj(
            qeye(4), dims=[[[2], [2]], [[2], [2]]], superrep='chi',
        ))
        # We need to restrict the number of iterations for things on the
        # boundary, such as perfectly distinguishable channels.
        assert 2 == pytest.approx(dnorm(id_chan, X_chan), abs=1e-7)
        assert 1.5 == pytest.approx(dnorm(id_chan, depol), abs=1e-7)
        # Finally, we add a known case from Johnston's QETLAB documentation,
        #   || Phi - I ||_♢,
        # where Phi(X) = UXU⁺ and U = [[1, 1], [-1, 1]] / sqrt(2).
        assert np.sqrt(2) == pytest.approx(dnorm(Qobj([[1, 1], [-1, 1]])
                                           / np.sqrt(2), qeye(2)))

    @pytest.mark.parametrize(["variable", 'target', 'matrix_creator'], [
        [1.000000e-03, 3.141591e-03, 'overrotation'],
        [3.100000e-03, 9.738899e-03, 'overrotation'],
        [1.000000e-02, 3.141463e-02, 'overrotation'],
        [3.100000e-02, 9.735089e-02, 'overrotation'],
        [1.000000e-01, 3.128689e-01, 'overrotation'],
        [3.100000e-01, 9.358596e-01, 'overrotation'],
        [1.000000e-03, 2.000000e-03, 'had_mixture'],
        [3.100000e-03, 6.200000e-03, 'had_mixture'],
        [1.000000e-02, 2.000000e-02, 'had_mixture'],
        [3.100000e-02, 6.200000e-02, 'had_mixture'],
        [1.000000e-01, 2.000000e-01, 'had_mixture'],
        [3.100000e-01, 6.200000e-01, 'had_mixture'],
        [1.000000e-03, 2.000000e-03, 'swap_map'],
        [3.100000e-03, 6.199997e-03, 'swap_map'],
        [1.000000e-02, 1.999992e-02, 'swap_map'],
        [3.100000e-02, 6.199752e-02, 'swap_map'],
        [1.000000e-01, 1.999162e-01, 'swap_map'],
        [3.100000e-01, 6.173918e-01, 'swap_map'],
    ])
    def test_qubit_known_cases(self, variable, target, matrix_creator):
        # Next, we'll generate some test cases based on comparisons to
        # pre-existing dnorm() implementations. In particular, the targets for
        # the following test cases were generated using QuantumUtils for MATLAB
        # (https://goo.gl/oWXhO9).
        id_chan = to_choi(qeye(2))
        if matrix_creator == 'overrotation':
            assert target == pytest.approx(dnorm(overrotation(variable),
                                                 id_chan), abs=1e-7)

        elif matrix_creator == 'had_mixture':
            assert target == pytest.approx(dnorm(had_mixture(variable),
                                                 id_chan), abs=1e-7)

        elif matrix_creator == 'swap_map':
            assert target == pytest.approx(dnorm(swap_map(variable),
                                                 id_chan), abs=1e-7)

    @pytest.mark.parametrize(["dim"], [
        pytest.param(2, id="dim2"),
        pytest.param(2, id="dim3"),
    ])
    def test_qubit_scalar(self, dim):
        """
        Metrics: checks that dnorm(a * A) == a * dnorm(A) for scalar a, qobj A.
        """
        a = np.random.random()
        A = rand_super_bcsz(dim)
        B = rand_super_bcsz(dim)
        assert dnorm(a * A, a * B) == pytest.approx(a * dnorm(A, B), abs=1e-7)

    @pytest.mark.parametrize(["dim"], [
        pytest.param(2, id="dim2"),
        pytest.param(3, id="dim3")
    ])
    def test_qubit_triangle(self, dim):
        """
        Metrics: checks that dnorm(A + B) ≤ dnorm(A) + dnorm(B).
        """
        A = rand_super_bcsz(dim)
        B = rand_super_bcsz(dim)
        assert dnorm(A + B) <= dnorm(A) + dnorm(B) + 1e-7

    @pytest.mark.repeat(10)
    @pytest.mark.parametrize(["dim", 'matrix_generator'], [
        pytest.param(2, 'bcsz', id="dim2"),
        pytest.param(3, 'bcsz', id="dim3"),
        pytest.param(2, 'haar', id="dim2"),
        pytest.param(3, 'haar', id="dim3"),
    ])
    def test_force_solve(self, dim, matrix_generator):
        """
        Metrics: checks that special cases for dnorm agree with SDP solutions.
        """
        if matrix_generator == 'bcsz':
            A = rand_super_bcsz(dim)
            B = rand_super_bcsz(dim)
        elif matrix_generator == 'haar':
            A = rand_unitary_haar(dim)
            B = rand_unitary_haar(dim)
        assert (
            dnorm(A, B, force_solve=False)
            == pytest.approx(dnorm(A, B, force_solve=True), abs=1e-5)
        )

    @pytest.mark.repeat(10)
    @pytest.mark.parametrize(["dim"], [
        pytest.param(2, id="dim2"),
        pytest.param(3, id="dim3"),
    ])
    def test_cptp(self, dim):
        """
        Metrics: checks that the diamond norm is one for CPTP maps.
        """
        A = rand_super_bcsz(dim)
        assert 1 == pytest.approx(dnorm(A, force_solve=True), abs=1e-7)
