# -*- coding: utf-8 -*-
"""
Simple tests for metrics and pseudometrics implemented in
the qutip.metrics module.
"""

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
from __future__ import division

import numpy as np
from numpy import abs, sqrt, ones, diag
from numpy.testing import (
    assert_, run_module_suite, assert_approx_equal,
    assert_almost_equal
)
import scipy

from qutip.operators import (
    create, destroy, jmat, identity, qdiags, sigmax, sigmay, sigmaz, qeye
)
from qutip.states import fock_dm, basis
from qutip.propagator import propagator
from qutip.random_objects import (
    rand_herm, rand_dm, rand_unitary, rand_ket, rand_super_bcsz,
    rand_ket_haar, rand_dm_ginibre, rand_unitary_haar
)
from qutip.qobj import Qobj
from qutip.superop_reps import to_super, to_choi
from qutip.qip.gates import hadamard_transform, swap
from qutip.tensor import tensor
from qutip.metrics import *

import qutip.settings

import platform
import unittest

try:
    import cvxpy
except:
    cvxpy = None


# Disable dnorm tests if MKL is present (see Issue #484).
if qutip.settings.has_mkl:
    dnorm_test = unittest.skipIf(True,
                                 "Known failure; CVXPY/MKL incompatibility.")
else:
    dnorm_test = unittest.skipIf(cvxpy is None, "CVXPY required for dnorm().")

#FIXME: Try to resolve the average_gate_fidelity issues on MACOS
avg_gate_fidelity_test = unittest.skipIf(platform.system().startswith("Darwin"),
                "average_gate_fidelity tests were failing on MacOS "
                "as of July 2019.")

"""
A test class for the metrics and pseudo-metrics included with QuTiP.
"""



def test_fid_trdist_limits():
    """
    Metrics: Fidelity / trace distance limiting cases
    """
    rho = rand_dm(25, 0.25)
    assert_(abs(fidelity(rho, rho)-1) < 1e-6)
    assert_(tracedist(rho, rho) < 1e-6)
    rho1 = fock_dm(5, 1)
    rho2 = fock_dm(5, 2)
    assert_(fidelity(rho1, rho2) < 1e-6)
    assert_(abs(tracedist(rho1, rho2)-1) < 1e-6)


def test_fidelity1():
    """
    Metrics: Fidelity, mixed state inequality
    """
    for k in range(10):
        rho1 = rand_dm(25, 0.25)
        rho2 = rand_dm(25, 0.25)
        F = fidelity(rho1, rho2)
        assert_(1-F <= sqrt(1-F**2))


def test_fidelity2():
    """
    Metrics: Fidelity, invariance under unitary trans.
    """
    for k in range(10):
        rho1 = rand_dm(25, 0.25)
        rho2 = rand_dm(25, 0.25)
        U = rand_unitary(25, 0.25)
        F = fidelity(rho1, rho2)
        FU = fidelity(U*rho1*U.dag(), U*rho2*U.dag())
        assert_(abs((F-FU)/F) < 1e-5)


def test_fidelity_max():
    """
    Metrics: Fidelity of a pure state w/ itself should be 1.
    """
    for _ in range(10):
        psi = rand_ket_haar(13)
        assert_almost_equal(fidelity(psi, psi), 1)


def test_fidelity_bounded_purepure(tol=1e-7):
    """
    Metrics: Fidelity of pure states within [0, 1].
    """
    for _ in range(10):
        psi = rand_ket_haar(17)
        phi = rand_ket_haar(17)
        F = fidelity(psi, phi)
        assert_(-tol <= F <= 1 + tol)


def test_fidelity_bounded_puremixed(tol=1e-7):
    """
    Metrics: Fidelity of pure states against mixed states within [0, 1].
    """
    for _ in range(10):
        psi = rand_ket_haar(11)
        sigma = rand_dm_ginibre(11)
        F = fidelity(psi, sigma)
        assert_(-tol <= F <= 1 + tol)


def test_fidelity_bounded_mixedmixed(tol=1e-7):
    """
    Metrics: Fidelity of mixed states within [0, 1].
    """
    for _ in range(10):
        rho = rand_dm_ginibre(11)
        sigma = rand_dm_ginibre(11)
        F = fidelity(rho, sigma)
        assert_(-tol <= F <= 1 + tol)


def test_fidelity_known_cases():
    """
    Metrics: Checks fidelity against known cases.
    """
    ket0 = basis(2, 0)
    ket1 = basis(2, 1)
    ketp = (ket0 + ket1).unit()
    # A state that almost overlaps with |+> should serve as
    # a nice test case, especially since we can analytically
    # calculate its overlap with |+>.
    ketpy = (ket0 + np.exp(1j * np.pi / 4) * ket1).unit()

    mms = qeye(2).unit()

    assert_almost_equal(fidelity(ket0, ketp), 1 / np.sqrt(2))
    assert_almost_equal(fidelity(ket0, ket1), 0)
    assert_almost_equal(fidelity(ket0, mms),  1 / np.sqrt(2))
    assert_almost_equal(fidelity(ketp, ketpy),
        np.sqrt(
            (1 / 8) + (1 / 2 + 1 / (2 * np.sqrt(2))) ** 2
        )
    )


def test_fidelity_overlap():
    """
    Metrics: Checks fidelity against pure-state overlap. (#631)
    """
    for _ in range(10):
        psi = rand_ket_haar(7)
        phi = rand_ket_haar(7)

        assert_almost_equal(
            fidelity(psi, phi),
            np.abs((psi.dag() * phi)[0, 0])
        )

def test_tracedist1():
    """
    Metrics: Trace dist., invariance under unitary trans.
    """
    for k in range(10):
        rho1 = rand_dm(25, 0.25)
        rho2 = rand_dm(25, 0.25)
        U = rand_unitary(25, 0.25)
        D = tracedist(rho1, rho2)
        DU = tracedist(U*rho1*U.dag(), U*rho2*U.dag())
        assert_(abs((D-DU)/D) < 1e-5)


def test_tracedist2():
    """
    Metrics: Trace dist. & Fidelity mixed/mixed inequality
    """
    for k in range(10):
        rho1 = rand_dm(25, 0.25)
        rho2 = rand_dm(25, 0.25)
        F = fidelity(rho1, rho2)
        D = tracedist(rho1, rho2)
        assert_(1-F <= D)


def test_tracedist3():
    """
    Metrics: Trace dist. & Fidelity mixed/pure inequality
    """
    for k in range(10):
        ket = rand_ket(25, 0.25)
        rho1 = ket*ket.dag()
        rho2 = rand_dm(25, 0.25)
        F = fidelity(rho1, rho2)
        D = tracedist(rho1, rho2)
        assert_(1-F**2 <= D)


def test_hellinger_corner():
    """
    Metrics: Hellinger dist.: check corner cases:
    same states, orthogonal states
    """
    orth1 = basis(40, 1)
    orth2 = basis(40, 3)
    orth3 = basis(40, 38)
    s2 = np.sqrt(2.0)
    assert_almost_equal(hellinger_dist(orth1, orth2), s2)
    assert_almost_equal(hellinger_dist(orth2, orth3), s2)
    assert_almost_equal(hellinger_dist(orth3, orth1), s2)
    for _ in range(10):
        ket = rand_ket(25, 0.25)
        rho = rand_dm(18, 0.75)
        assert_almost_equal(hellinger_dist(ket, ket), 0.)
        assert_almost_equal(hellinger_dist(rho, rho), 0.)


def test_hellinger_pure():
    """
    Metrics: Hellinger dist.: check against a simple
    expression which applies to pure states
    """
    for _ in range(10):
        ket1 = rand_ket(25, 0.25)
        ket2 = rand_ket(25, 0.25)
        hellinger = hellinger_dist(ket1, ket2)
        sqr_overlap = np.square(np.abs(ket1.overlap(ket2)))
        simple_expr = np.sqrt(2.0*(1.0-sqr_overlap))
        assert_almost_equal(hellinger, simple_expr)


def test_hellinger_inequality():
    """
    Metrics: Hellinger dist.: check whether Hellinger
    distance is indeed larger than Bures distance
    """
    for _ in range(10):
        rho1 = rand_dm(25, 0.25)
        rho2 = rand_dm(25, 0.25)
        hellinger = hellinger_dist(rho1, rho2)
        bures = bures_dist(rho1, rho2)
        assert_(hellinger >= bures)
        ket1 = rand_ket(40, 0.25)
        ket2 = rand_ket(40, 0.25)
        hellinger = hellinger_dist(ket1, ket2)
        bures = bures_dist(ket1, ket2)
        assert_(hellinger >= bures)


def test_hellinger_monotonicity():
    """
    Metrics: Hellinger dist.: check monotonicity
    w.r.t. tensor product, see. Eq. (45) in
    arXiv:1611.03449v2:
    hellinger_dist(rhoA*rhoB, sigmaA*sigmaB)>=
    hellinger_dist(rhoA, sigmaA)
    with equality iff sigmaB=rhoB
    """
    for _ in range(10):
        rhoA = rand_dm(8, 0.5)
        sigmaA = rand_dm(8, 0.5)
        rhoB = rand_dm(8, 0.5)
        sigmaB = rand_dm(8, 0.5)
        hellA = hellinger_dist(rhoA, sigmaA)
        hell_tensor = hellinger_dist(tensor(rhoA, rhoB),
                                     tensor(sigmaA, sigmaB))
        #inequality when sigmaB!=rhoB
        assert_(hell_tensor >= hellA)
        #equality iff sigmaB=rhoB
        rhoB = sigmaB
        hell_tensor = hellinger_dist(tensor(rhoA, rhoB),
                                     tensor(sigmaA, sigmaB))
        assert_almost_equal(hell_tensor, hellA)


def rand_super():
    h_5 = rand_herm(5)
    return propagator(h_5, scipy.rand(), [
        create(5), destroy(5), jmat(2, 'z')
    ])


@avg_gate_fidelity_test
def test_average_gate_fidelity():
    """
    Metrics: Check avg gate fidelities for random
    maps (equal to 1 for id maps).
    """
    for dims in range(2, 5):
        assert_(abs(average_gate_fidelity(identity(dims)) - 1) <= 1e-12)
    assert_(0 <= average_gate_fidelity(rand_super()) <= 1)

@avg_gate_fidelity_test
def test_average_gate_fidelity_target():
    """
    Metrics: Tests that for random unitaries U, AGF(U, U) = 1.
    """
    for _ in range(10):
        U = rand_unitary_haar(13)
        SU = to_super(U)
        assert_almost_equal(average_gate_fidelity(SU, target=U), 1)

def test_hilbert_dist():
    """
    Metrics: Hilbert distance.
    """
    diag1 = np.array([0.5, 0.5, 0, 0])
    diag2 = np.array([0, 0, 0.5, 0.5])
    r1 = qdiags(diag1, 0)
    r2 = qdiags(diag2, 0)
    assert_(abs(hilbert_dist(r1, r2)-1) <= 1e-6)


def test_unitarity_known():
    """
    Metrics: Unitarity for known cases.
    """
    def case(q_oper, known_unitarity):
        assert_almost_equal(unitarity(q_oper), known_unitarity)

    yield case, to_super(sigmax()), 1.0
    yield case, sum(map(
        to_super, [qeye(2), sigmax(), sigmay(), sigmaz()]
    )) / 4, 0.0
    yield case, sum(map(
        to_super, [qeye(2), sigmax()]
    )) / 2, 1 / 3.0

def test_unitarity_bounded(nq=3, n_cases=10):
    """
    Metrics: Unitarity in [0, 1].
    """
    def case(q_oper):
        assert_(0.0 <= unitarity(q_oper) <= 1.0)

    for _ in range(n_cases):
        yield case, rand_super_bcsz(2**nq)


@dnorm_test
def test_dnorm_bounded(n_cases=10):
    """
    Metrics: dnorm(A - B) in [0, 2] for random superops A, B.
    """
    def case(A, B, tol=1e-4):
        # We allow for a generous tolerance so that we don't kill off SCS.
        assert_(-tol <= dnorm(A, B) <= 2.0 + tol)

    for _ in range(n_cases):
        yield case, rand_super_bcsz(3), rand_super_bcsz(3)


@dnorm_test
def test_dnorm_qubit_known_cases():
    """
    Metrics: check agreement for known qubit channels.
    """
    def case(chan1, chan2, expected, significant=4):
        # We again take a generous tolerance so that we don't kill off
        # SCS solvers.
        assert_approx_equal(
            dnorm(chan1, chan2), expected,
            significant=significant
        )

    id_chan = to_choi(qeye(2))
    S_eye = to_super(id_chan)
    X_chan = to_choi(sigmax())
    depol = to_choi(Qobj(
        diag(ones((4,))),
        dims=[[[2], [2]], [[2], [2]]], superrep='chi'
    ))
    S_H = to_super(hadamard_transform())

    W = swap()

    # We need to restrict the number of iterations for things on the boundary,
    # such as perfectly distinguishable channels.
    yield case, id_chan, X_chan, 2
    yield case, id_chan, depol, 1.5

    # Next, we'll generate some test cases based on comparisons to pre-existing
    # dnorm() implementations. In particular, the targets for the following
    # test cases were generated using QuantumUtils for MATLAB (https://goo.gl/oWXhO9).

    def overrotation(x):
        return to_super((1j * np.pi * x * sigmax() / 2).expm())

    for x, target in {
        1.000000e-03: 3.141591e-03,
        3.100000e-03: 9.738899e-03,
        1.000000e-02: 3.141463e-02,
        3.100000e-02: 9.735089e-02,
        1.000000e-01: 3.128689e-01,
        3.100000e-01: 9.358596e-01
    }.items():
        yield case, overrotation(x), id_chan, target

    def had_mixture(x):
        return (1 - x) * S_eye + x * S_H

    for x, target in {
        1.000000e-03: 2.000000e-03,
        3.100000e-03: 6.200000e-03,
        1.000000e-02: 2.000000e-02,
        3.100000e-02: 6.200000e-02,
        1.000000e-01: 2.000000e-01,
        3.100000e-01: 6.200000e-01
    }.items():
        yield case, had_mixture(x), id_chan, target

    def swap_map(x):
        S = (1j * x * W).expm()
        S._type = None
        S.dims = [[[2], [2]], [[2], [2]]]
        S.superrep = 'super'
        return S

    for x, target in {
        1.000000e-03: 2.000000e-03,
        3.100000e-03: 6.199997e-03,
        1.000000e-02: 1.999992e-02,
        3.100000e-02: 6.199752e-02,
        1.000000e-01: 1.999162e-01,
        3.100000e-01: 6.173918e-01
    }.items():
        yield case, swap_map(x), id_chan, target

    # Finally, we add a known case from Johnston's QETLAB documentation,
    # || Phi - I ||,_♢ where Phi(X) = UXU⁺ and U = [[1, 1], [-1, 1]] / sqrt(2).
    yield case, Qobj([[1, 1], [-1, 1]]) / np.sqrt(2), qeye(2), np.sqrt(2)


@dnorm_test
def test_dnorm_qubit_scalar():
    """
    Metrics: checks that dnorm(a * A) == a * dnorm(A) for scalar a, qobj A.
    """
    def case(a, A, B, significant=5):
        assert_approx_equal(
            dnorm(a * A, a * B), a * dnorm(A, B),
            significant=significant
        )

    for dim in (2, 3):
        for _ in range(10):
            yield (
                case, np.random.random(),
                rand_super_bcsz(dim), rand_super_bcsz(dim)
            )


@dnorm_test
def test_dnorm_qubit_triangle():
    """
    Metrics: checks that dnorm(A + B) ≤ dnorm(A) + dnorm(B).
    """
    def case(A, B, tol=1e-4):
        assert (
            dnorm(A + B) <= dnorm(A) + dnorm(B) + tol
        )

    for dim in (2, 3):
        for _ in range(10):
            yield (
                case,
                rand_super_bcsz(dim), rand_super_bcsz(dim)
            )


@dnorm_test
def test_dnorm_force_solve():
    """
    Metrics: checks that special cases for dnorm agree with SDP solutions.
    """
    def case(A, B, significant=4):
        assert_approx_equal(
            dnorm(A, B, force_solve=False), dnorm(A, B, force_solve=True)
        )

    for dim in (2, 3):
        for _ in range(10):
            yield (
                case,
                rand_super_bcsz(dim), None
            )
        for _ in range(10):
            yield (
                case,
                rand_unitary_haar(dim), rand_unitary_haar(dim)
            )


@dnorm_test
def test_dnorm_cptp():
    """
    Metrics: checks that the diamond norm is one for CPTP maps.
    """
    # NB: It might be worth dropping test_dnorm_force_solve, and separating
    #     into cases for each optimization path.
    def case(A, significant=4):
        for force_solve in (False, True):
            assert_approx_equal(
                dnorm(A, force_solve=force_solve), 1
            )

    for dim in (2, 3):
        for _ in range(10):
            yield case, rand_super_bcsz(dim)


if __name__ == "__main__":
    run_module_suite()
