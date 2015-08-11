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
import numpy as np
from numpy import abs, sqrt
from numpy.testing import assert_, assert_almost_equal, run_module_suite
import scipy

from qutip.operators import (
    create, destroy, jmat, identity, qdiags, sigmax, sigmay, sigmaz, qeye
)
from qutip.states import fock_dm
from qutip.propagator import propagator
from qutip.random_objects import (
    rand_herm, rand_dm, rand_unitary, rand_ket, rand_super_bcsz
)
from qutip.qobj import Qobj
from qutip.superop_reps import to_super
from qutip.metrics import *

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


def rand_super():
    h_5 = rand_herm(5)
    return propagator(h_5, scipy.rand(), [
        create(5), destroy(5), jmat(2, 'z')
    ])


def test_average_gate_fidelity():
    """
    Metrics: Check avg gate fidelities for random
    maps (equal to 1 for id maps).
    """
    for dims in range(2, 5):
        assert_(abs(average_gate_fidelity(identity(dims)) - 1) <= 1e-12)
    assert_(0 <= average_gate_fidelity(rand_super()) <= 1)


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


if __name__ == "__main__":
    run_module_suite()
