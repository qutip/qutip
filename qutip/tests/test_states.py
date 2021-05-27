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

import pytest
import numpy as np
import qutip
from functools import partial
from itertools import combinations


_2PI = 2 * np.pi
N = 5


@pytest.mark.parametrize("size, n", [(2, 0), (2, 1), (100, 99)])
def test_basis_simple(size, n):
    qobj = qutip.basis(size, n)
    numpy = np.zeros((size, 1), dtype=complex)
    numpy[n, 0] = 1
    assert np.array_equal(qobj.full(), numpy)


@pytest.mark.parametrize("to_test", [qutip.basis, qutip.fock, qutip.fock_dm])
@pytest.mark.parametrize("size, n", [([2, 2], [0, 1]), ([2, 3, 4], [1, 2, 0])])
def test_implicit_tensor_basis_like(to_test, size, n):
    implicit = to_test(size, n)
    explicit = qutip.tensor(*[to_test(ss, nn) for ss, nn in zip(size, n)])
    assert implicit == explicit


@pytest.mark.parametrize("size, n, m", [
        ([2, 2], [0, 0], [1, 1]),
        ([2, 3, 4], [1, 2, 0], [0, 1, 3]),
    ])
def test_implicit_tensor_projection(size, n, m):
    implicit = qutip.projection(size, n, m)
    explicit = qutip.tensor(*[qutip.projection(ss, nn, mm)
                              for ss, nn, mm in zip(size, n, m)])
    assert implicit == explicit


@pytest.mark.parametrize("base, operator, args, opargs, eigenval", [
    pytest.param(qutip.basis, qutip.num, (10, 3), (10,), 3,
                 id="basis"),
    pytest.param(qutip.basis, qutip.num, (10, 3, 1), (10, 1), 3,
                 id="basis_offset"),
    pytest.param(qutip.fock, qutip.num, (10, 3), (10,), 3,
                 id="fock"),
    pytest.param(qutip.fock_dm, qutip.num, (10, 3), (10,), 3,
                 id="fock_dm"),
    pytest.param(qutip.fock_dm, qutip.num, (10, 3, 1), (10, 1), 3,
                 id="fock_dm_offset"),
    pytest.param(qutip.coherent, qutip.destroy, (20, 0.75), (20,), 0.75,
                 id="coherent"),
    pytest.param(qutip.coherent, qutip.destroy, (50, 4.25, 1), (50, 1), 4.25,
                 id="coherent_offset"),
    pytest.param(qutip.coherent_dm, qutip.destroy, (25, 1.25), (25,), 1.25,
                 id="coherent_dm"),
    pytest.param(qutip.phase_basis, qutip.phase, (10, 3), (10,), 3 * _2PI / 10,
                 id="phase_basis"),
    pytest.param(qutip.phase_basis, qutip.phase,
                 (10, 3, 1), (10, 1), 3 * _2PI / 10 + 1,
                 id="phase_basis_phi0"),
    pytest.param(qutip.spin_state, qutip.spin_Jz, (3, 2), (3,), 2,
                 id="spin_state"),
    pytest.param(qutip.zero_ket, qutip.qeye, (10,), (10,), 0,
                 id="zero_ket"),
])
def test_diverse_basis(base, operator, args, opargs, eigenval):
    # For state which are supposed to eigenvector of an operator
    # Verify that correspondance
    state = base(*args)
    oper = operator(*opargs)
    assert qutip.expect(oper, state) == pytest.approx(eigenval)


@pytest.mark.parametrize('dm', [
    partial(qutip.thermal_dm, n=1.),
    qutip.maximally_mixed_dm,
    partial(qutip.coherent_dm, alpha=0.5),
    partial(qutip.fock_dm, n=1),
    partial(qutip.spin_state, m=2, type='dm'),
    partial(qutip.spin_coherent, theta=1, phi=2, type='dm'),
], ids=[
    'thermal_dm', 'maximally_mixed_dm', 'coherent_dm',
    'fock_dm', 'spin_state', 'spin_coherent'
])
def test_dm(dm):
    N = 5
    rho = dm(N)
    # make sure rho has trace close to 1.0
    assert abs(rho.tr() - 1.0) < 1e-12


def test_CoherentState():
    N = 10
    alpha = 0.5
    c1 = qutip.coherent(N, alpha)  # displacement method
    c2 = qutip.coherent(7, alpha, offset=3)  # analytic method
    assert abs(qutip.expect(qutip.destroy(N), c1) - alpha) < 1e-10
    assert (qutip.Qobj(c1[3:]) - c2).norm() < 1e-7


def test_TripletStateNorm():
    for triplet in qutip.triplet_states():
        assert abs(triplet.norm() - 1.) < 1e-12


def test_ket2dm():
    N = 5
    state = qutip.coherent(N, 2)
    oper = qutip.ket2dm(state)
    assert np.abs(qutip.expect(oper, state) - 1) < 1e-12


@pytest.mark.parametrize('state', [[0, 1], [0, 0], [0, 1, 0, 1]])
def test_qstate(state):
    from_basis = qutip.basis([2] * len(state), state)
    from_qstate = qutip.qstate("".join({0: "d", 1: "u"}[i] for i in state))
    assert from_basis == from_qstate


@pytest.mark.parametrize('state', ["11000", "eeggg", "dduuu", "VVHHH"])
def test_bra_ket(state):
    from_basis = qutip.basis([2, 2, 2, 2, 2], [1, 1, 0, 0, 0])
    from_ket = qutip.ket(state)
    from_bra = qutip.bra(state).dag()
    assert from_basis == from_ket
    assert from_basis == from_bra


def test_w_states():
    state = (
        qutip.qstate("uddd") +
        qutip.qstate("dudd") +
        qutip.qstate("ddud") +
        qutip.qstate("dddu")
    ) / 2
    assert state == qutip.w_state(4)


def test_ghz_states():
    state = (qutip.qstate("uuu") + qutip.qstate("ddd")) * 0.5**0.5
    assert state == qutip.ghz_state(3)


def test_bell_state():
    states = [
        qutip.bell_state('00'),
        qutip.bell_state('01'),
        qutip.bell_state('10'),
        qutip.bell_state('11')
    ]
    exited = qutip.basis([2, 2], [1, 1])
    for state, overlap in zip(states, [0.5**0.5, -0.5**0.5, 0, 0]):
        assert state.norm() == pytest.approx(1.0)
        assert state.overlap(exited) == pytest.approx(overlap)

    for state1, state2 in combinations(states, 2):
        assert state1.overlap(state2) == pytest.approx(0.0)

    assert qutip.singlet_state() == qutip.bell_state('11')
