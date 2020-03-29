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
import pytest
import qutip


def pauli_spin_operators():
    return [qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()]


_simple_qubit_gamma = 0.25
_m_c_op = np.sqrt(_simple_qubit_gamma) * qutip.sigmam()
_z_c_op = np.sqrt(_simple_qubit_gamma) * qutip.sigmaz()
_x_a_op = [qutip.sigmax(), lambda w: _simple_qubit_gamma * (w >= 0)]


@pytest.mark.parametrize("me_c_ops, brme_c_ops, brme_a_ops", [
    pytest.param([_m_c_op], [], [_x_a_op], id="me collapse-br coupling"),
    pytest.param([_m_c_op], [_m_c_op], [], id="me collapse-br collapse"),
    pytest.param([_m_c_op, _z_c_op], [_z_c_op], [_x_a_op],
                 id="me collapse-br collapse-br coupling"),
])
def test_simple_qubit_system(me_c_ops, brme_c_ops, brme_a_ops):
    """
    Test that the BR solver handles collapse and coupling operators correctly
    relative to the standard ME solver.
    """
    delta = 0.0 * 2*np.pi
    epsilon = 0.5 * 2*np.pi
    e_ops = pauli_spin_operators()
    H = delta*0.5*qutip.sigmax() + epsilon*0.5*qutip.sigmaz()
    psi0 = (2*qutip.basis(2, 0) + qutip.basis(2, 1)).unit()
    times = np.linspace(0, 10, 100)
    me = qutip.mesolve(H, psi0, times, c_ops=me_c_ops, e_ops=e_ops).expect
    brme = qutip.brmesolve(H, psi0, times,
                           brme_a_ops, e_ops, brme_c_ops).expect
    for me_expectation, brme_expectation in zip(me, brme):
        np.testing.assert_allclose(me_expectation, brme_expectation, atol=1e-2)


def _harmonic_oscillator_spectrum_frequency(n_th, w0, kappa):
    if n_th == 0:
        return lambda w: kappa * (w >= 0)
    w_th = w0 / np.log(1 + 1/n_th)

    def f(w):
        scale = np.exp(w / w_th) if w < 0 else 1
        return (n_th + 1) * kappa * scale
    return f


def _harmonic_oscillator_c_ops(n_th, kappa, dimension):
    a = qutip.destroy(dimension)
    if n_th == 0:
        return [np.sqrt(kappa) * a]
    return [np.sqrt(kappa * (n_th+1)) * a,
            np.sqrt(kappa * n_th) * a.dag()]


@pytest.mark.parametrize("n_th", [0, 0.15])
def test_harmonic_oscillator(n_th):
    N = 10
    w0 = 1.0 * 2*np.pi
    g = 0.05 * w0
    kappa = 0.15
    S_w = _harmonic_oscillator_spectrum_frequency(n_th, w0, kappa)

    a = qutip.destroy(N)
    H = w0*a.dag()*a + g*(a+a.dag())
    psi0 = (qutip.basis(N, 4) + qutip.basis(N, 2) + qutip.basis(N, 0)).unit()
    psi0 = qutip.ket2dm(psi0)
    times = np.linspace(0, 25, 1000)

    c_ops = _harmonic_oscillator_c_ops(n_th, kappa, N)
    a_ops = [[a + a.dag(), S_w]]
    e_ops = [a.dag()*a, a+a.dag()]

    me = qutip.mesolve(H, psi0, times, c_ops, e_ops)
    brme = qutip.brmesolve(H, psi0, times, a_ops, e_ops)
    for me_expectation, brme_expectation in zip(me.expect, brme.expect):
        np.testing.assert_allclose(me_expectation, brme_expectation, atol=1e-2)

    num = qutip.num(N)
    me_num = qutip.expect(num, me.states)
    brme_num = qutip.expect(num, brme.states)
    np.testing.assert_allclose(me_num, brme_num, atol=1e-2)


def test_jaynes_cummings_zero_temperature():
    """
    brmesolve: Jaynes-Cummings model, zero temperature
    """
    N = 10
    a = qutip.tensor(qutip.destroy(N), qutip.qeye(2))
    sp = qutip.tensor(qutip.qeye(N), qutip.sigmap())
    psi0 = qutip.ket2dm(qutip.tensor(qutip.basis(N, 1), qutip.basis(2, 0)))
    a_ops = [[(a + a.dag()), lambda w: kappa * (w >= 0)]]
    e_ops = [a.dag()*a, sp.dag()*sp]

    w0 = 1.0 * 2*np.pi
    g = 0.05 * 2*np.pi
    kappa = 0.05
    times = np.linspace(0, 2 * 2*np.pi / g, 1000)

    c_ops = [np.sqrt(kappa) * a]
    H = w0*a.dag()*a + w0*sp.dag()*sp + g*(a+a.dag())*(sp+sp.dag())

    me = qutip.mesolve(H, psi0, times, c_ops, e_ops)
    brme = qutip.brmesolve(H, psi0, times, a_ops, e_ops)
    for me_expectation, brme_expectation in zip(me.expect, brme.expect):
        # Accept 5% error.
        np.testing.assert_allclose(me_expectation, brme_expectation, atol=5e-2)


def test_pull_572_error():
    """
    brmesolve: Check for #572 bug.
    """
    w1, w2, w3 = 1, 2, 3
    gamma2, gamma3 = 0.1, 0.1
    id2 = qutip.qeye(2)

    # Hamiltonian for three uncoupled qubits
    H = (w1/2. * qutip.tensor(qutip.sigmaz(), id2, id2)
         + w2/2. * qutip.tensor(id2, qutip.sigmaz(), id2)
         + w3/2. * qutip.tensor(id2, id2, qutip.sigmaz()))

    # White noise
    def S2(w):
        return gamma2

    def S3(w):
        return gamma3

    qubit_2_x = qutip.tensor(id2, qutip.sigmax(), id2)
    qubit_3_x = qutip.tensor(id2, id2, qutip.sigmax())
    # Bloch-Redfield tensor including dissipation for qubits 2 and 3 only
    R, ekets = qutip.bloch_redfield_tensor(H,
                                           [[qubit_2_x, S2], [qubit_3_x, S3]])
    # Initial state : first qubit is excited
    grnd2 = qutip.sigmam() * qutip.sigmap()  # 2x2 ground
    exc2 = qutip.sigmap() * qutip.sigmam()   # 2x2 excited state
    ini = qutip.tensor(exc2, grnd2, grnd2)   # Full system

    # Projector on the excited state of qubit 1
    proj_up1 = qutip.tensor(exc2, id2, id2)

    # Solution of the master equation
    times = np.linspace(0, 10./gamma3, 1000)
    sol = qutip.bloch_redfield_solve(R, ekets, ini, times, [proj_up1])
    np.testing.assert_allclose(sol[0], np.ones_like(times))


def test_solver_accepts_list_hamiltonian():
    """
    brmesolve: input list of Qobj
    """
    delta = 0.0 * 2*np.pi
    epsilon = 0.5 * 2*np.pi
    gamma = 0.25
    c_ops = [np.sqrt(gamma) * qutip.sigmam()]
    e_ops = pauli_spin_operators()
    H = [delta*0.5*qutip.sigmax(), epsilon*0.5*qutip.sigmaz()]
    psi0 = (2*qutip.basis(2, 0) + qutip.basis(2, 1)).unit()
    times = np.linspace(0, 10, 100)
    me = qutip.mesolve(H, psi0, times, c_ops=c_ops, e_ops=e_ops).expect
    brme = qutip.brmesolve(H, psi0, times, [], e_ops, c_ops).expect
    for me_expectation, brme_expectation in zip(me, brme):
        np.testing.assert_allclose(me_expectation, brme_expectation, atol=1e-8)
