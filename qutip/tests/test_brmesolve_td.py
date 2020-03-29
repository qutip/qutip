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

pytestmark = [
    pytest.mark.requires_cython,
    pytest.mark.usefixtures("in_temporary_directory"),
]

# A lot of this module is direct duplication of test_brmesolve.py, but using
# string time dependence rather than functional.


def pauli_spin_operators():
    return [qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()]


_simple_qubit_gamma = 0.25
_m_c_op = np.sqrt(_simple_qubit_gamma) * qutip.sigmam()
_z_c_op = np.sqrt(_simple_qubit_gamma) * qutip.sigmaz()
_x_a_op = [qutip.sigmax(), '{0} * (w >= 0)'.format(_simple_qubit_gamma)]


@pytest.mark.slow
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
    brme = qutip.brmesolve([[H, '1']], psi0, times,
                           brme_a_ops, e_ops, brme_c_ops).expect
    for me_expectation, brme_expectation in zip(me, brme):
        np.testing.assert_allclose(me_expectation, brme_expectation, atol=1e-2)


def _harmonic_oscillator_spectrum_frequency(n_th, w0, kappa):
    if n_th == 0:
        return "{kappa} * (w >= 0)".format(kappa=kappa)
    w_th = w0 / np.log(1 + 1/n_th)
    scale = "((w<0)*exp(w/{w_th}) + (w>=0))".format(w_th=w_th)
    return "({n_th}+1) * {kappa} * {scale}".format(
            n_th=n_th, kappa=kappa, scale=scale)


def _harmonic_oscillator_c_ops(n_th, kappa, dimension):
    a = qutip.destroy(dimension)
    if n_th == 0:
        return [np.sqrt(kappa) * a]
    return [np.sqrt(kappa * (n_th+1)) * a,
            np.sqrt(kappa * n_th) * a.dag()]


@pytest.mark.slow
@pytest.mark.parametrize("n_th", [0, 1.5])
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


@pytest.mark.slow
def test_jaynes_cummings_zero_temperature():
    N = 10
    a = qutip.tensor(qutip.destroy(N), qutip.qeye(2))
    sp = qutip.tensor(qutip.qeye(N), qutip.sigmap())
    psi0 = qutip.ket2dm(qutip.tensor(qutip.basis(N, 1), qutip.basis(2, 0)))
    kappa = 0.05
    a_ops = [[(a + a.dag()), "{kappa} * (w >= 0)".format(kappa=kappa)]]
    e_ops = [a.dag()*a, sp.dag()*sp]

    w0 = 1.0 * 2*np.pi
    g = 0.05 * 2*np.pi
    times = np.linspace(0, 2 * 2*np.pi / g, 1000)

    c_ops = [np.sqrt(kappa) * a]
    H = w0*a.dag()*a + w0*sp.dag()*sp + g*(a+a.dag())*(sp+sp.dag())

    me = qutip.mesolve(H, psi0, times, c_ops, e_ops)
    brme = qutip.brmesolve(H, psi0, times, a_ops, e_ops)
    for me_expectation, brme_expectation in zip(me.expect, brme.expect):
        # Accept 5% error.
        np.testing.assert_allclose(me_expectation, brme_expectation, atol=5e-2)


def _mixed_string(kappa, _):
    return "{kappa} * exp(-t) * (w >= 0)".format(kappa=kappa)


def _separate_strings(kappa, _):
    return ("{kappa} * (w >= 0)".format(kappa=kappa), "exp(-t)")


def _string_w_interpolating_t(kappa, times):
    spline = qutip.Cubic_Spline(times[0], times[-1], np.exp(-times))
    return ("{kappa} * (w >= 0)".format(kappa=kappa), spline)


@pytest.mark.slow
@pytest.mark.parametrize("time_dependence_tuple", [
        _mixed_string,
        _separate_strings,
        _string_w_interpolating_t,
    ])
def test_time_dependence_tuples(time_dependence_tuple):
    N = 10
    a = qutip.destroy(N)
    H = a.dag()*a
    psi0 = qutip.basis(N, 9)
    times = np.linspace(0, 10, 100)
    kappa = 0.2
    a_ops = [[a + a.dag(), time_dependence_tuple(kappa, times)]]
    exact = 9 * np.exp(-kappa * (1 - np.exp(-times)))
    brme = qutip.brmesolve(H, psi0, times, a_ops, e_ops=[a.dag()*a])
    assert np.mean(np.abs(brme.expect[0] - exact) / exact) < 1e-5


def test_time_dependent_spline_in_c_ops():
    N = 10
    a = qutip.destroy(N)
    H = a.dag()*a
    psi0 = qutip.basis(N, 9)
    times = np.linspace(0, 10, 100)
    kappa = 0.2
    exact = 9 * np.exp(-2 * kappa * (1 - np.exp(-times)))
    a_ops = [[a + a.dag(), _string_w_interpolating_t(kappa, times)]]
    collapse_points = np.sqrt(kappa) * np.exp(-0.5*times)
    c_ops = [[a, qutip.Cubic_Spline(times[0], times[-1], collapse_points)]]
    brme = qutip.brmesolve(H, psi0, times,
                           a_ops, e_ops=[a.dag()*a], c_ops=c_ops)
    assert np.mean(np.abs(brme.expect[0] - exact) / exact) < 1e-5


@pytest.mark.slow
def test_nonhermitian_e_ops():
    N = 5
    a = qutip.destroy(N)
    coefficient = np.random.random() + 1j*np.random.random()
    H = a.dag()*a + coefficient*a + np.conj(coefficient)*a.dag()
    H_brme = [[H, '1']]
    psi0 = qutip.basis(N, 2)
    times = np.linspace(0, 10, 10)
    me = qutip.mesolve(H, psi0, times, c_ops=[], e_ops=[a]).expect[0]
    brme = qutip.brmesolve(H_brme, psi0, times, a_ops=[], e_ops=[a]).expect[0]
    np.testing.assert_allclose(me, brme, atol=1e-4)


@pytest.mark.slow
def test_result_states():
    N = 5
    a = qutip.destroy(N)
    coefficient = np.random.random() + 1j*np.random.random()
    H = a.dag()*a + coefficient*a + np.conj(coefficient)*a.dag()
    H_brme = [[H, '1']]
    psi0 = qutip.fock_dm(N, 2)
    times = np.linspace(0, 10, 10)
    me = qutip.mesolve(H, psi0, times).states
    brme = qutip.brmesolve(H_brme, psi0, times).states
    assert max(np.abs((me_state - brme_state).full()).max()
               for me_state, brme_state in zip(me, brme)) < 1e-5


def _2_tuple_split(dimension, kappa, _):
    a = qutip.destroy(dimension)
    spectrum = "{kappa} * (w >= 0)".format(kappa=kappa)
    return ([np.sqrt(kappa)*a, np.sqrt(kappa)*a],
            [],
            [[a + a.dag(), spectrum],
             [(a, a.dag()), (spectrum, '1', '1')]])


def _4_tuple_split(dimension, kappa, _):
    a = qutip.destroy(dimension)
    spectrum = "{kappa} * (w >= 0)".format(kappa=kappa)
    return ([np.sqrt(kappa)*a, np.sqrt(4*kappa)*a],
            [],
            [[a + a.dag(), spectrum],
             [(a, a.dag(), a, a.dag()), (spectrum, '1', '1', '1', '1')]])


def _2_tuple_splines(dimension, kappa, times):
    a = qutip.destroy(dimension)
    spectrum = "{kappa} * (w >= 0)".format(kappa=kappa)
    spline = qutip.Cubic_Spline(times[0], times[-1], np.ones_like(times))
    return ([np.sqrt(kappa)*a, np.sqrt(kappa)*a, np.sqrt(kappa)*a],
            [np.sqrt(kappa)*a],
            [[a + a.dag(), spectrum],
             [(a, a.dag()), (spectrum, spline, spline)]])


def _2_list_entries_2_tuple_split(dimension, kappa, _):
    a = qutip.destroy(dimension)
    spectrum = "{kappa} * (w >= 0)".format(kappa=kappa)
    return ([np.sqrt(kappa)*a, np.sqrt(kappa)*a, np.sqrt(kappa)*a],
            [],
            [[a + a.dag(), spectrum],
             [(a, a.dag()), (spectrum, '1', '1')],
             [(a, a.dag()), (spectrum, '1', '1')]])


@pytest.mark.parametrize("collapse_operators", [
        _2_tuple_split,
        pytest.param(_4_tuple_split, marks=pytest.mark.slow),
        pytest.param(_2_tuple_splines, marks=pytest.mark.slow),
        pytest.param(_2_list_entries_2_tuple_split, marks=pytest.mark.slow),
    ])
def test_split_operators_maintain_answer(collapse_operators):
    N = 10
    w0 = 1.0 * 2*np.pi
    g = 0.05 * w0
    kappa = 0.15

    a = qutip.destroy(N)
    H = w0*a.dag()*a + g*(a+a.dag())
    psi0 = (qutip.basis(N, 4) + qutip.basis(N, 2) + qutip.basis(N, 0)).unit()
    psi0 = qutip.ket2dm(psi0)
    times = np.linspace(0, 25, 1000)
    e_ops = [a.dag()*a, a+a.dag()]

    me_c_ops, brme_c_ops, a_ops = collapse_operators(N, kappa, times)
    me = qutip.mesolve(H, psi0, times, me_c_ops, e_ops)
    brme = qutip.brmesolve(H, psi0, times, a_ops, e_ops, brme_c_ops)

    for me_expect, brme_expect in zip(me.expect, brme.expect):
        np.testing.assert_allclose(me_expect, brme_expect, atol=1e-2)


@pytest.mark.slow
def test_hamiltonian_taking_arguments():
    N = 10
    w0 = 1.0 * 2*np.pi
    g = 0.75 * 2*np.pi
    kappa = 0.05
    a = qutip.tensor(qutip.destroy(N), qutip.qeye(2))
    sp = qutip.tensor(qutip.qeye(N), qutip.sigmap())
    psi0 = qutip.tensor(qutip.basis(N, 1), qutip.basis(2, 0))
    psi0 = qutip.ket2dm(psi0)
    times = np.linspace(0, 5 * 2*np.pi / g, 1000)

    a_ops = [[(a + a.dag()), "{kappa}*(w > 0)".format(kappa=kappa)]]
    e_ops = [a.dag()*a, sp.dag()*sp]

    H = w0*a.dag()*a + w0*sp.dag()*sp + g*(a+a.dag())*(sp+sp.dag())
    args = {'ii': 1}

    no_args = qutip.brmesolve(H, psi0, times, a_ops, e_ops)
    args = qutip.brmesolve([[H, 'ii']], psi0, times, a_ops, e_ops, args=args)
    for arg, no_arg in zip(args.expect, no_args.expect):
        assert np.array_equal(arg, no_arg)
