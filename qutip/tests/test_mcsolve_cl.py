# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    Copyright (c) 2016 and later, Christian Wasserthal.
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
from qutip import *
from numpy.testing import assert_equal, run_module_suite, assert_

# average error for failure
mc_error = 5e-2  # 5%
ntraj = 750


def test_mcsolve_cl_const_h():
    "mcsolve_cl: Constant H with no collapse ops (expect)"
    error = 1e-8
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 9)  # initial state
    c_op_list = []
    tlist = np.linspace(0, 10, 100)
    mcdata = mcsolve_cl(H, psi0, tlist, c_op_list, [a.dag() * a], ntraj=ntraj)
    expt = mcdata.expect[0]
    actual_answer = 9.0 * np.ones(len(tlist))
    diff = np.mean(abs(actual_answer - expt) / actual_answer)
    assert_equal(diff < error, True)


def test_mcsolve_cl_multiple_const_h():
    "mcsolve_cl: Two constant H with no collapse ops (expect)"
    tlist = np.linspace(0, 10, 100)
    psi0 = tensor(fock(2, 0), fock(10, 5))
    a = tensor(qeye(2), destroy(10))
    sm = tensor(destroy(2), qeye(10))
    H1 = 2 * np.pi * a.dag() * a + 2 * np.pi * sm.dag() * sm
    H2 = 2 * np.pi * 0.25 * (sm * a.dag() + sm.dag() * a)
    c_ops = [np.sqrt(0.1) * a]
    e_ops = [a.dag() * a, sm.dag() * sm]
    mcdata = mcsolve_cl([H1, H2], psi0, tlist, c_ops, e_ops, ntraj=ntraj)
    medata = mesolve(H1 + H2, psi0, tlist, c_ops, e_ops)
    expt = mcdata.expect[0]
    actual_answer = medata.expect[0]
    diff = np.mean(abs(actual_answer - expt) / actual_answer)
    assert_equal(diff < mc_error, True)
    expt = mcdata.expect[1]
    actual_answer = medata.expect[1]
    diff = np.mean(abs(actual_answer - expt) / 1.0)
    assert_equal(diff < mc_error, True)


def test_mcsolve_cl_time_dep_h():
    "mcsolve_cl: Time-dependent H with no collapse ops (expect)"
    tlist = np.linspace(0.0, 10.0, 200)
    psi0 = basis(2, 0)
    H1 = sigmax()
    H2 = sigmay()
    H = [H1, [H2, 'sin(omega*t)']]
    c_ops = []
    e_ops = [sigmaz()]
    mcdata = mcsolve_cl(H, psi0, tlist, c_ops, e_ops, args={'omega': 0.3})
    medata = mesolve(H, psi0, tlist, c_ops, e_ops, args={'omega': 0.3})
    expt = mcdata.expect[0]
    actual_answer = medata.expect[0]
    diff = np.mean(abs(actual_answer - expt) / 1.0)
    assert_equal(diff < mc_error, True)


def test_mcsolve_cl_time_dep_h_complex():
    "mcsolve_cl: Complex valued time-dependent H with no collapse ops (expect)"
    tlist = np.linspace(0.0, 10.0, 200)
    psi0 = basis(2, 0)
    H1 = sigmax()
    H2 = sigmay()
    c_ops = []
    e_ops = [sigmaz()]
    mcdata = mcsolve_cl([H1, [H2, '1', 'sin(omega*t)']], psi0, tlist, c_ops,
                        e_ops, args={'omega': 0.4})
    medata = mesolve([H1, [H2, '1+1j*sin(omega*t)']], psi0, tlist, c_ops,
                     e_ops, args={'omega': 0.4})
    expt = mcdata.expect[0]
    actual_answer = medata.expect[0]
    diff = np.mean(abs(actual_answer - expt) / 1.0)
    assert_equal(diff < mc_error, True)


def test_mcsolve_cl_output_states():
    "mcsolve_cl: Constant H with no collapse ops (states)"
    error = 1e-8
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 9)  # initial state
    tlist = np.linspace(0, 10, 100)
    mcdata = mcsolve_cl(H, psi0, tlist, [], [], ntraj=ntraj)
    states = mcdata.states
    expt = expect(a.dag() * a, states)
    actual_answer = 9.0 * np.ones(len(tlist))
    diff = np.mean(abs(actual_answer - expt) / actual_answer)
    assert_equal(diff < error, True)


def test_mcsolve_cl_expect_and_states():
    "mcsolve_cl: Constant H with no collapse ops (expect and states)"
    error = 1e-8
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 9)  # initial state
    tlist = np.linspace(0, 10, 100)
    mcdata = mcsolve_cl(H, psi0, tlist, [], [a.dag() * a], ntraj=ntraj,
                        options=Options(store_states=True))
    actual_answer = 9.0 * np.ones(len(tlist))
    expt = mcdata.expect[0]
    diff = np.mean(abs(actual_answer - expt) / actual_answer)
    assert_equal(diff < error, True)
    assert_(len(mcdata.states) == len(tlist))
    assert_(isinstance(mcdata.states[0], Qobj))
    expt = expect(a.dag() * a, mcdata.states)
    diff = np.mean(abs(actual_answer - expt) / actual_answer)
    assert_equal(diff < error, True)


def test_mcsolve_cl_simple_collapse():
    "mcsolve_cl: Constant H with constant collapse (expect)"
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 9)  # initial state
    kappa = 0.2  # coupling to oscillator
    c_op_list = [np.sqrt(kappa) * a]
    tlist = np.linspace(0, 10, 100)
    mcdata = mcsolve_cl(H, psi0, tlist, c_op_list, [a.dag() * a], ntraj=ntraj)
    expt = mcdata.expect[0]
    actual_answer = 9.0 * np.exp(-kappa * tlist)
    avg_diff = np.mean(abs(actual_answer - expt) / actual_answer)
    assert_equal(avg_diff < mc_error, True)


def test_mcsolve_cl_simple_collapse_states():
    "mcsolve_cl: Constant H with constant collapse (states)"
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 9)  # initial state
    kappa = 0.2  # coupling to oscillator
    c_op_list = [np.sqrt(kappa) * a]
    tlist = np.linspace(0, 10, 100)
    mcdata = mcsolve_cl(H, psi0, tlist, c_op_list, [], ntraj=ntraj,
                        options=Options(average_states=True))
    assert_(len(mcdata.states) == len(tlist))
    assert_(isinstance(mcdata.states[0], Qobj))
    expt = expect(a.dag() * a, mcdata.states)
    actual_answer = 9.0 * np.exp(-kappa * tlist)
    avg_diff = np.mean(abs(actual_answer - expt) / actual_answer)
    assert_equal(avg_diff < mc_error, True)


def test_mcsolve_cl_simple_collapse_expect_and_states():
    "mcsolve_cl: Constant H with constant collapse (states)"
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 9)  # initial state
    kappa = 0.2  # coupling to oscillator
    c_op_list = [np.sqrt(kappa) * a]
    tlist = np.linspace(0, 10, 100)
    mcdata = mcsolve_cl(
        H, psi0, tlist, c_op_list, [a.dag() * a], ntraj=ntraj,
        options=Options(average_states=True, store_states=True)
    )
    actual_answer = 9.0 * np.exp(-kappa * tlist)
    expt1 = mcdata.expect[0]
    avg_diff = np.mean(abs(actual_answer - expt1) / actual_answer)
    assert_equal(avg_diff < mc_error, True)
    assert_(len(mcdata.states) == len(tlist))
    assert_(isinstance(mcdata.states[0], Qobj))
    expt2 = expect(a.dag() * a, mcdata.states)
    avg_diff = np.mean(abs(actual_answer - expt2) / actual_answer)
    assert_equal(avg_diff < mc_error, True)


def test_mcsolve_cl_qobj_collaps():
    """mcsolve_cl: Constant H with single collapse operator (not as list)"""
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 9)  # initial state
    kappa = 0.2  # coupling to oscillator
    c_op_list = np.sqrt(kappa) * a
    tlist = np.linspace(0, 10, 100)
    mcdata = mcsolve_cl(H, psi0, tlist, c_op_list, [a.dag() * a], ntraj=ntraj)
    expt = mcdata.expect[0]
    actual_answer = 9.0 * np.exp(-kappa * tlist)
    avg_diff = np.mean(abs(actual_answer - expt) / actual_answer)
    assert_equal(avg_diff < mc_error, True)


def test_mcsolve_cl_qobj_expect():
    """mcsolve_cl: Constant H with single expect operator (not as list)"""
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 9)  # initial state
    kappa = 0.2  # coupling to oscillator
    c_op_list = [np.sqrt(kappa) * a]
    tlist = np.linspace(0, 10, 100)
    mcdata = mcsolve_cl(H, psi0, tlist, c_op_list, a.dag() * a, ntraj=ntraj)
    expt = mcdata.expect[0]
    actual_answer = 9.0 * np.exp(-kappa * tlist)
    avg_diff = np.mean(abs(actual_answer - expt) / actual_answer)
    assert_equal(avg_diff < mc_error, True)


def test_mcsolve_cl_time_dep_collaps():
    """mcsolve_cl: Constant H, single time-dependent collapse operator"""
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 9)  # initial state
    kappa = 0.1  # coupling to oscillator
    c_op_list = [[a, 'sqrt(kappa*t)']]
    tlist = np.linspace(0, 10, 100)
    mcdata = mcsolve_cl(H, psi0, tlist, c_op_list, [a.dag() * a], ntraj=ntraj,
                        args={'kappa': kappa})
    medata = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a],
                     args={'kappa': kappa})
    expt = mcdata.expect[0]
    actual_answer = medata.expect[0]
    avg_diff = np.mean(abs(actual_answer - expt) / actual_answer)
    assert_equal(avg_diff < mc_error, True)


def test_mcsolve_cl_twin_peaks():
    """mcsolve_cl: Constant H, one time-dependent collapse operator"""
    N = 20
    a = destroy(N)
    H = a.dag() * a
    args = {'kappa': 0.2, 'omega': 0.5}
    psi0 = basis(N, 5)
    c_op_list = [a.dag() * 0.1, [a, 'kappa*sin(omega*t)'], qeye(N)]
    tlist = np.linspace(0, 10, 100)
    mcdata = mcsolve_cl(H, psi0, tlist, c_op_list, [a.dag() * a], ntraj=ntraj,
                        args=args)
    medata = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a], args=args)
    diff = abs(medata.expect[0] - mcdata.expect[0]) / medata.expect[0]
    assert_equal(np.mean(diff) < mc_error, True)


def test_mcsolve_cl_atol():
    """mcsolve_cl: Constant H, two collaps operators, fails with high atol."""
    N = 20  # number of basis states to consider
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 5)  # initial state
    c_op_list = [a.dag() * 0.2, a]
    tlist = np.linspace(0, 10, 100)
    mcdata = mcsolve_cl(H, psi0, tlist, c_op_list, [a.dag() * a], ntraj=ntraj,
                        options=Options(atol=1e-15, nsteps=4000))
    medata = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a])
    expt = mcdata.expect[0]
    actual_answer = medata.expect[0]
    avg_diff = np.mean(abs(actual_answer - expt) / 1.0)
    assert_equal(avg_diff < mc_error, True)


def test_mcsolve_cl_dtypes1():
    "mcsolve_cl: check for correct dtypes (average_states=True)"
    # set system parameters
    kappa = 2.0  # mirror coupling
    gamma = 0.2  # spontaneous emission rate
    g = 1  # atom/cavity coupling strength
    wc = 0  # cavity frequency
    w0 = 0  # atom frequency
    wl = 0  # driving frequency
    E = 0.5  # driving amplitude
    N = 5  # number of cavity energy levels (0->3 Fock states)
    tlist = np.linspace(0, 10, 5)  # times for expectation values
    # construct Hamiltonian
    ida = qeye(N)
    idatom = qeye(2)
    a = tensor(destroy(N), idatom)
    sm = tensor(ida, sigmam())
    H = (w0 - wl) * sm.dag() * sm + (wc - wl) * a.dag() * a + \
        1j * g * (a.dag() * sm - sm.dag() * a) + E * (a.dag() + a)
    # collapse operators
    C1 = np.sqrt(2 * kappa) * a
    C2 = np.sqrt(gamma) * sm
    C1dC1 = C1.dag() * C1
    C2dC2 = C2.dag() * C2
    # intial state
    psi0 = tensor(basis(N, 0), basis(2, 1))
    data = mcsolve_cl(H, psi0, tlist, [C1, C2], [C1dC1, C2dC2, a], ntraj=5,
                      options=Options(average_expect=True))
    assert_equal(isinstance(data.expect[0][1], float), True)
    assert_equal(isinstance(data.expect[1][1], float), True)
    assert_equal(isinstance(data.expect[2][1], complex), True)


def test_mcsolve_cl_dtypes2():
    "mcsolve_cl: check for correct dtypes (average_states=False)"
    # set system parameters
    kappa = 2.0  # mirror coupling
    gamma = 0.2  # spontaneous emission rate
    g = 1  # atom/cavity coupling strength
    wc = 0  # cavity frequency
    w0 = 0  # atom frequency
    wl = 0  # driving frequency
    E = 0.5  # driving amplitude
    N = 5  # number of cavity energy levels (0->3 Fock states)
    tlist = np.linspace(0, 10, 5)  # times for expectation values
    # construct Hamiltonian
    ida = qeye(N)
    idatom = qeye(2)
    a = tensor(destroy(N), idatom)
    sm = tensor(ida, sigmam())
    H = (w0 - wl) * sm.dag() * sm + (wc - wl) * a.dag() * a + \
        1j * g * (a.dag() * sm - sm.dag() * a) + E * (a.dag() + a)
    # collapse operators
    C1 = np.sqrt(2 * kappa) * a
    C2 = np.sqrt(gamma) * sm
    C1dC1 = C1.dag() * C1
    C2dC2 = C2.dag() * C2
    # intial state
    psi0 = tensor(basis(N, 0), basis(2, 1))
    data = mcsolve_cl(H, psi0, tlist, [C1, C2], [C1dC1, C2dC2, a], ntraj=5,
                      options=Options(average_expect=False))
    assert_equal(isinstance(data.expect[0][0][1], float), True)
    assert_equal(isinstance(data.expect[0][1][1], float), True)
    assert_equal(isinstance(data.expect[0][2][1], complex), True)


def test_mcsolve_cl_seed_reuse():
    "mcsolve_cl: check reusing seeds"
    N0 = 6
    N1 = 6
    N2 = 6
    # damping rates
    gamma0 = 0.1
    gamma1 = 0.4
    gamma2 = 0.1
    alpha = np.sqrt(2)  # initial coherent state param for mode 0
    tlist = np.linspace(0, 10, 2)
    ntraj = 500  # number of trajectories
    # define operators
    a0 = tensor(destroy(N0), qeye(N1), qeye(N2))
    a1 = tensor(qeye(N0), destroy(N1), qeye(N2))
    a2 = tensor(qeye(N0), qeye(N1), destroy(N2))
    # number operators for each mode
    num0 = a0.dag() * a0
    num1 = a1.dag() * a1
    num2 = a2.dag() * a2
    # dissipative operators for zero-temp. baths
    C0 = np.sqrt(2.0 * gamma0) * a0
    C1 = np.sqrt(2.0 * gamma1) * a1
    C2 = np.sqrt(2.0 * gamma2) * a2
    # initial state: coherent mode 0 & vacuum for modes #1 & #2
    psi0 = tensor(coherent(N0, alpha), basis(N1, 0), basis(N2, 0))
    # trilinear Hamiltonian
    H = 1j * (a0 * a1.dag() * a2.dag() - a0.dag() * a1 * a2)
    # run Monte-Carlo
    data1 = mcsolve_cl(H, psi0, tlist, [C0, C1, C2], [num0, num1, num2],
                       ntraj=ntraj)
    data2 = mcsolve_cl(H, psi0, tlist, [C0, C1, C2], [num0, num1, num2],
                       ntraj=ntraj, options=Options(seeds=data1.seeds))
    assert_equal(np.allclose(data1.expect, data2.expect), True)


def test_mcsolve_cl_seed_noreuse():
    "mcsolve_cl: check not reusing seeds"
    N0 = 6
    N1 = 6
    N2 = 6
    # damping rates
    gamma0 = 0.1
    gamma1 = 0.4
    gamma2 = 0.1
    alpha = np.sqrt(2)  # initial coherent state param for mode 0
    tlist = np.linspace(0, 10, 2)
    ntraj = 500  # number of trajectories
    # define operators
    a0 = tensor(destroy(N0), qeye(N1), qeye(N2))
    a1 = tensor(qeye(N0), destroy(N1), qeye(N2))
    a2 = tensor(qeye(N0), qeye(N1), destroy(N2))
    # number operators for each mode
    num0 = a0.dag() * a0
    num1 = a1.dag() * a1
    num2 = a2.dag() * a2
    # dissipative operators for zero-temp. baths
    C0 = np.sqrt(2.0 * gamma0) * a0
    C1 = np.sqrt(2.0 * gamma1) * a1
    C2 = np.sqrt(2.0 * gamma2) * a2
    # initial state: coherent mode 0 & vacuum for modes #1 & #2
    psi0 = tensor(coherent(N0, alpha), basis(N1, 0), basis(N2, 0))
    # trilinear Hamiltonian
    H = 1j * (a0 * a1.dag() * a2.dag() - a0.dag() * a1 * a2)
    # run Monte-Carlo
    data1 = mcsolve_cl(H, psi0, tlist, [C0, C1, C2], [num0, num1, num2],
                       ntraj=ntraj)
    data2 = mcsolve_cl(H, psi0, tlist, [C0, C1, C2], [num0, num1, num2],
                       ntraj=ntraj)
    assert_equal(np.allclose(data1.expect, data2.expect), False)


def test_mcsolve_cl_ntraj_list():
    "mcsolve_cl: list of trajectories"
    N = 5
    a = destroy(N)
    H = a.dag() * a       # Simple oscillator Hamiltonian
    psi0 = basis(N, 1)  # Initial Fock state with one photon
    kappa = 1.0 / 0.129   # Coupling rate to heat bath
    nth = 0.063         # Temperature with <n>=0.063
    # Build collapse operators for the thermal bath
    c_ops = []
    c_ops.append(np.sqrt(kappa * (1 + nth)) * a)
    c_ops.append(np.sqrt(kappa * nth) * a.dag())
    ntraj = [1, 5, 15, 904]  # number of MC trajectories
    tlist = np.linspace(0, 0.8, 100)
    mc = mcsolve_cl(H, psi0, tlist, c_ops, [a.dag() * a], ntraj)
    assert_equal(len(mc.expect), 4)


if __name__ == "__main__":
    run_module_suite()
