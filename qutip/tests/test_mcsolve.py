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
from numpy.testing import assert_equal, run_module_suite, assert_
import unittest
from qutip import *
from qutip import _version2int

# find Cython if it exists
try:
    import Cython
except:
    Cython_found = 0
else:
    Cython_found = 1

kappa = 0.2


def sqrt_kappa(t, args):
    return np.sqrt(kappa)


def sqrt_kappa2(t, args):
    return np.sqrt(kappa * np.exp(-t))


def const_H1_coeff(t, args):
    return 0.0

# average error for failure
mc_error = 5e-2  # 5%
ntraj = 750


def test_MCNoCollExpt():
    "Monte-carlo: Constant H with no collapse ops (expect)"
    error = 1e-8
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 9)  # initial state
    c_op_list = []
    tlist = np.linspace(0, 10, 100)
    mcdata = mcsolve(H, psi0, tlist, c_op_list, [a.dag() * a], ntraj=ntraj)
    expt = mcdata.expect[0]
    actual_answer = 9.0 * np.ones(len(tlist))
    diff = np.mean(abs(actual_answer - expt) / actual_answer)
    assert_equal(diff < error, True)


def test_MCNoCollStates():
    "Monte-carlo: Constant H with no collapse ops (states)"
    error = 1e-8
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 9)  # initial state
    c_op_list = []
    tlist = np.linspace(0, 10, 100)
    mcdata = mcsolve(H, psi0, tlist, c_op_list, [], ntraj=ntraj)
    states = mcdata.states
    expt = expect(a.dag() * a, states)
    actual_answer = 9.0 * np.ones(len(tlist))
    diff = np.mean(abs(actual_answer - expt) / actual_answer)
    assert_equal(diff < error, True)


def test_MCNoCollExpectStates():
    "Monte-carlo: Constant H with no collapse ops (expect and states)"
    error = 1e-8
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 9)  # initial state
    c_op_list = []
    tlist = np.linspace(0, 10, 100)
    mcdata = mcsolve(H, psi0, tlist, c_op_list, [a.dag() * a], ntraj=ntraj,
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


def test_MCNoCollStrExpt():
    "Monte-carlo: Constant H (str format) with no collapse ops (expect)"
    error = 1e-8
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = [a.dag() * a, [a.dag() * a, 'c']]
    psi0 = basis(N, 9)  # initial state
    c_op_list = []
    tlist = np.linspace(0, 10, 100)
    mcdata = mcsolve(H, psi0, tlist, c_op_list, [a.dag() * a], args={'c': 0.0},
                     ntraj=ntraj)
    expt = mcdata.expect[0]
    actual_answer = 9.0 * np.ones(len(tlist))
    diff = np.mean(abs(actual_answer - expt) / actual_answer)
    assert_equal(diff < error, True)


def test_MCNoCollFuncExpt():
    "Monte-carlo: Constant H (func format) with no collapse ops (expect)"
    error = 1e-8
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = [a.dag() * a, [a.dag() * a, const_H1_coeff]]
    psi0 = basis(N, 9)  # initial state
    c_op_list = []
    tlist = np.linspace(0, 10, 100)
    mcdata = mcsolve(H, psi0, tlist, c_op_list, [a.dag() * a], ntraj=ntraj)
    expt = mcdata.expect[0]
    actual_answer = 9.0 * np.ones(len(tlist))
    diff = np.mean(abs(actual_answer - expt) / actual_answer)
    assert_equal(diff < error, True)


def test_MCNoCollStrStates():
    "Monte-carlo: Constant H (str format) with no collapse ops (states)"
    error = 1e-8
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = [a.dag() * a, [a.dag() * a, 'c']]
    psi0 = basis(N, 9)  # initial state
    c_op_list = []
    tlist = np.linspace(0, 10, 100)
    mcdata = mcsolve(H, psi0, tlist, c_op_list, [], args={'c': 0.0})
    states = mcdata.states
    expt = expect(a.dag() * a, states)
    actual_answer = 9.0 * np.ones(len(tlist))
    diff = np.mean(abs(actual_answer - expt) / actual_answer)
    assert_equal(diff < error, True)


def test_MCNoCollFuncStates():
    "Monte-carlo: Constant H (func format) with no collapse ops (states)"
    error = 1e-8
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = [a.dag() * a, [a.dag() * a, const_H1_coeff]]
    psi0 = basis(N, 9)  # initial state
    c_op_list = []
    tlist = np.linspace(0, 10, 100)
    mcdata = mcsolve(H, psi0, tlist, c_op_list, [], ntraj=ntraj)
    states = mcdata.states
    expt = expect(a.dag() * a, states)
    actual_answer = 9.0 * np.ones(len(tlist))
    diff = np.mean(abs(actual_answer - expt) / actual_answer)
    assert_equal(diff < error, True)


def test_MCCollapseTimesOperators():
    "Monte-carlo: Check for stored collapse operators and times"
    N = 10
    kappa = 5.0
    times = np.linspace(0, 10, 100)
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 9)
    c_ops = [np.sqrt(kappa) * a, np.sqrt(kappa) * a]
    result = mcsolve(H, psi0, times, c_ops, [], ntraj=1)
    assert_(len(result.col_times[0]) > 0)
    assert_(len(result.col_which) == len(result.col_times))
    assert_(set(result.col_which[0]) == {0, 1})


def test_MCSimpleConst():
    "Monte-carlo: Constant H with constant collapse"
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 9)  # initial state
    kappa = 0.2  # coupling to oscillator
    c_op_list = [np.sqrt(kappa) * a]
    tlist = np.linspace(0, 10, 100)
    mcdata = mcsolve(H, psi0, tlist, c_op_list, [a.dag() * a], ntraj=ntraj)
    expt = mcdata.expect[0]
    actual_answer = 9.0 * np.exp(-kappa * tlist)
    avg_diff = np.mean(abs(actual_answer - expt) / actual_answer)
    assert_equal(avg_diff < mc_error, True)


def test_MCSimpleConstStates():
    "Monte-carlo: Constant H with constant collapse (states)"
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 9)  # initial state
    kappa = 0.2  # coupling to oscillator
    c_op_list = [np.sqrt(kappa) * a]
    tlist = np.linspace(0, 10, 100)
    mcdata = mcsolve(H, psi0, tlist, c_op_list, [], ntraj=ntraj,
                     options=Options(average_states=True))
    assert_(len(mcdata.states) == len(tlist))
    assert_(isinstance(mcdata.states[0], Qobj))
    expt = expect(a.dag() * a, mcdata.states)
    actual_answer = 9.0 * np.exp(-kappa * tlist)
    avg_diff = np.mean(abs(actual_answer - expt) / actual_answer)
    assert_equal(avg_diff < mc_error, True)


def test_MCSimpleConstExptStates():
    "Monte-carlo: Constant H with constant collapse (states)"
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 9)  # initial state
    kappa = 0.2  # coupling to oscillator
    c_op_list = [np.sqrt(kappa) * a]
    tlist = np.linspace(0, 10, 100)
    mcdata = mcsolve(H, psi0, tlist, c_op_list, [a.dag() * a], ntraj=ntraj,
                     options=Options(average_states=True, store_states=True))
    actual_answer = 9.0 * np.exp(-kappa * tlist)
    expt1 = mcdata.expect[0]
    avg_diff = np.mean(abs(actual_answer - expt1) / actual_answer)
    assert_equal(avg_diff < mc_error, True)
    assert_(len(mcdata.states) == len(tlist))
    assert_(isinstance(mcdata.states[0], Qobj))
    expt2 = expect(a.dag() * a, mcdata.states)
    avg_diff = np.mean(abs(actual_answer - expt2) / actual_answer)
    assert_equal(avg_diff < mc_error, True)


def test_MCSimpleSingleCollapse():
    """Monte-carlo: Constant H with single collapse operator"""
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 9)  # initial state
    kappa = 0.2  # coupling to oscillator
    c_op_list = [np.sqrt(kappa) * a]
    tlist = np.linspace(0, 10, 100)
    mcdata = mcsolve(H, psi0, tlist, c_op_list, [a.dag() * a], ntraj=ntraj)
    expt = mcdata.expect[0]
    actual_answer = 9.0 * np.exp(-kappa * tlist)
    avg_diff = np.mean(abs(actual_answer - expt) / actual_answer)
    assert_equal(avg_diff < mc_error, True)


def test_MCSimpleSingleExpect():
    """Monte-carlo: Constant H with single expect operator"""
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 9)  # initial state
    kappa = 0.2  # coupling to oscillator
    c_op_list = [np.sqrt(kappa) * a]
    tlist = np.linspace(0, 10, 100)
    mcdata = mcsolve(H, psi0, tlist, c_op_list, [a.dag() * a], ntraj=ntraj)
    expt = mcdata.expect[0]
    actual_answer = 9.0 * np.exp(-kappa * tlist)
    avg_diff = np.mean(abs(actual_answer - expt) / actual_answer)
    assert_equal(avg_diff < mc_error, True)


def test_MCSimpleConstFunc():
    "Monte-carlo: Collapse terms constant (func format)"
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 9)  # initial state
    kappa = 0.2  # coupling to oscillator
    c_op_list = [[a, sqrt_kappa]]
    tlist = np.linspace(0, 10, 100)
    mcdata = mcsolve(H, psi0, tlist, c_op_list, [a.dag() * a], ntraj=ntraj)
    expt = mcdata.expect[0]
    actual_answer = 9.0 * np.exp(-kappa * tlist)
    avg_diff = np.mean(abs(actual_answer - expt) / actual_answer)
    assert_equal(avg_diff < mc_error, True)


@unittest.skipIf(_version2int(Cython.__version__) < _version2int('0.14') or
                 Cython_found == 0, 'Cython not found or version too low.')
def test_MCSimpleConstStr():
    "Monte-carlo: Collapse terms constant (str format)"
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 9)  # initial state
    kappa = 0.2  # coupling to oscillator
    c_op_list = [[a, 'sqrt(k)']]
    args = {'k': kappa}
    tlist = np.linspace(0, 10, 100)
    mcdata = mcsolve(H, psi0, tlist, c_op_list, [a.dag() * a], args=args,
                     ntraj=ntraj)
    expt = mcdata.expect[0]
    actual_answer = 9.0 * np.exp(-kappa * tlist)
    avg_diff = np.mean(abs(actual_answer - expt) / actual_answer)
    assert_equal(avg_diff < mc_error, True)


def test_MCTDFunc():
    "Monte-carlo: Time-dependent H (func format)"
    error = 5e-2
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 9)  # initial state
    kappa = 0.2  # coupling to oscillator
    c_op_list = [[a, sqrt_kappa2]]
    tlist = np.linspace(0, 10, 100)
    mcdata = mcsolve(H, psi0, tlist, c_op_list, [a.dag() * a], ntraj=ntraj)
    expt = mcdata.expect[0]
    actual_answer = 9.0 * np.exp(-kappa * (1.0 - np.exp(-tlist)))
    diff = np.mean(abs(actual_answer - expt) / actual_answer)
    assert_equal(diff < error, True)


@unittest.skipIf(_version2int(Cython.__version__) < _version2int('0.14') or
                 Cython_found == 0, 'Cython not found or version too low.')
def test_TDStr():
    "Monte-carlo: Time-dependent H (str format)"
    error = 5e-2
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 9)  # initial state
    kappa = 0.2  # coupling to oscillator
    c_op_list = [[a, 'sqrt(k*exp(-t))']]
    args = {'k': kappa}
    tlist = np.linspace(0, 10, 100)
    mcdata = mcsolve(H, psi0, tlist, c_op_list, [a.dag() * a], args=args,
                     ntraj=ntraj)
    expt = mcdata.expect[0]
    actual = 9.0 * np.exp(-kappa * (1.0 - np.exp(-tlist)))
    diff = np.mean(abs(actual - expt) / actual)
    assert_equal(diff < error, True)


def test_mc_dtypes1():
    "Monte-carlo: check for correct dtypes (average_states=True)"
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
    opts = Options(average_expect=True)
    data = mcsolve(
        H, psi0, tlist, [C1, C2], [C1dC1, C2dC2, a], ntraj=5, options=opts)
    assert_equal(isinstance(data.expect[0][1], float), True)
    assert_equal(isinstance(data.expect[1][1], float), True)
    assert_equal(isinstance(data.expect[2][1], complex), True)


def test_mc_dtypes2():
    "Monte-carlo: check for correct dtypes (average_states=False)"
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
    opts = Options(average_expect=False)
    data = mcsolve(
        H, psi0, tlist, [C1, C2], [C1dC1, C2dC2, a], ntraj=5, options=opts)
    assert_equal(isinstance(data.expect[0][0][1], float), True)
    assert_equal(isinstance(data.expect[0][1][1], float), True)
    assert_equal(isinstance(data.expect[0][2][1], complex), True)


def test_mc_seed_reuse():
    "Monte-carlo: check reusing seeds"
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
    data1 = mcsolve(H, psi0, tlist, [C0, C1, C2], [num0, num1, num2],
                    ntraj=ntraj)
    data2 = mcsolve(H, psi0, tlist, [C0, C1, C2], [num0, num1, num2],
                    ntraj=ntraj, options=Options(seeds=data1.seeds))
    for k in range(ntraj):
        assert_equal(np.allclose(data1.col_times[k],data2.col_times[k]), True)
        assert_equal(np.allclose(data1.col_which[k],data2.col_which[k]), True)


def test_mc_seed_noreuse():
    "Monte-carlo: check not reusing seeds"
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
    data1 = mcsolve(H, psi0, tlist, [C0, C1, C2], [num0, num1, num2],
                    ntraj=ntraj)
    data2 = mcsolve(H, psi0, tlist, [C0, C1, C2], [num0, num1, num2],
                    ntraj=ntraj)
    diff_flag = False
    for k in range(ntraj):
        if len(data1.col_times[k]) != len(data2.col_times[k]):
            diff_flag = 1
            break
        else:
            if not np.allclose(data1.col_which[k],data2.col_which[k]):
                diff_flag = 1
                break
    assert_equal(diff_flag, 1)


def test_mc_ntraj_list():
    "Monte-carlo: list of trajectories"
    N = 5
    a = destroy(N)
    H = a.dag()*a       # Simple oscillator Hamiltonian
    psi0 = basis(N, 1)  # Initial Fock state with one photon
    kappa = 1.0/0.129   # Coupling rate to heat bath
    nth = 0.063         # Temperature with <n>=0.063
    # Build collapse operators for the thermal bath
    c_ops = []
    c_ops.append(np.sqrt(kappa * (1 + nth)) * a)
    c_ops.append(np.sqrt(kappa * nth) * a.dag())
    ntraj = [1, 5, 15, 904]  # number of MC trajectories
    tlist = np.linspace(0, 0.8, 100)
    mc = mcsolve(H, psi0, tlist, c_ops, [a.dag()*a], ntraj)
    assert_equal(len(mc.expect), 4)

def test_mc_functd_sum():
    "Monte-carlo: Test for #490"
    psi0 = (basis(2,0) + basis(2,1)).unit()   
    H0 = sigmax()
    H1 = sigmay()
    H2 = sigmaz()
    def H1_coeff(t,args):
        return t-1

    def H2_coeff(t,args):
        return -t

    h_t = [H0,[H1, H1_coeff],
           [H2, H2_coeff]]
    ntraj = 1

    tlist = np.linspace(0, 3, 10)
    medata = mesolve(h_t, psi0, tlist, [], [], args = {})
    mcdata = mcsolve(h_t, psi0, tlist, [], [], ntraj = ntraj, args = {})
    assert_(max([(medata.states[k]-mcdata.states[k]).norm() for k in range(10)]) < 1e-5)


if __name__ == "__main__":
    run_module_suite()
