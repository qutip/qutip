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

from numpy.testing import assert_equal, run_module_suite
import unittest
# find fortran files if they exist

from qutip import destroy, basis, expect, tensor, Options, sigmam, qeye

try:
    from qutip.fortran import mcsolve_f90
except:
    fortran_found = 0
else:
    fortran_found = 1

kappa = 0.2


def sqrt_kappa(t, args):
    return np.sqrt(kappa)


def sqrt_kappa2(t, args):
    return np.sqrt(kappa * np.exp(-t))


def const_H1_coeff(t, args):
    return 0.0

# average error for failure
mc_error = 5e-2  # 5% for ntraj=500


@unittest.skipIf(fortran_found == 0, 'fortran files not found')
def test_MCNoCollExpt():
    "mcsolve_f90: Constant H with no collapse ops (expect)"
    error = 1e-8
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 9)  # initial state
    c_op_list = []
    tlist = np.linspace(0, 10, 100)
    mcdata = mcsolve_f90(H, psi0, tlist, c_op_list, [a.dag() * a])
    expt = mcdata.expect[0]
    actual_answer = 9.0 * np.ones(len(tlist))
    diff = np.mean(abs(actual_answer - expt) / actual_answer)
    assert_equal(diff < error, True)


@unittest.skipIf(fortran_found == 0, 'fortran files not found')
def test_MCNoCollStates():
    "mcsolve_f90: Constant H with no collapse ops (states)"
    error = 1e-8
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 9)  # initial state
    c_op_list = []
    tlist = np.linspace(0, 10, 100)
    mcdata = mcsolve_f90(H, psi0, tlist, c_op_list, [])
    states = mcdata.states[0]
    expt = expect(a.dag() * a, states)
    actual_answer = 9.0 * np.ones(len(tlist))
    diff = np.mean(abs(actual_answer - expt) / actual_answer)
    assert_equal(diff < error, True)


@unittest.skipIf(fortran_found == 0, 'fortran files not found')
def test_MCSimpleConst():
    "mcsolve_f90: Constant H with constant collapse"
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 9)  # initial state
    kappa = 0.2  # coupling to oscillator
    c_op_list = [np.sqrt(kappa) * a]
    tlist = np.linspace(0, 10, 100)
    mcdata = mcsolve_f90(H, psi0, tlist, c_op_list, [a.dag() * a])
    expt = mcdata.expect[0]
    actual_answer = 9.0 * np.exp(-kappa * tlist)
    avg_diff = np.mean(abs(actual_answer - expt) / actual_answer)
    assert_equal(avg_diff < mc_error, True)


@unittest.skipIf(fortran_found == 0, 'fortran files not found')
def test_MCSimpleSingleCollapse():
    """mcsolve_f90: Constant H with single collapse operator"""
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 9)  # initial state
    kappa = 0.2  # coupling to oscillator
    c_op_list = [np.sqrt(kappa) * a]
    tlist = np.linspace(0, 10, 100)
    mcdata = mcsolve_f90(H, psi0, tlist, c_op_list, [a.dag() * a])
    expt = mcdata.expect[0]
    actual_answer = 9.0 * np.exp(-kappa * tlist)
    avg_diff = np.mean(abs(actual_answer - expt) / actual_answer)
    assert_equal(avg_diff < mc_error, True)


@unittest.skipIf(fortran_found == 0, 'fortran files not found')
def test_MCSimpleSingleExpect():
    """mcsolve_f90: Constant H with single expect operator"""
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 9)  # initial state
    kappa = 0.2  # coupling to oscillator
    c_op_list = [np.sqrt(kappa) * a]
    tlist = np.linspace(0, 10, 100)
    mcdata = mcsolve_f90(H, psi0, tlist, c_op_list, [a.dag() * a])
    expt = mcdata.expect[0]
    actual_answer = 9.0 * np.exp(-kappa * tlist)
    avg_diff = np.mean(abs(actual_answer - expt) / actual_answer)
    assert_equal(avg_diff < mc_error, True)


@unittest.skipIf(fortran_found == 0, 'fortran files not found')
def test_mcf90_dtypes1():
    "mcsolve_f90: check for correct dtypes (average_states=True)"
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
    data = mcsolve_f90(
        H, psi0, tlist, [C1, C2], [C1dC1, C2dC2, a], ntraj=5, options=opts)
    assert_equal(isinstance(data.expect[0][1], float), True)
    assert_equal(isinstance(data.expect[1][1], float), True)
    assert_equal(isinstance(data.expect[2][1], complex), True)


@unittest.skipIf(fortran_found == 0, 'fortran files not found')
def test_mcf90_dtypes2():
    "mcsolve_f90: check for correct dtypes (average_states=False)"
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
    data = mcsolve_f90(
        H, psi0, tlist, [C1, C2], [C1dC1, C2dC2, a], ntraj=5, options=opts)
    assert_equal(isinstance(data.expect[0][0][1], float), True)
    assert_equal(isinstance(data.expect[0][1][1], float), True)
    assert_equal(isinstance(data.expect[0][2][1], complex), True)


if __name__ == "__main__":
    run_module_suite()
