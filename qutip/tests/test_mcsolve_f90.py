# This file is part of QuTiP.
#
#    QuTiP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    QuTiP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTiP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011-2013, Paul D. Nation & Robert J. Johansson
#
###########################################################################

from qutip import *
from qutip.odechecks import _ode_checks
from numpy import allclose, linspace, mean, ones
from numpy.testing import assert_equal, run_module_suite
from numpy.testing.decorators import skipif
import unittest
# find fortran files if they exist
try:
    from qutip.fortran import qutraj_run
except:
    fortran_found = 0
else:
    fortran_found = 1

kappa = 0.2


def sqrt_kappa(t, args):
    return sqrt(kappa)


def sqrt_kappa2(t, args):
    return sqrt(kappa * exp(-t))


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
    kappa = 0.2  # coupling to oscillator
    c_op_list = []
    tlist = linspace(0, 10, 100)
    mcdata = mcsolve_f90(H, psi0, tlist, c_op_list,
                         [a.dag() * a], options=Odeoptions(gui=False))
    expt = mcdata.expect[0]
    actual_answer = 9.0 * ones(len(tlist))
    diff = mean(abs(actual_answer - expt) / actual_answer)
    assert_equal(diff < error, True)


@unittest.skipIf(fortran_found == 0, 'fortran files not found')
def test_MCNoCollStates():
    "mcsolve_f90: Constant H with no collapse ops (states)"
    error = 1e-8
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 9)  # initial state
    kappa = 0.2  # coupling to oscillator
    c_op_list = []
    tlist = linspace(0, 10, 100)
    mcdata = mcsolve_f90(
        H, psi0, tlist, c_op_list, [], options=Odeoptions(gui=False))
    states = mcdata.states
    expt = expect(a.dag() * a, states)
    actual_answer = 9.0 * ones(len(tlist))
    diff = mean(abs(actual_answer - expt) / actual_answer)
    assert_equal(diff < error, True)


@unittest.skipIf(fortran_found == 0, 'fortran files not found')
def test_MCSimpleConst():
    "mcsolve_f90: Constant H with constant collapse"
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 9)  # initial state
    kappa = 0.2  # coupling to oscillator
    c_op_list = [sqrt(kappa) * a]
    tlist = linspace(0, 10, 100)
    mcdata = mcsolve_f90(H, psi0, tlist, c_op_list,
                         [a.dag() * a], options=Odeoptions(gui=False))
    expt = mcdata.expect[0]
    actual_answer = 9.0 * exp(-kappa * tlist)
    avg_diff = mean(abs(actual_answer - expt) / actual_answer)
    assert_equal(avg_diff < mc_error, True)


@unittest.skipIf(fortran_found == 0, 'fortran files not found')
def test_MCSimpleSingleCollapse():
    """mcsolve_f90: Constant H with single collapse operator"""
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 9)  # initial state
    kappa = 0.2  # coupling to oscillator
    c_op_list = [sqrt(kappa) * a]
    tlist = linspace(0, 10, 100)
    mcdata = mcsolve_f90(H, psi0, tlist, c_op_list,
                         [a.dag() * a], options=Odeoptions(gui=False))
    expt = mcdata.expect[0]
    actual_answer = 9.0 * exp(-kappa * tlist)
    avg_diff = mean(abs(actual_answer - expt) / actual_answer)
    assert_equal(avg_diff < mc_error, True)


@unittest.skipIf(fortran_found == 0, 'fortran files not found')
def test_MCSimpleSingleExpect():
    """mcsolve_f90: Constant H with single expect operator"""
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 9)  # initial state
    kappa = 0.2  # coupling to oscillator
    c_op_list = [sqrt(kappa) * a]
    tlist = linspace(0, 10, 100)
    mcdata = mcsolve_f90(H, psi0, tlist, c_op_list,
                         [a.dag() * a], options=Odeoptions(gui=False))
    expt = mcdata.expect[0]
    actual_answer = 9.0 * exp(-kappa * tlist)
    avg_diff = mean(abs(actual_answer - expt) / actual_answer)
    assert_equal(avg_diff < mc_error, True)


@unittest.skipIf(fortran_found == 0, 'fortran files not found')
def test_mcf90_dtypes1():
    "mcsolve_f90: check for correct dtypes (mc_avg=True)"
    # set system parameters
    kappa = 2.0  # mirror coupling
    gamma = 0.2  # spontaneous emission rate
    g = 1  # atom/cavity coupling strength
    wc = 0  # cavity frequency
    w0 = 0  # atom frequency
    wl = 0  # driving frequency
    E = 0.5  # driving amplitude
    N = 5  # number of cavity energy levels (0->3 Fock states)
    tlist = linspace(0, 10, 5)  # times for expectation values
    # construct Hamiltonian
    ida = qeye(N)
    idatom = qeye(2)
    a = tensor(destroy(N), idatom)
    sm = tensor(ida, sigmam())
    H = (w0 - wl) * sm.dag() * sm + (wc - wl) * a.dag() * a + \
        1j * g * (a.dag() * sm - sm.dag() * a) + E * (a.dag() + a)
    # collapse operators
    C1 = sqrt(2 * kappa) * a
    C2 = sqrt(gamma) * sm
    C1dC1 = C1.dag() * C1
    C2dC2 = C2.dag() * C2
    # intial state
    psi0 = tensor(basis(N, 0), basis(2, 1))
    opts = Odeoptions(gui=False, mc_avg=True)
    data = mcsolve_f90(
        H, psi0, tlist, [C1, C2], [C1dC1, C2dC2, a], ntraj=5, options=opts)
    assert_equal(isinstance(data.expect[0][0], float), True)
    assert_equal(isinstance(data.expect[1][0], float), True)
    assert_equal(isinstance(data.expect[2][0], complex), True)


@unittest.skipIf(fortran_found == 0, 'fortran files not found')
def test_mcf90_dtypes2():
    "mcsolve_f90: check for correct dtypes (mc_avg=False)"
    # set system parameters
    kappa = 2.0  # mirror coupling
    gamma = 0.2  # spontaneous emission rate
    g = 1  # atom/cavity coupling strength
    wc = 0  # cavity frequency
    w0 = 0  # atom frequency
    wl = 0  # driving frequency
    E = 0.5  # driving amplitude
    N = 5  # number of cavity energy levels (0->3 Fock states)
    tlist = linspace(0, 10, 5)  # times for expectation values
    # construct Hamiltonian
    ida = qeye(N)
    idatom = qeye(2)
    a = tensor(destroy(N), idatom)
    sm = tensor(ida, sigmam())
    H = (w0 - wl) * sm.dag() * sm + (wc - wl) * a.dag() * a + \
        1j * g * (a.dag() * sm - sm.dag() * a) + E * (a.dag() + a)
    # collapse operators
    C1 = sqrt(2 * kappa) * a
    C2 = sqrt(gamma) * sm
    C1dC1 = C1.dag() * C1
    C2dC2 = C2.dag() * C2
    # intial state
    psi0 = tensor(basis(N, 0), basis(2, 1))
    opts = Odeoptions(gui=False, mc_avg=False)
    data = mcsolve_f90(
        H, psi0, tlist, [C1, C2], [C1dC1, C2dC2, a], ntraj=5, options=opts)
    assert_equal(isinstance(data.expect[0][0][0], float), True)
    assert_equal(isinstance(data.expect[0][1][0], float), True)
    assert_equal(isinstance(data.expect[0][2][0], complex), True)


if __name__ == "__main__":
    run_module_suite()
