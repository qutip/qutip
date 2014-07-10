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

from functools import partial
from numpy import allclose, linspace, mean, ones
from numpy.testing import assert_, run_module_suite

# disable the MC progress bar
import os
os.environ['QUTIP_GRAPHICS'] = "NO"

from qutip import *


class TestJCModelEvolution:
    """
    A test class for the QuTiP functions for the evolution of JC model
    """

    def qubit_integrate(self, tlist, psi0, epsilon, delta, g1, g2):

        H = epsilon / 2.0 * sigmaz() + delta / 2.0 * sigmax()

        c_op_list = []

        rate = g1
        if rate > 0.0:
            c_op_list.append(sqrt(rate) * sigmam())

        rate = g2
        if rate > 0.0:
            c_op_list.append(sqrt(rate) * sigmaz())

        output = mesolve(
            H, psi0, tlist, c_op_list, [sigmax(), sigmay(), sigmaz()])
        expt_list = output.expect[0], output.expect[1], output.expect[2]

        return expt_list[0], expt_list[1], expt_list[2]

    def jc_steadystate(self, N, wc, wa, g, kappa, gamma,
                       pump, psi0, use_rwa, tlist):

        # Hamiltonian
        a = tensor(destroy(N), qeye(2))
        sm = tensor(qeye(N), destroy(2))

        if use_rwa:
            # use the rotating wave approxiation
            H = wc * a.dag(
            ) * a + wa * sm.dag() * sm + g * (a.dag() * sm + a * sm.dag())
        else:
            H = wc * a.dag() * a + wa * sm.dag() * sm + g * (
                a.dag() + a) * (sm + sm.dag())

        # collapse operators
        c_op_list = []

        n_th_a = 0.0  # zero temperature

        rate = kappa * (1 + n_th_a)
        c_op_list.append(sqrt(rate) * a)

        rate = kappa * n_th_a
        if rate > 0.0:
            c_op_list.append(sqrt(rate) * a.dag())

        rate = gamma
        if rate > 0.0:
            c_op_list.append(sqrt(rate) * sm)

        rate = pump
        if rate > 0.0:
            c_op_list.append(sqrt(rate) * sm.dag())

        # find the steady state
        rho_ss = steadystate(H, c_op_list)

        return expect(a.dag() * a, rho_ss), expect(sm.dag() * sm, rho_ss)

    def jc_integrate(self, N, wc, wa, g, kappa, gamma,
                     pump, psi0, use_rwa, tlist):

        # Hamiltonian
        a = tensor(destroy(N), qeye(2))
        sm = tensor(qeye(N), destroy(2))

        if use_rwa:
            # use the rotating wave approxiation
            H = wc * a.dag(
            ) * a + wa * sm.dag() * sm + g * (a.dag() * sm + a * sm.dag())
        else:
            H = wc * a.dag() * a + wa * sm.dag() * sm + g * (
                a.dag() + a) * (sm + sm.dag())

        # collapse operators
        c_op_list = []

        n_th_a = 0.0  # zero temperature

        rate = kappa * (1 + n_th_a)
        c_op_list.append(sqrt(rate) * a)

        rate = kappa * n_th_a
        if rate > 0.0:
            c_op_list.append(sqrt(rate) * a.dag())

        rate = gamma
        if rate > 0.0:
            c_op_list.append(sqrt(rate) * sm)

        rate = pump
        if rate > 0.0:
            c_op_list.append(sqrt(rate) * sm.dag())

        # evolve and calculate expectation values
        output = mesolve(
            H, psi0, tlist, c_op_list, [a.dag() * a, sm.dag() * sm])
        expt_list = output.expect[0], output.expect[1]
        return expt_list[0], expt_list[1]

    def testQubitDynamics1(self):
        "mesolve: qubit with dissipation"

        epsilon = 0.0 * 2 * pi   # cavity frequency
        delta = 1.0 * 2 * pi   # atom frequency
        g2 = 0.1
        g1 = 0.0
        psi0 = basis(2, 0)        # initial state
        tlist = linspace(0, 5, 200)

        sx, sy, sz = self.qubit_integrate(tlist, psi0, epsilon, delta, g1, g2)

        sx_analytic = zeros(shape(tlist))
        sy_analytic = -sin(2 * pi * tlist) * exp(-tlist * g2)
        sz_analytic = cos(2 * pi * tlist) * exp(-tlist * g2)

        assert_(max(abs(sx - sx_analytic)) < 0.05)
        assert_(max(abs(sy - sy_analytic)) < 0.05)
        assert_(max(abs(sz - sz_analytic)) < 0.05)

    def testQubitDynamics1(self):
        "mesolve: qubit without dissipation"

        epsilon = 0.0 * 2 * pi   # cavity frequency
        delta = 1.0 * 2 * pi   # atom frequency
        g2 = 0.0
        g1 = 0.0
        psi0 = basis(2, 0)        # initial state
        tlist = linspace(0, 5, 200)

        sx, sy, sz = self.qubit_integrate(tlist, psi0, epsilon, delta, g1, g2)

        sx_analytic = zeros(shape(tlist))
        sy_analytic = -sin(2 * pi * tlist) * exp(-tlist * g2)
        sz_analytic = cos(2 * pi * tlist) * exp(-tlist * g2)

        assert_(max(abs(sx - sx_analytic)) < 0.05)
        assert_(max(abs(sy - sy_analytic)) < 0.05)
        assert_(max(abs(sz - sz_analytic)) < 0.05)

    def testCase1(self):
        "mesolve: cavity-qubit interaction, no dissipation"

        use_rwa = True
        N = 4           # number of cavity fock states
        wc = 2 * pi * 1.0   # cavity frequency
        wa = 2 * pi * 1.0   # atom frequency
        g = 2 * pi * 0.01  # coupling strength
        kappa = 0.0     # cavity dissipation rate
        gamma = 0.0     # atom dissipation rate
        pump = 0.0     # atom pump rate

        # start with an excited atom and maximum number of photons
        n = N - 2
        psi0 = tensor(basis(N, n), basis(2, 1))
        tlist = linspace(0, 1000, 2000)

        nc, na = self.jc_integrate(
            N, wc, wa, g, kappa, gamma, pump, psi0, use_rwa, tlist)

        nc_ex = (n + 0.5 * (1 - cos(2 * g * sqrt(n + 1) * tlist)))
        na_ex = 0.5 * (1 + cos(2 * g * sqrt(n + 1) * tlist))

        assert_(max(abs(nc - nc_ex)) < 0.005, True)
        assert_(max(abs(na - na_ex)) < 0.005, True)

    def testCase2(self):
        "mesolve: cavity-qubit without interaction, decay"

        use_rwa = True
        N = 4           # number of cavity fock states
        wc = 2 * pi * 1.0   # cavity frequency
        wa = 2 * pi * 1.0   # atom frequency
        g = 2 * pi * 0.0   # coupling strength
        kappa = 0.005   # cavity dissipation rate
        gamma = 0.01    # atom dissipation rate
        pump = 0.0     # atom pump rate

        # start with an excited atom and maximum number of photons
        n = N - 2
        psi0 = tensor(basis(N, n), basis(2, 1))
        tlist = linspace(0, 1000, 2000)

        nc, na = self.jc_integrate(
            N, wc, wa, g, kappa, gamma, pump, psi0, use_rwa, tlist)

        nc_ex = (n + 0.5 * (1 - cos(2 * g * sqrt(n + 1) * tlist))) * \
            exp(-kappa * tlist)
        na_ex = 0.5 * (1 + cos(2 * g * sqrt(n + 1) * tlist)) * \
            exp(-gamma * tlist)

        assert_(max(abs(nc - nc_ex)) < 0.005, True)
        assert_(max(abs(na - na_ex)) < 0.005, True)

    def testCase3(self):
        "mesolve: cavity-qubit with interaction, decay"

        use_rwa = True
        N = 4           # number of cavity fock states
        wc = 2 * pi * 1.0   # cavity frequency
        wa = 2 * pi * 1.0   # atom frequency
        g = 2 * pi * 0.1   # coupling strength
        kappa = 0.05    # cavity dissipation rate
        gamma = 0.001   # atom dissipation rate
        pump = 0.25    # atom pump rate

        # start with an excited atom and maximum number of photons
        n = N - 2
        psi0 = tensor(basis(N, n), basis(2, 1))
        tlist = linspace(0, 200, 500)

        nc, na = self.jc_integrate(
            N, wc, wa, g, kappa, gamma, pump, psi0, use_rwa, tlist)

        # we don't have any analytics for this parameters, so
        # compare with the steady state
        nc_ss, na_ss = self.jc_steadystate(
            N, wc, wa, g, kappa, gamma, pump, psi0, use_rwa, tlist)

        nc_ss = nc_ss * ones(shape(nc))
        na_ss = na_ss * ones(shape(na))

        assert_(abs(nc[-1] - nc_ss[-1]) < 0.005, True)
        assert_(abs(na[-1] - na_ss[-1]) < 0.005, True)

# percent error for failure
me_error = 1e-8


class TestMESolverConstDecay:
    """
    A test class for the time-dependent ode check function.
    """

    def testMESimpleConstDecay(self):
        "mesolve: simple constant decay"

        N = 10  # number of basis states to consider
        a = destroy(N)
        H = a.dag() * a
        psi0 = basis(N, 9)  # initial state
        kappa = 0.2  # coupling to oscillator
        c_op_list = [sqrt(kappa) * a]
        tlist = linspace(0, 10, 100)
        medata = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a])
        expt = medata.expect[0]
        actual_answer = 9.0 * exp(-kappa * tlist)
        avg_diff = mean(abs(actual_answer - expt) / actual_answer)
        assert_(avg_diff < me_error)

    def testMESimpleConstDecaySingleCollapse(self):
        "mesolve: simple constant decay"

        N = 10  # number of basis states to consider
        a = destroy(N)
        H = a.dag() * a
        psi0 = basis(N, 9)  # initial state
        kappa = 0.2  # coupling to oscillator
        c_op = sqrt(kappa) * a
        tlist = linspace(0, 10, 100)
        medata = mesolve(H, psi0, tlist, c_op, [a.dag() * a])
        expt = medata.expect[0]
        actual_answer = 9.0 * exp(-kappa * tlist)
        avg_diff = mean(abs(actual_answer - expt) / actual_answer)
        assert_(avg_diff < me_error)

    def testMESimpleConstDecaySingleExpect(self):
        "mesolve: simple constant decay"

        N = 10  # number of basis states to consider
        a = destroy(N)
        H = a.dag() * a
        psi0 = basis(N, 9)  # initial state
        kappa = 0.2  # coupling to oscillator
        c_op_list = [sqrt(kappa) * a]
        tlist = linspace(0, 10, 100)
        medata = mesolve(H, psi0, tlist, c_op_list, a.dag() * a)
        expt = medata.expect[0]
        actual_answer = 9.0 * exp(-kappa * tlist)
        avg_diff = mean(abs(actual_answer - expt) / actual_answer)
        assert_(avg_diff < me_error)

    def testMESimpleConstDecayAsFuncList(self):
        "mesolve: constant decay as function list"

        N = 10  # number of basis states to consider
        a = destroy(N)
        H = a.dag() * a
        psi0 = basis(N, 9)  # initial state
        kappa = 0.2  # coupling to oscillator

        def sqrt_kappa(t, args):
            return sqrt(kappa)
        c_op_list = [[a, sqrt_kappa]]
        tlist = linspace(0, 10, 100)
        medata = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a])
        expt = medata.expect[0]
        actual_answer = 9.0 * exp(-kappa * tlist)
        avg_diff = mean(abs(actual_answer - expt) / actual_answer)
        assert_(avg_diff < me_error)

    def testMESimpleConstDecayAsStrList(self):
        "mesolve: constant decay as string list"

        N = 10  # number of basis states to consider
        a = destroy(N)
        H = a.dag() * a
        psi0 = basis(N, 9)  # initial state
        kappa = 0.2  # coupling to oscillator
        c_op_list = [[a, 'sqrt(k)']]
        args = {'k': kappa}
        tlist = linspace(0, 10, 100)
        medata = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a], args=args)
        expt = medata.expect[0]
        actual_answer = 9.0 * exp(-kappa * tlist)
        avg_diff = mean(abs(actual_answer - expt) / actual_answer)
        assert_(avg_diff < me_error)


# average error for failure
me_error = 1e-6


class TestMESolveTDDecay:
    """
    A test class for the time-dependent odes.  Comparing to analytic answer

    N(t)=9 * exp[ -kappa*( 1-exp(-t) ) ]

    """

    def testMESimpleTDDecayAsFuncList(self):
        "mesolve: time-dependence as function list"

        N = 10  # number of basis states to consider
        a = destroy(N)
        H = a.dag() * a
        psi0 = basis(N, 9)  # initial state
        kappa = 0.2  # coupling to oscillator

        def sqrt_kappa(t, args):
            return sqrt(kappa * exp(-t))
        c_op_list = [[a, sqrt_kappa]]
        tlist = linspace(0, 10, 100)
        medata = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a])
        expt = medata.expect[0]
        actual_answer = 9.0 * exp(-kappa * (1.0 - exp(-tlist)))
        avg_diff = mean(abs(actual_answer - expt) / actual_answer)
        assert_(avg_diff < me_error)

    def testMESimpleTDDecayAsPartialFuncList(self):
        "mesolve: time-dependence as partial function list"

        N = 10
        a = destroy(N)
        H = num(N)
        psi0 = basis(N, 9)
        tlist = linspace(0, 10, 100)
        c_ops = [[[a, partial(lambda t, args, k: sqrt(k * exp(-t)), k=kappa)]]
                 for kappa in [0.05, 0.1, 0.2]]

        for idx, kappa in enumerate([0.05, 0.1, 0.2]):
            medata = mesolve(H, psi0, tlist, c_ops[idx], [H])
            ref = 9.0 * exp(-kappa * (1.0 - exp(-tlist)))
            avg_diff = mean(abs(ref - medata.expect[0]) / ref)
            assert_(avg_diff < me_error)

    def testMESimpleTDDecayAsStrList(self):
        "mesolve: time-dependence as string list"

        N = 10  # number of basis states to consider
        a = destroy(N)
        H = a.dag() * a
        psi0 = basis(N, 9)  # initial state
        kappa = 0.2  # coupling to oscillator
        c_op_list = [[a, 'sqrt(k*exp(-t))']]
        args = {'k': kappa}
        tlist = linspace(0, 10, 100)
        medata = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a], args=args)
        expt = medata.expect[0]
        actual_answer = 9.0 * exp(-kappa * (1.0 - exp(-tlist)))
        avg_diff = mean(abs(actual_answer - expt) / actual_answer)
        assert_(avg_diff < me_error)

    def testMESimpleTDDecayAsArray(self):
        "mesolve: time-dependence as array"

        N = 10
        a = destroy(N)
        H = a.dag() * a
        psi0 = basis(N, 9)
        kappa = 0.2
        tlist = linspace(0, 10, 1000)
        c_op_list = [[a, sqrt(kappa * exp(-tlist))]]
        medata = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a])
        expt = medata.expect[0]
        actual_answer = 9.0 * exp(-kappa * (1.0 - exp(-tlist)))
        avg_diff = mean(abs(actual_answer - expt) / actual_answer)
        assert_(avg_diff < 100 * me_error)

if __name__ == "__main__":
    run_module_suite()
