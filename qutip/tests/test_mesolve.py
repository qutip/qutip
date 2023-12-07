from functools import partial

import numpy as np
from numpy.testing import assert_, assert_allclose
import pytest

# disable the MC progress bar
import os

from qutip import *
from qutip.random_objects import rand_ket

os.environ['QUTIP_GRAPHICS'] = "NO"


class TestJCModelEvolution:
    """
    A test class for the QuTiP functions for the evolution of JC model
    """

    def qubit_integrate(self, tlist, psi0, epsilon, delta, g1, g2):

        H = epsilon / 2.0 * sigmaz() + delta / 2.0 * sigmax()

        c_op_list = []

        rate = g1
        if rate > 0.0:
            c_op_list.append(np.sqrt(rate) * sigmam())

        rate = g2
        if rate > 0.0:
            c_op_list.append(np.sqrt(rate) * sigmaz())

        output = mesolve(
            H, psi0, tlist, c_op_list, [sigmax(), sigmay(), sigmaz()])
        expt_list = output.expect[0], output.expect[1], output.expect[2]

        return expt_list[0], expt_list[1], expt_list[2]

    def jc_steadystate(self, N, wc, wa, g, kappa, gamma,
                       pump, psi0, use_rwa, tlist):

        # Hamiltonian
        a = tensor(destroy(N), identity(2))
        sm = tensor(identity(N), destroy(2))

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
        c_op_list.append(np.sqrt(rate) * a)

        rate = kappa * n_th_a
        if rate > 0.0:
            c_op_list.append(np.sqrt(rate) * a.dag())

        rate = gamma
        if rate > 0.0:
            c_op_list.append(np.sqrt(rate) * sm)

        rate = pump
        if rate > 0.0:
            c_op_list.append(np.sqrt(rate) * sm.dag())

        # find the steady state
        rho_ss = steadystate(H, c_op_list)

        return expect(a.dag() * a, rho_ss), expect(sm.dag() * sm, rho_ss)

    def jc_integrate(self, N, wc, wa, g, kappa, gamma,
                     pump, psi0, use_rwa, tlist):

        # Hamiltonian
        a = tensor(destroy(N), identity(2))
        sm = tensor(identity(N), destroy(2))

        if use_rwa:
            # use the rotating wave approxiation
            H = wc * a.dag() * a + wa * sm.dag() * sm + g * (
                a.dag() * sm + a * sm.dag())
        else:
            H = wc * a.dag() * a + wa * sm.dag() * sm + g * (
                a.dag() + a) * (sm + sm.dag())

        # collapse operators
        c_op_list = []

        n_th_a = 0.0  # zero temperature

        rate = kappa * (1 + n_th_a)
        c_op_list.append(np.sqrt(rate) * a)

        rate = kappa * n_th_a
        if rate > 0.0:
            c_op_list.append(np.sqrt(rate) * a.dag())

        rate = gamma
        if rate > 0.0:
            c_op_list.append(np.sqrt(rate) * sm)

        rate = pump
        if rate > 0.0:
            c_op_list.append(np.sqrt(rate) * sm.dag())

        # evolve and calculate expectation values
        output = mesolve(
            H, psi0, tlist, c_op_list, [a.dag() * a, sm.dag() * sm])
        expt_list = output.expect[0], output.expect[1]
        return expt_list[0], expt_list[1]

    def testQubitDynamics1(self):
        "mesolve: qubit with dissipation"

        epsilon = 0.0 * 2 * np.pi   # cavity frequency
        delta = 1.0 * 2 * np.pi   # atom frequency
        g2 = 0.1
        g1 = 0.0
        psi0 = basis(2, 0)        # initial state
        tlist = np.linspace(0, 5, 200)

        sx, sy, sz = self.qubit_integrate(tlist, psi0, epsilon, delta, g1, g2)

        sx_analytic = np.zeros(np.shape(tlist))
        sy_analytic = -np.sin(2 * np.pi * tlist) * np.exp(-tlist * g2)
        sz_analytic = np.cos(2 * np.pi * tlist) * np.exp(-tlist * g2)

        assert_(max(abs(sx - sx_analytic)) < 0.05)
        assert_(max(abs(sy - sy_analytic)) < 0.05)
        assert_(max(abs(sz - sz_analytic)) < 0.05)

    def testQubitDynamics2(self):
        "mesolve: qubit without dissipation"

        epsilon = 0.0 * 2 * np.pi   # cavity frequency
        delta = 1.0 * 2 * np.pi   # atom frequency
        g2 = 0.0
        g1 = 0.0
        psi0 = basis(2, 0)        # initial state
        tlist = np.linspace(0, 5, 200)

        sx, sy, sz = self.qubit_integrate(tlist, psi0, epsilon, delta, g1, g2)

        sx_analytic = np.zeros(np.shape(tlist))
        sy_analytic = -np.sin(2 * np.pi * tlist) * np.exp(-tlist * g2)
        sz_analytic = np.cos(2 * np.pi * tlist) * np.exp(-tlist * g2)

        assert_(max(abs(sx - sx_analytic)) < 0.05)
        assert_(max(abs(sy - sy_analytic)) < 0.05)
        assert_(max(abs(sz - sz_analytic)) < 0.05)

    def testCase1(self):
        "mesolve: cavity-qubit interaction, no dissipation"

        use_rwa = True
        N = 4           # number of cavity fock states
        wc = 2 * np.pi * 1.0   # cavity frequency
        wa = 2 * np.pi * 1.0   # atom frequency
        g = 2 * np.pi * 0.01  # coupling strength
        kappa = 0.0     # cavity dissipation rate
        gamma = 0.0     # atom dissipation rate
        pump = 0.0     # atom pump rate

        # start with an excited atom and maximum number of photons
        n = N - 2
        psi0 = tensor(basis(N, n), basis(2, 1))
        tlist = np.linspace(0, 1000, 2000)

        nc, na = self.jc_integrate(
            N, wc, wa, g, kappa, gamma, pump, psi0, use_rwa, tlist)

        nc_ex = (n + 0.5 * (1 - np.cos(2 * g * np.sqrt(n + 1) * tlist)))
        na_ex = 0.5 * (1 + np.cos(2 * g * np.sqrt(n + 1) * tlist))

        assert_(max(abs(nc - nc_ex)) < 0.005, True)
        assert_(max(abs(na - na_ex)) < 0.005, True)

    def testCase2(self):
        "mesolve: cavity-qubit without interaction, decay"

        use_rwa = True
        N = 4           # number of cavity fock states
        wc = 2 * np.pi * 1.0   # cavity frequency
        wa = 2 * np.pi * 1.0   # atom frequency
        g = 2 * np.pi * 0.0   # coupling strength
        kappa = 0.005   # cavity dissipation rate
        gamma = 0.01    # atom dissipation rate
        pump = 0.0     # atom pump rate

        # start with an excited atom and maximum number of photons
        n = N - 2
        psi0 = tensor(basis(N, n), basis(2, 1))
        tlist = np.linspace(0, 1000, 2000)

        nc, na = self.jc_integrate(
            N, wc, wa, g, kappa, gamma, pump, psi0, use_rwa, tlist)

        nc_ex = (n + 0.5 * (1 - np.cos(2 * g * np.sqrt(n + 1) * tlist))) * \
            np.exp(-kappa * tlist)
        na_ex = 0.5 * (1 + np.cos(2 * g * np.sqrt(n + 1) * tlist)) * \
            np.exp(-gamma * tlist)

        assert_(max(abs(nc - nc_ex)) < 0.005, True)
        assert_(max(abs(na - na_ex)) < 0.005, True)

    def testCase3(self):
        "mesolve: cavity-qubit with interaction, decay"

        use_rwa = True
        N = 4           # number of cavity fock states
        wc = 2 * np.pi * 1.0   # cavity frequency
        wa = 2 * np.pi * 1.0   # atom frequency
        g = 2 * np.pi * 0.1   # coupling strength
        kappa = 0.05    # cavity dissipation rate
        gamma = 0.001   # atom dissipation rate
        pump = 0.25    # atom pump rate

        # start with an excited atom and maximum number of photons
        n = N - 2
        psi0 = tensor(basis(N, n), basis(2, 1))
        tlist = np.linspace(0, 200, 500)

        nc, na = self.jc_integrate(
            N, wc, wa, g, kappa, gamma, pump, psi0, use_rwa, tlist)

        # we don't have any analytics for this parameters, so
        # compare with the steady state
        nc_ss, na_ss = self.jc_steadystate(
            N, wc, wa, g, kappa, gamma, pump, psi0, use_rwa, tlist)

        nc_ss = nc_ss * np.ones(np.shape(nc))
        na_ss = na_ss * np.ones(np.shape(na))

        assert_(abs(nc[-1] - nc_ss[-1]) < 0.005, True)
        assert_(abs(na[-1] - na_ss[-1]) < 0.005, True)

# percent error for failure
me_error = 1e-8


class TestMESolverConstDecay:
    """
    A test class for the time-dependent ode check function.
    """

    def testMEDecay(self):
        "mesolve: simple constant decay"

        N = 10  # number of basis states to consider
        a = destroy(N)
        H = a.dag() * a
        psi0 = basis(N, 9)  # initial state
        kappa = 0.2  # coupling to oscillator
        c_op_list = [np.sqrt(kappa) * a]
        tlist = np.linspace(0, 10, 100)
        medata = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a])
        expt = medata.expect[0]
        actual_answer = 9.0 * np.exp(-kappa * tlist)
        avg_diff = np.mean(abs(actual_answer - expt) / actual_answer)
        assert_(avg_diff < me_error)

    def testMEDecaySingleCollapse(self):
        "mesolve: simple constant decay"

        N = 10  # number of basis states to consider
        a = destroy(N)
        H = a.dag() * a
        psi0 = basis(N, 9)  # initial state
        kappa = 0.2  # coupling to oscillator
        c_op = np.sqrt(kappa) * a
        tlist = np.linspace(0, 10, 100)
        medata = mesolve(H, psi0, tlist, [c_op], [a.dag() * a])
        expt = medata.expect[0]
        actual_answer = 9.0 * np.exp(-kappa * tlist)
        avg_diff = np.mean(abs(actual_answer - expt) / actual_answer)
        assert_(avg_diff < me_error)


    def testMEDecayAsFuncList(self):
        "mesolve: constant decay as function list"

        N = 10  # number of basis states to consider
        a = destroy(N)
        H = a.dag() * a
        psi0 = basis(N, 9)  # initial state
        kappa = 0.2  # coupling to oscillator

        def sqrt_kappa(t, args):
            return np.sqrt(kappa)
        c_op_list = [[a, sqrt_kappa]]
        tlist = np.linspace(0, 10, 100)
        medata = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a])
        expt = medata.expect[0]
        actual_answer = 9.0 * np.exp(-kappa * tlist)
        avg_diff = np.mean(abs(actual_answer - expt) / actual_answer)
        assert_(avg_diff < me_error)

    def testMEDecayAsStrList(self):
        "mesolve: constant decay as string list"

        N = 10  # number of basis states to consider
        a = destroy(N)
        H = a.dag() * a
        psi0 = basis(N, 9)  # initial state
        kappa = 0.2  # coupling to oscillator
        c_op_list = [[a, 'sqrt(k)']]
        args = {'k': kappa}
        tlist = np.linspace(0, 10, 100)
        medata = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a], args=args)
        expt = medata.expect[0]
        actual_answer = 9.0 * np.exp(-kappa * tlist)
        avg_diff = np.mean(abs(actual_answer - expt) / actual_answer)
        assert_(avg_diff < me_error)


# average error for failure
me_error = 1e-6


class TestMESolveTDDecay:
    """
    A test class for the time-dependent odes.  Comparing to analytic answer

    N(t)=9 * exp[ -kappa*( 1-exp(-t) ) ]

    """

    def testMETDDecayAsFuncList(self):
        "mesolve: time-dependence as function list"

        N = 10  # number of basis states to consider
        a = destroy(N)
        H = a.dag() * a
        psi0 = basis(N, 9)  # initial state
        kappa = 0.2  # coupling to oscillator

        def sqrt_kappa(t, args):
            return np.sqrt(kappa * np.exp(-t))
        c_op_list = [[a, sqrt_kappa]]
        tlist = np.linspace(0, 10, 100)
        medata = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a])
        expt = medata.expect[0]
        actual_answer = 9.0 * np.exp(-kappa * (1.0 - np.exp(-tlist)))
        avg_diff = np.mean(abs(actual_answer - expt) / actual_answer)
        assert_(avg_diff < me_error)

    def testMETDDecayAsPartFuncList(self):
        "mesolve: time-dependence as partial function list"

        me_error = 1e-5
        N = 10
        a = destroy(N)
        H = num(N)
        psi0 = basis(N, 9)
        tlist = np.linspace(0, 10, 100)
        c_ops = [[[a, partial(lambda t, args, k:
                              np.sqrt(k * np.exp(-t)), k=kappa)]]
                 for kappa in [0.05, 0.1, 0.2]]

        for idx, kappa in enumerate([0.05, 0.1, 0.2]):
            medata = mesolve(H, psi0, tlist, c_ops[idx], [H])
            ref = 9.0 * np.exp(-kappa * (1.0 - np.exp(-tlist)))
            avg_diff = np.mean(abs(ref - medata.expect[0]) / ref)
            assert_(avg_diff < me_error)

    def testMETDDecayAsStrList(self):
        "mesolve: time-dependence as string list"

        N = 10  # number of basis states to consider
        a = destroy(N)
        H = a.dag() * a
        psi0 = basis(N, 9)  # initial state
        kappa = 0.2  # coupling to oscillator
        c_op_list = [[a, 'sqrt(k*exp(-t))']]
        args = {'k': kappa}
        tlist = np.linspace(0, 10, 100)
        medata = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a], args=args)
        expt = medata.expect[0]
        actual_answer = 9.0 * np.exp(-kappa * (1.0 - np.exp(-tlist)))
        avg_diff = np.mean(abs(actual_answer - expt) / actual_answer)
        assert_(avg_diff < me_error)

    def testMETDDecayAsArray(self):
        "mesolve: time-dependence as array"

        N = 10
        a = destroy(N)
        H = a.dag() * a
        psi0 = basis(N, 9)
        kappa = 0.2
        tlist = np.linspace(0, 10, 1000)
        c_op_list = [[a, np.sqrt(kappa * np.exp(-tlist))]]
        medata = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a])
        expt = medata.expect[0]
        actual_answer = 9.0 * np.exp(-kappa * (1.0 - np.exp(-tlist)))
        avg_diff = np.mean(abs(actual_answer - expt) / actual_answer)
        assert_(avg_diff < 100 * me_error)

    def testMETDDecayAsFunc(self):
        "mesolve: time-dependent Liouvillian as single function"

        N = 10  # number of basis states to consider
        a = destroy(N)
        H = a.dag() * a
        rho0 = ket2dm(basis(N, 9))  # initial state
        kappa = 0.2  # coupling to oscillator

        def Liouvillian_func(t, args):
            c = np.sqrt(kappa * np.exp(-t))*a
            return liouvillian(H, [c])

        tlist = np.linspace(0, 10, 100)
        args = {'kappa': kappa}
        out1 = mesolve(Liouvillian_func, rho0, tlist, [], [], args=args)
        expt = expect(a.dag()*a, out1.states)
        actual_answer = 9.0 * np.exp(-kappa * (1.0 - np.exp(-tlist)))
        avg_diff = np.mean(abs(actual_answer - expt) / actual_answer)
        assert_(avg_diff < me_error)


# average error for failure
# me_error = 1e-6

class TestMESolveSuperInit:
    """
    A test class comparing mesolve run with an identity super-operator and
    a density matrix as initial conditions, respectively.
    """

    def fidelitycheck(self, out1, out2, rho0vec):
        fid = np.zeros(len(out1.states))
        for i, E in enumerate(out2.states):
            rhot = vector_to_operator(E*rho0vec)
            fid[i] = fidelity(out1.states[i], rhot)
        return fid

    def jc_integrate(self, N, wc, wa, g, kappa, gamma,
                     pump, psi0, use_rwa, tlist):
        # Hamiltonian
        a = tensor(destroy(N), identity(2))
        sm = tensor(identity(N), destroy(2))

        # Identity super-operator
        E0 = sprepost(tensor(qeye(N), qeye(2)), tensor(qeye(N), qeye(2)))

        if use_rwa:
            # use the rotating wave approxiation
            H = wc * a.dag() * a + wa * sm.dag() * sm + g * (
                a.dag() * sm + a * sm.dag())
        else:
            H = wc * a.dag() * a + wa * sm.dag() * sm + g * (
                a.dag() + a) * (sm + sm.dag())

        # collapse operators
        c_op_list = []

        n_th_a = 0.0  # zero temperature

        rate = kappa * (1 + n_th_a)
        c_op_list.append(np.sqrt(rate) * a)

        rate = kappa * n_th_a
        if rate > 0.0:
            c_op_list.append(np.sqrt(rate) * a.dag())

        rate = gamma
        if rate > 0.0:
            c_op_list.append(np.sqrt(rate) * sm)

        rate = pump
        if rate > 0.0:
            c_op_list.append(np.sqrt(rate) * sm.dag())

        # evolve and calculate expectation values
        output1 = mesolve(H, psi0, tlist, c_op_list, [])
        output2 = mesolve(H, E0, tlist, c_op_list, [])
        return output1, output2

    def testSuperJC(self):
        "mesolve: super vs. density matrix as initial condition"
        me_error = 1e-6

        use_rwa = True
        N = 4           # number of cavity fock states
        wc = 2 * np.pi * 1.0   # cavity frequency
        wa = 2 * np.pi * 1.0   # atom frequency
        g = 2 * np.pi * 0.1   # coupling strength
        kappa = 0.05    # cavity dissipation rate
        gamma = 0.001   # atom dissipation rate
        pump = 0.25    # atom pump rate

        # start with an excited atom and maximum number of photons
        n = N - 2
        psi0 = tensor(basis(N, n), basis(2, 1))
        rho0vec = operator_to_vector(psi0*psi0.dag())
        tlist = np.linspace(0, 100, 50)

        out1, out2 = self.jc_integrate(
            N, wc, wa, g, kappa, gamma, pump, psi0, use_rwa, tlist)

        fid = self.fidelitycheck(out1, out2, rho0vec)
        assert_(max(abs(1.0-fid)) < me_error, True)

    def testMETDDecayAsFuncList(self):
        "mesolve: time-dependence as function list with super as init cond"
        me_error = 1e-6

        N = 10  # number of basis states to consider
        a = destroy(N)
        H = a.dag() * a
        psi0 = basis(N, 9)  # initial state
        rho0vec = operator_to_vector(psi0*psi0.dag())
        E0 = sprepost(qeye(N), qeye(N))
        kappa = 0.2  # coupling to oscillator

        def sqrt_kappa(t, args):
            return np.sqrt(kappa * np.exp(-t))
        c_op_list = [[a, sqrt_kappa]]
        tlist = np.linspace(0, 10, 100)
        out1 = mesolve(H, psi0, tlist, c_op_list, [])
        out2 = mesolve(H, E0, tlist, c_op_list, [])

        fid = self.fidelitycheck(out1, out2, rho0vec)
        assert_(max(abs(1.0-fid)) < me_error, True)

    def testMETDDecayAsPartFuncList(self):
        "mesolve: time-dep. as partial function list with super as init cond"
        me_error = 1e-5

        N = 10
        a = destroy(N)
        H = num(N)
        psi0 = basis(N, 9)
        rho0vec = operator_to_vector(psi0*psi0.dag())
        E0 = sprepost(qeye(N), qeye(N))
        tlist = np.linspace(0, 10, 100)
        c_ops = [[[a, partial(lambda t, args, k:
                              np.sqrt(k * np.exp(-t)), k=kappa)]]
                 for kappa in [0.05, 0.1, 0.2]]

        for idx, kappa in enumerate([0.05, 0.1, 0.2]):
            out1 = mesolve(H, psi0, tlist, c_ops[idx], [])
            out2 = mesolve(H, E0, tlist, c_ops[idx], [])
            fid = self.fidelitycheck(out1, out2, rho0vec)
            assert_(max(abs(1.0-fid)) < me_error, True)

    def testMETDDecayAsStrList(self):
        "mesolve: time-dependence as string list with super as init cond"
        me_error = 1e-6

        N = 10  # number of basis states to consider
        a = destroy(N)
        H = a.dag() * a
        psi0 = basis(N, 9)  # initial state
        rho0vec = operator_to_vector(psi0*psi0.dag())
        E0 = sprepost(qeye(N), qeye(N))
        kappa = 0.2  # coupling to oscillator
        c_op_list = [[a, 'sqrt(k*exp(-t))']]
        args = {'k': kappa}
        tlist = np.linspace(0, 10, 100)
        out1 = mesolve(H, psi0, tlist, c_op_list, [], args=args)
        out2 = mesolve(H, E0, tlist, c_op_list, [], args=args)
        fid = self.fidelitycheck(out1, out2, rho0vec)
        assert_(max(abs(1.0-fid)) < me_error, True)

    def testMETDDecayAsArray(self):
        "mesolve: time-dependence as array with super as init cond"
        me_error = 1e-5

        N = 10
        a = destroy(N)
        H = a.dag() * a
        psi0 = basis(N, 9)
        rho0vec = operator_to_vector(psi0*psi0.dag())
        E0 = sprepost(qeye(N), qeye(N))
        kappa = 0.2
        tlist = np.linspace(0, 10, 1000)
        c_op_list = [[a, np.sqrt(kappa * np.exp(-tlist))]]
        out1 = mesolve(H, psi0, tlist, c_op_list, [])
        out2 = mesolve(H, E0, tlist, c_op_list, [])
        fid = self.fidelitycheck(out1, out2, rho0vec)
        assert_(max(abs(1.0-fid)) < me_error, True)

    def testMETDDecayAsFunc(self):
        "mesolve: time-dependence as function with super as init cond"

        N = 10  # number of basis states to consider
        a = destroy(N)
        H = a.dag() * a
        rho0 = ket2dm(basis(N, 9))  # initial state
        rho0vec = operator_to_vector(rho0)
        E0 = sprepost(qeye(N), qeye(N))
        kappa = 0.2  # coupling to oscillator

        def Liouvillian_func(t, args):
            c = np.sqrt(kappa * np.exp(-t))*a
            data = liouvillian(H, [c])
            return data

        tlist = np.linspace(0, 10, 100)
        args = {'kappa': kappa}
        out1 = mesolve(Liouvillian_func, rho0, tlist, [], [], args=args)
        out2 = mesolve(Liouvillian_func, E0, tlist, [], [], args=args)

        fid = self.fidelitycheck(out1, out2, rho0vec)
        assert_(max(abs(1.0-fid)) < me_error, True)

    def test_me_interp1(self):
        "mesolve: interp time-dependent collapse operator #1"

        N = 10  # number of basis states to consider
        kappa = 0.2  # coupling to oscillator
        tlist = np.linspace(0, 10, 100)
        a = destroy(N)
        H = a.dag() * a
        psi0 = basis(N, 9)  # initial state
        S = Cubic_Spline(tlist[0],tlist[-1], np.sqrt(kappa*np.exp(-tlist)))
        c_op_list = [[a, S]]
        medata = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a])
        expt = medata.expect[0]
        actual_answer = 9.0 * np.exp(-kappa * (1.0 - np.exp(-tlist)))
        avg_diff = np.mean(abs(actual_answer - expt) / actual_answer)
        assert_(avg_diff < 1e-5)

    def test_me_interp2(self):
         "mesolve: interp time-dependent collapse operator #2"

         N = 10  # number of basis states to consider
         kappa = 0.2  # coupling to oscillator
         tlist = np.linspace(0, 10, 100)
         C = Cubic_Spline(tlist[0], tlist[-1], np.ones_like(tlist))
         S = Cubic_Spline(tlist[0],tlist[-1], np.sqrt(kappa*np.exp(-tlist)))
         a = destroy(N)
         H = [[a.dag() * a, C]]
         psi0 = basis(N, 9)  # initial state
         c_op_list = [[a, S]]
         medata = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a])
         expt = medata.expect[0]
         actual_answer = 9.0 * np.exp(-kappa * (1.0 - np.exp(-tlist)))
         avg_diff = np.mean(abs(actual_answer - expt) / actual_answer)
         assert_(avg_diff < 1e-5)

    def test_me_interp3(self):
        "mesolve: interp time-dependent collapse operator #3"

        N = 10  # number of basis states to consider
        kappa = 0.2  # coupling to oscillator
        tlist = np.linspace(0, 10, 100)
        C = Cubic_Spline(tlist[0], tlist[-1], np.ones_like(tlist))
        S = Cubic_Spline(tlist[0],tlist[-1], np.sqrt(kappa*np.exp(-tlist)))
        a = destroy(N)
        H = [a.dag() * a, [a.dag() * a, C]]
        psi0 = basis(N, 9)  # initial state
        c_op_list = [[a, S]]
        medata = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a])
        expt = medata.expect[0]
        actual_answer = 9.0 * np.exp(-kappa * (1.0 - np.exp(-tlist)))
        avg_diff = np.mean(abs(actual_answer - expt) / actual_answer)
        assert_(avg_diff < 1e-5)

    def test_me_interp4(self):
        "mesolve: interp time-dependent collapse operator #4"

        N = 10  # number of basis states to consider
        kappa = 0.2  # coupling to oscillator
        tlist = np.linspace(0, 10, 100)
        C = Cubic_Spline(tlist[0], tlist[-1], np.ones_like(tlist))
        S = Cubic_Spline(tlist[0],tlist[-1], np.sqrt(kappa*np.exp(-tlist)))
        a = destroy(N)
        H = [a.dag() * a, [a.dag() * a, C]]
        psi0 = basis(N, 9)  # initial state
        c_op_list = [[a, S],[a, S]]
        medata = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a])
        expt = medata.expect[0]
        actual_answer = 9.0 * np.exp(-2*kappa * (1.0 - np.exp(-tlist)))
        avg_diff = np.mean(abs(actual_answer - expt) / actual_answer)
        assert_(avg_diff < 1e-5)


class TestMESolverMisc:
    """
    A test class for the misc mesolve features.
    """
    def testMEFinalState(self):
        "mesolve: final_state has correct dims"

        N = 5
        psi0 = tensor(basis(N+1,0), basis(N+1,0), basis(N+1,N))
        a = tensor(destroy(N+1), qeye(N+1), qeye(N+1))
        b = tensor(qeye(N+1), destroy(N+1), qeye(N+1))
        c = tensor(qeye(N+1), qeye(N+1), destroy(N+1))
        H = a*b*c.dag() * c.dag() - a.dag()*b.dag()*c * c

        times = np.linspace(0.0, 2.0, 100)
        opts = Options(store_states=False, store_final_state=True)
        rho0 = ket2dm(psi0)
        result = mesolve(H, rho0, times, [], [a.dag()*a, b.dag()*b, c.dag()*c],
                         options=opts)
        assert_(rho0.dims == result.final_state.dims)


    def testSEFinalState(self):
        "sesolve: final_state has correct dims"

        N = 5
        psi0 = tensor(basis(N+1,0), basis(N+1,0), basis(N+1,N))
        a = tensor(destroy(N+1), qeye(N+1), qeye(N+1))
        b = tensor(qeye(N+1), destroy(N+1), qeye(N+1))
        c = tensor(qeye(N+1), qeye(N+1), destroy(N+1))
        H = a*b*c.dag() * c.dag() - a.dag()*b.dag()*c * c

        times = np.linspace(0.0, 2.0, 100)
        opts = Options(store_states=False, store_final_state=True)
        result = mesolve(H, psi0, times, [], [a.dag()*a, b.dag()*b, c.dag()*c],
                         options=opts)
        assert_(psi0.dims == result.final_state.dims)

    def test_num_collapse_set(self):
        H = sigmaz()
        psi = (basis(2, 0) + basis(2, 1)).unit()
        ts = [0, 1]
        for c_ops in (
            sigmax(),
            [sigmax()],
            [sigmay(), sigmax()],
        ):
            res = qutip.mesolve(H, psi, ts, c_ops=c_ops)
            if not isinstance(c_ops, list):
                c_ops = [c_ops]
            assert res.num_collapse == len(c_ops)

    # All Hamiltonians should have dimensions [3, 2], so "bad" states can have
    # [2, 3] instead - the aim is to test mismatch in most cases.
    @pytest.mark.parametrize(['state'], [
        pytest.param(basis([2, 3], [0, 0]), id='ket_bad_tensor'),
        pytest.param(basis([2, 3], [0, 0]).proj(), id='dm_bad_tensor'),
        pytest.param(basis([2, 3], [0, 0]).dag(), id='bra_bad_tensor'),
        pytest.param(to_super(basis([2, 3], [0, 0]).proj()),
                     id='super_bad_tensor'),
        pytest.param(basis([3, 2], [0, 0]).dag(), id='bra_good_tensor'),
        pytest.param(operator_to_vector(basis([3, 2], [0, 0]).proj()),
                     id='operket_good_tensor'),
        pytest.param(operator_to_vector(basis([3, 2], [0, 0]).proj()).dag(),
                     id='operbra_good_tensor'),
        pytest.param(tensor(basis(2, 0), qeye(3)), id='nonsquare_operator'),
    ])
    @pytest.mark.parametrize(['operator'], [
        pytest.param(qeye([3, 2]), id='constant_hamiltonian'),
        pytest.param(liouvillian(qeye([3, 2]), []), id='constant_liouvillian'),
        pytest.param([[qeye([3, 2]), lambda t, args: 1]], id='py_scalar'),
        pytest.param(lambda t, args: qeye([3, 2]), id='py_hamiltonian'),
        pytest.param(lambda t, args: liouvillian(qeye([3, 2]), []),
                     id='py_liouvillian'),
    ])
    def test_incorrect_state_caught(self, state, operator):
        """
        Test that mesolve will correctly catch an input state that is not a
        correctly shaped state, relative to the Hamiltonian or Liouvillian.

        Regression test for gh-1456.
        """
        times = [0, 1e-5]
        c_op = qeye([3, 2])
        with pytest.raises(ValueError):
            mesolve(operator, state, times, c_ops=[c_op])


class TestMESolveStepFuncCoeff:
    """
    A Test class for using time-dependent array coefficients
    as step functions instead of doing interpolation
    """
    def python_coeff(self, t, args):
        if t < np.pi/2:
            return 1.
        else:
            return 0.

    def test_py_coeff(self):
        """
        Test for Python function as coefficient as step function coeff
        """
        rho0 = rand_ket(2)
        tlist = np.array([0, np.pi/2])
        qu = QobjEvo([[sigmax(), self.python_coeff]],
                     tlist=tlist, args={"_step_func_coeff": 1})
        result = mesolve(qu, rho0=rho0, tlist=tlist)
        assert(qu.type == "func")
        assert_allclose(
            fidelity(result.states[-1], sigmax()*rho0), 1, rtol=1.e-7)

    def test_array_cte_coeff(self):
        """
        Test for Array coefficient with uniform tlist as step function coeff
        """
        rho0 = rand_ket(2)
        tlist = np.array([0., np.pi/2, np.pi], dtype=float)
        npcoeff = np.array([0.25, 0.75, 0.75])
        qu = QobjEvo([[sigmax(), npcoeff]],
                     tlist=tlist, args={"_step_func_coeff": 1})
        result = mesolve(qu, rho0=rho0, tlist=tlist)
        assert(qu.type == "array")
        assert_allclose(
            fidelity(result.states[-1], sigmax()*rho0), 1, rtol=1.e-7)

    def test_array_t_coeff(self):
        """
        Test for Array with non-uniform tlist as step function coeff
        """
        rho0 = rand_ket(2)
        tlist = np.array([0., np.pi/2, np.pi*3/2], dtype=float)
        npcoeff = np.array([0.5, 0.25, 0.25])
        qu = QobjEvo([[sigmax(), npcoeff]],
                     tlist=tlist, args={"_step_func_coeff": 1})
        result = mesolve(qu, rho0=rho0, tlist=tlist)
        assert(qu.type == "array")
        assert_allclose(
            fidelity(result.states[-1], sigmax()*rho0), 1, rtol=1.e-7)

    def test_array_str_coeff(self):
        """
        Test for Array and string as step function coeff.
        qobjevo_codegen is used and uniform tlist
        """
        rho0 = rand_ket(2)
        tlist = np.array([0., np.pi/2, np.pi], dtype=float)
        npcoeff1 = np.array([0.25, 0.75, 0.75], dtype=complex)
        npcoeff2 = np.array([0.5, 1.5, 1.5], dtype=float)
        strcoeff = "1."
        qu = QobjEvo(
            [[sigmax(), npcoeff1], [sigmax(), strcoeff], [sigmax(), npcoeff2]],
            tlist=tlist, args={"_step_func_coeff": 1})
        result = mesolve(qu, rho0=rho0, tlist=tlist)
        assert_allclose(
            fidelity(result.states[-1], sigmax()*rho0), 1, rtol=1.e-7)

    def test_array_str_py_coeff(self):
        """
        Test for Array, string and Python function as step function coeff.
        qobjevo_codegen is used and non non-uniform tlist
        """
        rho0 = rand_ket(2)
        tlist = np.array([0., np.pi/4, np.pi/2, np.pi], dtype=float)
        npcoeff1 = np.array([0.4, 1.6, 1.0, 1.0], dtype=complex)
        npcoeff2 = np.array([0.4, 1.6, 1.0, 1.0], dtype=float)
        strcoeff = "1."
        qu = QobjEvo(
            [[sigmax(), npcoeff1], [sigmax(), npcoeff2],
             [sigmax(), self.python_coeff], [sigmax(), strcoeff]],
            tlist=tlist, args={"_step_func_coeff": 1})
        result = mesolve(qu, rho0=rho0, tlist=tlist)
        assert(qu.type == "mixed_callable")
        assert_allclose(
            fidelity(result.states[-1], sigmax()*rho0), 1, rtol=1.e-7)

    def test_dynamic_args(self):
        "sesolve: state feedback"
        tol = 1e-3
        def f(t, args):
            return np.sqrt(args["state_vec"][3])

        H = [qeye(2), [destroy(2)+create(2), f]]
        res = mesolve(H, basis(2,1), tlist=np.linspace(0,10,11),
                      c_ops=[qeye(2)],
                      e_ops=[num(2)], args={"state_vec":basis(2,1)})
        assert_(max(abs(res.expect[0][5:])) < tol,
            msg="evolution with feedback not proceding as expected")

        def f(t, args):
            return np.sqrt(args["expect_op_0"])

        H = [qeye(2), [destroy(2)+create(2), f]]
        res = mesolve(H, basis(2,1), tlist=np.linspace(0,10,11),
                      c_ops=[qeye(2)],
                      e_ops=[num(2)], args={"expect_op_0":num(2)})
        assert_(max(abs(res.expect[0][5:])) < tol,
            msg="evolution with feedback not proceding as expected")


def test_non_hermitian_dm():
    """Test that mesolve works correctly for density matrices that are
    not Hermitian.
    See Issue #1460
    """
    N = 2
    a = destroy(N)
    x = (a + a.dag())/np.sqrt(2)
    H = a.dag() * a

    # Create non-Hermitian initial state.
    rho0 = x*fock_dm(N, 0)

    tlist = np.linspace(0, 0.1, 2)

    options = Options()
    options.store_final_state = True
    options.store_states = True

    result = mesolve(H, rho0, tlist, e_ops=[x], options=options)

    msg = ('Mesolve is not working properly with a non Hermitian density' +
       ' matrix as input. Check computation of '
      )

    imag_part = np.abs(np.imag(result.expect[0][-1]))
    # Since we used an initial state that is not Hermitian, the expectation of
    # x must be imaginary for t>0.
    assert_(imag_part > 0,
            msg + "expectation values. They should be imaginary")

    # Check that the output state is not hermitian since the input was not
    # Hermitian either.
    assert_(not result.final_state.isherm,
            msg + " final density  matrix. It should not be hermitian")
    assert_(not result.states[-1].isherm,
            msg + " states. They should not be hermitian.")

    # Check that when suing a callable we get imaginary expectation values.
    def callable_x(t, rho):
        "Dummy callable_x expectation operator."
        return expect(rho, x)
    result = mesolve(H, rho0, tlist, e_ops=callable_x)

    imag_part = np.abs(np.imag(result.expect[-1]))
    assert_(imag_part > 0,
            msg + "expectation values when using callable operator." +
            "They should be imaginary.")


def test_tlist_h_with_constant_c_ops():
    """
    Test that it's possible to mix a time-dependent Hamiltonian given as a
    QobjEvo with interpolated coefficients with time-independent collapse
    operators, if the solver times are not equal to the interpolation times of
    the Hamiltonian.

    See gh-1560.
    """
    state = basis(2, 0)
    all_times = np.linspace(0, 1, 11)
    few_times = np.linspace(0, 1, 3)
    dependence = np.cos(2*np.pi * all_times)
    hamiltonian = QobjEvo([[sigmax(), dependence]], tlist=all_times)
    collapse = qeye(2)
    result = mesolve(hamiltonian, state, few_times, c_ops=[collapse])
    assert result.num_collapse == 1
    assert len(result.states) == len(few_times)


def test_mixed_e_ops():
    """
    Test callable and Qobj e_ops can mix.

    See gh-2118.
    """
    state = basis(2, 0)
    hamiltonian = QobjEvo(sigmax())
    collapse = create(2)
    e_ops = [qeye(2), lambda t, qobj: qobj.norm()]
    result = mesolve(
        hamiltonian, state, [0, 1, 2], c_ops=[collapse], e_ops=e_ops
    )
    assert result.num_expect == 2


def test_tlist_h_with_other_tlist_c_ops_raises():
    state = basis(2, 0)
    all_times = np.linspace(0, 1, 11)
    few_times = np.linspace(0, 1, 3)
    dependence = np.cos(2*np.pi * all_times)
    hamiltonian = QobjEvo([[sigmax(), dependence]], tlist=all_times)
    collapse = [qeye(2), np.cos(2*np.pi * few_times)]
    with pytest.raises(ValueError) as exc:
        mesolve(hamiltonian, state, few_times, c_ops=[collapse])
    assert str(exc.value) == "Time lists are not compatible"


def test_mesolve_bad_e_ops():
    H = sigmaz()
    c_ops = [sigmax()]
    psi0 = basis(2, 0)
    tlist = np.linspace(0, 20, 200)
    with pytest.raises(TypeError) as exc:
        mesolve(H, psi0, tlist=tlist, c_ops=c_ops, e_ops=[qeye(3)])
