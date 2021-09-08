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
from numpy.testing import assert_allclose
from types import FunctionType

from qutip import *
from qutip.solver import *
from qutip.random_objects import rand_ket
import pickle
import pytest
all_ode_method = integrator_collection.list_keys('methods', time_dependent=True)


def fidelitycheck(out1, out2, rho0vec):
    fid = np.zeros(len(out1.states))
    for i, E in enumerate(out2.states):
        rhot = vector_to_operator(E*rho0vec)
        fid[i] = fidelity(out1.states[i], rhot)
    return fid


class TestMESolveDecay:
    N = 10
    a = destroy(N)
    kappa = 0.2
    tlist = np.linspace(0, 10, 201)
    ada = a.dag() * a

    @pytest.fixture(params=[
        pytest.param([ada, lambda t, args: 1], id='Hlist_func'),
        pytest.param([ada, '1'], id='Hlist_str'),
        pytest.param([ada, Cubic_Spline(0, 10, np.ones_like(tlist))],
                      id='Hlist_cubic_spline'),
        pytest.param([ada, np.ones_like(tlist)], id='Hlist_array'),
        pytest.param(QobjEvo([ada, '1']), id='HQobjEvo'),
        pytest.param(lambda t, args: create(10) * destroy(10),
                     id='func'),
    ])
    def H(self, request):
        return request.param

    @pytest.fixture(params=[
        pytest.param(np.sqrt(kappa) * a,
                  id='const'),
        pytest.param(lambda t, args: np.sqrt(args['kappa']) * destroy(10),
                     id='func'),
        pytest.param([a, lambda t, args: np.sqrt(args['kappa'])],
                  id='list_func'),
        pytest.param([a, 'sqrt(kappa)'],
                  id='list_str'),
        pytest.param([a, Cubic_Spline(0, 10, np.sqrt(kappa) *
                                      np.ones_like(tlist))],
                  id='list_cubic_spline'),
        pytest.param([a, np.sqrt(kappa) * np.ones_like(tlist)],
                  id='list_array'),
        pytest.param(QobjEvo([a, 'sqrt(kappa)'], args={'kappa': kappa}),
                  id='QobjEvo'),
    ])
    def cte_c_ops(self, request):
        return request.param

    @pytest.fixture(params=[
        pytest.param([a, lambda t, args: np.sqrt(args['kappa'] * np.exp(-t))],
                  id='list_func'),
        pytest.param([a, 'sqrt(kappa * exp(-t))'],
                  id='list_str'),
        pytest.param([a, Cubic_Spline(0, 10, np.sqrt(kappa * np.exp(-tlist)))],
                  id='list_cubic_spline'),
        pytest.param([a, np.sqrt(kappa * np.exp(-tlist))],
                  id='list_array'),
        pytest.param(QobjEvo([a, 'sqrt(kappa * exp(-t))'],
                          args={'kappa': kappa}),
                  id='QobjEvo'),
        pytest.param(lambda t, args: (np.sqrt(args['kappa'] * np.exp(-t)) *
                                      destroy(10)),
                     id='func'),
    ])
    def c_ops(self, request):
        return request.param

    c_ops_1 = c_ops

    def testME_CteDecay(self, cte_c_ops):
        "mesolve: simple constant decay"
        me_error = 1e-6
        H = self.a.dag() * self.a
        psi0 = basis(self.N, 9)  # initial state
        c_op_list = [cte_c_ops]
        options = SolverOptions(progress_bar=None)
        medata = mesolve(H, psi0, self.tlist, c_op_list, [H],
                         args={"kappa": self.kappa},
                         options=options)
        expt = medata.expect[0]
        actual_answer = 9.0 * np.exp(-self.kappa * self.tlist)
        assert_allclose(actual_answer, expt, me_error)

    @pytest.mark.parametrize('method',
                             all_ode_method, ids=all_ode_method)
    def testME_TDDecay(self, c_ops, method):
        "mesolve: time-dependence as function list"
        me_error = 1e-5
        H = self.a.dag() * self.a
        psi0 = basis(self.N, 9)  # initial state
        c_op_list = [c_ops]
        options = SolverOptions(method=method, progress_bar=None)
        medata = mesolve(H, psi0, self.tlist, c_op_list, [H],
                         args={"kappa": self.kappa}, options=options)
        expt = medata.expect[0]
        actual_answer = 9.0 * np.exp(-self.kappa * (1.0 - np.exp(-self.tlist)))
        assert_allclose(actual_answer, expt, me_error)

    def testME_2TDDecay(self, c_ops, c_ops_1):
        "mesolve: time-dependence as function list"
        me_error = 1e-5
        H = self.a.dag() * self.a
        psi0 = basis(self.N, 9)  # initial state
        c_op_list = [c_ops, c_ops_1]
        options = SolverOptions(progress_bar=None)
        medata = mesolve(H, psi0, self.tlist, c_op_list, [H],
                         args={"kappa": self.kappa},
                         options=options)
        expt = medata.expect[0]
        actual_answer = 9.0 * np.exp(-2 * self.kappa *
                                     (1.0 - np.exp(-self.tlist)))
        assert_allclose(actual_answer, expt, me_error)

    def testME_TDH_TDDecay(self, H, c_ops):
        "mesolve: time-dependence as function list"
        me_error = 5e-6
        psi0 = basis(self.N, 9)  # initial state
        c_op_list = [c_ops]
        options = SolverOptions(progress_bar=None)
        medata = mesolve(H, psi0, self.tlist, c_op_list, [self.ada],
                         args={"kappa": self.kappa},
                         options=options)
        expt = medata.expect[0]
        actual_answer = 9.0 * np.exp(-self.kappa *
                                     (1.0 - np.exp(-self.tlist)))
        np.testing.assert_allclose(actual_answer, expt, me_error)

    def testME_TDH_longTDDecay(self, H, c_ops):
        "mesolve: time-dependence as function list"
        me_error = 2e-5
        psi0 = basis(self.N, 9)  # initial state
        if isinstance(c_ops, FunctionType):
            return
        if isinstance(c_ops, QobjEvo):
            c_op_list = [c_ops + c_ops]
        else:
            c_op_list = [[c_ops, c_ops]]
        options = SolverOptions(progress_bar=None)
        medata = mesolve(H, psi0, self.tlist, c_op_list, [self.ada],
                         args={"kappa": self.kappa},
                         options=options)
        expt = medata.expect[0]
        actual_answer = 9.0 * np.exp(-4 * self.kappa *
                                     (1.0 - np.exp(-self.tlist)))
        np.testing.assert_allclose(actual_answer, expt, me_error)

    def testME_TDDecayUnitary(self, c_ops):
        "mesolve: time-dependence as function list with super as init cond"
        me_error = 5e-6

        H = self.ada
        psi0 = basis(self.N, 9)  # initial state
        rho0vec = operator_to_vector(psi0*psi0.dag())
        E0 = sprepost(qeye(self.N), qeye(self.N))
        options = SolverOptions(progress_bar=None)
        c_op_list = [c_ops]
        out1 = mesolve(H, psi0, self.tlist, c_op_list, [],
                       args={"kappa": self.kappa},
                       options=options)
        out2 = mesolve(H, E0, self.tlist, c_op_list, [],
                       args={"kappa": self.kappa},
                       options=options)

        fid = fidelitycheck(out1, out2, rho0vec)
        assert max(abs(1.0-fid)) < me_error

    def testME_TDDecayliouvillian(self, c_ops):
        "mesolve: time-dependence as function list with super as init cond"
        me_error = 5e-6

        H = self.ada
        L = liouvillian(H)
        psi0 = basis(self.N, 9)  # initial state
        rho0vec = operator_to_vector(psi0*psi0.dag())
        E0 = sprepost(qeye(self.N), qeye(self.N))
        options = SolverOptions(progress_bar=None)
        c_op_list = [c_ops]
        out1 = mesolve(L, psi0, self.tlist, c_op_list, [],
                       args={"kappa": self.kappa},
                       options=options)
        out2 = mesolve(L, E0, self.tlist, c_op_list, [],
                       args={"kappa": self.kappa},
                       options=options)

        fid = fidelitycheck(out1, out2, rho0vec)
        assert max(abs(1.0-fid)) < me_error

    def test_mesolver_pickling(self):
        options = SolverOptions(progress_bar=None)
        solver_obj = MeSolver(self.ada, c_ops=[self.a], e_ops=[self.ada],
                              options=options)
        copy = pickle.loads(pickle.dumps(solver_obj))
        e1 = solver_obj.run(basis(self.N, 9), [0, 1, 2, 3], {}).expect
        e2 = solver_obj.run(basis(self.N, 9), [0, 1, 2, 3], {}).expect
        assert_allclose(e1, e2)

    @pytest.mark.parametrize('method',
                             all_ode_method, ids=all_ode_method)
    def test_mesolver_steping(self, method):
        options = SolverOptions(method=method, progress_bar=None)
        solver_obj = MeSolver(self.ada,
                              c_ops=[[self.a,
                                     lambda t, args: np.sqrt(args['kappa'] *
                                                     np.exp(-t))]],
                              args={"kappa":1.}, options=options)
        solver_obj.start(basis(self.N, 9), 0)
        state1 = solver_obj.step(1)
        assert expect(self.ada, state1) != expect(self.ada, basis(self.N, 9))
        state2 = solver_obj.step(2, {"kappa":0.})
        assert_allclose(expect(self.ada, state1),
                        expect(self.ada, state2), 1e-6)


@pytest.mark.parametrize('super_', ["ket", "dm", "liouvillian"],
                         ids=["ket", "dm", "liouvillian"])
def testME_SesolveFallback(super_):
    "mesolve: final_state has correct dims"
    N = 5
    a = tensor(destroy(N+1), qeye(N+1), qeye(N+1))
    b = tensor(qeye(N+1), destroy(N+1), qeye(N+1))
    c = tensor(qeye(N+1), qeye(N+1), destroy(N+1))
    psi0 = tensor(basis(N+1,0), basis(N+1,0), basis(N+1,N))
    H = a * b * c.dag() * c.dag() + a.dag() * b.dag() * c * c
    if super_ == "dm":
        state0 = ket2dm(psi0)
    else:
        state0 = psi0
    if super_ == "liouvillian":
        H = liouvillian(H)

    times = np.linspace(0.0, 0.1, 3)
    options = SolverOptions(store_states=False, store_final_state=True,
                            progress_bar=None)
    result = mesolve(H, state0, times, [], e_ops=[a], options=options)
    if super_ == "ket":
        assert result.final_state.dims == psi0.dims
    else:
        assert result.final_state.dims == ket2dm(psi0).dims


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

        output = mesolve(H, psi0, tlist, c_op_list,
                         e_ops=[sigmax(), sigmay(), sigmaz()])

        return output.expect[0], output.expect[1], output.expect[2]

    def jc_steadystate(self, N, wc, wa, g, kappa, gamma,
                       pump, psi0, use_rwa, tlist):

        # Hamiltonian
        a = tensor(destroy(N), identity(2))
        sm = tensor(identity(N), destroy(2))

        if use_rwa:
            # use the rotating wave approxiation
            H = (wc * a.dag() * a + wa * sm.dag() * sm
                 + g * (a.dag() * sm + a * sm.dag()))
        else:
            H = (wc * a.dag() * a + wa * sm.dag() * sm
                 + g * (a.dag() + a) * (sm + sm.dag()))

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
                     pump, psi0, use_rwa, tlist, oper_evo=False):

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

        options = SolverOptions(store_states=True, progress_bar=None)

        # evolve and calculate expectation values
        output = mesolve(
            H, psi0, tlist, c_op_list, [a.dag() * a, sm.dag() * sm],
            options=options)
        if oper_evo:
            output2 = mesolve(H, E0, tlist, c_op_list, [])
            return output, output2
        return output.expect[0], output.expect[1]

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
        psi0 = basis([N, 2], [n, 1])
        rho0vec = operator_to_vector(psi0.proj())
        tlist = np.linspace(0, 100, 50)

        out1, out2 = self.jc_integrate(
            N, wc, wa, g, kappa, gamma, pump, psi0, use_rwa, tlist, True)

        fid = fidelitycheck(out1, out2, rho0vec)
        assert np.max(np.abs(1.0 - fid)) < me_error

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

        assert np.max(np.abs(sx - sx_analytic)) < 0.05
        assert np.max(np.abs(sy - sy_analytic)) < 0.05
        assert np.max(np.abs(sz - sz_analytic)) < 0.05

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

        assert np.max(np.abs(sx - sx_analytic)) < 0.05
        assert np.max(np.abs(sy - sy_analytic)) < 0.05
        assert np.max(np.abs(sz - sz_analytic)) < 0.05

    def testCase1(self):
        "mesolve: cavity-qubit interaction, no dissipation"

        use_rwa = True
        N = 4           # number of cavity fock states
        wc = 2 * np.pi * 1.0   # cavity frequency
        wa = 2 * np.pi * 1.0   # atom frequency
        g = 2 * np.pi * 0.01   # coupling strength
        kappa = 0.0     # cavity dissipation rate
        gamma = 0.0     # atom dissipation rate
        pump = 0.0      # atom pump rate

        # start with an excited atom and maximum number of photons
        n = N - 2
        psi0 = tensor(basis(N, n), basis(2, 1))
        tlist = np.linspace(0, 1000, 2000)

        nc, na = self.jc_integrate(
            N, wc, wa, g, kappa, gamma, pump, psi0, use_rwa, tlist)

        nc_ex = 0.5 * (1 - np.cos(2 * g * np.sqrt(n + 1) * tlist)) + n
        na_ex = 0.5 * (1 + np.cos(2 * g * np.sqrt(n + 1) * tlist))

        assert np.max(np.abs(nc - nc_ex)) < 0.005
        assert np.max(np.abs(na - na_ex)) < 0.005

    def testCase2(self):
        "mesolve: cavity-qubit without interaction, decay"

        use_rwa = True
        N = 4           # number of cavity fock states
        wc = 2 * np.pi * 1.0   # cavity frequency
        wa = 2 * np.pi * 1.0   # atom frequency
        g = 2 * np.pi * 0.0    # coupling strength
        kappa = 0.005   # cavity dissipation rate
        gamma = 0.01    # atom dissipation rate
        pump = 0.0      # atom pump rate

        # start with an excited atom and maximum number of photons
        n = N - 2
        psi0 = tensor(basis(N, n), basis(2, 1))
        tlist = np.linspace(0, 1000, 2000)

        nc, na = self.jc_integrate(
            N, wc, wa, g, kappa, gamma, pump, psi0, use_rwa, tlist)

        nc_ex = (0.5 * (1 - np.cos(2 * g * np.sqrt(n + 1) * tlist)) + n) * \
            np.exp(-kappa * tlist)
        na_ex = 0.5 * (1 + np.cos(2 * g * np.sqrt(n + 1) * tlist)) * \
            np.exp(-gamma * tlist)

        assert np.max(np.abs(nc - nc_ex)) < 0.005
        assert np.max(np.abs(na - na_ex)) < 0.005

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

        assert abs(nc[-1] - nc_ss[-1]) < 0.005
        assert abs(na[-1] - na_ss[-1]) < 0.005


class TestMESolveStepFuncCoeff:
    """
    A Test class for using time-dependent array coefficients
    as step functions instead of doing interpolation
    """
    # Runge-Kutta method (dop853) behave better with step function evolution
    # than multi-step methods (adams, qutip 4's default)
    options = SolverOptions(method="dop853", nsteps=1e8, progress_bar=None)

    def python_coeff(self, t, args):
        if t < np.pi/2:
            return 1.
        else:
            return 0.

    @pytest.mark.parametrize('method',
                             all_ode_method, ids=all_ode_method)
    def test_py_coeff(self, method):
        """
        Test for Python function as coefficient as step function coeff
        """
        rho0 = rand_ket(2)
        tlist = np.array([0, np.pi/2])
        options = SolverOptions(method=method, nsteps=1e5)
        qu = QobjEvo([[sigmax(), self.python_coeff]],
                     tlist=tlist, args={"_step_func_coeff": 1})
        result = mesolve(qu, rho0=rho0, tlist=tlist, options=options)
        assert_allclose(
            fidelity(result.states[-1], sigmax()*rho0), 1, rtol=1.e-6)

    def test_array_cte_coeff(self):
        """
        Test for Array coefficient with uniform tlist as step function coeff
        """
        rho0 = rand_ket(2)
        tlist = np.array([0., np.pi/2, np.pi], dtype=float)
        npcoeff = np.array([0.25, 0.75, 0.75])
        qu = QobjEvo([[sigmax(), npcoeff]],
                     tlist=tlist, args={"_step_func_coeff": 1})
        result = mesolve(qu, rho0=rho0, tlist=tlist, options=self.options)
        assert_allclose(
            fidelity(result.states[-1], sigmax()*rho0), 1, rtol=1.e-6)

    def test_array_t_coeff(self):
        """
        Test for Array with non-uniform tlist as step function coeff
        """
        rho0 = rand_ket(2)
        tlist = np.array([0., np.pi/2, np.pi*3/2], dtype=float)
        npcoeff = np.array([0.5, 0.25, 0.25])
        qu = QobjEvo([[sigmax(), npcoeff]],
                     tlist=tlist, args={"_step_func_coeff": 1})
        result = mesolve(qu, rho0=rho0, tlist=tlist, options=self.options)
        assert_allclose(
            fidelity(result.states[-1], sigmax()*rho0), 1, rtol=1.e-6)

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
        result = mesolve(qu, rho0=rho0, tlist=tlist, options=self.options)
        assert_allclose(
            fidelity(result.states[-1], sigmax()*rho0), 1, rtol=1.e-6)

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
        result = mesolve(qu, rho0=rho0, tlist=tlist, options=self.options)
        assert_allclose(
            fidelity(result.states[-1], sigmax()*rho0), 1, rtol=1.e-6)


def test_num_collapse_set():
    H = sigmaz()
    psi = (basis(2, 0) + basis(2, 1)).unit()
    ts = [0, 1]
    for c_ops in (
        sigmax(),
        [sigmax()],
        [sigmay(), sigmax()],
    ):
        res = mesolve(H, psi, ts, c_ops=c_ops)
        if not isinstance(c_ops, list):
            c_ops = [c_ops]
        assert res.num_collapse == len(c_ops)
