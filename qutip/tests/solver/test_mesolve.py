import numpy as np
from types import FunctionType

import qutip
from qutip.solver.mesolve import mesolve, MESolver, MESolverMatrixForm
from qutip.solver.solver_base import Solver
import pickle
import pytest

# Deactivate warning for test without cython
from qutip.core.coefficient import WARN_MISSING_MODULE
WARN_MISSING_MODULE[0] = 0


all_ode_method = [
    method for method, integrator in MESolver.avail_integrators().items()
    if integrator.support_time_dependant
]

def fidelitycheck(out1, out2, rho0vec):
    fid = np.zeros(len(out1.states))
    for i, E in enumerate(out2.states):
        rhot = qutip.vector_to_operator(E*rho0vec)
        fid[i] = qutip.fidelity(out1.states[i], rhot)
    return fid


class TestMESolveDecay:
    N = 10
    a = qutip.destroy(N)
    kappa = 0.2
    tlist = np.linspace(0, 10, 201)
    ada = a.dag() * a

    @pytest.fixture(params=[
        pytest.param(False, id='superop'),
        pytest.param(True, id='matrix_form'),
    ])
    def matrix_form(self, request):
        return request.param

    @pytest.fixture(params=[
        pytest.param([ada, lambda t, args: 1], id='Hlist_func'),
        pytest.param([ada, '1'], id='Hlist_str'),
        pytest.param([ada, np.ones_like(tlist)], id='Hlist_array'),
        pytest.param(qutip.QobjEvo([ada, '1']), id='HQobjEvo'),
        pytest.param(lambda t, args: qutip.create(10) * qutip.destroy(10),
                     id='func'),
    ])
    def H(self, request):
        return request.param

    @pytest.fixture(params=[
        pytest.param(np.sqrt(kappa) * a,
                     id='const'),
        pytest.param(lambda t, args: (np.sqrt(args['kappa'])
                                      * qutip.destroy(10)),
                     id='func'),
        pytest.param([a, lambda t, args: np.sqrt(args['kappa'])],
                     id='list_func'),
        pytest.param([a, 'sqrt(kappa)'],
                     id='list_str'),
        pytest.param([a, np.sqrt(kappa) * np.ones_like(tlist)],
                     id='list_array'),
        pytest.param(qutip.QobjEvo([a, 'sqrt(kappa)'], args={'kappa': kappa}),
                     id='QobjEvo'),
    ])
    def cte_c_ops(self, request):
        return request.param

    @pytest.fixture(params=[
        pytest.param([a, lambda t, args: np.sqrt(args['kappa'] * np.exp(-t))],
                  id='list_func'),
        pytest.param([a, 'sqrt(kappa * exp(-t))'],
                  id='list_str'),
        pytest.param([a, np.sqrt(kappa * np.exp(-tlist))],
                  id='list_array'),
        pytest.param(qutip.QobjEvo([a, 'sqrt(kappa * exp(-t))'],
                          args={'kappa': kappa}),
                  id='QobjEvo'),
        pytest.param(lambda t, args: (np.sqrt(args['kappa'] * np.exp(-t)) *
                                      qutip.destroy(10)),
                     id='func'),
    ])
    def c_ops(self, request):
        return request.param

    c_ops_1 = c_ops

    def testME_CteDecay(self, cte_c_ops, matrix_form):
        "mesolve: simple constant decay"
        me_error = 1e-6
        H = self.a.dag() * self.a
        psi0 = qutip.basis(self.N, 9)  # initial state
        c_op_list = [cte_c_ops]
        options = {"progress_bar": None, "matrix_form": matrix_form}
        medata = mesolve(H, psi0, self.tlist, c_op_list, e_ops=[H],
                         args={"kappa": self.kappa}, options=options)
        expt = medata.expect[0]
        actual_answer = 9.0 * np.exp(-self.kappa * self.tlist)
        np.testing.assert_allclose(actual_answer, expt, atol=me_error)

    @pytest.mark.parametrize('method',
                             all_ode_method, ids=all_ode_method)
    def testME_TDDecay(self, c_ops, method, matrix_form):
        "mesolve: time-dependence as function list"
        me_error = 1e-5
        H = self.a.dag() * self.a
        psi0 = qutip.basis(self.N, 9)  # initial state
        c_op_list = [c_ops]
        options = {"method": method, "progress_bar": None,
                   "matrix_form": matrix_form}
        medata = mesolve(H, psi0, self.tlist, c_op_list, e_ops=[H],
                         args={"kappa": self.kappa}, options=options)
        expt = medata.expect[0]
        actual_answer = 9.0 * np.exp(-self.kappa * (1.0 - np.exp(-self.tlist)))
        np.testing.assert_allclose(actual_answer, expt, rtol=me_error)

    def testME_2TDDecay(self, c_ops, c_ops_1, matrix_form):
        "mesolve: time-dependence as function list"
        me_error = 1e-5
        H = self.a.dag() * self.a
        psi0 = qutip.basis(self.N, 9)  # initial state
        c_op_list = [c_ops, c_ops_1]
        options = {"progress_bar": None, "matrix_form": matrix_form}
        medata = mesolve(H, psi0, self.tlist, c_op_list, e_ops=[H],
                         args={"kappa": self.kappa}, options=options)
        expt = medata.expect[0]
        actual_answer = 9.0 * np.exp(-2 * self.kappa *
                                     (1.0 - np.exp(-self.tlist)))
        np.testing.assert_allclose(actual_answer, expt, atol=me_error)

    def testME_TDH_TDDecay(self, H, c_ops, matrix_form):
        "mesolve: time-dependence as function list"
        me_error = 5e-6
        psi0 = qutip.basis(self.N, 9)  # initial state
        c_op_list = [c_ops]
        options = {"progress_bar": None, "matrix_form": matrix_form}
        medata = mesolve(H, psi0, self.tlist, c_op_list, e_ops=[self.ada],
                         args={"kappa": self.kappa}, options=options)
        expt = medata.expect[0]
        actual_answer = 9.0 * np.exp(-self.kappa *
                                     (1.0 - np.exp(-self.tlist)))
        np.testing.assert_allclose(actual_answer, expt, atol=me_error)

    def testME_TDH_longTDDecay(self, H, c_ops, matrix_form):
        "mesolve: time-dependence as function list"
        me_error = 2e-5
        psi0 = qutip.basis(self.N, 9)  # initial state
        if isinstance(c_ops, FunctionType):
            return
        if isinstance(c_ops, qutip.QobjEvo):
            c_op_list = [c_ops + c_ops]
        else:
            c_op_list = [[c_ops, c_ops]]
        options = {"progress_bar": None, "matrix_form": matrix_form}
        medata = mesolve(H, psi0, self.tlist, c_op_list, e_ops=[self.ada],
                         args={"kappa": self.kappa}, options=options)
        expt = medata.expect[0]
        actual_answer = 9.0 * np.exp(-4 * self.kappa *
                                     (1.0 - np.exp(-self.tlist)))
        np.testing.assert_allclose(actual_answer, expt, atol=me_error)

    def testME_TDDecayUnitary(self, c_ops):
        "mesolve: time-dependence as function list with super as init cond"
        me_error = 5e-6

        H = self.ada
        psi0 = qutip.basis(self.N, 9)  # initial state
        rho0vec = qutip.operator_to_vector(psi0*psi0.dag())
        E0 = qutip.sprepost(qutip.qeye(self.N), qutip.qeye(self.N))
        options = {"progress_bar": None}
        c_op_list = [c_ops]
        out1 = mesolve(H, psi0, self.tlist, c_op_list,
                       args={"kappa": self.kappa},
                       options=options)
        out2 = mesolve(H, E0, self.tlist, c_op_list,
                       args={"kappa": self.kappa},
                       options=options)

        fid = fidelitycheck(out1, out2, rho0vec)
        assert fid == pytest.approx(1., abs=me_error)

    def testME_TDDecayliouvillian(self, c_ops):
        "mesolve: time-dependence as function list with super as init cond"
        me_error = 5e-6

        H = self.ada
        L = qutip.liouvillian(H)
        psi0 = qutip.basis(self.N, 9)  # initial state
        rho0vec = qutip.operator_to_vector(psi0*psi0.dag())
        E0 = qutip.sprepost(qutip.qeye(self.N), qutip.qeye(self.N))
        options = {"progress_bar": None}
        c_op_list = [c_ops]
        out1 = mesolve(L, psi0, self.tlist, c_op_list,
                       args={"kappa": self.kappa},
                       options=options)
        out2 = mesolve(L, E0, self.tlist, c_op_list,
                       args={"kappa": self.kappa},
                       options=options)

        fid = fidelitycheck(out1, out2, rho0vec)
        assert fid == pytest.approx(1., abs=me_error)

    @pytest.mark.parametrize(['state_type', 'use_matrix_form'], [
        pytest.param("ket", False, id="ket-superop"),
        pytest.param("dm", False, id="dm-superop"),
        pytest.param("dm", True, id="dm-matrix_form"),
        pytest.param("unitary", False, id="unitary-superop"),
        # ket and unitary not supported by matrix_form (requires superoperator H)
    ])
    def test_mesolve_normalization(self, state_type, use_matrix_form):
        # non-hermitean H causes state to evolve non-unitarily
        H = qutip.Qobj([[1, -0.1j], [-0.1j, 1]])
        psi0 = qutip.basis(2, 0)
        options = {
            "normalize_output": True,
            "progress_bar": None,
            "atol": 1e-5,
            "nsteps": 1e5,
            "matrix_form": use_matrix_form,
        }

        if state_type in {"ket", "dm"}:
            if state_type == "dm":
                psi0 = qutip.ket2dm(psi0)
            if not use_matrix_form:
                # -i(H ρ - ρ dag(H)) matches matrix form evolution
                H = -1j * qutip.spre(H) + 1j * qutip.spost(H.dag())
            output = mesolve(H, psi0, self.tlist, e_ops=[], options=options)
            # density matrices are normalized by trace
            traces = [state.tr() for state in output.states]
            np.testing.assert_allclose(
                traces, [1.0 for _ in self.tlist], atol=1e-15,
            )
        else:
            # evolution of unitaries should not be normalized (superop only)
            H = -1j * qutip.spre(H) + 1j * qutip.spost(H.dag())
            U = qutip.sprepost(qutip.qeye(2), qutip.qeye(2))
            output = mesolve(H, U, self.tlist, e_ops=[], options=options)
            norms = [state.norm() for state in output.states]
            assert all(norm > 4 for norm in norms[1:])

    @pytest.mark.parametrize('solver_cls', [MESolver, MESolverMatrixForm],
                             ids=['MESolver', 'MESolverMatrixForm'])
    def test_mesolver_pickling(self, solver_cls):
        options = {"progress_bar": None}
        solver_obj = solver_cls(self.ada, c_ops=[self.a], options=options)
        solver_copy = pickle.loads(pickle.dumps(solver_obj))
        e1 = solver_obj.run(qutip.basis(self.N, 9), [0, 1, 2, 3],
                            e_ops=[self.ada]).expect[0]
        e2 = solver_copy.run(qutip.basis(self.N, 9), [0, 1, 2, 3],
                             e_ops=[self.ada]).expect[0]
        np.testing.assert_allclose(e1, e2)

    @pytest.mark.parametrize('solver_cls', [MESolver, MESolverMatrixForm],
                             ids=['MESolver', 'MESolverMatrixForm'])
    def test_mesolver_pickling_after_use(self, solver_cls):
        """Test that solvers can be pickled after being used."""
        options = {"progress_bar": None}
        solver_obj = solver_cls(self.ada, c_ops=[self.a], options=options)
        # Run solver first to populate any internal buffers
        e1 = solver_obj.run(qutip.basis(self.N, 9), [0, 1, 2, 3],
                            e_ops=[self.ada]).expect[0]
        # Pickle after use
        solver_copy = pickle.loads(pickle.dumps(solver_obj))
        # Run unpickled solver
        e2 = solver_copy.run(qutip.basis(self.N, 9), [0, 1, 2, 3],
                             e_ops=[self.ada]).expect[0]
        np.testing.assert_allclose(e1, e2)

    @pytest.mark.parametrize('method',
                             all_ode_method, ids=all_ode_method)
    @pytest.mark.parametrize('solver_cls', [MESolver, MESolverMatrixForm],
                             ids=['MESolver', 'MESolverMatrixForm'])
    def test_mesolver_stepping(self, method, solver_cls):
        options = {"method": method, "progress_bar": None}
        solver_obj = solver_cls(
            self.ada,
            c_ops=qutip.QobjEvo(
                [self.a, lambda t, kappa: np.sqrt(kappa * np.exp(-t))],
                args={'kappa': 1.}
            ),
            options=options
        )
        solver_obj.start(qutip.basis(self.N, 9), 0)
        state1 = solver_obj.step(1)
        assert qutip.expect(self.ada, state1) != (
            qutip.expect(self.ada, qutip.basis(self.N, 9))
        )
        state2 = solver_obj.step(2, args={"kappa": 0.})
        np.testing.assert_allclose(qutip.expect(self.ada, state1),
                                   qutip.expect(self.ada, state2), 1e-6)


@pytest.mark.parametrize('super_', ["ket", "dm", "liouvillian"],
                         ids=["ket", "dm", "liouvillian"])
def testME_SesolveFallback(super_):
    "mesolve: final_state has correct dims"
    N = 3
    a = qutip.tensor(qutip.destroy(N), qutip.qeye(N), qutip.qeye(N))
    b = qutip.tensor(qutip.qeye(N), qutip.destroy(N), qutip.qeye(N))
    c = qutip.tensor(qutip.qeye(N), qutip.qeye(N), qutip.destroy(N))
    psi0 = qutip.tensor(qutip.basis(N, 0),
                        qutip.basis(N, 0),
                        qutip.basis(N, N - 1))
    H = a * b * c.dag() * c.dag() + a.dag() * b.dag() * c * c
    if super_ == "dm":
        state0 = qutip.ket2dm(psi0)
    else:
        state0 = psi0
    if super_ == "liouvillian":
        H = qutip.liouvillian(H)

    times = np.linspace(0.0, 0.1, 3)
    options = {
        "store_states": False,
        "store_final_state": True,
        "progress_bar": None
    }
    result = mesolve(H, state0, times, [], e_ops=[a], options=options)
    if super_ == "ket":
        assert result.final_state.dims == psi0.dims
    else:
        assert result.final_state.dims == qutip.ket2dm(psi0).dims


class TestJCModelEvolution:
    """
    A test class for the QuTiP functions for the evolution of JC model
    """
    def qubit_integrate(self, tlist, psi0, epsilon, delta, g1, g2,
                        matrix_form=False):

        H = epsilon / 2.0 * qutip.sigmaz() + delta / 2.0 * qutip.sigmax()

        c_op_list = []

        rate = g1
        if rate > 0.0:
            c_op_list.append(np.sqrt(rate) * qutip.sigmam())

        rate = g2
        if rate > 0.0:
            c_op_list.append(np.sqrt(rate) * qutip.sigmaz())

        output = mesolve(H, psi0, tlist, c_op_list,
            e_ops=[qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()],
            options={"matrix_form": matrix_form}
        )

        return output.expect[0], output.expect[1], output.expect[2]

    def jc_steadystate(self, N, wc, wa, g, kappa, gamma,
                       pump, psi0, use_rwa, tlist):

        # Hamiltonian
        a = qutip.tensor(qutip.destroy(N), qutip.identity(2))
        sm = qutip.tensor(qutip.identity(N), qutip.destroy(2))

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
        rho_ss = qutip.steadystate(H, c_op_list)

        return (qutip.expect(a.dag() * a, rho_ss),
                qutip.expect(sm.dag() * sm, rho_ss))

    def jc_integrate(self, N, wc, wa, g, kappa, gamma,
                     pump, psi0, use_rwa, tlist, oper_evo=False,
                     matrix_form=False):

        # Hamiltonian
        a = qutip.tensor(qutip.destroy(N), qutip.identity(2))
        sm = qutip.tensor(qutip.identity(N), qutip.destroy(2))

        # Identity super-operator
        E0 = qutip.sprepost(
            qutip.tensor(qutip.qeye(N), qutip.qeye(2)),
            qutip.tensor(qutip.qeye(N), qutip.qeye(2))
        )

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

        options = {
            "store_states": True,
            "progress_bar": None,
            "matrix_form": matrix_form
        }

        # evolve and calculate expectation values
        output = mesolve(
            H, psi0, tlist, c_op_list, e_ops=[a.dag() * a, sm.dag() * sm],
            options=options)
        if oper_evo:
            output2 = mesolve(H, E0, tlist, c_op_list)
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
        psi0 = qutip.basis([N, 2], [n, 1])
        rho0vec = qutip.operator_to_vector(psi0.proj())
        tlist = np.linspace(0, 100, 50)

        out1, out2 = self.jc_integrate(
            N, wc, wa, g, kappa, gamma, pump, psi0, use_rwa, tlist, True)

        fid = fidelitycheck(out1, out2, rho0vec)
        assert fid == pytest.approx(1., abs=me_error)

    @pytest.mark.parametrize('matrix_form', [False, True])
    def testQubitDynamics1(self, matrix_form):
        "mesolve: qubit with dissipation"

        epsilon = 0.0 * 2 * np.pi   # cavity frequency
        delta = 1.0 * 2 * np.pi   # atom frequency
        g2 = 0.1
        g1 = 0.0
        psi0 = qutip.basis(2, 0)        # initial state
        tlist = np.linspace(0, 5, 200)

        sx, sy, sz = self.qubit_integrate(tlist, psi0, epsilon, delta, g1, g2,
                                          matrix_form=matrix_form)

        sx_analytic = np.zeros(np.shape(tlist))
        sy_analytic = -np.sin(2 * np.pi * tlist) * np.exp(-tlist * g2)
        sz_analytic = np.cos(2 * np.pi * tlist) * np.exp(-tlist * g2)

        np.testing.assert_allclose(sx, sx_analytic, atol=0.05)
        np.testing.assert_allclose(sy, sy_analytic, atol=0.05)
        np.testing.assert_allclose(sz, sz_analytic, atol=0.05)

    @pytest.mark.parametrize('matrix_form', [False, True])
    def testQubitDynamics2(self, matrix_form):
        "mesolve: qubit without dissipation"

        epsilon = 0.0 * 2 * np.pi   # cavity frequency
        delta = 1.0 * 2 * np.pi   # atom frequency
        g2 = 0.0
        g1 = 0.0
        psi0 = qutip.basis(2, 0)        # initial state
        tlist = np.linspace(0, 5, 200)

        sx, sy, sz = self.qubit_integrate(tlist, psi0, epsilon, delta, g1, g2,
                                          matrix_form=matrix_form)

        sx_analytic = np.zeros(np.shape(tlist))
        sy_analytic = -np.sin(2 * np.pi * tlist) * np.exp(-tlist * g2)
        sz_analytic = np.cos(2 * np.pi * tlist) * np.exp(-tlist * g2)

        np.testing.assert_allclose(sx, sx_analytic, atol=0.05)
        np.testing.assert_allclose(sy, sy_analytic, atol=0.05)
        np.testing.assert_allclose(sz, sz_analytic, atol=0.05)

    @pytest.mark.parametrize('matrix_form', [False, True])
    def testCavity1(self, matrix_form):
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
        psi0 = qutip.tensor(qutip.basis(N, n), qutip.basis(2, 1))
        tlist = np.linspace(0, 1000, 2000)

        nc, na = self.jc_integrate(
            N, wc, wa, g, kappa, gamma, pump, psi0, use_rwa, tlist,
            matrix_form=matrix_form)

        nc_ex = 0.5 * (1 - np.cos(2 * g * np.sqrt(n + 1) * tlist)) + n
        na_ex = 0.5 * (1 + np.cos(2 * g * np.sqrt(n + 1) * tlist))

        np.testing.assert_allclose(nc[-1], nc_ex[-1], atol=0.005)
        np.testing.assert_allclose(na[-1], na_ex[-1], atol=0.005)

    @pytest.mark.parametrize('matrix_form', [False, True])
    def testCavity2(self, matrix_form):
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
        psi0 = qutip.tensor(qutip.basis(N, n), qutip.basis(2, 1))
        tlist = np.linspace(0, 1000, 2000)

        nc, na = self.jc_integrate(
            N, wc, wa, g, kappa, gamma, pump, psi0, use_rwa, tlist,
            matrix_form=matrix_form)

        nc_ex = (0.5 * (1 - np.cos(2 * g * np.sqrt(n + 1) * tlist)) + n) * \
            np.exp(-kappa * tlist)
        na_ex = 0.5 * (1 + np.cos(2 * g * np.sqrt(n + 1) * tlist)) * \
            np.exp(-gamma * tlist)

        np.testing.assert_allclose(nc[-1], nc_ex[-1], atol=0.005)
        np.testing.assert_allclose(na[-1], na_ex[-1], atol=0.005)

    @pytest.mark.parametrize('matrix_form', [False, True])
    def testCavity3(self, matrix_form):
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
        psi0 = qutip.tensor(qutip.basis(N, n), qutip.basis(2, 1))
        tlist = np.linspace(0, 200, 500)

        nc, na = self.jc_integrate(
            N, wc, wa, g, kappa, gamma, pump, psi0, use_rwa, tlist,
            matrix_form=matrix_form)

        # we don't have any analytics for this parameters, so
        # compare with the steady state
        nc_ss, na_ss = self.jc_steadystate(
            N, wc, wa, g, kappa, gamma, pump, psi0, use_rwa, tlist)

        nc_ss = nc_ss * np.ones(np.shape(nc))
        na_ss = na_ss * np.ones(np.shape(na))

        np.testing.assert_allclose(nc[-1], nc_ss[-1], atol=0.005)
        np.testing.assert_allclose(na[-1], na_ss[-1], atol=0.005)


class TestMESolveStepFuncCoeff:
    """
    A Test class for using time-dependent array coefficients
    as step functions instead of doing interpolation
    """
    # Runge-Kutta method (dop853) behave better with step function evolution
    # than multi-step methods (adams, qutip 4's default)
    options = {"method": "dop853", "nsteps": 1e8, "progress_bar": None}

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
        rho0 = qutip.rand_ket(2)
        tlist = np.array([0, np.pi/2])
        options = {"method": method, "nsteps": 1e5, "rtol": 1e-7}
        qu = qutip.QobjEvo([[qutip.sigmax(), self.python_coeff]],
                     tlist=tlist, order=0)
        result = mesolve(qu, rho0=rho0, tlist=tlist, options=options)
        fid = qutip.fidelity(result.states[-1], qutip.sigmax()*rho0)
        assert fid == pytest.approx(1)

    def test_array_cte_coeff(self):
        """
        Test for Array coefficient with uniform tlist as step function coeff
        """
        rho0 = qutip.rand_ket(2)
        tlist = np.array([0., np.pi/2, np.pi], dtype=float)
        npcoeff = np.array([0.25, 0.75, 0.75])
        qu = qutip.QobjEvo([[qutip.sigmax(), npcoeff]], tlist=tlist, order=0)
        result = mesolve(qu, rho0=rho0, tlist=tlist, options=self.options)
        fid = qutip.fidelity(result.states[-1], qutip.sigmax()*rho0)
        assert fid == pytest.approx(1)

    def test_array_t_coeff(self):
        """
        Test for Array with non-uniform tlist as step function coeff
        """
        rho0 = qutip.rand_ket(2)
        tlist = np.array([0., np.pi/2, np.pi*3/2], dtype=float)
        npcoeff = np.array([0.5, 0.25, 0.25])
        qu = qutip.QobjEvo([[qutip.sigmax(), npcoeff]], tlist=tlist, order=0)
        result = mesolve(qu, rho0=rho0, tlist=tlist, options=self.options)
        fid = qutip.fidelity(result.states[-1], qutip.sigmax()*rho0)
        assert fid == pytest.approx(1)

    def test_array_str_coeff(self):
        """
        Test for Array and string as step function coeff.
        qobjevo_codegen is used and uniform tlist
        """
        rho0 = qutip.rand_ket(2)
        tlist = np.array([0., np.pi/2, np.pi], dtype=float)
        npcoeff1 = np.array([0.25, 0.75, 0.75], dtype=complex)
        npcoeff2 = np.array([0.5, 1.5, 1.5], dtype=float)
        strcoeff = "1."
        qu = qutip.QobjEvo(
            [[qutip.sigmax(), npcoeff1],
             [qutip.sigmax(), strcoeff],
             [qutip.sigmax(), npcoeff2]],
            tlist=tlist, order=0)
        result = mesolve(qu, rho0=rho0, tlist=tlist, options=self.options)
        fid = qutip.fidelity(result.states[-1], qutip.sigmax()*rho0)
        assert fid == pytest.approx(1)

    def test_array_str_py_coeff(self):
        """
        Test for Array, string and Python function as step function coeff.
        qobjevo_codegen is used and non non-uniform tlist
        """
        rho0 = qutip.rand_ket(2)
        tlist = np.array([0., np.pi/4, np.pi/2, np.pi], dtype=float)
        npcoeff1 = np.array([0.4, 1.6, 1.0, 1.0], dtype=complex)
        npcoeff2 = np.array([0.4, 1.6, 1.0, 1.0], dtype=float)
        strcoeff = "1."
        qu = qutip.QobjEvo(
            [[qutip.sigmax(), npcoeff1], [qutip.sigmax(), npcoeff2],
             [qutip.sigmax(), self.python_coeff], [qutip.sigmax(), strcoeff]],
            tlist=tlist, order=0)
        result = mesolve(qu, rho0=rho0, tlist=tlist, options=self.options)
        fid = qutip.fidelity(result.states[-1], qutip.sigmax()*rho0)
        assert fid == pytest.approx(1)


@pytest.mark.parametrize('matrix_form', [False, True], ids=['superop', 'matrix_form'])
def test_num_collapse_set(matrix_form):
    H = qutip.sigmaz()
    psi = (qutip.basis(2, 0) + qutip.basis(2, 1)).unit()
    ts = [0, 1]
    for c_ops in (
        qutip.sigmax(),
        [qutip.sigmax()],
        [qutip.sigmay(), qutip.sigmax()],
    ):
        res = mesolve(H, psi, ts, c_ops=c_ops,
                      options={"matrix_form": matrix_form})
        if not isinstance(c_ops, list):
            c_ops = [c_ops]
        assert res.stats["num_collapse"] == len(c_ops)


def test_mesolve_bad_H():
    with pytest.raises(TypeError):
        MESolver([qutip.qeye(3), 't'], [])
    with pytest.raises(TypeError):
        MESolver(qutip.qeye(3), [[qutip.qeye(3), 't']])


def test_mesolve_bad_state():
    solver = MESolver(qutip.qeye(4), [])
    with pytest.raises(TypeError):
        solver.start(qutip.basis(2,1) & qutip.basis(2,0), 0)


def test_mesolve_bad_options():
    with pytest.raises(TypeError):
        MESolver(qutip.qeye(4), [], options=False)


@pytest.mark.parametrize('solver_cls', [MESolver, MESolverMatrixForm])
def test_feedback(solver_cls):

    def f(t, A):
        return (A-4.)

    N = 10
    tol = 1e-14
    psi0 = qutip.basis(N, 7)
    # MESolverMatrixForm uses operators, MESolver uses superoperators
    if solver_cls is MESolverMatrixForm:
        feedback_op = qutip.num(N)
    else:
        feedback_op = qutip.spre(qutip.num(N))
    a = qutip.QobjEvo(
        [qutip.destroy(N), f],
        args={"A": solver_cls.ExpectFeedback(feedback_op)}
    )
    H = qutip.QobjEvo(qutip.num(N))
    solver = solver_cls(H, c_ops=[a])
    result = solver.run(psi0, np.linspace(0, 30, 301), e_ops=[qutip.num(N)])
    assert np.all(result.expect[0] > 4. - tol)


@pytest.mark.parametrize(
    'rho0',
    [qutip.sigmax(), qutip.sigmaz(), qutip.qeye(2)],
    ids=["sigmax", "sigmaz", "tr=2"]
)
@pytest.mark.parametrize('solver_cls', [MESolver, MESolverMatrixForm])
def test_non_normalized_dm(rho0, solver_cls):
    H = qutip.QobjEvo(qutip.num(2))
    solver = solver_cls(H, c_ops=[qutip.sigmaz()])
    result = solver.run(rho0, np.linspace(0, 1, 10), e_ops=[qutip.qeye(2)])
    np.testing.assert_allclose(result.expect[0], rho0.tr(), atol=1e-7)


class TestMESolveMatrixForm:
    """
    Tests comparing matrix-form solver to superoperator solver.

    Note: Basic correctness is tested via the matrix_form parameterization
    in TestMESolveDecay. These tests directly compare the two solver forms.
    """
    N = 20
    H = qutip.num(N)
    kappa = 0.1
    c_ops = [np.sqrt(kappa) * qutip.destroy(N)]
    tlist = np.linspace(0, 5, 20)

    def test_matrix_form_vs_superop_no_collapse(self):
        """Test matrix_form vs superop with no collapse operators."""
        psi0 = qutip.fock(self.N, self.N // 2)
        rho0 = qutip.ket2dm(psi0)
        e_ops = [qutip.num(self.N)]

        # Matrix-form with no collapse operators
        result_matrix = mesolve(
            self.H, rho0, self.tlist, c_ops=[],
            e_ops=e_ops,
            options={"matrix_form": True, "method": "vern7", "progress_bar": None}
        )

        # Superoperator with no collapse operators
        result_super = mesolve(
            self.H, rho0, self.tlist, c_ops=[],
            e_ops=e_ops,
            options={"matrix_form": False, "method": "vern7", "progress_bar": None}
        )

        # Should agree (unitary evolution preserves expectation values)
        np.testing.assert_allclose(
            result_matrix.expect[0], result_super.expect[0],
            rtol=1e-7, atol=1e-9
        )

        # Expectation value should be constant (no decay without c_ops)
        np.testing.assert_allclose(
            result_matrix.expect[0], self.N // 2, rtol=1e-7
        )

    def test_matrix_form_vs_superop_with_collapse(self):
        """
        Test matrix_form vs superop with collapse operators and various options.
        
        This test exercises non-default solver options to ensure they are
        processed correctly by both solver forms.
        """
        psi0 = qutip.fock(self.N, self.N // 2)
        rho0 = qutip.ket2dm(psi0)
        e_ops = [qutip.num(self.N), qutip.destroy(self.N).dag() * qutip.destroy(self.N)]

        common_options = {
            "method": "vern7",
            "progress_bar": None,
            "store_states": True,
            "store_final_state": True,
            "normalize_output": True,
            "atol": 1e-9,
            "rtol": 1e-7,
        }

        # Matrix-form solver
        result_matrix = mesolve(
            self.H, rho0, self.tlist, c_ops=self.c_ops,
            e_ops=e_ops,
            options={**common_options, "matrix_form": True}
        )

        # Superoperator solver
        result_super = mesolve(
            self.H, rho0, self.tlist, c_ops=self.c_ops,
            e_ops=e_ops,
            options={**common_options, "matrix_form": False}
        )

        # Compare expectation values
        for i in range(len(e_ops)):
            np.testing.assert_allclose(
                result_matrix.expect[i], result_super.expect[i],
                rtol=1e-6, atol=1e-8,
                err_msg=f"Expectation values differ for e_op {i}"
            )

        # Compare stored states
        assert len(result_matrix.states) == len(result_super.states)
        for i, (state_m, state_s) in enumerate(
            zip(result_matrix.states, result_super.states)
        ):
            np.testing.assert_allclose(
                state_m.full(), state_s.full(),
                rtol=1e-6, atol=1e-8,
                err_msg=f"States differ at time index {i}"
            )

        # Compare final states
        assert result_matrix.final_state is not None
        assert result_super.final_state is not None
        np.testing.assert_allclose(
            result_matrix.final_state.full(), result_super.final_state.full(),
            rtol=1e-6, atol=1e-8
        )

        # Verify normalization (trace should be 1)
        for state in result_matrix.states:
            np.testing.assert_allclose(state.tr(), 1.0, atol=1e-10)
