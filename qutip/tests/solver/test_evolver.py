


from qutip.solver.evolver import *
from qutip.solver.evolver import all_ode_method
from qutip.solver.options import SolverOptions
import qutip as qt
import pytest
import numpy as np
from qutip.core.qobjevofunc import QobjEvoFunc


class TestEvolverCte():
    def pytest_generate_tests(self, metafunc):
        cases = []
        system = qt.QobjEvo(-1j*qt.sigmax()*np.pi)

        for method in all_ode_method:
            cases.append(pytest.param(method, system, {}, id=method))
        cases.append(pytest.param("diag", system, {}, id="diag"))
        metafunc.parametrize(['method', 'system', 'optimization'],
                             cases)

    def _analytical(self, t):
        return np.cos(t*np.pi)

    def test_run(self, method, system, optimization):
        tol = 1e-5
        opt = SolverOptions(method=method, **optimization)
        evol = get_evolver(system, opt, {}, {})
        evol.set(qt.basis(2,0).to(qt.data.Dense).data, 0)
        for t, state in evol.run(np.linspace(0,2,21)):
            assert np.abs(self._analytical(t) - state.to_array()[0,0]) < tol

    def test_step(self, method, system, optimization):
        tol = 1e-5
        opt = SolverOptions(method=method, **optimization)
        evol = get_evolver(system, opt, {}, {})
        evol.set(qt.basis(2,0).to(qt.data.Dense).data, 0)
        for t in np.linspace(0,2,21):
            state = evol.step(t)
            assert np.abs(self._analytical(t) - state.to_array()[0,0]) < tol

    def test_backstep(self, method, system, optimization):
        tol = 1e-5
        opt = SolverOptions(method=method, **optimization)
        evol = get_evolver(system, opt, {}, {})
        evol.set(qt.basis(2,0).to(qt.data.Dense).data, 0)
        for i in range(1, 21):
            t_traget = i*0.05
            while evol.t < t_traget:
                t_old = evol.t
                y_old = evol.get_state()
                state = evol.step(t_traget, step=True)
                t = evol.t
                assert t <= t_traget
                assert t > t_old
                assert np.abs(self._analytical(t) -
                              state.to_array()[0,0]) < tol
                t_back = (t-t_old) * np.random.rand() + t_old
                bstate = evol.backstep(t_back, t_old, y_old)
                assert np.abs(self._analytical(t_back) -
                              bstate.to_array()[0,0]) < tol
                state = evol.step(t)

    def test_shape(self, method, system, optimization):
        tol = 1e-5
        opt = SolverOptions(method=method, **optimization)
        evol = get_evolver(system, opt, {}, {})
        evol.set(qt.core.unstack_columns(
            qt.coherent(6,3).to(qt.data.Dense).data, (2,3)), 0)
        for t, state in evol.run(np.linspace(0,0.1,3)):
            assert state.shape == (2,3)


def f(t, args):
    return t * args["cte"]


def func(t, args):
    return qt.sigmax()* (-1j * t * args["cte"] * np.pi )


class TestEvolver(TestEvolverCte):
    """
    Test that `mcsolve` correctly solves the system when the collapse operators
    are time-dependent.
    """
    def pytest_generate_tests(self, metafunc):
        cases = []
        systems = [
            (qt.QobjEvo([-1j*qt.sigmax()*np.pi,f], args={"cte":1}), "_qevo"),
            (QobjEvoFunc(func, args={"cte":1}), "_func"),
        ]
        for method in all_ode_method:
            for system, name in systems:
                for ahs in [True, False]:
                    ahs_str = "_AHS" if ahs else ""
                    cases.append(
                        pytest.param(method, system, {'ahs':ahs},
                            id=method + name + ahs_str,
                            marks=[pytest.mark.slow])
                        )
        metafunc.parametrize(['method', 'system', 'optimization'],
                             cases)

    def _analytical(self, t):
        return np.cos(t**2/2*np.pi)

    def _analytical_args(self, t_eff):
        return np.cos(t_eff/2*np.pi)

    def test_step_args(self, method, system, optimization):
        tol = 5e-5
        opt = SolverOptions(method=method, **optimization)
        evol = get_evolver(system, opt, {"cte":-1}, {})
        evol.set(qt.basis(2,0).to(qt.data.Dense).data, 0)
        t_eff = 0
        dt = 0.1
        for t_i in range(1,21):
            t = t_i * dt
            factor = t_i%5
            t_eff += factor * dt**2 * (2*t_i-1)
            # stepping with args change is a Solver options
            evol.update_args({"cte": factor})
            evol.set(evol.get_state(), evol.t)
            state = evol.step(t)
            assert np.abs(self._analytical_args(t_eff) -
                          state.to_array()[0,0]) < tol
