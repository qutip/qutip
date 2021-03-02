
from qutip.solver.evolver import *
from qutip.solver.options import SolverOptions
import qutip as qt
import pytest
import numpy as np
from qutip.core.qobjevofunc import QobjEvoFunc

def id_from_pair(pair):
    if pair[1] == "":
        return pair[0]
    else:
        return pair[0] + "_" + pair[1]

class TestEvolverCte():
    system = qt.QobjEvo(-1j*qt.sigmax()*np.pi)
    all_evolvers = evolver_collection.list_keys('pairs')
    backstep_evolvers = evolver_collection.list_keys('pairs', backstep=True)

    @pytest.fixture(params=[qt.QobjEvo(-1j*qt.sigmax()*np.pi)], ids=[""])
    def system(self, request):
        return request.param

    @pytest.fixture(
        params=all_evolvers,
        ids=id_from_pair
    )
    def all_evol(self, request):
        return request.param

    @pytest.fixture(
        params=backstep_evolvers,
        ids=id_from_pair
    )
    def backstep_evol(self, request):
        return request.param

    def _analytical(self, t):
        return np.cos(t*np.pi)

    def test_run(self, system, all_evol):
        method, rhs = all_evol
        tol = 1e-5
        opt = SolverOptions(method=method, rhs=rhs)
        evol = evolver_collection[method, rhs](system, opt, {}, {})
        evol.set_state(0, qt.basis(2,0).to(qt.data.Dense).data)
        for t, state in evol.run(np.linspace(0,2,21)):
            assert np.abs(self._analytical(t) - state.to_array()[0,0]) < tol

    def test_step(self, system, all_evol):
        tol = 1e-5
        method, rhs = all_evol
        opt = SolverOptions(method=method, rhs=rhs)
        evol = evolver_collection[method, rhs](system, opt, {}, {})
        evol.set_state(0, qt.basis(2,0).to(qt.data.Dense).data)
        for t in np.linspace(0,2,21):
            _, state = evol.step(t)
            assert np.abs(self._analytical(t) - state.to_array()[0,0]) < tol

    def test_backstep(self, system, backstep_evol):
        tol = 1e-5
        method, rhs = backstep_evol
        opt = SolverOptions(method=method, rhs=rhs)
        evol = evolver_collection[method, rhs](system, opt, {}, {})
        evol.set_state(0, qt.basis(2,0).to(qt.data.Dense).data)
        t = 0
        for i in range(1, 21):
            t_target = i*0.05
            while t < t_target:
                t_old, y_old = evol.get_state()
                t, state = evol.one_step(t_target)
                assert t <= t_target
                assert t > t_old
                assert np.abs(self._analytical(t) -
                              state.to_array()[0,0]) < tol
                t_back = (t - t_old) * np.random.rand() + t_old
                _, bstate = evol.backstep(t_back)
                assert np.abs(self._analytical(t_back) -
                              bstate.to_array()[0,0]) < tol
                t, state = evol.step(t)

    def test_shape(self, system, all_evol):
        tol = 1e-5
        method, rhs = all_evol
        opt = SolverOptions(method=method, rhs=rhs)
        evol = evolver_collection[method, rhs](system, opt, {}, {})
        evol.set_state(0,
            qt.core.unstack_columns(
                qt.coherent(6,3).to(qt.data.Dense).data, (2,3)))
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
    all_evolvers = evolver_collection.list_keys('pairs', time_dependent=True)
    backstep_evolvers = evolver_collection.list_keys('pairs', backstep=True,
                                                     time_dependent=True)
    args_evolvers = evolver_collection.list_keys('pairs', update_args=True,
                                                     time_dependent=True)
    feedback_evolvers = evolver_collection.list_keys('pairs', feedback=True,
                                                     time_dependent=True)

    @pytest.fixture(params=[
        qt.QobjEvo([-1j*qt.sigmax()*np.pi,f], args={"cte":1}),
        QobjEvoFunc(func, args={"cte":1})
        ], ids=["qevo", "func"])
    def system(self, request):
        return request.param

    @pytest.fixture(
        params=all_evolvers,
        ids=id_from_pair
    )
    def all_evol(self, request):
        return request.param

    @pytest.fixture(
        params=backstep_evolvers,
        ids=id_from_pair
    )
    def backstep_evol(self, request):
        return request.param

    @pytest.fixture(
        params=args_evolvers,
        ids=id_from_pair
    )
    def args_evol(self, request):
        return request.param

    def _analytical(self, t):
        return np.cos(t**2/2*np.pi)

    def _analytical_args(self, t_eff):
        return np.cos(t_eff/2*np.pi)

    def test_step_args(self, system, args_evol):
        tol = 5e-5
        method, rhs = args_evol
        opt = SolverOptions(method=method, rhs=rhs)
        evol = evolver_collection[method, rhs](system, opt, {}, {})
        evol.set_state(0, qt.basis(2,0).to(qt.data.Dense).data)
        t_eff = 0
        dt = 0.1
        for t_i in range(1,21):
            t = t_i * dt
            factor = t_i%5
            t_eff += factor * dt**2 * (2*t_i-1)
            # stepping with args change is a Solver options
            evol.update_args({"cte": factor})
            t_, state_ = evol.get_state()
            evol.set_state(t_, state_)
            t, state = evol.step(t)
            assert np.abs(self._analytical_args(t_eff) -
                          state.to_array()[0,0]) < tol
