from qutip.solver.integrator import *
from qutip.solver.options import SolverOptions
from qutip.core import QobjEvo, liouvillian
import qutip as qt
import numpy as np
from numpy.testing import assert_allclose
import pytest


def id_from_keys(keys):
    if keys[1] == "":
        return keys[0]
    else:
        return keys[0] + "_" + keys[1]


class TestIntegratorCte():
    _analytical_se = lambda _, t: np.cos(t * np.pi)
    se_system = QobjEvo(-1j * qt.sigmax() * np.pi)
    _analytical_me = lambda _, t: 1 - np.exp(-t)
    me_system = liouvillian(QobjEvo(qt.qeye(2)), c_ops=[qt.destroy(2)])

    @pytest.fixture(
        params=integrator_collection._list_keys('pairs', solver="sesolve"),
        ids=id_from_keys
    )
    def se_keys(self, request):
        return request.param

    @pytest.fixture(
        params=integrator_collection._list_keys('pairs', solver="mesolve"),
        ids=id_from_keys
    )
    def me_keys(self, request):
        return request.param

    @pytest.fixture(
        params=integrator_collection._list_keys('pairs', solver="mcsolve"),
        ids=id_from_keys
    )
    def mc_keys(self, request):
        return request.param

    def test_se_integration(self, se_keys):
        method, rhs = se_keys
        opt = SolverOptions(method=method, rhs=rhs)
        evol = integrator_collection[method, rhs](self.se_system, opt)
        state0 = qt.core.unstack_columns(qt.basis(6,0).data, (2, 3))
        evol.set_state(0, state0)
        for t, state in evol.run(np.linspace(0, 2, 21)):
            assert_allclose(self._analytical_se(t),
                            state.to_array()[0, 0], atol=2e-5)
            assert state.shape == (2, 3)

    def test_me_integration(self, me_keys):
        method, rhs = me_keys
        opt = SolverOptions(method=method, rhs=rhs)
        evol = integrator_collection[method, rhs](self.me_system, opt)
        state0 = qt.operator_to_vector(qt.fock_dm(2,1)).data
        evol.set_state(0, state0)
        for t in np.linspace(0, 2, 21):
            t_, state = evol.integrate(t)
            assert t_ == t
            assert_allclose(self._analytical_me(t),
                            state.to_array()[0, 0], atol=2e-5)

    def test_mc_integration(self, mc_keys):
        method, rhs = mc_keys
        opt = SolverOptions(method=method, rhs=rhs)
        evol = integrator_collection[method, rhs](self.se_system, opt)
        state = qt.basis(2,0).data
        evol.set_state(0, state)
        t = 0
        for i in range(1, 21):
            t_target = i * 0.05
            while t < t_target:
                t_old, y_old = evol.get_state()
                t, state = evol.integrate(t_target, step=True)
                assert t <= t_target
                assert t > t_old
                assert_allclose(self._analytical_se(t),
                                state.to_array()[0, 0], atol=2e-5)
                t_back = (t + t_old) / 2
                t_got, bstate = evol.integrate(t_back)
                assert t_back == t_got
                assert_allclose(self._analytical_se(t),
                                state.to_array()[0, 0], atol=2e-5)
                t, state = evol.integrate(t)


class TestIntegrator(TestIntegratorCte):
    _analytical_se = lambda _, t: np.cos(t**2/2 * np.pi)
    se_system = QobjEvo([-1j * qt.sigmax() * np.pi, "t"])
    _analytical_me = lambda _, t: 1 - np.exp(-(t**3) / 3)
    me_system = liouvillian(QobjEvo(qt.qeye(2)),
                            c_ops=[QobjEvo([qt.destroy(2), 't'])])

    @pytest.fixture(
        params=integrator_collection._list_keys('pairs', solver="sesolve",
                                                time_dependent=True),
        ids=id_from_keys
    )
    def se_keys(self, request):
        return request.param

    @pytest.fixture(
        params=integrator_collection._list_keys('pairs', solver="mesolve",
                                                time_dependent=True),
        ids=id_from_keys
    )
    def me_keys(self, request):
        return request.param

    @pytest.fixture(
        params=integrator_collection._list_keys('pairs', solver="mcsolve",
                                                time_dependent=True),
        ids=id_from_keys
    )
    def mc_keys(self, request):
        return request.param
