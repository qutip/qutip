from qutip.solver.integrator import (sesolve_integrators,
                                     mesolve_integrators, mcsolve_integrators)
from qutip.solver.options import SolverOdeOptions
import qutip
import numpy as np
from numpy.testing import assert_allclose
import pytest


class TestIntegratorCte():
    _analytical_se = lambda _, t: np.cos(t * np.pi)
    se_system = qutip.QobjEvo(-1j * qutip.sigmax() * np.pi)
    _analytical_me = lambda _, t: 1 - np.exp(-t)
    me_system = qutip.liouvillian(qutip.QobjEvo(qutip.qeye(2)),
                                  c_ops=[qutip.destroy(2)])

    @pytest.fixture(params=list(sesolve_integrators.keys()))
    def se_method(self, request):
        return request.param

    @pytest.fixture(params=list(mesolve_integrators.keys()))
    def me_method(self, request):
        return request.param

    @pytest.fixture(params=list(mcsolve_integrators.keys()))
    def mc_method(self, request):
        return request.param

    def test_se_integration(self, se_method):
        opt = SolverOdeOptions(method=se_method)
        evol = sesolve_integrators[se_method](self.se_system, opt)
        state0 = qutip.core.unstack_columns(qutip.basis(6,0).data, (2, 3))
        evol.set_state(0, state0)
        for t, state in evol.run(np.linspace(0, 2, 21)):
            assert_allclose(self._analytical_se(t),
                            state.to_array()[0, 0], atol=2e-5)
            assert state.shape == (2, 3)

    def test_me_integration(self, me_method):
        opt = SolverOdeOptions(method=me_method)
        evol = mesolve_integrators[me_method](self.me_system, opt)
        state0 = qutip.operator_to_vector(qutip.fock_dm(2,1)).data
        evol.set_state(0, state0)
        for t in np.linspace(0, 2, 21):
            t_, state = evol.integrate(t)
            assert t_ == t
            assert_allclose(self._analytical_me(t),
                            state.to_array()[0, 0], atol=2e-5)

    def test_mc_integration(self, mc_method):
        opt = SolverOdeOptions(method=mc_method)
        evol = mcsolve_integrators[mc_method](self.se_system, opt)
        state = qutip.basis(2,0).data
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
    se_system = qutip.QobjEvo([-1j * qutip.sigmax() * np.pi, "t"])
    _analytical_me = lambda _, t: 1 - np.exp(-(t**3) / 3)
    me_system = qutip.liouvillian(
        qutip.QobjEvo(qutip.qeye(2)),
        c_ops=[qutip.QobjEvo([qutip.destroy(2), 't'])]
    )

    @pytest.fixture(
        params=[key for key, integrator in sesolve_integrators.items()
                if integrator.support_time_dependant]
    )
    def se_method(self, request):
        return request.param

    @pytest.fixture(
        params=[key for key, integrator in mesolve_integrators.items()
                if integrator.support_time_dependant]
    )
    def me_method(self, request):
        return request.param

    @pytest.fixture(
        params=[key for key, integrator in mcsolve_integrators.items()
                if integrator.support_time_dependant]
    )
    def mc_method(self, request):
        return request.param
