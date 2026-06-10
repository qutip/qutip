from qutip.solver.sesolve import SESolver
from qutip.solver.mesolve import MESolver
from qutip.solver.mcsolve import MCSolver
from qutip.solver.solver_base import Solver
from qutip.solver.integrator import *
from qutip.solver.integrator._rhs import RHS
import qutip
import qutip.core.data as _data

import functools
import numpy as np
from numpy.testing import assert_allclose
import pytest

# Deactivate warning for test without cython
from qutip.core.coefficient import WARN_MISSING_MODULE
WARN_MISSING_MODULE[0] = 0


def derivative_1(t, state, out=None):
    x = state.to_array()
    der = np.array([
        [1, t],
        [x[1, 0] * -0.1, t**2],
        [-0.1j * t * x[2, 0], t**3],
    ])
    der = _data.Dense(der)
    if out is not None:
        der += out
    return der

def analytical_1(t, x0):
    x0 = x0.to_array()
    out = np.array([
        [x0[0, 0] + t, x0[0, 1] + t**2/2],
        [np.exp(t * -0.1) * x0[1, 0] , x0[1, 1] + t**3/3],
        [np.exp(-0.1j * t**2/2) * x0[2, 0], x0[2, 1] + t**4/4],
    ])
    return out

mat = qutip.rand_herm(3, density=0.75) * -1j

derivative_2 = qutip.QobjEvo(mat).matmul_data

def analytical_2(t, x0):
    return ((t * mat).expm() @ qutip.Qobj(x0)).full()

derivative_3 = RHS(derivative_1, inplace=True)


class TestIntegratorCallable:
    @pytest.fixture(params=[
        method
        for method, integrator in Solver.avail_integrators().items()
        if integrator.RHS_format == "callable"
    ])
    def method(self, request):
        # Callable are the general method that should all be registered with
        # the general Solver
        return request.param

    def _check_integrator(self, integrator_instance, state0, analytical):
        integrator_instance.set_state(0, state0)
        tlist = np.linspace(0, 2, 21)
        for (t, state), t_in in zip(integrator_instance.run(tlist), tlist[1:]):
            assert t == t_in
            assert_allclose(analytical(t, state0), state.to_array(), atol=5e-5)
            assert state.shape == state0.shape
        t, state = integrator_instance.integrate(3)
        assert t == 3.
        assert_allclose(analytical(t, state0), state.to_array(), atol=5e-5)

        t_front, state = integrator_instance.mcstep(5)
        t_target = (3 + t_front) / 2
        t_in, state = integrator_instance.mcstep(t_target)
        assert t_in == t_target
        assert_allclose(analytical(t_in, state0), state.to_array(), atol=5e-5)

    # When the derivative is C ordered, the messa
    # @pytest.mark.filterwarnings("ignore:cannot stack columns")

    @pytest.mark.parametrize(['derivative', "analytical"], [
        (derivative_1, analytical_1),
        (derivative_2, analytical_2),
        (derivative_3, analytical_1),
    ], ids=["function", "QobjEvo", "RHS"])
    def test_integration_func(self, derivative, analytical, method):
        integrator = Solver.avail_integrators()[method]
        integrator_instance = integrator(derivative, {})
        state0 = _data.Dense(np.array([
            [1, 1j],
            [1., 0.25 + 0.75j],
            [0.5 - 0.5j, 0.],
        ]))
        self._check_integrator(integrator_instance, state0, analytical)


class TestIntegratorMatrix:
    hermitian = qutip.rand_herm(10).data
    non_hermitian = (
        qutip.rand_stochastic(10, density=0.4) +
        qutip.rand_stochastic(10, density=0.4) *1j
    ).data

    @pytest.fixture(params=[
        (IntegratorKrylov, {"krylov_dim": 5}),
        (IntegratorDiag, {}),
    ], ids=["krylov", "diag"])
    def integrator(self, request):
        # Matrix could be limited to hermitian only and not registered with
        # all solver.
        return request.param

    @staticmethod
    def analytical(matrix, t, state0):
        return ((qutip.Qobj(matrix) * t).expm() @ qutip.Qobj(state0)).full()

    def _check_integrator(self, integrator_instance, state0, analytical):
        integrator_instance.set_state(0, state0)
        tlist = np.linspace(0, 2, 21)
        for (t, state), t_in in zip(integrator_instance.run(tlist), tlist[1:]):
            assert t == t_in
            assert_allclose(analytical(t, state0), state.to_array(), atol=2e-5)
            assert state.shape == state0.shape
        t, state = integrator_instance.integrate(3)
        assert t == 3.
        assert_allclose(analytical(t, state0), state.to_array(), atol=2e-5)

    def test_integration_hermitian(self, integrator):
        integrator, options = integrator
        integrator_instance = integrator(self.hermitian, options)
        state0 = qutip.rand_ket(10).data
        self._check_integrator(
            integrator_instance, state0,
            functools.partial(self.analytical, self.hermitian)
        )

    def test_integration_non_hermitian(self, integrator):
        integrator, options = integrator
        integrator_instance = integrator(self.non_hermitian, options)
        state0 = qutip.rand_ket(10).data
        self._check_integrator(
            integrator_instance, state0,
            functools.partial(self.analytical, self.non_hermitian)
        )


class TestIntegratorMCstep():
    analytical = lambda _, t: np.cos(t * np.pi)
    system = qutip.QobjEvo(-1j * qutip.sigmax() * np.pi)

    @pytest.fixture(params=list(MCSolver.avail_integrators().keys()))
    def mc_method(self, request):
        return request.param

    def test_mc_integration(self, mc_method):
        integrator = MCSolver.avail_integrators()[mc_method]
        if integrator.RHS_format == "callable":
            evol = integrator(self.system.matmul_data, {})
        elif integrator.RHS_format == "matrix":
            evol = integrator(self.system(0).data, {})
        else:
            pytest.skip()
        state = qutip.basis(2,0).data
        evol.set_state(0, state)
        t = 0
        for i in range(1, 21):
            t_target = i * 0.05
            while t < t_target:
                t_old, y_old = evol.get_state()
                t, state = evol.mcstep(t_target)
                assert t <= t_target
                assert t > t_old
                assert_allclose(self.analytical(t),
                                state.to_array()[0, 0], atol=2e-5)
                t_back = (t + t_old) / 2
                t_got, bstate = evol.mcstep(t_back)
                assert t_back == t_got
                assert_allclose(self.analytical(t),
                                state.to_array()[0, 0], atol=2e-5)
                t, state = evol.mcstep(t)


    @pytest.mark.parametrize('start', [1, -1])
    def test_mc_integration_mixed(self, start, mc_method):
        system = qutip.QobjEvo(qutip.qeye(1))
        integrator = MCSolver.avail_integrators()[mc_method]
        if integrator.RHS_format == "callable":
            evol = integrator(system.matmul_data, {})
        elif integrator.RHS_format == "matrix":
            evol = integrator(system(0).data, {})
        else:
            pytest.skip()

        state = qutip.basis(1,0).data
        evol.set_state(start, state)
        t = start
        t_target = start + .1
        while t < t_target:
            t, _ = evol.mcstep(start + .1)
        _ = evol.mcstep(start + .2)
        t_target = (start + .1 + t) / 2
        t, state = evol.mcstep(t_target)
        assert (
            state.to_array()[0, 0]
            == pytest.approx(np.exp(t - start), abs=1e-5)
        )


@pytest.mark.parametrize('sizes', [(1, 100), (10, 10), (100, 0)],
                     ids=["large", "multiple subspaces", "diagonal"])
def test_krylov(sizes):
    # Krylov solve act differently for large systems composed tensored
    # sub systems.
    N, M = sizes
    H = qutip.qeye(N)
    if M:
        H = H & (qutip.num(M) + qutip.create(M) + qutip.destroy(M))
    H = -1j * H
    integrator = IntegratorKrylov(H.data, {"krylov_dim": 30})
    ref_integrator = IntegratorDiag(H.data, {})
    psi = qutip.basis(100, 95).data
    integrator.set_state(0, psi)
    ref_integrator.set_state(0, psi)
    for t in np.linspace(0.25, 1, 4):
        out = integrator.integrate(t)[1]
        ref = ref_integrator.integrate(t)[1]
        assert qutip.data.norm.l2(out - ref) == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize('integrator',
    [IntegratorScipyAdams, IntegratorScipyBDF, IntegratorScipylsoda],
    ids=["adams", 'bdf', "lsoda"]
)
def test_concurent_usage(integrator):
    opt = {'atol':1e-10, 'rtol':1e-7}

    sys1 = qutip.QobjEvo(0.5 * qutip.qeye(1))
    inter1 = integrator(sys1.matmul_data, opt)
    inter1.set_state(0, qutip.basis(1,0).data)

    sys2 = qutip.QobjEvo(-0.5 * qutip.qeye(1))
    inter2 = integrator(sys2.matmul_data, opt)
    inter2.set_state(0, qutip.basis(1,0).data)

    for t in np.linspace(0,1,6):
        expected1 = pytest.approx(np.exp(t/2), abs=1e-5)
        assert inter1.integrate(t)[1].to_array()[0, 0] == expected1
        expected2 = pytest.approx(np.exp(-t/2), abs=1e-5)
        assert inter2.integrate(t)[1].to_array()[0, 0] == expected2

@pytest.mark.parametrize('integrator',
    [IntegratorVern7, IntegratorVern9, IntegratorTsit5],
    ids=["vern7", 'vern9', 'tsit5']
)
def test_pickling_rk_methods(integrator):
    """Test whether VernN and Tsitoura's methods can be pickled and"
    " hence used in multiprocessing"""
    opt = {'atol':1e-10, 'rtol':1e-7}

    sys = qutip.QobjEvo(0.5 * qutip.qeye(1))
    inter = integrator(sys.matmul_data, opt)
    inter.set_state(0, qutip.basis(1,0).data)

    import pickle
    pickled = pickle.dumps(inter, -1)
    recreated = pickle.loads(pickled)
    recreated.set_state(0, qutip.basis(1,0).data)

    for t in np.linspace(0, 1, 6):
        expected = pytest.approx(np.exp(t/2), abs=1e-5)
        result1 = inter.integrate(t)[1].to_array()[0, 0]
        result2 = recreated.integrate(t)[1].to_array()[0, 0]
        assert result1 == result2 == expected

@pytest.mark.parametrize('integrator',
    [IntegratorVern7, IntegratorVern9, IntegratorTsit5],
    ids=["vern7", 'vern9', 'tsit5']
)
def test_rk_options(integrator):
    """Test whether VernN and Tsitoura's methods with no dense output."""
    opt = {
        'atol':1e-10, 'rtol':1e-7, 'interpolate':False, 'first_step':0.5
    }

    sys = qutip.QobjEvo(qutip.qeye(1)).matmul_data
    inter = integrator(sys, opt)
    inter.set_state(0, qutip.basis(1,0).data)

    for t in np.linspace(0, 1, 6):
        expected = pytest.approx(np.exp(t), abs=1e-5)
        result1 = inter.integrate(t)[1].to_array()[0, 0]
        assert result1 == expected
