import numpy as np
import pytest
import pickle

import qutip
from qutip.core.environment import (
    BosonicEnvironment, DrudeLorentzEnvironment, OhmicEnvironment,
    UnderDampedEnvironment
)
from qutip.solver.ulmesolve import ULMESolver, ulmesolve, UL_transform
from qutip.solver.ulmesolve import _make_operators
from qutip import sigmax, sigmay, sigmaz, basis, qeye, num


# ---------------------------------------------------------------------------
# Reference operators
# ---------------------------------------------------------------------------

def _make_dissipator_sum(H, X, env, T=15, Nt=300):
    """
    Reference implementation: simple integration of the operator using
    trapezoidal rule.
    """
    U = qutip.Propagator(H)
    ts = np.linspace(-T, T, Nt)
    # Single vectorized call to jump_correlator
    g = qutip.coefficient(env.jump_correlator(ts), tlist=ts)

    def _trapezoid(func, t, T, Nt):
        """Trapezoidal integral of ``func(s, t)`` for ``s`` in ``[-T, T]``."""
        ts = np.linspace(-T, T, Nt)
        out = func(ts[0], t) / 2
        out += func(ts[-1], t) / 2
        for s in ts[1:-1]:
            out += func(s, t)
        return out * (2 * T / (Nt - 1))

    def func(s, t):
        Us = U(t + s, t)
        return g(-s) * Us.dag() @ X(t + s) @ Us

    def L(t):
        return _trapezoid(func, t, T, Nt)

    if H.isconstant and X.isconstant:
        return qutip.QobjEvo(L(0))
    else:
        return qutip.QobjEvo(L)


def _make_ULME_lamb_shift_sum(H, X, env, T=15, Nt=300):

    U = qutip.Propagator(H, memoize=Nt+1, tol=1e-12)

    class _Lamb:
        def __init__(self, U, X, env, T, Nt):
            self.U = U
            self.X = X
            self.g = env.jump_correlator
            self.ts = np.linspace(-T, T, Nt)

        def __call__(self, t):
            Xs = []
            gs = self.g(self.ts)
            for s in self.ts:
                Us = U(t, s + t)
                Xs.append(Us @ self.X(s + t) @ Us.dag())

            dt = self.ts[1] - self.ts[0]

            out = Xs[0] @ Xs[-1] * (gs[0] * gs[0] * -0.25)
            out += Xs[-1] @ Xs[0] * (gs[-1] * gs[-1] * 0.25)

            for i in range(1, Nt - 1):
                out -= Xs[0] @ Xs[i] * (gs[0] * gs[-i - 1] * 0.5)
                out += Xs[-1] @ Xs[i] * (gs[-1] * gs[-i - 1] * 0.5)
                out -= Xs[i] @ Xs[0] * (gs[i] * gs[-1] * 0.5)
                out += Xs[i] @ Xs[-1] * (gs[i] * gs[0] * 0.5)

            for i in range(1, Nt-1):
                for j in range(i+1, Nt-1):
                    out -= Xs[i] @ Xs[j] * (gs[i] * gs[-j - 1])
                    out += Xs[j] @ Xs[i] * (gs[j] * gs[-i - 1])

            return out * (dt ** 2 * -0.5j)

    op = _Lamb(U, X, env, T, Nt)

    if H.isconstant and X.isconstant:
        return qutip.QobjEvo(op(0))

    return qutip.QobjEvo(op)


# ---------------------------------------------------------------------------
# a_ops input parsing
# ---------------------------------------------------------------------------

class TestAOpsParsing:
    env = UnderDampedEnvironment(T=1.0, lam=0.005, gamma=5.0, w0=1.)

    @pytest.mark.parametrize("a_ops_formater", [
        pytest.param(tuple, id="tuple"),
        pytest.param(list, id="list"),
        pytest.param(lambda x: (list(x),), id="tuple[list]"),
        pytest.param(lambda x: (tuple(x),), id="tuple[tuple]"),
        pytest.param(lambda x: [tuple(x)], id="list[tuple]"),
        pytest.param(lambda x: [list(x)], id="list[list]"),
    ])
    def test_good_a_ops(self, a_ops_formater):
        tlist = np.linspace(0, 0.1, 5)
        env = UnderDampedEnvironment(T=1.0, lam=0.005, gamma=5.0, w0=1.)
        a_ops = a_ops_formater((sigmax(), env))
        ulmesolve(sigmaz(), basis(2, 1), tlist, a_ops)

    def test_empty_raises(self):
        with pytest.raises(ValueError) as err:
            ulmesolve(sigmaz(), basis(2, 1), np.linspace(0, 1, 5), [])
        assert "At least one (operator, environment) pair" in str(err.value)

    def test_bad_environment_type_raises(self):
        with pytest.raises(TypeError) as err:
            ulmesolve(
                sigmaz(), basis(2, 1), np.linspace(0, 1, 5),
                [(sigmax(), "not an environment")],
            )
        assert "BosonicEnvironment instance" in str(err.value)

    def test_dimension_mismatch_raises(self):
        op_3level = qeye(3)
        env = UnderDampedEnvironment(T=1.0, lam=0.005, gamma=5.0, w0=1.)
        with pytest.raises(ValueError) as err:
            ULMESolver(sigmax(), [(op_3level, env)])
        assert "Dimension mismatch" in str(err.value)

    def test_superoperator_H_raises(self):
        env = UnderDampedEnvironment(T=1.0, lam=0.005, gamma=5.0, w0=1.)
        with pytest.raises(TypeError) as err:
            ULMESolver(qutip.liouvillian(sigmaz()), [(sigmax(), env)])
        assert "ULME cannot be used with superoperator" in str(err.value)

    @pytest.mark.parametrize("a_ops, exc, msg", [
        ((sigmax(), env, 1), TypeError, "pair"),
        ("abc", TypeError, "list of"),
        ((env, sigmax()), TypeError, "operator, environment")
    ])
    def test_bad_a_ops(self, a_ops, exc, msg):
        with pytest.raises(exc, match=msg):
            ulmesolve(sigmaz(), basis(2, 1), [0, 1], a_ops)


# ---------------------------------------------------------------------------
# Cross-checks between construction methods.
# ---------------------------------------------------------------------------
class TestMethodsAgree:
    def make_H(self, td):
        H = [0.5 * qutip.sigmaz()]
        if td:
            H += [[0.3 * qutip.sigmax(), lambda t: np.cos(0.7 * t)]]
        return qutip.QobjEvo(H)

    def make_X(self, td):
        X = [0.8 * qutip.sigmax()]
        if td:
            X += [[0.4 * qutip.sigmay(), lambda t: np.sin(t + 0.3)]]
        return qutip.QobjEvo(X)

    @pytest.mark.parametrize("H_td", [True, False])
    @pytest.mark.parametrize("X_td", [True, False])
    def test_sum_vs_prop_jump_operator(self, H_td, X_td):
        H = self.make_H(H_td)
        X = self.make_X(X_td)
        env = OhmicEnvironment(T=0.1, alpha=1.0, wc=0.5, s=1.)
        L_prop, _ = _make_operators(
            H, [(X, env)], False, {"ULME_creation": "propagator"}
        )
        L_sum = _make_dissipator_sum(H, X, env, T=25, Nt=3000)

        np.testing.assert_allclose(
            L_sum(0.6).full(),
            L_prop[0](0.6).full(),
            atol = 2e-3
        )

    @pytest.mark.parametrize("H_td", [True, False])
    @pytest.mark.parametrize("X_td", [True, False])
    def test_sum_vs_prop_lamb_shift(self, H_td, X_td):
        H = self.make_H(H_td)
        X = self.make_X(X_td)
        env = OhmicEnvironment(T=0.1, alpha=1.0, wc=0.5, s=1.)
        _, lamb_prop = _make_operators(
            H, [(X, env)], True, {"ULME_creation": "propagator"}
        )
        lamb_sum = _make_ULME_lamb_shift_sum(H, X, env, T=25, Nt=1000)
        with qutip.CoreOptions(atol=1e-5):
            assert lamb_prop[0](0.6).isherm
            assert lamb_sum(0.6).isherm
        np.testing.assert_allclose(
            lamb_sum(0.6).full(),
            lamb_prop[0](0.6).full(),
            atol = 1e-3
        )

    @pytest.mark.parametrize("lamb_shift", [True, False])
    def test_eigen_vs_prop(self, lamb_shift):
        H = self.make_H(False)
        X = self.make_X(False)
        env = OhmicEnvironment(T=0.1, alpha=1.0, wc=0.5, s=1.)
        L_eigen , lamb_eigen = _make_operators(
            H, [(X, env)], lamb_shift, {
                "ULME_creation": "eigen",
                "eigen_pv_integral_limits": 50,
            },
        )
        L_prop, lamb_prop = _make_operators(
            H, [(X, env)], lamb_shift, {"ULME_creation": "propagator"},
        )
        np.testing.assert_allclose(
            L_eigen[0](0.6).full(),
            L_prop[0](0.6).full(),
            atol = 1e-3
        )
        if lamb_shift:
            np.testing.assert_allclose(
                lamb_eigen[0](0.6).full(),
                lamb_prop[0](0.6).full(),
                atol = 1e-3
            )
            with qutip.CoreOptions(atol=1e-5):
                assert lamb_prop[0](0.6).isherm
                assert lamb_eigen[0](0.6).isherm


class TestEvolution:
    def make_H(self, td, random_generator):
        H = [qutip.rand_herm(4, density=0.6, seed=random_generator)]
        if td:
            H += [[
                qutip.rand_herm(4, seed=random_generator),
                lambda t: np.cos(0.7 * t + .2)
            ]]
        return qutip.QobjEvo(H)

    def make_X(self, td, random_generator):
        X = [qutip.rand_herm(4, density=0.25, seed=random_generator) * 0.25]
        if td:
            X += [
                lambda t: np.exp(-10 * (t-2)**2)
            ]
        return qutip.QobjEvo(X)

    @pytest.mark.parametrize("lamb_shift", [True, False])
    def test_runs_and_normalizes(self, lamb_shift):
        env = OhmicEnvironment(T=0.5, alpha=1.0, wc=0.5, s=1.)

        H = (sigmaz() & sigmaz())
        X1 = (sigmax() & qeye(2)) * qutip.coefficient("a*t", args={"a": 0})
        X2 = (qeye(2) & sigmax()) * qutip.coefficient("b*t", args={"b": 1})

        result = ulmesolve(
            H, basis([2, 2], [1, 1]), np.linspace(0, 1, 51),
            a_ops = [(X1, env), (X2, env)], c_ops=(sigmay() & sigmay()),
            args={"a": 1}, e_ops=[num(2) & qeye(2), qeye(2) & num(2)],
            options={"use_lamb_shift": lamb_shift, "store_states": True},
        )
        for state in result.states:
            assert np.isclose(state.tr(), 1.0, atol=1e-6)
            assert state.isherm
            assert state.dims == H.dims
            assert min(state.eigenenergies()) >= 0

        np.testing.assert_allclose(result.expect[0], result.expect[1])

    @pytest.mark.parametrize("H_td", [True, False])
    @pytest.mark.parametrize("X_td", [True, False])
    def test_matches_brmesolve(self, H_td, X_td, random_generator):
        tlist = np.linspace(0, 5, 501)
        H = self.make_H(H_td, random_generator)
        X = self.make_X(X_td, random_generator)
        env = OhmicEnvironment(T=0.1, alpha=1.0, wc=0.5, s=1.)
        e_ops = [H(0), H, qutip.num(4), qutip.destroy(4)]
        rho0 = qutip.fock_dm(4, 2)

        result_ulme = ulmesolve(
            H, rho0, tlist, (X, env), e_ops=e_ops,
        )

        result_br = qutip.brmesolve(
            H, rho0, tlist, a_ops=[(X, env)], e_ops=e_ops, sec_cutoff=-1,
        )

        for i in range(4):
            np.testing.assert_allclose(
                result_ulme.expect[i],
                result_br.expect[i],
                atol=1e-2, rtol=1e-2
            )

    def test_matches_UL_transform(self, random_generator):
        tlist = np.linspace(0, 10, 50)
        N = 5
        H = qutip.QobjEvo([
            qutip.rand_herm(N, density=0.6, seed=random_generator),
            [qutip.rand_herm(N, seed=random_generator), lambda t: np.sin(t)],
        ])
        c_op = qutip.rand_herm(N, seed=random_generator) * 0.5
        env = OhmicEnvironment(T=0.5, alpha=1.0, wc=0.5, s=1.)
        e_ops = {
            "num": qutip.num(N),
            "energy": H,
        }
        rho0 = qutip.rand_dm(5, dtype="dense", seed=random_generator)

        H_eff, c_ops = UL_transform(H, (c_op, env))
        result_transform = qutip.mesolve(
            H_eff, rho0, tlist, c_ops=c_ops, e_ops=e_ops,
        )
        result_solver = ulmesolve(
            H, rho0, tlist, (c_op, env), e_ops=e_ops,
        )

        np.testing.assert_allclose(
            result_transform.e_data["num"],
            result_solver.e_data["num"],
            atol=1e-8
        )
        np.testing.assert_allclose(
            result_transform.e_data["energy"],
            result_solver.e_data["energy"],
            atol=1e-8
        )

    def test_mcsolve(self, random_generator):
        tlist = np.linspace(0, 2, 21)
        N = 5
        H = qutip.rand_herm(N, density=0.6, seed=random_generator)
        c_op = qutip.rand_herm(N, seed=random_generator)
        env = OhmicEnvironment(T=0.5, alpha=1.0, wc=0.5, s=1.)
        options = {"ULME_creation": "eigen", "use_lamb_shift": False}
        e_ops = {"num": qutip.num(N), "energy": H}
        rho0 = qutip.rand_dm(N, dtype="dense", seed=random_generator)

        H_eff, c_ops = UL_transform(H, (c_op, env), options=options)
        result_transform = qutip.mcsolve(
            H_eff, rho0, tlist, c_ops=c_ops, e_ops=e_ops, ntraj=100
        )
        result_solver = ulmesolve(
            H, rho0, tlist, (c_op, env), e_ops=e_ops, options=options
        )

        np.testing.assert_allclose(
            result_transform.average_e_data["num"],
            result_solver.e_data["num"],
            atol=0.1, rtol=0.1,
        )
        np.testing.assert_allclose(
            result_transform.average_e_data["energy"],
            result_solver.e_data["energy"],
            atol=0.1, rtol=0.1,
        )

    @pytest.mark.parametrize("method", ["eigen", "propagator"])
    def test_degen_H(self, method):
        H = qutip.qdiags([-1., -0.1, -0.01, -0.01, 0, 0, 0.01, 0.01, 0.1, 1.])
        a = qutip.destroy(10)
        X = a + a.dag()

        env = UnderDampedEnvironment(T=1.0, lam=0.005, gamma=5.0, w0=1.)

        solver = ULMESolver(
            H, a_ops = (X, env), c_ops = a.dag() * 0.1,
            options={"ULME_creation": method, "use_lamb_shift": False}
        )
        tlist = np.linspace(1, 2, 11)

        e_ops = [qutip.num(10)]
        rho0 = qutip.coherent_dm(10, 3)
        result = solver.run(rho0, tlist=tlist, e_ops=e_ops)

        ref = qutip.brmesolve(
            H, rho0, tlist = tlist,
            a_ops = [(X, env)], c_ops = a.dag() * 0.1, e_ops = e_ops,
            sec_cutoff = -1,
        )
        np.testing.assert_allclose(
            result.expect[0], ref.expect[0], atol=1e-4, rtol=1e-4
        )

    @pytest.mark.parametrize("method", ["eigen", "propagator"])
    def test_multi_bath(self, method, random_generator):
        H = qutip.rand_herm([2, 2], density=0.5, seed=random_generator)
        X0 = qeye(2) & qutip.rand_herm(2, density=0.1, seed=random_generator)
        X1 = qutip.rand_herm(2, density=0.1, seed=random_generator) & qeye(2)

        env0 = UnderDampedEnvironment(T=1.0, lam=0.05, gamma=2.0, w0=3.)
        env1 = UnderDampedEnvironment(T=1.0, lam=0.005, gamma=5.0, w0=1.)
        a_ops = [(X0, env0), (X1, env1)]

        solver = ULMESolver(H, a_ops=a_ops, options={"ULME_creation": method})

        e_ops = [qutip.num(2) & qeye(2), qeye(2) & qutip.num(2)]
        rho0 = qutip.basis([2, 2], [1, 1])
        result = solver.run(rho0, tlist=np.linspace(0, 1, 11), e_ops=e_ops)

        ref = qutip.brmesolve(
            H, rho0, tlist=np.linspace(0, 1, 11), a_ops=a_ops, e_ops=e_ops,
            sec_cutoff = -1,
        )

        np.testing.assert_allclose(
            result.expect[0], ref.expect[0], atol=1e-4, rtol=1e-4,
        )

        np.testing.assert_allclose(
            result.expect[1], ref.expect[1], atol=1e-4, rtol=1e-4,
        )

    @pytest.mark.filterwarnings(
        "ignore:overflow encountered in exp:RuntimeWarning"
    )
    def test_pulse(self):
        H = sigmaz()
        # Need an environment with thin jump_correlator
        env = OhmicEnvironment(T=0.1, alpha=1.0, wc=100, s=1.5)
        pulse = qutip.coefficient(lambda t: 5 * np.exp(-1000 * (t - 0.5)**2) )
        a_ops = (qutip.sigmax() * pulse, env)

        psi0 = qutip.basis(2, 0)
        tlist = np.linspace(0, 1.0, 101)
        result = qutip.ulmesolve(
            H, psi0, tlist, a_ops, e_ops={"z": qutip.sigmaz()},
            options={"use_lamb_shift": True, "max_step": 0.05}
        )
        # test flat outside pulse
        np.testing.assert_allclose(
            result.e_data["z"][0], result.e_data["z"][:20], atol=1e-7,
        )
        np.testing.assert_allclose(
            result.e_data["z"][-1], result.e_data["z"][-20:], atol=1e-7,
        )
        # Moved during pulse
        assert abs(result.e_data["z"][0] - result.e_data["z"][-1]) > 0.5


@pytest.mark.filterwarnings(
    "ignore:overflow encountered in exp:RuntimeWarning"
)
def test_UL_transform_no_lamb():
    H = sigmaz()
    env = OhmicEnvironment(T=0.1, alpha=1.0, wc=100, s=1.5)
    X = sigmax()
    H_eff, _ = UL_transform(H, (X, env), options={"use_lamb_shift": False})
    assert H == H_eff(0)


def test_pickle(random_generator):
    N = 3

    H = qutip.QobjEvo([
        qutip.rand_herm(N, density=0.6, seed=random_generator),
        [qutip.rand_herm(N, density=0.1, seed=random_generator), "t"],
    ])
    c_op = qutip.rand_herm(N, density=0.1, seed=random_generator)
    env = OhmicEnvironment(T=0.5, alpha=1.0, wc=0.5, s=1.)

    H_eff, c_ops = UL_transform(H, (c_op, env))

    H_rebuild = pickle.loads(pickle.dumps(H_eff))
    L_rebuild = pickle.loads(pickle.dumps(c_ops[0]))

    assert H_eff(0.2) == H_rebuild(0.2)
    assert c_ops[0](0.2) == L_rebuild(0.2)


@pytest.mark.filterwarnings("ignore:DrudeLorentzEnvironment")
@pytest.mark.filterwarnings("ignore:overflow encountered in exp:RuntimeWarning")
def test_JC_env():
    env = DrudeLorentzEnvironment(T=0.1, lam=10., gamma=1.)

    jc = env.jump_correlator_function()
    t_Max = jc.tMax

    env_from_jc = BosonicEnvironment.from_jump_correlator(jc)
    env_from_func = BosonicEnvironment.from_jump_correlator(
        lambda t: jc(t), tMax = t_Max
    )
    tlist = np.linspace(0, t_Max, 2001)
    env_from_array = BosonicEnvironment.from_jump_correlator(
        jc(tlist), tlist = tlist
    )

    np.testing.assert_allclose(jc(tlist), jc(-tlist).conj())
    assert jc(t_Max * 1.01) == 0.

    ts = np.linspace(-t_Max, t_Max, 11)
    for approx_env in [env_from_jc, env_from_func, env_from_array]:
        # Same function everywhere
        np.testing.assert_allclose(
            env.jump_correlator(ts),
            approx_env.jump_correlator(ts),
            atol=1e-10, rtol= 1e-8
        )

    ts = np.linspace(t_Max/20, t_Max/2, 101)
    for approx_env in [env_from_jc, env_from_func, env_from_array]:
        np.testing.assert_allclose(
            env.correlation_function(ts),
            approx_env.correlation_function(ts),
            atol=1e-5, rtol= 0.05
        )


@pytest.mark.filterwarnings("ignore:DrudeLorentzEnvironment")
@pytest.mark.filterwarnings("ignore:overflow encountered in exp:RuntimeWarning")
def test_jump_correlator():
    env = DrudeLorentzEnvironment(T=0.1, lam=10., gamma=1.)

    jc = env.jump_correlator_function()
    ts = np.linspace(0, 1, 11)
    np.testing.assert_allclose(jc(ts), jc(-ts).conj(), atol=1e-14)

    assert isinstance(jc(1), complex)

    t2d = np.linspace(0, 1, 4).reshape((2, 2))
    assert jc(t2d).shape == t2d.shape


@pytest.mark.filterwarnings(
    "ignore:overflow encountered in exp:RuntimeWarning"
)
def test_options_propagator():
    H = sigmaz()
    X = sigmax()
    options = {
        "ULME_creation": "propagator",
        "use_lamb_shift": True,
        "propagator_method": "vern7",
        "propagator_ODE_options": {"atol": 1e-7},
        "JC_options": {"atol": 1e-4, "rtol":1e-3, "tMax": 10},
        "conv_tol": 1e-3,
        "matrix_form": True,
        "store_final_state": True,
    }

    def jc(t):
        if np.any(t > 10):
            raise RuntimeError("Should have converged.")
        t = np.array(t)
        out = np.zeros_like(t)
        out[t < 5] = np.exp(-2 * np.abs(t[t < 5]))
        return out

    env = BosonicEnvironment.from_jump_correlator(jc, tMax=15)
    solver = ULMESolver(H, (X, env), options=options)
    result = solver.run(basis(2, 1), [0, 1], e_ops=[lambda t, state: 0])
    assert result.final_state.dims == H.dims


@pytest.mark.filterwarnings(
    "ignore:overflow encountered in exp:RuntimeWarning"
)
@pytest.mark.filterwarnings(
    "ignore:The maximum number of subdivisions:"
    "scipy.integrate._quadpack_py.IntegrationWarning"
)
def test_options_eigen():
    H = sigmaz()
    X = sigmax()
    env = OhmicEnvironment(T=0.1, alpha=1.0, wc=100, s=1.1)

    H_ref = UL_transform(
        H, (X, env),
        options={"ULME_creation": "eigen", "eigen_pv_integral_limits": 100},
    )[0](0)

    diff = []
    for limit in [1, 5, 15, 30, 60]:
        H_eff = UL_transform(
            H, (X, env),
            options={
                "ULME_creation": "eigen",
                "eigen_pv_integral_limits": limit,
        },
        )[0](0)
        diff.append((H_ref - H_eff).norm())

    assert np.all(np.diff(diff) < 0)


class TestErrors:
    env = UnderDampedEnvironment(T=1.0, lam=0.005, gamma=5.0, w0=1.)

    @pytest.mark.parametrize("td", ["H", "X"])
    def test_eigen_needs_constant(self, td):
        H = sigmaz() if td == "X" else [sigmaz(), [sigmax(), "t"]]
        X = sigmax() if td == "H" else [sigmax(), [sigmay(), "t"]]
        with pytest.raises(TypeError, match="time-independent"):
            ULMESolver(qutip.QobjEvo(H), [(qutip.QobjEvo(X), self.env)],
                       options={"ULME_creation": "eigen"})

    def test_bad_creation(self):
        H = sigmaz()
        with pytest.raises(ValueError, match="Unknown ULME_creation"):
            ULMESolver(H, [(H, self.env)], options={"ULME_creation": "None"})
