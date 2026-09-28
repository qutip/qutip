import numpy as np
import pytest

import qutip
from qutip.core.environment import DrudeLorentzEnvironment, OhmicEnvironment
from qutip.solver.ulmesolve import ULMESolver, ulmesolve, UL_transform
from qutip.solver.ulmesolve import _make_operators, ULOP
from qutip import sigmax, sigmay, sigmaz, basis, qeye


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
        env = DrudeLorentzEnvironment(T=1.0, lam=0.005, gamma=5.0)
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
        env = DrudeLorentzEnvironment(T=1.0, lam=0.005, gamma=5.0)
        with pytest.raises(ValueError) as err:
            ULMESolver(sigmax(), [(op_3level, env)])
        assert "Dimension mismatch" in str(err.value)

    def test_superoperator_H_raises(self):
        env = DrudeLorentzEnvironment(T=1.0, lam=0.005, gamma=5.0)
        with pytest.raises(TypeError) as err:
            ULMESolver(qutip.liouvillian(sigmaz()), [(sigmax(), env)])
        assert "ULME cannot be used with superoperator" in str(err.value)


@pytest.mark.parametrize("method", ["eigen", "propagator"])
@pytest.mark.parametrize("lamb_shift", [True, False])
def test_runs_and_normalizes(method, lamb_shift):
    env = OhmicEnvironment(T=0.5, alpha=1.0, wc=0.5, s=1.)

    result = ulmesolve(
        sigmaz(), basis(2, 1), np.linspace(0, 1, 51), (sigmax(), env),
        options={"ULME_creation": method, "use_lamb_shift": lamb_shift},
    )
    for state in result.states:
        assert np.isclose(state.tr(), 1.0, atol=1e-6)
        assert state.isherm


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
        X = self.make_H(X_td)
        env = OhmicEnvironment(T=0.1, alpha=1.0, wc=0.5, s=1.)
        L_prop, _ = _make_operators(
            H, X, env,
            {"ULME_creation": "propagator", "use_lamb_shift": False},
        )
        L_sum = _make_dissipator_sum(H, X, env)

        np.testing.assert_allclose(
            L_sum(0.6).full(),
            L_prop(0.6).full(),
            atol = 1e-3
        )

    @pytest.mark.parametrize("H_td", [True, False])
    @pytest.mark.parametrize("X_td", [True, False])
    def test_sum_vs_prop_lamb_shift(self, H_td, X_td):
        H = self.make_H(H_td)
        X = self.make_H(X_td)
        env = OhmicEnvironment(T=0.1, alpha=1.0, wc=0.5, s=1.)
        _, lamb_prop = _make_operators(
            H, X, env,
            {"ULME_creation": "propagator", "use_lamb_shift": True},
        )
        lamb_sum = _make_ULME_lamb_shift_sum(H, X, env)
        with qutip.CoreOptions(atol=1e-5):
            assert lamb_prop(0.6).isherm
            assert lamb_sum(0.6).isherm
        np.testing.assert_allclose(
            lamb_sum(0.6).full(),
            lamb_prop(0.6).full(),
            atol = 1e-3
        )

    @pytest.mark.parametrize("lamb_shift", [True, False])
    def test_eigen_vs_prop(self, lamb_shift):
        H = self.make_H(False)
        X = self.make_H(False)
        env = OhmicEnvironment(T=0.1, alpha=1.0, wc=0.5, s=1.)
        L_eigen , lamb_eigen = _make_operators(
            H, X, env,
            {"ULME_creation": "eigen", "use_lamb_shift": lamb_shift},
        )
        L_prop, lamb_prop = _make_operators(
            H, X, env,
            {"ULME_creation": "propagator", "use_lamb_shift": lamb_shift},
        )
        np.testing.assert_allclose(
            L_eigen(0.6).full(),
            L_prop(0.6).full(),
            atol = 1e-3
        )
        if lamb_shift:
            np.testing.assert_allclose(
                lamb_eigen(0.6).full(),
                lamb_prop(0.6).full(),
                atol = 1e-3
            )
            with qutip.CoreOptions(atol=1e-5):
                assert lamb_prop(0.6).isherm
                assert lamb_eigen(0.6).isherm


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
            H, rho0, tlist, a_ops=[(X, env)], e_ops=e_ops,
        )

        for i in range(3):
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
        c_op = qutip.rand_herm(N, seed=random_generator)
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
            result_transform.e_data["num"], result_solver.e_data["num"],
            atol=1e-8
        )
        np.testing.assert_allclose(
            result_transform.e_data["energy"], result_solver.e_data["energy"],
            atol=1e-8
        )
