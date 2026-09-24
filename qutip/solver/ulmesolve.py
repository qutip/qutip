import numpy as np
import scipy
import scipy.integrate as integrate
from scipy.interpolate import CubicSpline
import functools
import itertools

from numpy.typing import ArrayLike
from .. import Qobj, QobjEvo
from .mesolve import MESolver
from .solver_base import Solver
from ..typing import EopsLike, QobjEvoLike
from . import Result
import qutip.core.data as _data
from .propagator import Propagator
from ..core.environment import BosonicEnvironment
from typing import Any
from .cy._ulme import split_dense, merge_dense
import qutip as qt


__all__ = ["ulmesolve", "ULMESolver", "UL_transform"]


def ulmesolve(
    H: QobjEvoLike,
    rho0: Qobj,
    tlist: ArrayLike,
    a_ops: list[tuple[QobjEvoLike, BosonicEnvironment]] | tuple[QobjEvoLike, BosonicEnvironment],
    *,
    e_ops: EopsLike | list[EopsLike] | dict[Any, EopsLike] = None,
    args: dict[str, Any] = None,
    options: dict[str, Any] = None,
) -> Result:
    """
    Evolution of a density matrix using the Universal Lindblad Master Equation
    (ULME). This solver is suitable for open systems where the secular
    approximation is not fully justified.

    Parameters
    ----------
    H : :obj:`.Qobj`, :obj:`.QobjEvo` or :obj:`.QobjEvo` compatible format.
        System Hamiltonian. Can be time-independent or time-dependent.

    rho0 : :obj:`.Qobj`
        Initial state density matrix or state vector (ket).

    tlist : array_like
        List of times for which to save the state.

    a_ops : [list of] tuple[:obj:`.Qobj` | :obj:`.QobjEvo`, BosonicEnvironment]
        Single or list of system-bath coupling operators and their
        corresponding environments. All baths must be independent.

    e_ops : list of :obj:`.Qobj` / callback function, optional
        Single operator or list of operators for which to evaluate
        expectation values.

    args : dict, optional
        Dictionary of parameters for time-dependent Hamiltonians or coupling
        operators.

    options : dict, optional
        Options for the solver. All options for mesolve are supported.
        ULME-specific are:

        - | ULME_creation : str {"eigen", "prop"}
          | Method used to construct the Lindblad jump operators.
            "eigen": faster method for constant system.
            "prop": computationally intensive general method.
        - | use_lamb_shift : bool, True
          | Whether to calculate and include the Lamb shift correction in the
            effective Hamiltonian.

        All options are listed in ``ULMESolver.options``'s docstring.

    Returns
    -------
    result : :obj:`.Result`
        An instance of the class :obj:`.Result`, containing expectation
        values and/or states.
    """
    # Backward compatibility warnings

    H = QobjEvo(H, args=args, tlist=tlist)
    if not a_ops:
        raise ValueError(
            "ULMESolve requires at least one "
            "(operator, environment) pair in a_ops."
        )

    if (
        isinstance(a_ops, tuple)
        and len(a_ops) == 2
        and isinstance(a_ops[1], BosonicEnvironment)
    ):
        # Single (op, bath) pair
        a_ops = [a_ops]
    elif isinstance(a_ops, tuple) and all(isinstance(x, tuple) for x in a_ops):
        # Tuple of tuples passed instead of list
        a_ops = list(a_ops)
    elif not isinstance(a_ops, list):
        raise TypeError(
            "a_ops must be a list of (operator, environment) tuples."
        )
    a_ops = [
        (QobjEvo(op, args=args, tlist=tlist), bath)
        for op, bath in a_ops
    ]

    solver = ULMESolver(H, a_ops, options=options)
    return solver.run(rho0, tlist, e_ops=e_ops)


class ULMESolver(MESolver):
    """
    Universal Lindblad Master Equation evolution of a density matrix for a
    given Hamiltonian and set of bath with their coupling operators.

    The ULME provides a completely positive and trace-preserving (CPTP)
    description of open system dynamics that remains valid beyond the
    standard secular approximation.

    See: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.102.115109

    Parameters
    ----------
    H : :obj:`.Qobj`, :obj:`.QobjEvo`
        Possibly time-dependent system Liouvillian or Hamiltonian as a Qobj or
        QobjEvo. List of [:obj:`.Qobj`, :obj:`.Coefficient`] or callable that
        can be made into :obj:`.QobjEvo` are also accepted.
        Note: H cannot be a superoperator.

    a_ops : list of tuple[:obj:`.Qobj` | :obj:`.QobjEvo`, BosonicEnvironment]
        List of tuples where each contains a system-bath coupling
        operator and its corresponding :obj:`BosonicEnvironment`. All baths are
        assumed to be independent.

    options : dict, optional
        Options for the solver,
        ULME-specific settings are:
        - "ULME_creation": Method for creating jump operators.
        - "use_lamb_shift": Whether to include the Lamb shift.
        See :obj:`ULMESolver.options` and
        `Integrator <./classes.html#classes-ode>`_ for a list of all options.

    Attributes
    ----------
    stats: dict
        Diverse diagnostic statistics of the evolution.
    """
    name = "Universal lindblad equation"
    solver_options = {
        "progress_bar": "",
        "progress_kwargs": {"chunk_size": 10},
        "store_final_state": False,
        "store_states": None,
        "normalize_output": True,
        "method": "adams",
        "matrix_form": False,
        "ULME_creation": None,
        "use_lamb_shift": True,
        "Nt": 500,
        "T": 20,
        "prop_options": {},
    }
    def __init__(self, H, a_ops, *, options=None):
        self.H = QobjEvo(H)
        if H.issuper:
            raise TypeError(
                "ULME cannot be used with superoperator H"
            )
        self.a_ops = []
        for i, (op, env) in enumerate(a_ops):
            op = QobjEvo(op)
            if not isinstance(env, BosonicEnvironment):
                raise TypeError(
                    f"The environment for a_ops[{i}] must be a "
                    f"BosonicEnvironment instance, but got {type(env)}."
                )
            if not self.H._dims == op._dims:
                raise ValueError(
                    f"Dimension mismatch in a_ops[{i}]: "
                    f"Hamiltonian dims {self.H.dims} "
                    f"do not match coupling operator dims {op.dims}."
                )
            self.a_ops.append((op, env))

        self._num_collapse = len(self.a_ops)

        self.options = options

        self.c_ops = []
        self.lamb_shifts = []

        for op, env in self.a_ops:
            L, Lamb = _make_l_lambda(self.H, op, env, self.options)
            self.c_ops.append(L)
            self.lamb_shifts.append(Lamb)

        super().__init__(
            self.H + sum(self.lamb_shifts), self.c_ops, options=self.options
        )

    @property
    def options(self) -> dict:
        """
        Solver's options:

        store_final_state: bool, default: False
            Whether or not to store the final state of the evolution in the
            :obj:`.Result` object.

        store_states: bool, default: None
            Whether or not to store the state vectors or density matrices.
            If `None` the states will be saved if no expectation operators are
            given.

        normalize_output: bool, default: True
            Normalize output state to hide ODE numerical errors.

        progress_bar: str {"text", "enhanced", "tqdm", ""}, default: ""
            The type of progress bar to use. 'tqdm' requires the installation
            of the ``tqdm`` module. An empty string or ``False``
            disables the progress bar.

        progress_kwargs: dict, default: {"chunk_size": 10}
            Arguments to pass to the progress_bar. Qutip's bars use
            ``chunk_size``.

        method: str, default: "adams"
            Which ODE integration method to use. All available ODE method can
            be listed with the ``avail_integrators`` method.

        ULME_creation: str {"eigen", "prop"}, default: None
            Method used to construct the Lindblad jump operators:

            - "eigen": Constructs dissipators via the eigen-decomposition
              of the Hamiltonian. This requires a time-independent system
              (Hamiltonian and coupling operators) and utilizes the bath's
              ``power_spectrum``.

            - "prop": Constructs dissipators by convolving the coupling
              operator with the bath's ``jump_correlator`` in the interaction
              picture. This is generally more computationally intensive but
              more flexible.

            Per default, "eigen" will be used for constant system, and "prop"
            otherwise.

        use_lamb_shift: bool, default: True
            Whether to calculate and include the Lamb shift correction in the
            effective Hamiltonian.
        """
        return self._options

    @options.setter
    def options(self, new_options):
        Solver.options.fset(self, new_options)


def UL_transform(
    H: Qobj | QobjEvo,
    a_ops:
        tuple[Qobj | QobjEvo, BosonicEnvironment]
        | list[tuple[Qobj | QobjEvo, BosonicEnvironment]],
    options: dict=None
) -> tuple[Qobj | QobjEvo, list[Qobj | QobjEvo]]:
    """
    Transform the system according to universal lindblad equation.
    It shift the Hamiltonian energies according to the lamb shift and compute
    the dissipators.

    Parameters
    ----------
    H: Qobj or QobjEvo
        Hamiltonian of the system
    a_ops: list of tuple of (Qobj | QobjEvo, BosonicEnvironment)
        One or multiple coupling operators with their bath as
        BosonicEnvironment. The bath must be independent.

        .. note::

            The power_spectrum and jump_correlator of the baths are used.
            When possible, computing analytically the jump_correlator will
            provide less numerical error.

        .. note::

            Coupling operators must be Hermitian

    options: dict, optional
        Options used to compute the operators. The following options are used:
        - "ULME_creation": {"eigen", "prop"}
          Method used to compute the operators, either eigen decomposition or
          integrating over convolution of the operators with the jump_correlator.
        - "use_lamb_shift": True,
          Compute the lamb shift and add it to the Hamiltonian.
        - "Nt": 500,
          Number of time point used in the integration
        - "T": 20
          Range of the integration, the jump_correlator is expected to be
          effectively 0 after this.
        - "prop_options": {},
          Option passed to sesolve used to compute the propagators.

    Returns
    -------
    H, c_ops:
        The corrected Hamiltonian and collapse operators.
        These are formated so they can be used directly in mesolve or mcsolve.
    """
    opt = {
        "ULME_creation": None,
        "use_lamb_shift": True,
        "Nt": 500,
        "T": 20,
        "prop_options": {}
    }
    H_evo = QobjEvo(H, copy=False)
    if isinstance(options, dict):
        opt.update(options)
    if isinstance(a_ops, tuple):
        a_ops = [a_ops]
    c_ops = []
    lamb_shifts = []
    for op, env in a_ops:
        op = QobjEvo(op, copy=False)
        L, Lamb = _make_l_lambda(H_evo, op, env, opt)
        c_ops.append(L)
        lamb_shifts.append(Lamb)

    return H + sum(lamb_shifts), c_ops


def _make_l_lambda(H, X, env, options=None):
    options = options or {}
    method = options.get("ULME_creation", None)
    if method is None:
        if H.isconstant and X.isconstant:
            method = "eigen"
        else:
            method = "prop"

    if not options["use_lamb_shift"]:
        if method == "eigen":
            return _make_L_eigs(H, X, env), 0
        elif method == "prop":
            return _make_L_prop(H, X, env, options), 0
        raise NotImplementedError(...)

    else:
        if method == "eigen":
            return _make_L_lambda_eigen(H, X, env)
        elif method == "prop":
            return _make_l_lambda_prop(H, X, env, options)
        elif method == "prev":
            return _make_l_lambda_prop_prev(H, X, env, options)
        raise NotImplementedError(...)


def _make_L_prop(
    H: QobjEvo,
    X: QobjEvo,
    env: BosonicEnvironment,
    options=None,
):
    options = options or {}
    T = options.get("T", 15)
    Nt = options.get("Nt", 500)
    Nt = options.get("Nt", 500)
    U = Propagator(H)

    def integrate_1d_flat(op, t, T, Nt):
        ts = np.linspace(-T, T, Nt)
        out = op(ts[0], t) / 2
        out += op(ts[-1], t) / 2
        for s in ts[1:-1]:
            out += op(s, t)
        return out * (2 * T / (Nt - 1))

    def op(s, t):
        Us = U(t+s, t)
        return env.jump_correlator(-s) * Us.dag() @ X(t+s) @ Us

    def L(t):
        return integrate_1d_flat(op, t, T, Nt)

    if H.isconstant and X.isconstant:
        return QobjEvo(L(0))
    else:
        return QobjEvo(L)


def _make_L_eigs(H: QobjEvo, X: QobjEvo, env: BosonicEnvironment):
    if not (H.isconstant and X.isconstant):
        raise TypeError

    vals, vecs = H(0).eigenstates(output_type="oper")
    X_H = (vecs.dag() @ X(0) @ vecs).data
    L_H = _data.multiply(X_H, _data.Dense(env._g_w(-np.subtract.outer(vals, vals))))
    L = vecs @ Qobj(L_H, dims=X.dims) @ vecs.dag() * (np.pi * 2)
    return L


def _make_L_lambda_eigen(H: QobjEvo, X: QobjEvo, env: BosonicEnvironment):
    if not (H.isconstant and X.isconstant):
        raise TypeError

    @functools.lru_cache
    def f(e1, e2, eps=1e-5):
        return integrate.quad(
            lambda w: env._g_w(w -e1) * env._g_w(w + e2),
            -50, 50, weight='cauchy', wvar=0
        )[0] * (-2 * np.pi)

    vals, vecs = H(0).eigenstates(output_type="oper")
    X_diag = (vecs.dag() @ X(0) @ vecs)
    X_np = X_diag.full()
    X_data = X_diag.data
    L_responce = _data.Dense(env._g_w(-np.subtract.outer(vals, vals)))
    L_H = _data.multiply(X_data, L_responce) * (np.pi * 2)

    N = len(vals)
    fs = np.zeros((N, N, N), dtype=float)
    for i, j, k in itertools.product(range(N), repeat=3):
        #TODO: optimize
        fs[i, j, k] = f(vals[j] - vals[i], vals[k] - vals[j])

    LL = np.einsum("ijk,ij,jk->ik", fs, X_np, X_np)
    return (
        vecs @ Qobj(L_H) @ vecs.dag(),
        vecs @ Qobj(LL) @ vecs.dag()
    )



class ULOP():
    def __init__(self, H, X, env, options={}):
        self.H = H
        self.X = X
        self.size = H.shape[0]
        self.options = options
        self.t = None
        self._L = None
        self._Lamd = None
        self.integrator = qt.solver.integrator.IntegratorTsit5(self.derr, {})
        self.prepare_g(env.jump_correlator)

    def prepare_g(self, jc):
        """
        Create a spline for the jump_correlator.
        The jump_correlator is expected to decrease exponentially:
           jc(t) = f(t) * exp(-t*alpha)
        but we don't know the units so have to estimate the cutoff.
        """
        ts = np.logspace(-8, 3, 201)
        jcs = jc(ts)
        idx = np.where(np.abs(jcs) > self.options.get("tol", 1e-6))[0]

        if len(idx) == 0:
            t_max = 1000.
        else:
            t_max = ts[np.max(idx)]

        ts = np.linspace(0, t_max, 1001)
        jcs = jc(ts)
        self.g = qt.coefficient(jcs, tlist=ts)
        self.t_scale = t_max / 100

    @staticmethod
    def merge_states(list_state):
        N = len(list_state)
        if isinstance(list_state, qt.Qobj):
            state0 = [op.data for op in list_state]
        else:
            state0 = list_state
        state0 = [qt.core.data.column_stack(state) for state in state0]
        state = qt.core.data.dense.zeros(state0[0].shape[0], N, fortran=True)
        for i in range(N):
            state.as_ndarray()[:, i] = state0[i].to_array()[:, 0]
        return state

    @staticmethod
    def split_states(state, ncol):
        if isinstance(state, qt.Qobj):
            state = state.data
        # split = qt.core.data.split_columns(state, copy=False)
        out = []
        for op in state.as_ndarray().T:
            data = qt.data.dense.fast_from_numpy(op)
            out.append(qt.data.column_unstack_dense(data, ncol, inplace=True))
        return out

    def initial(self, t):
        Id = qt.data.dense.identity(self.size)
        zero = qt.core.data.dense.zeros(self.size, self.size, fortran=True)
        self._tmp = zero.copy()
        self._Xp = zero.copy()
        self._Xm = zero.copy()
        self._derr = self.merge_states([zero, zero, zero, zero, zero, zero])

        return self.merge_states([Id, Id, zero, zero, zero, zero])

    def derr(self, s, state):
        states = self.split_states(state, self.size)
        # derrivative = self.split_states(self._derr, self.size)
        Xp = states[0].adjoint() @ self.X._call(s + self.t) @ states[0]
        Xm = states[1].adjoint() @ self.X._call(-s + self.t) @ states[1]
        g = self.g(s)
        derr = [
            -1j * self.H._call(s + self.t) @ states[0],
            1j * states[1] @ self.H._call(self.t - s),
            Xp * g.conjugate(),
            Xm * g,
            Xp @ states[2] * (2 * g),
            Xm @ states[3] * (-2 * g.conjugate()),
        ]
        return self.merge_states(derr)

    def derr2(self, s, state):
        #states = self.split_states(state, self.size)
        states = split_dense(state, self.size)
        _derr = _data.dense.zeros(self.size**2, 6, fortran=True)
        #derrivative = self.split_states(_derr, self.size)
        derrivative = split_dense(_derr, self.size)
        g = self.g(s)

        # Inplace is needed for this to work

        #Propagator

        self.H.adjoint_rmatmul_data(self.t + s, states[0], out=derrivative[0], scale=1j)
        self.H.matmul_data(self.t - s, states[1], out=derrivative[1], scale=-1j)
        """
        self.H.matmul_data(self.t + s, states[0], out=derrivative[0], scale=-1j)
        self.H.adjoint_rmatmul_data(self.t - s, states[1], out=derrivative[1], scale=1j)
        """

        # diffusion correction terms

        self._tmp = _data.imul_dense(self._tmp, 0)
        self._tmp = self.X.adjoint_rmatmul_data(self.t + s, states[0], out=self._tmp)
        _data.matmul_dag_dense(self._tmp, states[0], out=derrivative[2], scale=g.conjugate())
        self._tmp = _data.imul_dense(self._tmp, 0)
        self._tmp = self.X.adjoint_rmatmul_data(self.t - s, states[1], out=self._tmp)
        _data.matmul_dag_dense(self._tmp, states[1], out=derrivative[3], scale=g)
        """
        self._tmp = _data.imul_dense(self._tmp, 0)
        self._tmp = self.X.adjoint_rmatmul_data(self.t + s, states[0].adjoint(), out=self._tmp)
        _data.matmul_dag_dense(self._tmp, states[0].adjoint(), out=derrivative[2], scale=g.conjugate())
        self._tmp = _data.imul_dense(self._tmp, 0)
        self._tmp = self.X.adjoint_rmatmul_data(self.t - s, states[1].adjoint(), out=self._tmp)
        _data.matmul_dag_dense(self._tmp, states[1].adjoint(), out=derrivative[3], scale=g)
        """

        # second order terms for lamb shift
        _data.matmul_dense(derrivative[2], states[2], out=derrivative[4], scale=(2 * g / g.conjugate()))
        _data.matmul_dense(derrivative[3], states[3], out=derrivative[5], scale=(-2 * g.conjugate() / g))

        return _derr

    def L(self, t):
        if t != self.t:
            self.compute(t)
        return qt.Qobj(self._L, dims=self.H._dims)

    def Lamd(self, t):
        if t != self.t:
            self.compute(t)
        return qt.Qobj(self._Lamd, dims=self.H._dims)

    def compute(self, t, tol = 1e-4):
        prev = self.initial(t)
        self.t = t
        self.integrator.set_state(0, prev)
        t_scale = self.t_scale / 10

        diff = tol + 1
        s = 0
        while diff > tol:
            s += t_scale
            _, state = self.integrator.integrate(s)
            diff = np.linalg.norm(
                state.to_array()[:, 2] - prev.to_array()[:, 2], 2
            )
            prev = state
        state = self.split_states(state, self.size)
        self._L = (state[2] + state[3])
        self._Lamd = ((state[2].adjoint() + state[3].adjoint()) @ (-state[2] + state[3]) + (state[-1] + state[-2])) * -0.5j



def _make_l_lambda_prop(H, X, env, options=None):
    op = ULOP(H, X, env, options)

    if H.isconstant and X.isconstant:
        return QobjEvo(op.L(0)), QobjEvo(op.Lamd(0))

    return QobjEvo(op.L), QobjEvo(op.Lamd)


# ------------- Development utility functions ------------

class OP:
    def __init__(self, U, X, env, T, Nt):
        self.U1 = U[0]
        self.U2 = U[1]
        self.X = X
        self.g = env.jump_correlator
        self.ts = np.linspace(0, T, Nt)
        self.t = None
        self._L = None
        self._Lamd = None

    def compute(self, t):
        # TODO: auto detect T, Nt according to the jump_correlator convergence
        dt = self.ts[1] - self.ts[0]
        Nt = len(self.ts)

        Xp = self.X(t)
        Xm = self.X(t)
        g = self.g(0)

        Lp = Xp * (g.conjugate() * 0.5)
        Lm = Xm * (g * 0.5)
        Ip = Xp * (g * 0.5)
        Im = Xm * (g.conjugate() * 0.5)
        Yp = Xp @ Lp * g
        Ym = Xm @ Lm * (g.conjugate() * -1)
        Ut = self.U1(t)
        Ut_inv = Ut.dag()

        for i, s in enumerate(self.ts[1:]):
            Us = self.U1(s + t) @ Ut_inv
            Xp = Us.dag() @ self.X(s + t) @ Us
            Us = Ut @ self.U2(0, t - s)
            Xm = Us @ self.X(t - s) @ Us.dag()
            g = self.g(s)

            Lp += Xp * g.conjugate()
            Lm += Xm * g
            Ip += Xp * g
            Im += Xm * g.conjugate()
            Yp += Xp @ Lp * (g * 2)
            Ym += Xm @ Lm * (g.conjugate() * -2)

        self._L = (Lp + Lm) * dt
        self._Lamd = ((Ip + Im) @ (-Lp + Lm) + (Yp + Ym)) * (dt**2 * -0.5j)
        self.t = t

    def L(self, t):
        if t != self.t:
            self.compute(t)
        return self._L

    def Lamd(self, t):
        if t != self.t:
            self.compute(t)
        return self._Lamd


def _make_l_lambda_prop_prev(H, X, env, options=None):
    options = options or {}
    T = options.get("T", 15)
    Nt = options.get("Nt", 300)
    # Propagator memoization scheme is bad with tracking 2 evolutions at once...
    U1 = Propagator(H, tol=1e-12, options={"method": "tsit5"})
    U2 = Propagator(H, tol=1e-12, options={"method": "tsit5"})
    op = OP([U1, U2], X, env, T, Nt)

    if H.isconstant and X.isconstant:
        return op.L(0), op.Lamd(0)

    return QobjEvo(op.L), QobjEvo(op.Lamd)


def cont_t2w_fft(ft, t_max, Nt):
    dt = t_max * 2**(1 - Nt)
    N = 2**Nt
    ts = np.linspace(-t_max, t_max - dt, N)
    vals = ft(ts)
    ffts = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(vals)) * 2 * t_max / N)[::-1]
    ws = np.fft.fftshift(2 * np.pi  * np.fft.fftfreq(N, dt))
    return CubicSpline(ws[1:], ffts[:-1])


def cont_w2t_fft(fw, t_max, Nt):
    dt = t_max * 2**(1 - Nt)
    N = 2**Nt
    ws = np.linspace(-t_max, t_max - dt, N)
    vals = fw(ws)
    ffts = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(vals)) * t_max / np.pi)[::-1]
    ts = np.fft.fftshift(2 * np.pi  * np.fft.fftfreq(N, dt))
    return CubicSpline(ts[1:], ffts[:-1])


def _make_lambda_prop_old(
    H: QobjEvo, X: QobjEvo, env: BosonicEnvironment, T=15, Nt=300
):
    U = Propagator(H, memoize=Nt+1, tol=1e-12)

    class OP:
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

            for i in range(1, Nt-1):
                out -= Xs[0] @ Xs[i] * (gs[0] * gs[-i - 1] * 0.5)
                out += Xs[-1] @ Xs[i] * (gs[-1] * gs[-i - 1] * 0.5)
                out -= Xs[i] @ Xs[0] * (gs[i] * gs[-1] * 0.5)
                out += Xs[i] @ Xs[-1] * (gs[i] * gs[0] * 0.5)

            for i in range(1, Nt-1):
                for j in range(i+1, Nt-1):
                    out -= Xs[i] @ Xs[j] * (gs[i] * gs[-j-1])
                    out += Xs[j] @ Xs[i] * (gs[j] * gs[-i-1])

            return out * (dt ** 2 * -0.5j)

    op = OP(U, X, env, T, Nt)

    if H.isconstant and X.isconstant:
        return op(0)

    return QobjEvo(op)
