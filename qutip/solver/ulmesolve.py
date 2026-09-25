import functools
import itertools
import warnings
from typing import Any

import numpy as np
import scipy.integrate as integrate
from numpy.typing import ArrayLike

from .. import Qobj, QobjEvo
from ..core import data as _data
from ..core.coefficient import coefficient
from ..core.environment import BosonicEnvironment
from ..typing import EopsLike, QobjEvoLike
from .mesolve import MESolver
from .result import Result
from .solver_base import Solver
from .cy._ulme import split_dense, merge_dense


__all__ = ["ulmesolve", "ULMESolver", "UL_transform"]


_ULME_DEFAULT_OPTIONS = {
    "ULME_creation": "propagator",
    "use_lamb_shift": True,
    "tol": 1e-6,
    "eigen pv integral limits": 30,
}


def ulmesolve(
    H: QobjEvoLike,
    rho0: Qobj,
    tlist: ArrayLike,
    a_ops: (
        list[tuple[QobjEvoLike, BosonicEnvironment]]
        | tuple[QobjEvoLike, BosonicEnvironment]
    ),
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

        - | ULME_creation : str {"eigen", "propagator"}
          | Method used to construct the Lindblad jump operators.
          | "eigen": eigen-decomposition of the Hamiltonian, constant system
            only.
          | "propagator": integration in the interaction picture, general.
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
    a_ops = _parse_a_ops(a_ops, args, tlist)
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
    name = "Universal Lindblad equation"
    solver_options = {**MESolver.solver_options, **_ULME_DEFAULT_OPTIONS}
    def __init__(self, H, a_ops, *, options=None):
        self.H = QobjEvo(H)
        if self.H.issuper:
            raise TypeError("ULME cannot be used with superoperator H")
        self.a_ops = _parse_a_ops(a_ops)
        for i, (op, _) in enumerate(self.a_ops):
            if op.dims != H.dims:
                raise ValueError(
                    f"Dimension mismatch in a_ops[{i}]: "
                    f"Hamiltonian dims {H.dims} "
                    f"do not match coupling operator dims {op.dims}."
                )

        self._num_collapse = len(self.a_ops)
        self.options = options

        self.c_ops = []
        self.lamb_shifts = []

        for op, env in self.a_ops:
            L, lamb_shift = _make_operators(self.H, op, env, self.options)
            self.c_ops.append(L)
            self.lamb_shifts.append(lamb_shift)

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

        ULME_creation: str {"eigen", "propagator"}, default: None
            Method used to construct the Lindblad jump operators:

            - "eigen": Constructs dissipators via the eigen-decomposition
              of the Hamiltonian. This requires a time-independent system
              (Hamiltonian and coupling operators) and utilizes the bath's
              ``power_spectrum``.

            - "propagator": Constructs dissipators by convolving the coupling
              operator with the bath's ``jump_correlator`` in the interaction
              picture. Works for time-dependent systems and is
              generally faster than "eigen" when the Lamb shift is included.

            Per default, "eigen" will be used for constant system, and "propagator"
            otherwise.

        use_lamb_shift: bool, default: True
            Whether to calculate and include the Lamb shift correction in the
            effective Hamiltonian.

        ... TODO: more ULOP options to come.

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
    It shifts the Hamiltonian energies according to the Lamb shift and
    computes the dissipators.

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

        - "ULME_creation": {"eigen", "propagator"}
          Method used to compute the operators, either eigen decomposition or
          integration of the convolution of the operators with the
          jump_correlator.
        - "use_lamb_shift": True,
          Compute the lamb shift and add it to the Hamiltonian.
        - "prop_options": {},
          Option passed to sesolve used to compute the propagators.
        - ... TODO: Add more.

    Returns
    -------
    H, c_ops:
        The corrected Hamiltonian and collapse operators.
        These are formated so they can be used directly in mesolve or mcsolve.
    """
    options = {
        "ULME_creation": "propagator",
        "use_lamb_shift": True,
        "tol": 1e-6,
        "prop_options": {},
        **(options or {}),
    }
    H_evo = QobjEvo(H, copy=False)
    c_ops = []
    lamb_shifts = []
    for op, env in _parse_a_ops(a_ops):
        L, Lamb = _make_operators(H_evo, op, env, options)
        c_ops.append(L)
        lamb_shifts.append(Lamb)

    return H + sum(lamb_shifts), c_ops


def _parse_a_ops(a_ops, args=None, tlist=None):
    """
    Normalize ``a_ops`` to a list of ``(QobjEvo, BosonicEnvironment)``.
    Accepts a single pair, a list of pairs or a tuple of pairs.
    """
    if not a_ops:
        raise ValueError(
            "At least one (operator, environment) pair is required in a_ops."
        )
    if (
        isinstance(a_ops, (tuple, list))
        and len(a_ops) == 2
        and isinstance(a_ops[1], BosonicEnvironment)
    ):
        a_ops = [a_ops]
    elif isinstance(a_ops, tuple):
        a_ops = list(a_ops)
    if not isinstance(a_ops, list):
        raise TypeError(
            "a_ops must be a list of (operator, environment) tuples."
        )

    parsed = []
    for i, pair in enumerate(a_ops):
        if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
            raise TypeError(
                f"a_ops[{i}] must be an (operator, environment) pair."
            )
        op, env = pair
        if not isinstance(env, BosonicEnvironment):
            raise TypeError(
                f"The environment for a_ops[{i}] must be a "
                f"BosonicEnvironment instance, but got {type(env)}."
            )
        parsed.append((QobjEvo(op, args=args, tlist=tlist), env))
    return parsed


def _make_operators(
    H: QobjEvo,
    X: QobjEvo,
    env: BosonicEnvironment,
    options: dict=None
):
    """
    Build the jump operator and Lamb shift for one system-bath coupling.

    Returns
    -------
    L : QobjEvo
        Jump operator. A Qobj when the system is constant.
    lamb_shift : QobjEvo or {0}
        Lamb shift Hamiltonian, 0 when ``use_lamb_shift`` is False.
    """
    method = options.get("ULME_creation")
    use_lamb_shift = options.get("use_lamb_shift", True)

    if method == "eigen":
        return _operators_eigen(H, X, env, use_lamb_shift, options)
    if method == "propagator":
        return _operators_prop(H, X, env, use_lamb_shift, options)
    raise ValueError(
        f"Unknown ULME_creation method {method!r}, "
        "expected 'eigen' or 'propagator'."
    )


# ---------------------------------------------------------------------------
# "eigen" method: constant system, eigen-decomposition of H.
# ---------------------------------------------------------------------------

def _operators_eigen(
    H: QobjEvo,
    X: QobjEvo,
    env: BosonicEnvironment,
    use_lamb_shift: bool,
    options: dict,
):
    if not (H.isconstant and X.isconstant):
        raise TypeError(
            "ULME_creation='eigen' requires a time-independent Hamiltonian "
            "and coupling operator."
        )

    limits = options.get("eigen pv integral limits", 50)  # Add to solver options
    @functools.lru_cache(maxsize=None)
    def _integral(e1, e2):
        return integrate.quad(
            lambda w: env._g_w(w -e1) * env._g_w(w + e2),
            -limits, limits, weight='cauchy', wvar=0
        )[0] * (-2 * np.pi)

    vals, vecs = H(0).eigenstates(output_type="oper")
    X_diag = (vecs.dag() @ X(0) @ vecs)
    X_np = X_diag.full()
    X_data = X_diag.data
    L_responce = _data.Dense(env._g_w(-np.subtract.outer(vals, vals)))
    L_H = _data.multiply(X_data, L_responce) * (np.pi * 2)
    if not use_lamb_shift:
        return QobjEvo(vecs @ Qobj(L_H) @ vecs.dag()), 0

    N = len(vals)
    fs = np.zeros((N, N, N), dtype=float)
    for i, j, k in itertools.product(range(N), repeat=3):
        #TODO: optimize
        fs[i, j, k] = _integral(vals[j] - vals[i], vals[k] - vals[j])

    LL = np.einsum("ijk,ij,jk->ik", fs, X_np, X_np)
    return (
        QobjEvo(vecs @ Qobj(L_H) @ vecs.dag()),
        QobjEvo(vecs @ Qobj(LL) @ vecs.dag()),
    )

# ---------------------------------------------------------------------------
# "propagator" method: integration in the interaction picture.
# ---------------------------------------------------------------------------

def _operators_prop(H, X, env, use_lamb_shift, options):
    op = ULOP(H, X, env, options)
    if H.isconstant and X.isconstant:
        return (
            QobjEvo(op.L(0)),
            (QobjEvo(op.lamb_shift(0)) if use_lamb_shift else 0),
        )
    return QobjEvo(op.L), (QobjEvo(op.lamb_shift) if use_lamb_shift else 0)


class ULOP():
    """
    Jump operator and Lamb shift of the ULME at time ``t``, computed by
    integrating, over the relative time ``s``, the coupling operator in the
    interaction picture (with reference time ``t``) against the bath's
    ``jump_correlator`` ``g``:

    .. math::

        L(t) = \\int_{-\\infty}^\\infty ds g(s) X_I(t - s)

    The Lamb shift needs nested (second order) integrals of the same
    quantities.

    All integrals are computed together in one ODE in ``s``. The integrated
    variables are stacked as columns of a single dense matrix, ordered as
    ``[Up, Um, Lp, Lm, Yp, Ym]`` (the last two only with the Lamb shift):

    - ``Up``, ``Um``: propagators forward (``t -> t + s``) and backward.
    - ``Lp``, ``Lm``: integrals of ``g*(s) X_I(t + s)`` and ``g(s) X_I(t - s)``.
    - ``Yp``, ``Ym``: second order integrals for the Lamb shift.

    Parameters
    ----------
    H, X : QobjEvo
        System Hamiltonian and Hermitian coupling operator.
    env : BosonicEnvironment
        Bath defining the jump_correlator
    options : dict, optional
        ``tol`` is used to find the time window over which ``g`` is not zero.
    """
    def __init__(self, H, X, env, options=None):
        self.H = H
        self.X = X
        self.size = H.shape[0]
        self.options = options or {}
        self._with_lamb = options.get("use_lamb_shift", True)
        self._ncols = 6 if self._with_lamb else 4

        self._g_tol = options.get("g tol", 1e-6)
        self._conv_tol = options.get("conv tol", 1e-4)

        self.t = None
        self._L = None
        self._lamb = None

        integrator = Solver.avail_integrators()[
            options.get("prop method", "tsit5")
        ]
        integrator_options = options.get("prop ODE options", {})
        self._integrator = integrator(self._rhs, {})
        self._prepare_g(env.jump_correlator)

    def _prepare_g(self, jump_correlator):
        """
        Create a spline for the jump_correlator.
        The jump_correlator is expected to decrease exponentially:
           jc(t) = f(t) * exp(-t*alpha)
        but we don't know the units so have to estimate the cutoff.
        """
        ts = np.logspace(-8, 3, 201)
        correlator = jump_correlator(ts)
        above = np.where(np.abs(correlator) > self._g_tol)[0]
        # TODO: is 1000 enough?
        # Look farter if not converged yet?
        # Add time scale options?
        # TODO: at least document this as a hard limit somewhere.
        t_max = ts[above[-1]] if above.size else 1000.

        ts = np.linspace(0, t_max, 1001)
        self.g = coefficient(jump_correlator(ts), tlist=ts)
        self._t_max = t_max
        self._t_scale = t_max / 100

    def _initial_state(self):
        eye = _data.dense.identity(self.size)
        zero = _data.dense.zeros(self.size, self.size, fortran=True)
        self._tmp = zero.copy()
        self._Xp = zero.copy()
        self._Xm = zero.copy()
        if self._with_lamb:
            self._derr = merge_dense([zero, zero, zero, zero, zero, zero])
            initial = merge_dense([eye, eye, zero, zero, zero, zero])
        else:
            self._derr = merge_dense([zero, zero, zero, zero])
            initial = merge_dense([eye, eye, zero, zero])
        return initial

    def _rhs(self, s, state):
        Up, Um, Lp, Lm, *_ = split_dense(state, self.size)
        _derr = _data.dense.zeros(self.size**2, self._ncols, fortran=True)
        dUp, dUm, dLp, dLm, *dY = split_dense(_derr, self.size)

        g = self.g(s)

        # Inplace is needed for this to work

        #Propagator
        self.H.adjoint_rmatmul_data(self.t + s, Up, out=dUp, scale=1j)
        # FIXME: check the ordering for a time-dependent H. The physical
        # backward propagator P(s) = U(t-s, t) obeys dP/ds = 1j H(t-s) P
        # (left multiplication), with Xm = P^dag X P. Both forms agree
        # when H is constant.
        self.H.matmul_data(self.t - s, Um, out=dUm, scale=-1j)

        # diffusion correction terms
        self._tmp = _data.imul_dense(self._tmp, 0)
        self._tmp = self.X.adjoint_rmatmul_data(self.t + s, Up, out=self._tmp)
        _data.matmul_dag_dense(self._tmp, Up, out=dLp, scale=g.conjugate())
        self._tmp = _data.imul_dense(self._tmp, 0)
        self._tmp = self.X.adjoint_rmatmul_data(self.t - s, Um, out=self._tmp)
        _data.matmul_dag_dense(self._tmp, Um, out=dLm, scale=g)

        # second order terms for lamb shift
        if self._with_lamb:
            dYp, dYm = dY
            phase = g / g.conjugate()
            _data.matmul_dense(dLp, Lp, out=dYp, scale=(2 * phase))
            _data.matmul_dense(dLm, Lm, out=dYm, scale=(-2 / phase))

        return _derr

    def L(self, t):
        if t != self.t:
            self.compute(t)
        return Qobj(self._L, dims=self.H._dims)

    def lamb_shift(self, t):
        if not self._with_lamb:
            raise RuntimeError("ULOP was created with lamb_shift=False.")
        if t != self.t:
            self.compute(t)
        return Qobj(self._lamb, dims=self.H._dims)

    def compute(self, t):
        prev = self._initial_state()
        self.t = t
        self._integrator.set_state(0, prev)
        t_scale = self._t_scale
        tol = self._conv_tol

        diff = np.inf
        s = 0
        # TODO: tol should scale with state size and use absolute scale
        while diff > tol and s < self._t_max:
            s += t_scale
            _, state = self._integrator.integrate(s)
            diff = np.linalg.norm(
                state.to_array()[:, 2:] - prev.to_array()[:, 2:], 2
            )
            prev = state

        if diff > tol:
            warnings.warn(
                "ULME operators did not converge within the support of "
                f"the jump correlator (t_max={self._t_max:.3g}).",
                RuntimeWarning,
            )

        _, _, Lp, Lm, *Ys = split_dense(state, self.size)
        self._L = Lp + Lm
        if self._with_lamb:
            Yp, Ym = Ys
            self._lamb = (
                (Lp.adjoint() + Lm.adjoint()) @ (-Lp + Lm) + (Yp + Ym)
            ) * -0.5j
