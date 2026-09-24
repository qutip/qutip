"""
Universal Lindblad master equation (ULME).

Implementation of the master equation of F. Nathan and M. S. Rudner,
Phys. Rev. B 102, 115109 (2020), https://arxiv.org/abs/2004.01469
"""
from __future__ import annotations

import functools
import itertools
import warnings
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import integrate

from .. import Qobj, QobjEvo
from ..core import data as _data
from ..core.coefficient import coefficient
from ..core.environment import BosonicEnvironment
from ..typing import EopsLike, QobjEvoLike
from .integrator import IntegratorTsit5
from .mesolve import MESolver
from .result import Result
from .solver_base import Solver


__all__ = ["ulmesolve", "ULMESolver", "UL_transform"]


# Options that are specific to the ULME.
_ULME_DEFAULT_OPTIONS = {
    "ULME_creation": None,
    "use_lamb_shift": True,
    "tol": 1e-6,
}

# Half-width of the frequency window used for the principal value integrals
# of the Lamb shift in the "eigen" method (same units as the bath frequencies).
_PV_LIMIT = 50


def ulmesolve(
    H: QobjEvoLike,
    rho0: Qobj,
    tlist: ArrayLike,
    a_ops: (
        tuple[QobjEvoLike, BosonicEnvironment]
        | list[tuple[QobjEvoLike, BosonicEnvironment]]
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

        - | ULME_creation : str {"eigen", "prop"}
          | Method used to construct the Lindblad jump operators.
            "eigen": eigen-decomposition of the Hamiltonian, constant system
            only. "prop": integration in the interaction picture, general.
        - | use_lamb_shift : bool, True
          | Whether to calculate and include the Lamb shift correction in the
            effective Hamiltonian.
        - | tol : float, 1e-6
          | Only for "prop". Threshold below which the bath's
            ``jump_correlator`` is considered to be zero.

        All options are listed in ``ULMESolver.options``'s docstring.

    Returns
    -------
    result : :obj:`.Result`
        An instance of the class :obj:`.Result`, containing expectation
        values and/or states.
    """
    H = QobjEvo(H, args=args, tlist=tlist)
    a_ops = _parse_a_ops(a_ops, args=args, tlist=tlist)
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
        Possibly time-dependent system Hamiltonian as a Qobj or QobjEvo.
        List of [:obj:`.Qobj`, :obj:`.Coefficient`] or callable that
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
        - "tol": Cutoff of the jump correlator (``"prop"`` only).

        See :obj:`ULMESolver.options` and
        `Integrator <./classes.html#classes-ode>`_ for a list of all options.

    Attributes
    ----------
    stats: dict
        Diverse diagnostic statistics of the evolution.
    """
    name = "Universal Lindblad master equation"
    solver_options = {**MESolver.solver_options, **_ULME_DEFAULT_OPTIONS}

    def __init__(self, H, a_ops, *, options=None):
        H = QobjEvo(H)
        if H.issuper:
            raise TypeError("ULME cannot be used with superoperator H.")
        a_ops = _parse_a_ops(a_ops)
        for i, (op, _) in enumerate(a_ops):
            if op.dims != H.dims:
                raise ValueError(
                    f"Dimension mismatch in a_ops[{i}]: "
                    f"Hamiltonian dims {H.dims} "
                    f"do not match coupling operator dims {op.dims}."
                )

        self.H = H
        self.a_ops = a_ops
        self._num_collapse = len(a_ops)
        self.options = options

        self.c_ops = []
        self.lamb_shifts = []
        H_eff = H
        for op, env in a_ops:
            L, lamb_shift = _make_operators(H, op, env, self.options)
            self.c_ops.append(L)
            self.lamb_shifts.append(lamb_shift)
            if lamb_shift is not None:
                H_eff = H_eff + lamb_shift

        super().__init__(H_eff, self.c_ops, options=self.options)

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
              spectral filter.

            - "prop": Constructs dissipators by integrating the coupling
              operator, in the interaction picture, against the bath's
              ``jump_correlator``. Works for time-dependent systems and is
              generally faster than "eigen" when the Lamb shift is included.

            Per default, "eigen" will be used for constant system, and "prop"
            otherwise.

        use_lamb_shift: bool, default: True
            Whether to calculate and include the Lamb shift correction in the
            effective Hamiltonian.

        tol: float, default: 1e-6
            Only used by "prop". The ``jump_correlator`` is considered to be
            zero where its magnitude is below ``tol``. This sets the time
            window over which the operators are integrated.
        """
        return self._options

    @options.setter
    def options(self, new_options):
        Solver.options.fset(self, new_options)


def UL_transform(
    H: Qobj | QobjEvo,
    a_ops: (
        tuple[Qobj | QobjEvo, BosonicEnvironment]
        | list[tuple[Qobj | QobjEvo, BosonicEnvironment]]
    ),
    options: dict = None,
) -> tuple[Qobj | QobjEvo, list[Qobj | QobjEvo]]:
    """
    Transform the system according to the universal Lindblad equation.
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

        - "ULME_creation": {"eigen", "prop"}
          Method used to compute the operators, either eigen decomposition or
          integration of the coupling operator with the jump_correlator.
        - "use_lamb_shift": True,
          Compute the lamb shift and add it to the Hamiltonian.
        - "tol": 1e-6
          Only for "prop". Threshold below which the jump_correlator is
          considered to be zero.

    Returns
    -------
    H, c_ops:
        The corrected Hamiltonian and collapse operators.
        These are formated so they can be used directly in mesolve or mcsolve.
    """
    options = {**_ULME_DEFAULT_OPTIONS, **(options or {})}
    H_evo = QobjEvo(H, copy=False)
    H_eff = H
    c_ops = []
    for op, env in _parse_a_ops(a_ops):
        L, lamb_shift = _make_operators(H_evo, op, env, options)
        c_ops.append(L)
        if lamb_shift is not None:
            H_eff = H_eff + lamb_shift
    return H_eff, c_ops


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
        isinstance(a_ops, tuple)
        and len(a_ops) == 2
        and isinstance(a_ops[1], BosonicEnvironment)
    ):
        a_ops = [a_ops]
    elif isinstance(a_ops, tuple):
        a_ops = list(a_ops)
    if not isinstance(a_ops, list):
        raise TypeError("a_ops must be a list of (operator, environment) tuples.")

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


def _make_operators(H, X, env, options):
    """
    Build the jump operator and Lamb shift for one system-bath coupling.

    Parameters
    ----------
    H, X : QobjEvo
        System Hamiltonian and coupling operator.
    env : BosonicEnvironment
    options : dict
        See ``ULMESolver.options``.

    Returns
    -------
    L : Qobj or QobjEvo
        Jump operator. A Qobj when the system is constant.
    lamb_shift : Qobj, QobjEvo or None
        Lamb shift Hamiltonian, None when ``use_lamb_shift`` is False.
    """
    method = options.get("ULME_creation")
    if method is None:
        method = "eigen" if (H.isconstant and X.isconstant) else "prop"
    use_lamb_shift = options.get("use_lamb_shift", True)

    if method == "eigen":
        return _operators_eigen(H, X, env, use_lamb_shift)
    if method == "prop":
        return _operators_prop(H, X, env, use_lamb_shift, options)
    raise ValueError(
        f"Unknown ULME_creation method {method!r}, "
        "expected 'eigen' or 'prop'."
    )


# ---------------------------------------------------------------------------
# "eigen" method: constant system, eigen-decomposition of H.
# ---------------------------------------------------------------------------

def _operators_eigen(
    H: QobjEvo,
    X: QobjEvo,
    env: BosonicEnvironment,
    use_lamb_shift: bool
    ):
    if not (H.isconstant and X.isconstant):
        raise TypeError(
            "ULME_creation='eigen' requires a time-independent Hamiltonian "
            "and coupling operator."
        )
    dims = X.dims
    vals, vecs = H(0).eigenstates(output_type="oper")
    X_eig = vecs.dag() @ X(0) @ vecs

    response = _data.Dense(env._g_w(-np.subtract.outer(vals, vals)))
    L_eig = _data.multiply(X_eig.data, response) * (2 * np.pi)
    L = vecs @ Qobj(L_eig, dims=dims) @ vecs.dag()
    if not use_lamb_shift:
        return L, None

    lamb_eig = _lamb_shift_eigenbasis(vals, X_eig.full(), env)
    return L, vecs @ Qobj(lamb_eig, dims=dims) @ vecs.dag()


def _lamb_shift_eigenbasis(vals, X_eig, env):
    """
    Lamb shift in the eigenbasis of H:
    ``sum_j f(E_j - E_i, E_k - E_j) X_ij X_jk`` where ``f`` is a principal
    value integral over the product of two spectral filters.
    """
    @functools.lru_cache(maxsize=None)
    def pv_integral(e1, e2):
        value = integrate.quad(
            lambda w: env._g_w(w - e1) * env._g_w(w + e2),
            -_PV_LIMIT, _PV_LIMIT, weight="cauchy", wvar=0,
        )[0]
        return -2 * np.pi * value

    N = len(vals)
    f = np.zeros((N, N, N), dtype=float)
    # TODO: this loop is O(N^3) calls to quad. The "prop" method avoids it.
    for i, j, k in itertools.product(range(N), repeat=3):
        f[i, j, k] = pv_integral(vals[j] - vals[i], vals[k] - vals[j])
    return np.einsum("ijk,ij,jk->ik", f, X_eig, X_eig)


# ---------------------------------------------------------------------------
# "prop" method: integration in the interaction picture.
# ---------------------------------------------------------------------------

def _operators_prop(H, X, env, use_lamb_shift, options):
    op = ULOP(H, X, env, options, lamb_shift=use_lamb_shift)
    if H.isconstant and X.isconstant:
        return op.L(0), (op.lamb_shift(0) if use_lamb_shift else None)
    return QobjEvo(op.L), (QobjEvo(op.lamb_shift) if use_lamb_shift else None)


class ULOP:
    """
    Jump operator and Lamb shift of the ULME at time ``t``, computed by
    integrating, over the relative time ``s``, the coupling operator in the
    interaction picture (with reference time ``t``) against the bath's
    ``jump_correlator`` ``g``:

    .. math::

        L(t) = \\int_0^\\infty ds \\left[
            g^*(s) X_I(t + s) + g(s) X_I(t - s)
        \\right]

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
    options : dict, optional
        ``tol`` is used to find the time window over which ``g`` is not zero.
    lamb_shift : bool
        Whether to also compute the Lamb shift.
    """

    # Convergence threshold on the change of ``Lp`` between checks.
    # TODO: make it an option, and consider a relative criterion.
    _conv_tol = 1e-4

    def __init__(self, H, X, env, options=None, *, lamb_shift=True):
        options = options or {}
        self.H = H
        self.X = X
        self.size = H.shape[0]
        self.tol = options.get("tol", _ULME_DEFAULT_OPTIONS["tol"])
        self._with_lamb = lamb_shift
        self._ncols = 6 if lamb_shift else 4

        self.t = None
        self._L = None
        self._lamb = None
        self._integrator = IntegratorTsit5(self._rhs, {})
        self._prepare_g(env.jump_correlator)

    def _prepare_g(self, jump_correlator):
        """
        Sample the jump correlator on a regular grid and build a spline.

        The jump_correlator is expected to decrease exponentially:
           jc(t) = f(t) * exp(-t*alpha)
        but we don't know the units so we have to estimate the cutoff.
        """
        ts = np.logspace(-8, 3, 201)
        above = np.flatnonzero(np.abs(jump_correlator(ts)) > self.tol)
        t_max = ts[above[-1]] if above.size else 1000.

        ts = np.linspace(0, t_max, 1001)
        self._g = coefficient(jump_correlator(ts), tlist=ts)
        self._t_max = t_max
        self._t_scale = t_max / 100

    def _merge_states(self, matrices):
        """Stack square Data matrices as the columns of one dense matrix."""
        columns = [_data.column_stack(mat) for mat in matrices]
        merged = _data.dense.zeros(
            columns[0].shape[0], len(columns), fortran=True
        )
        array = merged.as_ndarray()
        for i, column in enumerate(columns):
            array[:, i] = column.to_array()[:, 0]
        return merged

    def _split_states(self, state):
        """Inverse of ``_merge_states``."""
        out = []
        for column in state.as_ndarray().T:
            data = _data.dense.fast_from_numpy(column)
            out.append(_data.column_unstack_dense(data, self.size, inplace=True))
        return out

    def _initial_state(self):
        eye = _data.dense.identity(self.size)
        zero = _data.dense.zeros(self.size, self.size, fortran=True)
        return self._merge_states([eye, eye] + [zero] * (self._ncols - 2))

    def _rhs(self, s, state):
        Up, Um, Lp, Lm, *_ = self._split_states(state)
        Xp = Up.adjoint() @ self.X._call(self.t + s) @ Up
        Xm = Um.adjoint() @ self.X._call(self.t - s) @ Um
        g = self._g(s)
        derivatives = [
            -1j * self.H._call(self.t + s) @ Up,
            # FIXME: check the ordering for a time-dependent H. The physical
            # backward propagator P(s) = U(t-s, t) obeys dP/ds = 1j H(t-s) P
            # (left multiplication), with Xm = P^dag X P. Both forms agree
            # when H is constant.
            1j * Um @ self.H._call(self.t - s),
            Xp * g.conjugate(),
            Xm * g,
        ]
        if self._with_lamb:
            derivatives += [
                Xp @ Lp * (2 * g),
                Xm @ Lm * (-2 * g.conjugate()),
            ]
        return self._merge_states(derivatives)

    def _compute(self, t):
        self.t = t
        previous = self._initial_state()
        self._integrator.set_state(0, previous)

        step = self._t_scale / 10
        s = 0.
        diff = np.inf
        while diff > self._conv_tol:
            if s >= self._t_max:
                warnings.warn(
                    "ULME operators did not converge within the support of "
                    f"the jump correlator (t_max={self._t_max:.3g}).",
                    RuntimeWarning,
                )
                break
            s = min(s + step, self._t_max)
            _, state = self._integrator.integrate(s)
            diff = np.linalg.norm(
                state.to_array()[:, 2] - previous.to_array()[:, 2], 2
            )
            previous = state

        _, _, Lp, Lm, *second = self._split_states(state)
        self._L = Lp + Lm
        if self._with_lamb:
            Yp, Ym = second
            self._lamb = (
                (Lp.adjoint() + Lm.adjoint()) @ (-Lp + Lm) + (Yp + Ym)
            ) * -0.5j

    def L(self, t):
        """Jump operator at time ``t``."""
        if t != self.t:
            self._compute(t)
        return Qobj(self._L, dims=self.H.dims)

    def lamb_shift(self, t):
        """Lamb shift Hamiltonian at time ``t``."""
        if not self._with_lamb:
            raise RuntimeError("ULOP was created with lamb_shift=False.")
        if t != self.t:
            self._compute(t)
        return Qobj(self._lamb, dims=self.H.dims)
