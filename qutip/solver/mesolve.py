"""
This module provides solvers for the Lindblad master equation and von Neumann
equation.
"""

# Required for Sphinx to follow autodoc_type_aliases
from __future__ import annotations

__all__ = ['mesolve', 'MESolver', 'MESolverMatrixForm']

import warnings
from numpy.typing import ArrayLike
from typing import Any
from time import time
from .. import Qobj, QobjEvo, liouvillian, lindblad_dissipator, ket2dm
from ..typing import EopsLike, QobjEvoLike
from ..core import data as _data
from .solver_base import Solver, _solver_deprecation, _kwargs_migration
from .sesolve import sesolve, SESolver
from ._feedback import _QobjFeedback, _DataFeedback
from . import Result


def mesolve(
    H: QobjEvoLike,
    rho0: Qobj,
    tlist: ArrayLike,
    c_ops: Qobj | QobjEvo | list[QobjEvoLike] = None,
    _e_ops=None,
    _args=None,
    _options=None,
    *,
    e_ops: EopsLike | list[EopsLike] | dict[Any, EopsLike] = None,
    args: dict[str, Any] = None,
    options: dict[str, Any] = None,
    **kwargs
) -> Result:
    """
    Master equation evolution of a density matrix for a given Hamiltonian and
    set of collapse operators, or a Liouvillian.

    Evolve the state vector or density matrix (``rho0``) using a given
    Hamiltonian or Liouvillian (``H``) and an optional set of collapse
    operators (``c_ops``), by integrating the set of ordinary differential
    equations that define the system. In the absence of collapse operators
    the system is
    evolved according to the unitary evolution of the Hamiltonian.

    The output is either the state vector at arbitrary points in time
    (``tlist``), or the expectation values of the supplied operators
    (``e_ops``). If e_ops is a callback function, it is invoked for each
    time in ``tlist`` with time and the state as arguments, and the function
    does not use any return values.

    If either ``H`` or the Qobj elements in ``c_ops`` are superoperators, they
    will be treated as direct contributions to the total system Liouvillian.
    This allows the solution of master equations that are not in standard
    Lindblad form.

    **Time-dependent operators**

    For time-dependent problems, ``H`` and ``c_ops`` can be a :obj:`.QobjEvo`
    or object that can be interpreted as :obj:`.QobjEvo` such as a list of
    (Qobj, Coefficient) pairs or a function.

    **Additional options**

    Additional options to mesolve can be set via the ``options`` argument. Many
    ODE integration options can be set this way, and the ``store_states`` and
    ``store_final_state`` options can be used to store states even though
    expectation values are requested via the ``e_ops`` argument.

    Notes
    -----
    When no collapse operator are given and the `H` is not a superoperator,
    it will defer to :func:`sesolve`.

    Parameters
    ----------

    H : :obj:`.Qobj`, :obj:`.QobjEvo`, :obj:`.QobjEvo` compatible format.
        Possibly time-dependent system Liouvillian or Hamiltonian as a Qobj or
        QobjEvo. List of [:obj:`.Qobj`, :obj:`.Coefficient`] or callable that
        can be made into :obj:`.QobjEvo` are also accepted.

    rho0 : :obj:`.Qobj`
        initial density matrix or state vector (ket).

    tlist : *list* / *array*
        list of times for :math:`t`.

    c_ops : list of (:obj:`.QobjEvo`, :obj:`.QobjEvo` compatible format)
        Single collapse operator, or list of collapse operators, or a list
        of Liouvillian superoperators. None is equivalent to an empty list.

    e_ops : :obj:`.Qobj`, callable, list or dict, optional
        Single operator, or list or dict of operators, for which to evaluate
        expectation values. Operator can be Qobj, QobjEvo or callables with the
        signature `f(t: float, state: Qobj) -> Any`.

    args : dict, optional
        dictionary of parameters for time-dependent Hamiltonians and
        collapse operators.

    options : dict, optional
        Dictionary of options for the solver.

        - | store_final_state : bool
          | Whether or not to store the final state of the evolution in the
            result class.
        - | store_states : bool, None
          | Whether or not to store the state vectors or density matrices.
            On `None` the states will be saved if no expectation operators are
            given.
        - | normalize_output : bool
          | Normalize output state to hide ODE numerical errors. Only normalize
            the state if the initial state is already normalized.
        - | progress_bar : str {'text', 'enhanced', 'tqdm', ''}
          | How to present the solver progress.
            'tqdm' uses the python module of the same name and raise an error
            if not installed. Empty string or False will disable the bar.
        - | progress_kwargs : dict
          | kwargs to pass to the progress_bar. Qutip's bars use `chunk_size`.
        - | method : str ["adams", "bdf", "lsoda", "dop853", "vern9", etc.]
          | Which differential equation integration method to use.
        - | atol, rtol : float
          | Absolute and relative tolerance of the ODE integrator.
        - | nsteps : int
          | Maximum number of (internally defined) steps allowed in one
            ``tlist`` step.
        - | max_step : float
          | Maximum length of one internal step. When using pulses, it
            should be
            less than half the width of the thinnest pulse.
        - | matrix_form : bool
          | Use matrix-form Lindblad solver instead of superoperator form.
            The matrix-form solver can be faster for denser systems.
            Default: False.

        Other options could be supported depending on the integration method,
        see `Integrator <./classes.html#classes-ode>`_.

    Returns
    -------
    result: :obj:`.Result`

        An instance of the class :obj:`.Result`, which contains a *list of
        array* ``result.expect`` of expectation values for the times specified
        by ``tlist``, and/or a *list* ``result.states`` of state vectors or
        density matrices corresponding to the times in ``tlist`` [if ``e_ops``
        is an empty list of ``store_states=True`` in options].

    """
    e_ops = _kwargs_migration(_e_ops, e_ops, "e_ops")
    args = _kwargs_migration(_args, args, "args")
    options = _kwargs_migration(_options, options, "options")
    options = _solver_deprecation(kwargs, options)
    H = QobjEvo(H, args=args, tlist=tlist)

    c_ops = c_ops if c_ops is not None else []
    if not isinstance(c_ops, (list, tuple)):
        c_ops = [c_ops]
    c_ops = [QobjEvo(c_op, args=args, tlist=tlist) for c_op in c_ops]

    use_mesolve = len(c_ops) > 0 or (not rho0.isket) or H.issuper

    # Extract matrix_form option (not a solver option, it's a solver
    # selection option)
    if options is None:
        options = {}
    use_matrix_form = options.pop('matrix_form', False)

    if not use_mesolve:
        return sesolve(H, rho0, tlist, e_ops=e_ops, args=args,
                       options=options)

    if use_matrix_form:
        solver = MESolverMatrixForm(H, c_ops, options=options)
    else:
        solver = MESolver(H, c_ops, options=options)

    return solver.run(rho0, tlist, e_ops=e_ops)


class MESolver(SESolver):
    """
    Master equation evolution of a density matrix for a given Hamiltonian and
    set of collapse operators, or a Liouvillian.

    Evolve the density matrix (`rho0`) using a given
    Hamiltonian or Liouvillian (`H`) and an optional set of collapse operators
    (`c_ops`), by integrating the set of ordinary differential equations
    that define the system.

    If either `H` or the Qobj elements in `c_ops` are superoperators, they
    will be treated as direct contributions to the total system Liouvillian.
    This allows the solution of master equations that are not in standard
    Lindblad form.

    Parameters
    ----------
    H : :obj:`.Qobj`, :obj:`.QobjEvo`
        Possibly time-dependent system Liouvillian or Hamiltonian as a Qobj or
        QobjEvo. List of [:obj:`.Qobj`, :obj:`.Coefficient`] or callable that
        can be made into :obj:`.QobjEvo` are also accepted.

    c_ops : list of :obj:`.Qobj`, :obj:`.QobjEvo`
        Single collapse operator, or list of collapse operators, or a list
        of Liouvillian superoperators. None is equivalent to an empty list.

    options : dict, optional
        Options for the solver, see :obj:`MESolver.options` and
        `Integrator <./classes.html#classes-ode>`_ for a list of all options.

    Attributes
    ----------
    stats: dict
        Diverse diagnostic statistics of the evolution.
    """
    name = "mesolve"
    _avail_integrators: dict[str, object] = {}
    solver_options = {
        "progress_bar": "",
        "progress_kwargs": {"chunk_size": 10},
        "store_final_state": False,
        "store_states": None,
        "normalize_output": True,
        'method': 'adams',
    }

    def __init__(
        self,
        H: Qobj | QobjEvo,
        c_ops: Qobj | QobjEvo | list[Qobj | QobjEvo] = None,
        *,
        options: dict = None,
    ):
        _time_start = time()

        if not isinstance(H, (Qobj, QobjEvo)):
            raise TypeError("The Hamiltonian must be a Qobj or QobjEvo")
        c_ops = c_ops or []
        c_ops = [c_ops] if isinstance(c_ops, (Qobj, QobjEvo)) else c_ops
        for c_op in c_ops:
            if not isinstance(c_op, (Qobj, QobjEvo)):
                raise TypeError("All `c_ops` must be a Qobj or QobjEvo")

        self._num_collapse = len(c_ops)

        rhs = H if H.issuper else liouvillian(H)
        rhs += sum(c_op if c_op.issuper else lindblad_dissipator(c_op)
                   for c_op in c_ops)

        Solver.__init__(self, rhs, options=options)

    def _initialize_stats(self):
        stats = super()._initialize_stats()
        stats.update({
            "solver": "Master Equation Evolution",
            "num_collapse": self._num_collapse,
        })
        return stats

    @classmethod
    def StateFeedback(
        cls,
        default: Qobj | _data.Data = None,
        raw_data: bool = False,
        prop: bool = False
    ):
        """
        State of the evolution to be used in a time-dependent operator.

        When used as an args:

            ``QobjEvo([op, func], args={"state": MESolver.StateFeedback()})``

        The ``func`` will receive the density matrix as ``state`` during the
        evolution.

        Parameters
        ----------
        default : Qobj or qutip.core.data.Data, default : None
            Initial value to be used at setup of the system.

        prop : bool, default : False
            Set to True when computing propagators.
            The default with take the shape of the propagator instead of a
            state.

        raw_data : bool, default : False
            If True, the raw matrix will be passed instead of a Qobj.
            For density matrices, the matrices can be column stacked or square
            depending on the integration method.
        """
        if raw_data:
            return _DataFeedback(default, open=True, prop=prop)
        return _QobjFeedback(default, open=True, prop=prop)


class MESolverMatrixForm(Solver):
    """
    Master equation solver using matrix form of Lindblad equation.

    Instead of building an n^2 x n^2 Liouvillian superoperator and vectorizing
    the density matrix, this solver keeps rho as an n x n matrix and computes
    the RHS using matrix-matrix products:

    .. math::

        \\frac{d\\rho}{dt} = -i[H,\\rho] + \\sum_i \\left(c_i \\rho c_i^\\dagger 
        - \\frac{1}{2}\\{c_i^\\dagger c_i, \\rho\\}\\right)

    This approach can be more memory-efficient for large systems and avoids
    building the full superoperator. For dense operators, the time complexity
    also scales better with system size (n³ versus n⁴).
    However, it requires more matrix multiplications per RHS evaluation.

    Parameters
    ----------
    H : Qobj or QobjEvo
        Hamiltonian of the system (n x n operator).

    c_ops : list of Qobj or QobjEvo, optional
        List of collapse operators (each n x n operator).

    options : dict, optional
        Options for the solver. See :obj:`MESolverMatrixForm.options` for
        available options.

    Attributes
    ----------
    stats : dict
        Diagnostic statistics of the evolution.

    Examples
    --------
    >>> import qutip as qt
    >>> import numpy as np
    >>> N = 10
    >>> H = qt.num(N)
    >>> c_ops = [np.sqrt(0.1) * qt.destroy(N)]
    >>> rho0 = qt.fock_dm(N, 5)
    >>> tlist = np.linspace(0, 10, 100)
    >>> solver = qt.MESolverMatrixForm(H, c_ops, options={'method': 'vern7'})
    >>> result = solver.run(rho0, tlist)
    """
    name = "mesolve_matrix"
    _avail_integrators: dict[str, object] = {}
    solver_options = {
        "progress_bar": "",
        "progress_kwargs": {"chunk_size": 10},
        "store_final_state": False,
        "store_states": None,
        "normalize_output": True,
        'method': 'adams',
    }

    def __init__(
        self,
        H: Qobj | QobjEvo,
        c_ops: Qobj | QobjEvo | list[Qobj | QobjEvo] = None,
        *,
        options: dict = None,
    ):
        if not isinstance(H, (Qobj, QobjEvo)):
            raise TypeError("The Hamiltonian must be a Qobj or QobjEvo")

        c_ops = c_ops or []
        c_ops = [c_ops] if isinstance(c_ops, (Qobj, QobjEvo)) else c_ops
        for c_op in c_ops:
            if not isinstance(c_op, (Qobj, QobjEvo)):
                raise TypeError("All `c_ops` must be a Qobj or QobjEvo")

        self._num_collapse = len(c_ops)

        # Convert to QobjEvo if needed
        H = QobjEvo(H) if isinstance(H, Qobj) else H
        c_ops = [QobjEvo(c) if isinstance(c, Qobj) else c for c in c_ops]

        # Create matrix-form RHS (not a QobjEvo, but has compatible
        # interface)
        from qutip.core.cy.lindblad_matrix_form import LindbladMatrixForm
        self.rhs = LindbladMatrixForm(H, c_ops)

        # Initialize solver state (not calling super().__init__ since rhs
        # is not QobjEvo)
        self.options = options
        self._integrator = self._get_integrator()
        self._state_metadata = {}
        self.stats = self._initialize_stats()
        self.rhs._register_feedback({}, solver=self.name)

    def _initialize_stats(self):
        """ Return the initial values for the solver stats. """
        stats = super()._initialize_stats()
        stats.update({
            "solver": "Master Equation Evolution (Matrix Form)",
            "num_collapse": self._num_collapse,
        })
        return stats

    def _prepare_state(self, state):
        """
        Prepare state for integration.

        Unlike the superoperator form, we do NOT vectorize the state.
        The density matrix is kept as an n x n matrix throughout integration.
        """
        if self.rhs.issuper:
            raise TypeError(
                "MESolverMatrixForm cannot handle superoperator RHS. "
                "Use standard MESolver instead."
            )

        if state.isket:
            state = ket2dm(state)

        if not state.dtype.sparcity() == "dense":
            dtype = _data._parse_default_dtype(None, "dense")
            state = state.to(dtype)

        if self.rhs._dims[0] != state._dims[0]:
            raise TypeError(
                f"incompatible dimensions {self.rhs._dims} and {state.dims}"
            )

        self._state_metadata = {
            'dims': state._dims,
            'isherm': state._isherm,
        }

        # Determine if normalization should be applied
        if state.isket:
            norm = state.norm()
        elif state._dims.issquare:
            norm = state.tr()
        else:
            norm = -1

        self._normalize_output = (
            self._options.get("normalize_output", False)
            and abs(norm - 1) <= 1e-12
            and state.isoper
        )

        return state.data

    def _restore_state(self, data, *, copy=True):
        """
        Restore Qobj from Data - no unstacking needed.

        Since we keep the state as n x n throughout, just wrap in Qobj.
        """
        state = Qobj(data, **self._state_metadata, copy=copy)

        if self._normalize_output:
            state = state * (1 / state.tr())

        return state

    @classmethod
    def StateFeedback(
        cls,
        default: Qobj | _data.Data = None,
        raw_data: bool = False,
        prop: bool = False
    ):
        """
        State of the evolution to be used in a time-dependent operator.

        When used as an args:

            ``QobjEvo([op, func], args={"state":
            MESolverMatrixForm.StateFeedback()})``

        The ``func`` will receive the density matrix as ``state`` during
        the evolution.

        Parameters
        ----------
        default : Qobj or qutip.core.data.Data, default : None
            Initial value to be used at setup of the system.

        prop : bool, default : False
            Set to True when computing propagators.
            The default will take the shape of the propagator instead of a
            state.

        raw_data : bool, default : False
            If True, the raw matrix will be passed instead of a Qobj.
            For the matrix form solver, this will be an n x n dense matrix.
        """
        if raw_data:
            return _DataFeedback(default, open=True, prop=prop)
        return _QobjFeedback(default, open=True, prop=prop)
