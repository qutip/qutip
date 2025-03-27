"""
This module provides solvers for the unitary Schrodinger equation.
"""

# Required for Sphinx to follow autodoc_type_aliases
from __future__ import annotations

__all__ = ['sesolve', 'SESolver']

from numpy.typing import ArrayLike
from time import time
from typing import Any, Callable
from .. import Qobj, QobjEvo
from ..core import data as _data
from ..typing import QobjEvoLike, EopsLike
from .solver_base import Solver, _solver_deprecation, _kwargs_migration
from ._feedback import _QobjFeedback, _DataFeedback
from . import Result


def sesolve(
    H: QobjEvoLike,
    psi0: Qobj,
    tlist: ArrayLike,
    _e_ops = None,
    _args = None,
    _options = None,
    *,
    e_ops: EopsLike | list[EopsLike] | dict[Any, EopsLike] = None,
    args: dict[str, Any] = None,
    options: dict[str, Any] = None,
    **kwargs
) -> Result:
    """
    Schrodinger equation evolution of a state vector or unitary matrix
    for a given Hamiltonian.

    Evolve the state vector (``psi0``) using a given
    Hamiltonian (``H``), by integrating the set of ordinary differential
    equations that define the system. Alternatively evolve a unitary matrix in
    solving the Schrodinger operator equation.

    The output is either the state vector or unitary matrix at arbitrary points
    in time (``tlist``), or the expectation values of the supplied operators
    (``e_ops``). If e_ops is a callback function, it is invoked for each
    time in `tlist` with time and the state as arguments, and the function
    does not use any return values. e_ops cannot be used in conjunction
    with solving the Schrodinger operator equation

    **Time-dependent operators**

    For time-dependent problems, ``H`` and ``c_ops`` can be a :obj:`.QobjEvo`
    or object that can be interpreted as :obj:`.QobjEvo` such as a list of
    (Qobj, Coefficient) pairs or a function.

    Parameters
    ----------
    H : :obj:`.Qobj`, :obj:`.QobjEvo`, :obj:`.QobjEvo` compatible format.
        System Hamiltonian as a Qobj or QobjEvo for time-dependent
        Hamiltonians. List of [:obj:`.Qobj`, :obj:`.Coefficient`] or callable
        that can be made into :obj:`.QobjEvo` are also accepted.

    psi0 : :obj:`.Qobj`
        initial state vector (ket)
        or initial unitary operator `psi0 = U`

    tlist : *list* / *array*
        list of times for :math:`t`.

    e_ops : :obj:`.Qobj`, callable, list or dict, optional
        Single operator, or list or dict of operators, for which to evaluate
        expectation values. Operator can be Qobj, QobjEvo or callables with the
        signature `f(t: float, state: Qobj) -> Any`.

    args : dict, optional
        dictionary of parameters for time-dependent Hamiltonians

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
          | Maximum number of (internally defined) steps allowed in one ``tlist``
            step.
        - | max_step : float
          | Maximum lenght of one internal step. When using pulses, it should be
            less than half the width of the thinnest pulse.

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
    solver = SESolver(H, options=options)
    return solver.run(psi0, tlist, e_ops=e_ops)


class SESolver(Solver):
    """
    Schrodinger equation evolution of a state vector or unitary matrix
    for a given Hamiltonian.

    Parameters
    ----------
    H : :obj:`.Qobj`, :obj:`.QobjEvo`
        System Hamiltonian as a Qobj or QobjEvo for time-dependent
        Hamiltonians. List of [:obj:`.Qobj`, :obj:`.Coefficient`] or callable
        that can be made into :obj:`.QobjEvo` are also accepted.

    options : dict, optional
        Options for the solver, see :obj:`SESolver.options` and
        `Integrator <./classes.html#classes-ode>`_ for a list of all options.

    attributes
    ----------
    stats: dict
        Diverse diagnostic statistics of the evolution.
    """
    name = "sesolve"
    _avail_integrators = {}
    solver_options = {
        "progress_bar": "",
        "progress_kwargs": {"chunk_size":10},
        "store_final_state": False,
        "store_states": None,
        "normalize_output": True,
        'method': 'adams',
    }

    def __init__(self, H: Qobj | QobjEvo, *, options: dict[str, Any] = None):
        _time_start = time()

        if not isinstance(H, (Qobj, QobjEvo)):
            raise TypeError("The Hamiltonian must be a Qobj or QobjEvo")

        rhs = -1j * H
        if not rhs.isoper:
            raise ValueError("The hamiltonian must be an operator")
        super().__init__(rhs, options=options)

    def _initialize_stats(self):
        stats = super()._initialize_stats()
        stats.update({
            "solver": "Schrodinger Evolution",
        })
        return stats

    @property
    def options(self) -> dict:
        """
        Solver's options:

        store_final_state: bool, default: False
            Whether or not to store the final state of the evolution in the
            result class.

        store_states: bool, default: None
            Whether or not to store the state vectors or density matrices.
            On `None` the states will be saved if no expectation operators are
            given.

        normalize_output: bool, default: True
            Normalize output state to hide ODE numerical errors.

        progress_bar: str {"text", "enhanced", "tqdm", ""}, default: ""
            How to present the solver progress.
            'tqdm' uses the python module of the same name and raise an error
            if not installed. Empty string or False will disable the bar.

        progress_kwargs: dict, default: {"chunk_size": 10}
            Arguments to pass to the progress_bar. Qutip's bars use
            ``chunk_size``.

        method: str, default: "adams"
            Which ordinary differential equation integration method to use.
        """
        return self._options

    @options.setter
    def options(self, new_options: dict[str, Any]):
        Solver.options.fset(self, new_options)

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

            ``QobjEvo([op, func], args={"state": SESolver.StateFeedback()})``

        The ``func`` will receive the ket as ``state`` during the evolution.

        Parameters
        ----------
        default : Qobj or qutip.core.data.Data, default : None
            Initial value to be used at setup of the system.

        prop : bool, default : False
            Set to True when using sesolve for computing propagators.

        raw_data : bool, default : False
            If True, the raw matrix will be passed instead of a Qobj.
            For density matrices, the matrices can be column stacked or square
            depending on the integration method.
        """
        if raw_data:
            return _DataFeedback(default, open=False, prop=prop)
        return _QobjFeedback(default, open=False, prop=prop)
