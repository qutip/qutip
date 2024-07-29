# Required for Sphinx to follow autodoc_type_aliases
from __future__ import annotations

__all__ = ['krylovsolve']

from .. import QobjEvo, Qobj
from .sesolve import SESolver
from .result import Result
from .solver_base import _kwargs_migration
from numpy.typing import ArrayLike
from typing import Any, Callable


def krylovsolve(
    H: Qobj,
    psi0: Qobj,
    tlist: ArrayLike,
    krylov_dim: int,
    _e_ops = None,
    _args = None,
    _options = None,
    *,
    e_ops: dict[Any, Qobj | QobjEvo | Callable[[float, Qobj], Any]] = None,
    args: dict[str, Any] = None,
    options: dict[str, Any] = None,
) -> Result:
    """
    Schrodinger equation evolution of a state vector for time independent
    Hamiltonians using Krylov method.

    Evolve the state vector ("psi0") finding an approximation for the time
    evolution operator of Hamiltonian ("H") by obtaining the projection of
    the time evolution operator on a set of small dimensional Krylov
    subspaces (m << dim(H)).

    The output is either the state vector or unitary matrix at arbitrary points
    in time (`tlist`), or the expectation values of the supplied operators
    (`e_ops`). If e_ops is a callback function, it is invoked for each
    time in `tlist` with time and the state as arguments, and the function
    does not use any return values. e_ops cannot be used in conjunction
    with solving the Schrodinger operator equation

    Parameters
    ----------
    H : :obj:`.Qobj`, :obj:`.QobjEvo`, :obj:`.QobjEvo` compatible format.
        System Hamiltonian as a Qobj or QobjEvo for time-dependent
        Hamiltonians. List of [:obj:`.Qobj`, :obj:`.Coefficient`] or callable
        that can be made into :obj:`.QobjEvo` are also accepted.

    psi0 : :class:`.Qobj`
        Initial state vector (ket)

    tlist : *list* / *array*
        list of times for :math:`t`.

    krylov_dim: int
        Dimension of Krylov approximation subspaces used for the time
        evolution approximation.

    e_ops : :class:`.Qobj`, callable, or list, optional
        Single operator or list of operators for which to evaluate
        expectation values or callable or list of callable.
        Callable signature must be, `f(t: float, state: Qobj)`.
        See :func:`~qutip.core.expect.expect` for more detail of operator
        expectation.

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
        - | atol: float
          | Absolute tolerance of the ODE integrator.
        - | nsteps : int
          | Maximum number of (internally defined) steps allowed in one
            ``tlist`` step.
        - | min_step, max_step : float
          | Miniumum and maximum lenght of one internal step.
        - | always_compute_step: bool
          | If True, the step lenght is computed each time a new Krylov
            subspace is computed. Otherwise it is computed only once when
            creating the integrator.
        - | sub_system_tol: float
          | Tolerance to detect an happy breakdown. An happy breakdown happens
            when the initial ket is in a subspace of the Hamiltonian smaller
            than ``krylov_dim``.

    Returns
    -------
    result: :class:`.Result`

        An instance of the class :class:`.Result`, which contains
        a *list of array* ``result.expect`` of expectation values for the times
        specified by ``tlist``, and/or a *list* ``result.states`` of state
        vectors or density matrices corresponding to the times in ``tlist`` [if
        ``e_ops`` is an empty list of ``store_states=True`` in options].
    """
    e_ops = _kwargs_migration(_e_ops, e_ops, "e_ops")
    args = _kwargs_migration(_args, args, "args")
    options = _kwargs_migration(_options, options, "options")
    H = QobjEvo(H, args=args, tlist=tlist)
    options = options or {}
    options["method"] = "krylov"
    options["krylov_dim"] = krylov_dim
    solver = SESolver(H, options=options)
    return solver.run(psi0, tlist, e_ops=e_ops)
