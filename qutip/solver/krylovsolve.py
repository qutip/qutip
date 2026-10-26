# Required for Sphinx to follow autodoc_type_aliases
from __future__ import annotations

__all__ = ['krylovsolve']

from .. import QobjEvo, Qobj
from .mesolve import mesolve
from .result import Result
from numpy.typing import ArrayLike
from ..typing import EopsLike, QobjEvoLike
from typing import Any


def krylovsolve(
    H: Qobj,
    rho0: Qobj,
    tlist: ArrayLike,
    krylov_dim: int,
    algorithm: str = "auto",
    c_ops: Qobj | QobjEvo | list[QobjEvoLike] = None,
    *,
    e_ops: EopsLike | list[EopsLike] | dict[Any, EopsLike] = None,
    args: dict[str, Any] = None,
    options: dict[str, Any] = None,
) -> Result:
    """
    Master equation evolution of a (density) operator or pure state for time
    independent Hamiltonians using the Krylov method.

    Evolve the state vector or density matrix ("rho0") finding a reduced form
    of the time evolution operator of the Hamiltonian or Liouvillian ("H") by
    obtaining the projection of the time evolution operator on a set of small
    dimensional Krylov subspaces (``krylov_dim`` << dim(H)).

    The output is either the state vector or density matrix at arbitrary
    points in time (`tlist`), or the expectation values of the supplied
    operators (`e_ops`). If e_ops is a callback function, it is invoked for
    each time in `tlist` with time and the state as arguments, and the
    function does not use any return values.

    Parameters
    ----------
    H : :obj:`.Qobj`
        System Hamiltonian or Liouvillian. It must be time-independent.

    rho0 : :class:`.Qobj`
        Initial density matrix or state vector (ket).

    tlist : *list* / *array*
        list of times for :math:`t`.

    krylov_dim: int
        Dimension of Krylov subspaces used for the time evolution calculation.
        Note that for mixed states, the Krylov dimension can be larger than the
        dimension of the Hamiltonian since we are computing the Liouvillian.

    algorithm: str, default: "auto"
        The algorithm to use for Krylov space construction. If
        ``options["algorithm"]`` is set, it takes precedence over this
        argument. Options are:

        - "auto": use "lanczos_fro" for closed (Hermitian) systems and
          "arnoldi" for open (non-Hermitian) systems.
        - "lanczos": use the Lanczos algorithm (closed systems only).
        - "lanczos_fro": use the fully reorthogonalized Lanczos algorithm
          (closed systems only).
        - "arnoldi": use the Arnoldi iteration.

    c_ops : list of :obj:`.Qobj`, optional
        Single collapse operator, or list of collapse operators, or a list
        of Liouvillian superoperators. They must be time-independent. None is
        equivalent to an empty list.

    e_ops : :obj:`.Qobj`, callable, list or dict, optional
        Single operator, or list or dict of operators, for which to evaluate
        expectation values. Operators can be Qobj, QobjEvo or callables.
        Callable signature must be, `f(t: float, state: Qobj)`.
        See :func:`~qutip.core.expect.expect` for more detail of operator
        expectation.

    args : dict, optional
        Dictionary of parameters for Hamiltonian.

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
          | Absolute tolerance of the integrator. It bounds the estimated
            error of each Krylov step and thereby sets the maximal step
            length.
        - | nsteps : int
          | Maximum number of (internally defined) steps allowed in one
            ``tlist`` step.
        - | min_step, max_step : float
          | Minimum and maximum length of one internal step.
        - | always_compute_step: bool
          | If True, the step length is computed each time a new Krylov
            subspace is computed. Otherwise it is computed only once, for the
            first Krylov subspace.
        - | sub_system_tol: float
          | Tolerance to detect a happy breakdown. A happy breakdown happens
            when the initial state is in a subspace of the Hamiltonian or
            Liouvillian smaller than ``krylov_dim``.
        - | matrix_form : bool
          | Use matrix-form Lindblad solver instead of superoperator form.
            Only used for density matrix evolution. Default: False.

    Returns
    -------
    result: :class:`.Result`

        An instance of the class :class:`.Result`, which contains
        a *list of array* ``result.expect`` of expectation values for the times
        specified by ``tlist``, and/or a *list* ``result.states`` of state
        vectors or density matrices corresponding to the times in ``tlist`` [if
        ``e_ops`` is an empty list or ``store_states=True`` in options].
    """
    options = dict(options or {})
    options["method"] = "krylov"
    options["krylov_dim"] = krylov_dim
    # option setting overwrites argument
    options.setdefault("algorithm", algorithm)

    return mesolve(
        H, rho0, tlist, c_ops=c_ops, e_ops=e_ops, args=args, options=options
    )
