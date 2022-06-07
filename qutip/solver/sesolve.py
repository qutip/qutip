"""
This module provides solvers for the unitary Schrodinger equation.
"""

__all__ = ['sesolve', 'SeSolver']

import numpy as np
from time import time
from .. import Qobj, QobjEvo
from .solver_base import Solver
from .options import known_solver


def sesolve(H, psi0, tlist, e_ops=None, args=None, options=None):
    """
    Schrodinger equation evolution of a state vector or unitary matrix
    for a given Hamiltonian.

    Evolve the state vector (`psi0`) using a given
    Hamiltonian (`H`), by integrating the set of ordinary differential
    equations that define the system. Alternatively evolve a unitary matrix in
    solving the Schrodinger operator equation.

    The output is either the state vector or unitary matrix at arbitrary points
    in time (`tlist`), or the expectation values of the supplied operators
    (`e_ops`). If e_ops is a callback function, it is invoked for each
    time in `tlist` with time and the state as arguments, and the function
    does not use any return values. e_ops cannot be used in conjunction
    with solving the Schrodinger operator equation

    **Time-dependent operators**

    For time-dependent problems, `H` and `c_ops` can be a :class:`QobjEvo` or
    object that can be interpreted as :class:`QobjEvo` such as a list of
    (Qobj, Coefficient) pairs or a function.

    Parameters
    ----------
    H : :class:`Qobj`, :class:`QobjEvo`, :class:`QobjEvo` compatible format.
        System Hamiltonian as a Qobj or QobjEvo for time-dependent Hamiltonians.
        list of [:class:`Qobj`, :class:`Coefficient`] or callable that can be
        made into :class:`QobjEvo` are also accepted.

    psi0 : :class:`qutip.qobj`
        initial state vector (ket)
        or initial unitary operator `psi0 = U`

    tlist : *list* / *array*
        list of times for :math:`t`.

    e_ops : :class:`qutip.qobj`, callable, or list.
        Single operator or list of operators for which to evaluate
        expectation values or callable or list of callable.
        Callable signature must be, `f(t: float, state: Qobj)`.
        See :func:`expect` for more detail of operator expectation.

    args : None / *dictionary*
        dictionary of parameters for time-dependent Hamiltonians

    options : None / dict / :class:`Options`
        Options for the solver.

    Returns
    -------
    result: :class:`qutip.Result`

        An instance of the class :class:`qutip.Result`, which contains
        a *list of array* `result.expect` of expectation values for the times
        specified by `tlist`, and/or a *list* `result.states` of state vectors
        or density matrices corresponding to the times in `tlist` [if `e_ops`
        is an empty list of `store_states=True` in options].
    """
    H = QobjEvo(H, args=args, tlist=tlist)
    solver = SeSolver(H, options=options)
    return solver.run(psi0, tlist, e_ops=e_ops)


class SeSolver(Solver):
    """
    Schrodinger equation evolution of a state vector or unitary matrix
    for a given Hamiltonian.

    Parameters
    ----------
    H : :class:`Qobj`, :class:`QobjEvo`
        System Hamiltonian as a Qobj or QobjEvo for time-dependent Hamiltonians.
        list of [:class:`Qobj`, :class:`Coefficient`] or callable that can be
        made into :class:`QobjEvo` are also accepted.

    options : None / dict / :class:`Options`
        Options for the solver

    attributes
    ----------
    stats: dict
        Diverse diagnostic statistics of the evolution.
    """
    name = "sesolve"
    _avail_integrators = {}
    solver_options = {
        "progress_bar": "text",
        "progress_kwargs": {"chunk_size":10},
        "store_final_state": False,
        "store_states": None,
        "normalize_output": True,
        'method': 'adams',
    }

    def __init__(self, H, *, options=None):
        _time_start = time()

        if not isinstance(H, (Qobj, QobjEvo)):
            raise TypeError("The Hamiltonian must be a Qobj or QobjEvo")

        rhs = -1j * H
        if not rhs.isoper:
            raise ValueError("The hamiltonian must be an operator")
        super().__init__(rhs, options=options)

        self.stats['solver'] = "Schrodinger Evolution"
        self.stats["preparation time"] = time() - _time_start
        self.stats["run time"] = 0

known_solver['sesolve'] = SeSolver
known_solver['Sesolver'] = SeSolver
known_solver['SeSolver'] = SeSolver
known_solver['Schrodinger Evolution'] = SeSolver
