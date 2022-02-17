"""
This module provides solvers for the Lindblad master equation and von Neumann
equation.
"""

__all__ = ['mesolve', 'MeSolver', 'MeOptions']

import numpy as np
from time import time
from .. import (Qobj, QobjEvo, isket, liouvillian, ket2dm, lindblad_dissipator)
from ..core import stack_columns, unstack_columns
from ..core.data import to
from .solver_base import Solver
from .options import SolverOptions
from .sesolve import sesolve, SeOptions


class MeOptions(SolverOptions):
    """
    Class of options for :func:`mesolve` and :class:`MeSolver`. Options can be
    specified either as arguments to the constructor::

        opts = MeOptions(progress_bar='enhanced', ...)

    or by changing the class attributes after creation::

        opts = MeOptions()
        opts['progress_bar'] = 'enhanced'

    Returns options class to be used as options in evolution solvers.

    The default can be changed by changing the key of the class::

        MeOptions['progress_bar'] = 'enhanced'

    Options
    -------
    store_final_state : bool {False, True}
        Whether or not to store the final state of the evolution in the
        result class.

    store_states : bool {False, True, None}
        Whether or not to store the state vectors or density matrices.
        On `None` the states will be saved if no expectation operators are
        given.

    normalize_output : str {"", "ket", "all"}
        normalize output state to hide ODE numerical errors.
        "all" will normalize both ket and dm.
        On "ket", only 'ket' output are normalized.
        Leave empty for no normalization.

    progress_bar : str {'text', 'enhanced', 'tqdm', ''}
        How to present the solver progress.
        True will result in 'text'.
        'tqdm' uses the python module of the same name and raise an error if
        not installed.
        Empty string or False will disable the bar.

    progress_kwargs : dict
        kwargs to pass to the progress_bar. Qutip's bars use `chunk_size`.

    operator_data_type: str {""}
        Data type of the operator to used during the ODE evolution, such as
        'CSR' or 'Dense'. Use an empty string to keep the input state type.

    state_data_type: str {""}
        Name of the data type of the state used during the ODE evolution.
        Use an empty string to keep the input state type. Some integrator can
        only work with specific data type and will ignore this options.
    """
    default = {
        "progress_bar": "text",
        "progress_kwargs": {"chunk_size":10},
        "store_final_state": False,
        "store_states": None,
        "normalize_output": "ket",
        "operator_data_type": "",
        "state_data_type": "",
        'method': 'adams',
    }


def mesolve(H, rho0, tlist, c_ops=None, e_ops=None, args=None, options=None):
    """
    Master equation evolution of a density matrix for a given Hamiltonian and
    set of collapse operators, or a Liouvillian.

    Evolve the state vector or density matrix (`rho0`) using a given
    Hamiltonian or Liouvillian (`H`) and an optional set of collapse operators
    (`c_ops`), by integrating the set of ordinary differential equations
    that define the system. In the absence of collapse operators the system is
    evolved according to the unitary evolution of the Hamiltonian.

    The output is either the state vector at arbitrary points in time
    (`tlist`), or the expectation values of the supplied operators
    (`e_ops`). If e_ops is a callback function, it is invoked for each
    time in `tlist` with time and the state as arguments, and the function
    does not use any return values.

    If either `H` or the Qobj elements in `c_ops` are superoperators, they
    will be treated as direct contributions to the total system Liouvillian.
    This allows the solution of master equations that are not in standard
    Lindblad form.

    **Time-dependent operators**

    For time-dependent problems, `H` and `c_ops` can be a :class:`QobjEvo` or
    object that can be interpreted as :class:`QobjEvo` such as a list of
    (Qobj, Coefficient) pairs or a function.

    **Additional options**

    Additional options to mesolve can be set via the `options` argument, which
    should be an instance of :class:`qutip.solver.SolverOptions`. Many ODE
    integration options can be set this way, and the `store_states` and
    `store_final_state` options can be used to store states even though
    expectation values are requested via the `e_ops` argument.

    .. note::

        When no collapse operator are given and the `H` is not a superoperator,
        it will defer to :func:`sesolve`.

    Parameters
    ----------

    H : :class:`Qobj`, :class:`QobjEvo`, :class:`QobjEvo` compatible format.
        Possibly time-dependent system Liouvillian or Hamiltonian as a Qobj or QobjEvo.
        list of [:class:`Qobj`, :class:`Coefficient`] or callable that can be
        made into :class:`QobjEvo` are also accepted.

    rho0 : :class:`Qobj`
        initial density matrix or state vector (ket).

    tlist : *list* / *array*
        list of times for :math:`t`.

    c_ops : list of (:class:`QobjEvo`, :class:`QobjEvo` compatible format)
        Single collapse operator, or list of collapse operators, or a list
        of Liouvillian superoperators. None is equivalent to an empty list.

    e_ops : list of :class:`Qobj` / callback function
        Single operator or list of operators for which to evaluate
        expectation values or callable or list of callable.
        Callable signature must be, `f(t: float, state: Qobj)`.
        See :func:`expect` for more detail of operator expectation.

    args : None / *dictionary*
        dictionary of parameters for time-dependent Hamiltonians and
        collapse operators.

    options : None / dict / :class:`Options` / :class:`Options`
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

    c_ops = c_ops if c_ops is not None else []
    if not isinstance(c_ops, (list, tuple)):
        c_ops = [c_ops]
    c_ops = [QobjEvo(c_op, args=args, tlist=tlist) for c_op in c_ops]

    use_mesolve = len(c_ops) > 0 or (not rho0.isket) or H.issuper

    if not use_mesolve:
        return sesolve(H, rho0, tlist, e_ops=e_ops, args=args,
                       options=SeOptions(options, _strict=False))

    solver = MeSolver(H, c_ops, options=options)

    return solver.run(rho0, tlist, e_ops=e_ops)


class MeSolver(Solver):
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
    H : :class:`Qobj`, :class:`QobjEvo`
        Possibly time-dependent system Liouvillian or Hamiltonian as a Qobj or QobjEvo.
        list of [:class:`Qobj`, :class:`Coefficient`] or callable that can be
        made into :class:`QobjEvo` are also accepted.

    c_ops : list of :class:`Qobj`, :class:`QobjEvo`
        Single collapse operator, or list of collapse operators, or a list
        of Liouvillian superoperators. None is equivalent to an empty list.

    options : SolverOptions
        Options for the solver

    attributes
    ----------
    stats: dict
        Diverse diagnostic statistics of the evolution.
    """
    name = "mesolve"
    _avail_integrators = {}
    optionsclass = MeOptions

    def __init__(self, H, c_ops=None, *, options=None):
        _time_start = time()

        if not isinstance(H, (Qobj, QobjEvo)):
            raise TypeError("The Hamiltonian must be a Qobj or QobjEvo")
        c_ops = c_ops or []
        c_ops = [c_ops] if isinstance(c_ops, (Qobj, QobjEvo)) else c_ops
        for c_op in c_ops:
            if not isinstance(c_op, (Qobj, QobjEvo)):
                raise TypeError("All `c_ops` must be a Qobj or QobjEvo")

        rhs = H if H.issuper else liouvillian(H)
        rhs += sum(c_op if c_op.issuper else lindblad_dissipator(c_op)
                   for c_op in c_ops)
        super().__init__(rhs, options=options)

        self.stats['solver'] = "Master Equation Evolution"
        self.stats['num_collapse'] = len(c_ops)
        self.stats["preparation time"] = time() - _time_start
        self.stats["run time"] = 0
