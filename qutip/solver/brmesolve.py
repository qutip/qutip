"""
This module provides solvers for the Lindblad master equation and von Neumann
equation.
"""

__all__ = ['brmesolve', 'BRSolver']

import numpy as np
import inspect
from time import time
from .. import Qobj, QobjEvo, coefficient, Coefficient
from ..core.blochredfield import bloch_redfield_tensor, SpectraCoefficient
from ..core.cy.coefficient import InterCoefficient
from ..core import data as _data
from .solver_base import Solver
from .options import _SolverOptions


def brmesolve(H, psi0, tlist, a_ops=[], e_ops=[], c_ops=[],
              args={}, sec_cutoff=0.1, options=None):
    """
    Solves for the dynamics of a system using the Bloch-Redfield master
    equation, given an input Hamiltonian, Hermitian bath-coupling terms and
    their associated spectral functions, as well as possible Lindblad collapse
    operators.

    Parameters
    ----------
    H : :class:`Qobj`, :class:`QobjEvo`
        Possibly time-dependent system Liouvillian or Hamiltonian as a Qobj or
        QobjEvo. list of [:class:`Qobj`, :class:`Coefficient`] or callable that
        can be made into :class:`QobjEvo` are also accepted.

    psi0: Qobj
        Initial density matrix or state vector (ket).

    tlist : array_like
        List of times for evaluating evolution

    a_ops : list of (a_op, spectra)
        Nested list of system operators that couple to the environment,
        and the corresponding bath spectra.

        a_op : :class:`qutip.Qobj`, :class:`qutip.QobjEvo`
            The operator coupling to the environment. Must be hermitian.

        spectra : :class:`Coefficient`, str, func
            The corresponding bath spectral responce.
            Can be a `Coefficient` using an 'w' args, a function of the
            frequence or a string. Coefficient build from a numpy array are
            understood as a function of ``w`` instead of ``t``. Function are
            expected to be of the signature ``f(w)`` or ``f(t, w, **args)``.

            The spectra function can depend on ``t`` if the corresponding
            ``a_op`` is a :class:`QobjEvo`.

        Example:

        .. code-block::

            a_ops = [
                (a+a.dag(), ('w>0', args={"w": 0})),
                (QobjEvo(a+a.dag()), 'w > exp(-t)'),
                (QobjEvo([b+b.dag(), lambda t: ...]), lambda w: ...)),
                (c+c.dag(), SpectraCoefficient(coefficient(array, tlist=ws))),
            ]

        .. note:
            ``Cubic_Spline`` has been replaced by :class:`Coefficient`:
                ``spline = qutip.coefficient(array, tlist=times)``

            Whether the ``a_ops`` is time dependent is decided by the type of
            the operator: :class:`Qobj` vs :class:`QobjEvo` instead of the type
            of the spectra.

    e_ops : list of :class:`Qobj` / callback function
        Single operator or list of operators for which to evaluate
        expectation values or callable or list of callable.
        Callable signature must be, `f(t: float, state: Qobj)`.
        See :func:`expect` for more detail of operator expectation

    c_ops : list of (:class:`QobjEvo`, :class:`QobjEvo` compatible format)
        List of collapse operators.

    args : dict
        Dictionary of parameters for time-dependent Hamiltonians and
        collapse operators. The key ``w`` is reserved for the spectra function.

    sec_cutoff : float {0.1}
        Cutoff for secular approximation. Use ``-1`` if secular approximation
        is not used when evaluating bath-coupling terms.

    options : None / dict
        Dictionary of options for the solver.

        - store_final_state : bool
          Whether or not to store the final state of the evolution in the
          result class.
        - store_states : bool, None
          Whether or not to store the state vectors or density matrices.
          On `None` the states will be saved if no expectation operators are
          given.
        - normalize_output : bool
          Normalize output state to hide ODE numerical errors.
        - progress_bar : str {'text', 'enhanced', 'tqdm', ''}
          How to present the solver progress.
          'tqdm' uses the python module of the same name and raise an error
          if not installed. Empty string or False will disable the bar.
        - progress_kwargs : dict
          kwargs to pass to the progress_bar. Qutip's bars use `chunk_size`.
        - tensor_type : str ['sparse', 'dense', 'data']
          Which data type to use when computing the brtensor.
          With a cutoff 'sparse' is usually the most efficient.
        - sparse_eigensolver : bool {False}
          Whether to use the sparse eigensolver
        - method : str ["adams", "bdf", "lsoda", "dop853", "vern9", etc.]
          Which differential equation integration method to use.
        - atol, rtol : float
          Absolute and relative tolerance of the ODE integrator.
        - nsteps :
          Maximum number of (internally defined) steps allowed in one ``tlist``
          step.
        - max_step : float, 0
          Maximum lenght of one internal step. When using pulses, it should be
          less than half the width of the thinnest pulse.

        Other options could be supported depending on the integration method,
        see `Integrator <./classes.html#classes-ode>`_.

    Returns
    -------
    result: :class:`qutip.solver.Result`

        An instance of the class :class:`qutip.solver.Result`, which contains
        either an array of expectation values, for operators given in e_ops,
        or a list of states for the times specified by `tlist`.

    .. note:
        The option ``operator_data_type`` is used to determine in which format
        the bloch redfield tensor is computed. Use 'csr' for sparse and 'dense'
        for dense array. With 'data', it will try to use the same data type as
        the ``a_ops``, but it is usually less efficient than manually choosing
        it.
    """
    H = QobjEvo(H, args=args, tlist=tlist)

    c_ops = c_ops if c_ops is not None else []
    if not isinstance(c_ops, (list, tuple)):
        c_ops = [c_ops]
    c_ops = [QobjEvo(c_op, args=args, tlist=tlist) for c_op in c_ops]

    new_a_ops = []
    for (a_op, spectra) in a_ops:
        aop = QobjEvo(a_op, args=args, tlist=tlist)
        if isinstance(spectra, str):
            new_a_ops.append(
                (aop, coefficient(spectra, args={**args, 'w':0})))
        elif isinstance(spectra, InterCoefficient):
            new_a_ops.append((aop, SpectraCoefficient(spectra)))
        elif isinstance(spectra, Coefficient):
            new_a_ops.append((aop, spectra))
        elif callable(spectra):
            sig = inspect.signature(spectra)
            if tuple(sig.parameters.keys()) == ("w",):
                spec = SpectraCoefficient(coefficient(spectra))
            else:
                spec = coefficient(spectra, args={**args, 'w':0})
            new_a_ops.append((aop, spec))
        else:
            raise TypeError("a_ops's spectra not known")

    solver = BRSolver(H, new_a_ops, c_ops, sec_cutoff, options=options)

    return solver.run(psi0, tlist, e_ops=e_ops)


class BRSolver(Solver):
    """
    Bloch Redfield equation evolution of a density matrix for a given
    Hamiltonian and set of bath coupling operators.

    Parameters
    ----------
    H : :class:`Qobj`, :class:`QobjEvo`
        Possibly time-dependent system Liouvillian or Hamiltonian as a Qobj or
        QobjEvo. list of [:class:`Qobj`, :class:`Coefficient`] or callable that
        can be made into :class:`QobjEvo` are also accepted.

    a_ops : list of (a_op, spectra)
        Nested list of system operators that couple to the environment,
        and the corresponding bath spectra.

        a_op : :class:`qutip.Qobj`, :class:`qutip.QobjEvo`
            The operator coupling to the environment. Must be hermitian.

        spectra : :class:`Coefficient`
            The corresponding bath spectra. As a `Coefficient` using an 'w'
            args. Can depend on ``t`` only if a_op is a :class:`qutip.QobjEvo`.
            :class:`SpectraCoefficient` can be used to conver a coefficient
            depending on ``t`` to one depending on ``w``.

        Example:

        .. code-block::

            a_ops = [
                (a+a.dag(), coefficient('w>0', args={'w':0})),
                (QobjEvo([b+b.dag(), lambda t: ...]),
                 coefficient(lambda t, w: ...), args={"w": 0}),
                (c+c.dag(), SpectraCoefficient(coefficient(array, tlist=ws))),
            ]

    c_ops : list of :class:`Qobj`, :class:`QobjEvo`
        Single collapse operator, or list of collapse operators, or a list
        of Lindblad dissipator. None is equivalent to an empty list.

    options : dict, optional
        Options for the solver, see :obj:`BRSolver.options` and
        `Integrator <./classes.html#classes-ode>`_ for a list of all options.

    sec_cutoff : float {0.1}
        Cutoff for secular approximation. Use ``-1`` if secular approximation
        is not used when evaluating bath-coupling terms.

    attributes
    ----------
    stats: dict
        Diverse diagnostic statistics of the evolution.
    """
    name = "brmesolve"
    solver_options = {
        "progress_bar": "text",
        "progress_kwargs": {"chunk_size":10},
        "store_final_state": False,
        "store_states": None,
        "normalize_output": False,
        'method': 'adams',
        'tensor_type': 'sparse',
        'sparse_eigensolver': False,
    }
    _avail_integrators = {}

    def __init__(self, H, a_ops, c_ops=None, sec_cutoff=0.1, *, options=None):
        _time_start = time()

        self.rhs = None
        self.sec_cutoff = sec_cutoff
        self.options = options

        if not isinstance(H, (Qobj, QobjEvo)):
            raise TypeError("The Hamiltonian must be a Qobj or QobjEvo")
        H = QobjEvo(H)

        c_ops = c_ops or []
        c_ops = [c_ops] if isinstance(c_ops, (Qobj, QobjEvo)) else c_ops
        for c_op in c_ops:
            if not isinstance(c_op, (Qobj, QobjEvo)):
                raise TypeError("All `c_ops` must be a Qobj or QobjEvo")

        a_ops = a_ops or []
        if not hasattr(a_ops, "__iter__"):
            TypeError("`a_ops` must be a list of (operator, spectra)")
        if a_ops and isinstance(a_ops[0], (Qobj, QobjEvo)):
            a_ops = [a_ops]
        for oper, spectra in a_ops:
            if not isinstance(oper, (Qobj, QobjEvo)):
                raise TypeError("All `a_ops` operators "
                                "must be a Qobj or QobjEvo")
            if not isinstance(spectra, Coefficient):
                raise TypeError("All `a_ops` spectra "
                                "must be a Coefficient.")

        self._system = H, a_ops, c_ops
        self._num_collapse = len(c_ops)
        self._num_a_ops = len(a_ops)
        self.rhs = self._prepare_rhs()
        self._integrator = self._get_integrator()
        self._state_metadata = {}
        self.stats = self._initialize_stats()


    def _initialize_stats(self):
        stats = super()._initialize_stats()
        stats.update({
            "solver": "Bloch Redfield Equation Evolution",
            "init time": stats["init time"] + self._init_rhs_time,
            "num_collapse": self._num_collapse,
            "num_a_ops": self._num_a_ops,
        })
        return stats

    def _prepare_rhs(self):
        _time_start = time()
        rhs = bloch_redfield_tensor(
            *self._system,
            fock_basis=True,
            sec_cutoff=self.sec_cutoff,
            sparse_eigensolver=self.options['sparse_eigensolver'],
            br_dtype=self.options['tensor_type']
        )
        self._init_rhs_time = time() - _time_start
        return rhs

    @property
    def options(self):
        """
        Options for bloch redfield solver:

        store_final_state: bool, default=False
            Whether or not to store the final state of the evolution in the
            result class.

        store_states: bool, default=None
            Whether or not to store the state vectors or density matrices.
            On `None` the states will be saved if no expectation operators are
            given.

        normalize_output: bool, default=False
            Normalize output state to hide ODE numerical errors.

        progress_bar: str {'text', 'enhanced', 'tqdm', ''}, default="text"
            How to present the solver progress.
            'tqdm' uses the python module of the same name and raise an error if
            not installed. Empty string or False will disable the bar.

        progress_kwargs: dict, default={"chunk_size":10}
            Arguments to pass to the progress_bar. Qutip's bars use
            ``chunk_size``.

        tensor_type: str ['sparse', 'dense', 'data'], default="sparse"
            Which data type to use when computing the brtensor.
            With a cutoff 'sparse' is usually the most efficient.

        sparse_eigensolver: bool, default=False
            Whether to use the sparse eigensolver

        method: str, default="adams"
            Which ODE integrator methods are supported.
        """
        return self._options

    @options.setter
    def options(self, new_options):
        Solver.options.fset(self, new_options)

    def _apply_options(self, keys):
        need_new_rhs = self.rhs is not None and not self.rhs.isconstant
        need_new_rhs &= (
            'sparse_eigensolver' in keys or 'tensor_type' in keys
        )
        if need_new_rhs:
            self.rhs = self._prepare_rhs()

        if self._integrator is None or not keys:
            pass
        elif 'method' in keys or need_new_rhs:
            state = self._integrator.get_state()
            self._integrator = self._get_integrator()
            self._integrator.set_state(*state)
        else:
            self._integrator.options = self._options
            self._integrator.reset(hard=True)
