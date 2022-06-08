"""
This module provides solvers for the Lindblad master equation and von Neumann
equation.
"""

__all__ = ['brmesolve', 'BrSolver']

import numpy as np
import inspect
from time import time
from .. import Qobj, QobjEvo, coefficient, Coefficient
from ..core.blochredfield import bloch_redfield_tensor, SpectraCoefficient
from ..core.cy.coefficient import InterCoefficient
from ..core import data as _data
from .solver_base import Solver
from .options import known_solver, SolverOptions


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
            ``a_op`` is a :cls:`QobjEvo`.

        Example:
            a_ops = [
                (a+a.dag(), ('w>0', args={"w": 0})),
                (QobjEvo(a+a.dag()), 'w > exp(-t)'),
                (QobjEvo([b+b.dag(), lambda t: ...]), lambda w: ...)),
                (c+c.dag(), SpectraCoefficient(coefficient(array, tlist=ws))),
            ]

    .. note:
        ``Cubic_Spline`` have been replaced by :cls:`Coefficient`:
            ``spline = qutip.coefficient(array, tlist=times)``
        Whether the ``a_ops`` is time dependent is deceided by the type of the
        operator: :cls:`Qobj` vs :cls:`QobjEvo` instead of the type of the
        spectra.

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

    options : None / dict / :class:`Options`
        Options for the solver.

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

    solver = BrSolver(H, new_a_ops, c_ops, sec_cutoff, options=options)

    return solver.run(psi0, tlist, e_ops=e_ops)


class BrSolver(Solver):
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
            a_ops = [
                (a+a.dag(), coefficient('w>0', args={'w':0})),
                (QobjEvo([b+b.dag(), lambda t: ...]),
                 coefficient(lambda t, w: ...), args={"w": 0}),
                (c+c.dag(), SpectraCoefficient(coefficient(array, tlist=ws))),
            ]

    c_ops : list of :class:`Qobj`, :class:`QobjEvo`
        Single collapse operator, or list of collapse operators, or a list
        of Lindblad dissipator. None is equivalent to an empty list.

    sec_cutoff : float {0.1}
        Cutoff for secular approximation. Use ``-1`` if secular
        approximation is not used when evaluating bath-coupling terms.

    options : None / dict / :class:`Options`
        Options for the solver

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
        self._options = self.solver_options.copy()
        self.options = options

        if not isinstance(H, (Qobj, QobjEvo)):
            raise TypeError("The Hamiltonian must be a Qobj or QobjEvo")

        c_ops = c_ops or []
        c_ops = [c_ops] if isinstance(c_ops, (Qobj, QobjEvo)) else c_ops
        for c_op in c_ops:
            if not isinstance(c_op, (Qobj, QobjEvo)):
                raise TypeError("All `c_ops` must be a Qobj or QobjEvo")

        self._system = H, a_ops, c_ops
        rhs = self._prepare_rhs()
        super().__init__(rhs, options=self.options)

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
        store_final_state: bool
            Whether or not to store the final state of the evolution in the
            result class.

        store_states": bool, None
            Whether or not to store the state vectors or density matrices.
            On `None` the states will be saved if no expectation operators are
            given.

        normalize_output: bool
            Normalize output state to hide ODE numerical errors.

        progress_bar: str {'text', 'enhanced', 'tqdm', ''}
            How to present the solver progress.
            'tqdm' uses the python module of the same name and raise an error if
            not installed. Empty string or False will disable the bar.

        progress_kwargs: dict
            kwargs to pass to the progress_bar. Qutip's bars use `chunk_size`.

        tensor_type: str ['sparse', 'dense', 'data']
            Which data type to use when computing the brtensor.
            With a cutoff 'sparse' is usually the most efficient.

        sparse_eigensolver: bool {False}
            Whether to use the sparse eigensolver

        method: str
            Which ODE integrator methods are supported.
        """
        return self._options.copy()

    @options.setter
    def options(self, new_options):
        if not new_options:
            return  # Nothing to do
        change_method = (
            'method' in new_options
            and new_options['method'] != self._options['method']
        )
        kept_options = self._options
        if change_method:
            kept_options = self._options.solver_options

        self._options = SolverOptions('brmesolve', **{**kept_options, **new_options})

        need_new_rhs = (
            self.rhs is not None and self.rhs.isconstant
            and (
                'sparse_eigensolver' in new_options
                or 'tensor_type' in new_options
            )
        )
        if need_new_rhs:
            self.rhs = self._prepare_rhs()

        if self._integrator is None:
            pass
        elif change_method or need_new_rhs:
            state = self._integrator.get_state()
            self._integrator = self._get_integrator()
            self._integrator.set_state(*state)
        else:
            self._integrator.options = new_options

known_solver['brmesolve'] = BrSolver
known_solver['Brmesolver'] = BrSolver
known_solver['Brsolver'] = BrSolver
known_solver['BrSolver'] = BrSolver
known_solver['Bloch Redfield Equation Evolution'] = BrSolver
