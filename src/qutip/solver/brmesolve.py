"""
This module provides solvers for the Lindblad master equation and von Neumann
equation.
"""
# Required for Sphinx to follow autodoc_type_aliases
from __future__ import annotations

__all__ = ['brmesolve', 'BRSolver']

from typing import Any
import warnings
import numpy as np
from numpy.typing import ArrayLike
import inspect
from time import time
from .. import Qobj, QobjEvo, coefficient, Coefficient
from ..core.blochredfield import bloch_redfield_tensor, SpectraCoefficient
from ..core.cy.coefficient import InterCoefficient
from ..core import data as _data
from .solver_base import Solver, _solver_deprecation
from .options import _SolverOptions
from ._feedback import _QobjFeedback, _DataFeedback
from ..typing import EopsLike, QobjEvoLike, CoefficientLike
from ..core.environment import Environment


def brmesolve(
    H: QobjEvoLike,
    psi0: Qobj,
    tlist: ArrayLike,
    a_ops: list[tuple[QobjEvoLike, CoefficientLike]] = None,
    sec_cutoff: float = 0.1,
    *_pos_args,
    c_ops: list[QobjEvoLike] = None,
    e_ops: EopsLike | list[EopsLike] | dict[Any, EopsLike] = None,
    args: dict[str, Any] = None,
    options: dict[str, Any] = None,
    **kwargs
):
    r"""
    Solves for the dynamics of a system using the Bloch-Redfield master
    equation, given an input Hamiltonian, Hermitian bath-coupling terms and
    their associated spectral functions, as well as possible Lindblad collapse
    operators.

    Parameters
    ----------
    H : :obj:`.Qobj`, :obj:`.QobjEvo`
        Possibly time-dependent system Liouvillian or Hamiltonian as a Qobj or
        QobjEvo. list of [:obj:`.Qobj`, :obj:`.Coefficient`] or callable that
        can be made into :obj:`.QobjEvo` are also accepted.

    psi0: :obj:`.Qobj`
        Initial density matrix or state vector (ket).

    tlist : array_like
        List of times for evaluating evolution

    a_ops : list of (a_op, spectra)
        Nested list of system operators that couple to the environment,
        and the corresponding bath spectra.

        a_op : :obj:`.Qobj`, :obj:`.QobjEvo`, :obj:`.QobjEvo` compatible format
            The operator coupling to the environment. Must be hermitian.

        spectra : :obj:`.Coefficient`, str, func, Environment
            The corresponding bath spectra.
            Bath can be provided as :class:`.BosonicEnvironment`, 
            :class:`.FermionicEnvironment` or power spectra function.  These
            can be a :obj:`.Coefficient`, function or string. For coefficient,
            the frequency is passed as the 'w' args. The
            :class:`SpectraCoefficient` can be used for array based
            coefficient.

            The spectra can depend on ``t`` if the corresponding ``a_op`` is a
            :obj:`.QobjEvo`.

        Example:

        .. code-block::

            a_ops = [
                (a+a.dag(), BosonicEnvironment(...)),
                (a+a.dag(), ('w>0', args={"w": 0})),
                (QobjEvo(a+a.dag()), 'w > exp(-t)'),
                ([[b+b.dag(), lambda t: ...]], lambda w: ...)),
                (c+c.dag(), SpectraCoefficient(coefficient(array, tlist=ws))),
            ]

        .. note::

            ``Cubic_Spline`` has been replaced by:
                ``spline = qutip.coefficient(array, tlist=times)``

        Whether the ``a_ops`` is time dependent is decided by the type of
        the operator: :obj:`.Qobj` vs :obj:`.QobjEvo` instead of the type
        of the spectra.

    sec_cutoff : float, default: 0.1
        Cutoff for secular approximation. Use ``-1`` if secular approximation
        is not used when evaluating bath-coupling terms.

    *_pos_args :
        | Temporary shim to update the signature from
        | ``(..., a_ops, e_ops, c_ops, args, sec_cutoff, options)``
        | to
        | ``(..., a_ops, sec_cutoff, *, e_ops, c_ops, args, options)``
        | making ``e_ops``, ``c_ops``, ``args`` and ``options`` keyword only
          parameter from qutip 5.3.

    c_ops : list of (:obj:`.QobjEvo`, :obj:`.QobjEvo` compatible format), optional
        List of collapse operators.

    args : dict, optional
        Dictionary of parameters for time-dependent Hamiltonians and
        collapse operators. The key ``w`` is reserved for the spectra function.

    e_ops : list, dict, :obj:`.Qobj` or callback function, optional
        Single operator, or list or dict of operators, for which to evaluate
        expectation values. Operator can be Qobj, QobjEvo or callables with the
        signature `f(t: float, state: Qobj) -> Any`.
        Callable signature must be, `f(t: float, state: Qobj)`.

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
        - | computation_type : str ['sparse', 'dense', 'matrix']
          | With 'dense', the secular cutoff is checked first and only
            elements within the approximation are computed and stored in an
            array. With 'matrix', the tensor is build using matrix operation
            and the secular cutoff is applied at the end.
            With a cutoff 'sparse' is usually the most efficient.
        - | method : str ["adams", "bdf", "lsoda", "dop853", "vern9", etc.]
            Which differential equation integration method to use.
        - | atol, rtol : float
          | Absolute and relative tolerance of the ODE integrator.
        - | nsteps : int
          | Maximum number of (internally defined) steps allowed in one
            ``tlist`` step.
        - | max_step : float, 0
          | Maximum length of one internal step. When using pulses, it should
            be less than half the width of the thinnest pulse.

        Other options could be supported depending on the integration method,
        see `Integrator <./classes.html#classes-ode>`_.

    Returns
    -------
    result: :obj:`.Result`

        An instance of the class :obj:`qutip.solver.Result`, which contains
        either an array of expectation values, for operators given in e_ops,
        or a list of states for the times specified by ``tlist``.
    """
    if _pos_args or not isinstance(sec_cutoff, (int, float)):
        # Old signature used
        warnings.warn(
            "c_ops, e_ops, args and options will be keyword only"
            " from qutip 5.3",
            FutureWarning
        )
        # Re order for previous signature
        e_ops = sec_cutoff
        sec_cutoff = 0.1
        if len(_pos_args) >= 1:
            c_ops = _pos_args[0]
        if len(_pos_args) >= 2:
            args = _pos_args[1]
        if len(_pos_args) >= 3:
            sec_cutoff = _pos_args[2]
        if len(_pos_args) >= 4:
            options = _pos_args[3]

    options = _solver_deprecation(kwargs, options, "br")
    args = args or {}
    H = QobjEvo(H, args=args, tlist=tlist)

    c_ops = c_ops if c_ops is not None else []
    if not isinstance(c_ops, (list, tuple)):
        c_ops = [c_ops]
    c_ops = [QobjEvo(c_op, args=args, tlist=tlist) for c_op in c_ops]

    new_a_ops = []
    a_ops = a_ops or []
    for (a_op, spectra) in a_ops:
        if not isinstance(a_op, Qobj):
            a_op = QobjEvo(a_op, args=args, tlist=tlist)
        if isinstance(spectra, str):
            new_a_ops.append(
                (a_op, coefficient(spectra, args={**args, 'w': 0})))
        elif isinstance(spectra, InterCoefficient):
            new_a_ops.append((a_op, SpectraCoefficient(spectra)))
        elif isinstance(spectra, Coefficient):
            new_a_ops.append((a_op, spectra))
        elif isinstance(spectra, Environment):
            new_a_ops.append((a_op, spectra))
        elif callable(spectra):
            sig = inspect.signature(spectra)
            if tuple(sig.parameters.keys()) == ("w",):
                spec = SpectraCoefficient(coefficient(spectra))
            else:
                spec = coefficient(spectra, args={**args, 'w': 0})
            new_a_ops.append((a_op, spec))
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
    H : :obj:`.Qobj`, :obj:`.QobjEvo`
        Possibly time-dependent system Liouvillian or Hamiltonian as a Qobj or
        QobjEvo. list of [:obj:`.Qobj`, :obj:`.Coefficient`] or callable that
        can be made into :obj:`.QobjEvo` are also accepted.

    a_ops : list of (a_op, spectra)
        Nested list of system operators that couple to the environment,
        and the corresponding bath spectra.

        a_op : :obj:`.Qobj`, :obj:`.QobjEvo`
            The operator coupling to the environment. Must be hermitian.

        spectra : :obj:`.Coefficient`, :obj:`.Environment`
            The corresponding bath spectra. As a `Coefficient` using an 'w'
            args. Can depend on ``t`` only if a_op is a :obj:`.QobjEvo`.
            :obj:`SpectraCoefficient` can be used to convert a coefficient
            depending on ``t`` to one depending on ``w``.
            :class:`.BosonicEnvironment` or :class:`.FermionicEnvironment` are
            also valid spectrum.

        Example:

        .. code-block::

            a_ops = [
                (a+a.dag(), coefficient('w>0', args={'w':0})),
                (QobjEvo([b+b.dag(), lambda t: ...]),
                 coefficient(lambda t, w: ...), args={"w": 0}),
                (c+c.dag(), SpectraCoefficient(coefficient(array, tlist=ws))),
            ]

    c_ops : list of :obj:`.Qobj`, :obj:`.QobjEvo`
        Single collapse operator, or list of collapse operators, or a list
        of Lindblad dissipator. None is equivalent to an empty list.

    options : dict, optional
        Options for the solver, see :obj:`BRSolver.options` and
        `Integrator <./classes.html#classes-ode>`_ for a list of all options.

    sec_cutoff : float {0.1}
        Cutoff for secular approximation. Use ``-1`` if secular approximation
        is not used when evaluating bath-coupling terms.

    Attributes
    ----------
    stats: dict
        Diverse diagnostic statistics of the evolution.
    """
    name = "brmesolve"
    solver_options = {
        "progress_bar": "",
        "progress_kwargs": {"chunk_size": 10},
        "store_final_state": False,
        "store_states": None,
        "normalize_output": False,
        'method': 'adams',
        'tensor_type': 'sparse',
    }
    _avail_integrators = {}

    def __init__(
        self,
        H: Qobj | QobjEvo,
        a_ops: list[tuple[Qobj | QobjEvo, Coefficient | Environment]],
        c_ops: Qobj | QobjEvo | list[Qobj | QobjEvo] = None,
        sec_cutoff: float = 0.1,
        *,
        options: dict[str, Any] = None,
    ):
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
            raise TypeError("`a_ops` must be a list of (operator, spectra)")
        if a_ops and isinstance(a_ops[0], (Qobj, QobjEvo)):
            a_ops = [a_ops]
        for oper, spectra in a_ops:
            if not isinstance(oper, (Qobj, QobjEvo)):
                raise TypeError("All `a_ops` operators "
                                "must be a Qobj or QobjEvo")
            if not isinstance(spectra, (Coefficient, Environment)):
                raise TypeError("All `a_ops` spectra "
                                "must be a Coefficient or Environment.")

        self._system = H, a_ops, c_ops
        self._num_collapse = len(c_ops)
        self._num_a_ops = len(a_ops)
        self.rhs = self._prepare_rhs()
        self._integrator = self._get_integrator()
        self._state_metadata = {}
        self.stats = self._initialize_stats()
        self.rhs._register_feedback({}, solver=self.name)

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
            sparse_eigensolver=False,
            br_computation_method=self.options['tensor_type']
        )
        self._init_rhs_time = time() - _time_start
        return rhs

    @property
    def options(self):
        """
        Options for bloch redfield solver:

        store_final_state: bool, default: False
            Whether or not to store the final state of the evolution in the
            result class.

        store_states: bool, default: None
            Whether or not to store the state vectors or density matrices.
            On `None` the states will be saved if no expectation operators are
            given.

        normalize_output: bool, default: False
            Normalize output state to hide ODE numerical errors.

        progress_bar: str {'text', 'enhanced', 'tqdm', ''}, default: ""
            How to present the solver progress.
            'tqdm' uses the python module of the same name and raise an error if
            not installed. Empty string or False will disable the bar.

        progress_kwargs: dict, default: {"chunk_size":10}
            Arguments to pass to the progress_bar. Qutip's bars use
            ``chunk_size``.

        tensor_type: str ['sparse', 'dense', 'data'], default: "sparse"
            Which data type to use when computing the brtensor.
            With a cutoff 'sparse' is usually the most efficient.

        method: str, default: "adams"
            Which ODE integrator methods are supported.
        """
        return self._options

    @options.setter
    def options(self, new_options):
        Solver.options.fset(self, new_options)

    def _apply_options(self, keys):
        need_new_rhs = self.rhs is not None and not self.rhs.isconstant
        need_new_rhs &= 'tensor_type' in keys
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

    @classmethod
    def StateFeedback(cls, default=None, raw_data=False):
        """
        State of the evolution to be used in a time-dependent operator.

        When used as an args:

            ``QobjEvo([op, func], args={"state": BRMESolver.StateFeedback()})``

        The ``func`` will receive the density matrix as ``state`` during the
        evolution.

        .. note::

            The state will not be in the lab basis, but in the evolution basis.

        Parameters
        ----------
        default : Qobj or qutip.core.data.Data, default : None
            Initial value to be used at setup of the system.

        raw_data : bool, default : False
            If True, the raw matrix will be passed instead of a Qobj.
            For density matrices, the matrices can be column stacked or square
            depending on the integration method.
        """
        if raw_data:
            return _DataFeedback(default, open=True)
        return _QobjFeedback(default, open=True)
