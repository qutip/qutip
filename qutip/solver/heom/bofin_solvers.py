"""
This module provides solvers for system-bath evoluation using the
HEOM (hierarchy equations of motion).

See https://en.wikipedia.org/wiki/Hierarchical_equations_of_motion for a very
basic introduction to the technique.

The implementation is derived from the BoFiN library (see
https://github.com/tehruhn/bofin) which was itself derived from an earlier
implementation in QuTiP itself.
"""

from time import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from qutip.settings import settings
from qutip import state_number_enumerate, CoreOptions
from qutip.core import data as _data
from qutip.core.data import csr as _csr
from qutip.core.environment import (
    BosonicEnvironment, FermionicEnvironment,
    ExponentialBosonicEnvironment, ExponentialFermionicEnvironment)
from qutip.core import Qobj, QobjEvo
from qutip.core.superoperator import liouvillian, spre, spost
from .bofin_baths import (
    Bath, BathExponent, BosonicBath, DrudeLorentzBath, FermionicBath,
)
from ..solver_base import Solver
from .. import Result

# Load MKL spsolve if avaiable
if settings.has_mkl:
    from qutip._mkl.spsolve import mkl_spsolve
else:
    mkl_spsolve = None

__all__ = [
    "heomsolve",
    "HierarchyADOs",
    "HierarchyADOsState",
    "HEOMResult",
    "HEOMSolver",
    "HSolverDL",
]


class HierarchyADOs:
    """
    A description of ADOs (auxilliary density operators) with the
    hierarchical equations of motion.

    The list of ADOs is constructed from a list of bath exponents
    (corresponding to one or more baths). Each ADO is referred to by a label
    that lists the number of "excitations" of each bath exponent. The
    level of a label within the hierarchy is the sum of the "excitations"
    within the label.

    For example the label ``(0, 0, ..., 0)`` represents the density matrix
    of the system being solved and is the only 0th level label.

    The labels with a single 1, i.e. ``(1, 0, ..., 0)``, ``(0, 1, 0, ... 0)``,
    etc. are the 1st level labels.

    The second level labels all have either two 1s or a single 2, and so on
    for the third and higher levels of the hierarchy.

    Parameters
    ----------
    exponents : list of :class:`.BathExponent`
        The exponents of the correlation function describing the bath or
        baths.

    max_depth : int
        The maximum depth of the hierarchy (i.e. the maximum sum of
        "excitations" in the hierarchy ADO labels or maximum ADO level).

    Attributes
    ----------
    exponents : list of :class:`.BathExponent`
        The exponents of the correlation function describing the bath or
        baths.

    max_depth : int
        The maximum depth of the hierarchy (i.e. the maximum sum of
        "excitations" in the hierarchy ADO labels).

    dims : list of int
        The dimensions of each exponent within the bath(s).

    vk : list of complex
        The frequency of each exponent within the bath(s).

    ck : list of complex
        The coefficient of each exponent within the bath(s).

    ck2: list of complex
        For exponents of type "RI", the coefficient of the exponent within
        the imaginary expansion. For other exponent types, the entry is None.

    sigma_bar_k_offset: list of int
        For exponents of type "+" or "-" the offset within the list of modes
        of the corresponding "-" or "+" exponent. For other exponent types,
        the entry is None.

    labels: list of tuples
        A list of the ADO labels within the hierarchy.
    """

    def __init__(self, exponents, max_depth):
        self.exponents = exponents
        self.max_depth = max_depth

        self.dims = [exp.dim or (max_depth + 1) for exp in self.exponents]
        self.vk = [exp.vk for exp in self.exponents]
        self.ck = [exp.ck for exp in self.exponents]
        self.ck2 = [exp.ck2 for exp in self.exponents]
        self.sigma_bar_k_offset = [
            exp.sigma_bar_k_offset for exp in self.exponents
        ]

        self.labels = list(state_number_enumerate(self.dims, max_depth))
        self._label_idx = {s: i for i, s in enumerate(self.labels)}
        self.idx = self._label_idx.__getitem__

    def idx(self, label):
        """
        Return the index of the ADO label within the list of labels,
        i.e. within ``self.labels``.

        Parameters
        ----------
        label : tuple
            The label to look up.

        Returns
        -------
        int
            The index of the label within the list of ADO labels.

        Notes
        -----
        This implementation of the ``.idx(...)`` method is just for
        reference and documentation. To avoid the cost of a Python
        function call, it is replaced with
        ``self._label_idx.__getitem__`` when the instance is created.
        """
        return self._label_idx[label]

    def next(self, label, k):
        """
        Return the ADO label with one more excitation in the k'th exponent
        dimension or ``None`` if adding the excitation would exceed the
        dimension or maximum depth of the hierarchy.

        Parameters
        ----------
        label : tuple
            The ADO label to add an excitation to.
        k : int
            The exponent to add the excitation to.

        Returns
        -------
        tuple or None
            The next label.
        """
        if label[k] >= self.dims[k] - 1:
            return None
        if sum(label) >= self.max_depth:
            return None
        return label[:k] + (label[k] + 1,) + label[k + 1:]

    def prev(self, label, k):
        """
        Return the ADO label with one fewer excitation in the k'th
        exponent dimension or ``None`` if the label has no exciations in the
        k'th exponent.

        Parameters
        ----------
        label : tuple
            The ADO label to remove the excitation from.
        k : int
            The exponent to remove the excitation from.

        Returns
        -------
        tuple or None
            The previous label.
        """
        if label[k] <= 0:
            return None
        return label[:k] + (label[k] - 1,) + label[k + 1:]

    def exps(self, label):
        """
        Converts an ADO label into a tuple of exponents, with one exponent
        for each "excitation" within the label.

        The number of exponents returned is always equal to the level of the
        label within the hierarchy (i.e. the sum of the indices within the
        label).

        Parameters
        ----------
        label : tuple
            The ADO label to convert to a list of exponents.

        Returns
        -------
        tuple of :class:`.BathExponent`
            A tuple of BathExponents.

        Examples
        --------

        ``ados.exps((1, 0, 0))`` would return ``[ados.exponents[0]]``

        ``ados.exps((2, 0, 0))`` would return
        ``[ados.exponents[0], ados.exponents[0]]``.

        ``ados.exps((1, 2, 1))`` would return
        ``[ados.exponents[0], ados.exponents[1], ados.exponents[1], \
           ados.exponents[2]]``.
        """
        return sum(
            ((exp,) * n for (n, exp) in zip(label, self.exponents) if n > 0),
            (),
        )

    def filter(self, level=None, tags=None, dims=None, types=None):
        """
        Return a list of ADO labels for ADOs whose "excitations"
        match the given patterns.

        Each of the filter parameters (tags, dims, types) may be either
        unspecified (None) or a list. Unspecified parameters are excluded
        from the filtering.

        All specified filter parameters must be lists of the same length.
        Each position in the lists describes a particular excitation and
        any exponent that matches the filters may supply that excitation.
        The level of all labels returned is thus equal to the length of
        the filter parameter lists.

        Within a filter parameter list, items that are None represent
        wildcards and match any value of that exponent attribute

        Parameters
        ----------
        level : int
            The hierarchy depth to return ADOs from.

        tags : list of object or None
            Filter parameter that matches the ``.tag`` attribute of
            exponents.

        dims : list of int
            Filter parameter that matches the ``.dim`` attribute of
            exponents.

        types : list of BathExponent types or list of str
            Filter parameter that matches the ``.type`` attribute
            of exponents. Types may be supplied by name (e.g. "R", "I", "+")
            instead of by the actual type (e.g. ``CFExponent.types.R``).

        Returns
        -------
        list of tuple
            The ADO label for each ADO whose exponent excitations
            (i.e. label) match the given filters or level.
        """
        if types is not None:
            types = [
                t if t is None or isinstance(t, BathExponent.types)
                else BathExponent.types[t]
                for t in types
            ]
        filters = [("tag", tags), ("type", types), ("dim", dims)]
        filters = [(attr, f) for attr, f in filters if f is not None]
        n = max((len(f) for _, f in filters), default=0)
        if any(len(f) != n for _, f in filters):
            raise ValueError(
                "The tags, dims and types filters must all be the same length."
            )
        if n > self.max_depth:
            raise ValueError(
                f"The maximum depth for the hierarchy is {self.max_depth} but"
                f" {n} levels of excitation filters were given."
            )
        if level is None:
            if not filters:
                # fast path for when there are no excitation filters
                return self.labels[:]
        else:
            if not filters:
                # fast path for when there are no excitation filters
                return [label for label in self.labels if sum(label) == level]
            if level != n:
                raise ValueError(
                    f"The level parameter is {level} but {n} levels of"
                    " excitation filters were given."
                )

        filtered_dims = [1] * len(self.exponents)
        for lvl in range(n):
            level_filters = [
                (attr, f[lvl]) for attr, f in filters
                if f[lvl] is not None
            ]
            for j, exp in enumerate(self.exponents):
                if any(getattr(exp, attr) != f for attr, f in level_filters):
                    continue
                filtered_dims[j] += 1
                filtered_dims[j] = min(self.dims[j], filtered_dims[j])

        return [
            label for label in state_number_enumerate(filtered_dims, n)
            if sum(label) == n
        ]


class HierarchyADOsState:
    """
    Provides convenient access to the full hierarchy ADO state at a particular
    point in time, ``t``.

    Parameters
    ----------
    rho : :class:`.Qobj`
        The current state of the system (i.e. the 0th component of the
        hierarchy).
    ados : :class:`HierarchyADOs`
        The description of the hierarchy.
    ado_state : numpy.array
        The full state of the hierarchy.

    Attributes
    ----------
    rho : Qobj
        The system state.

    Notes
    -----
    In addition, all of the attributes of the hierarchy description,
    i.e. ``HierarchyADOs``, are provided directly on this class for
    convenience. E.g. one can access ``.labels``, or ``.exponents`` or
    call ``.idx(label)`` directly.

    See :class:`HierarchyADOs` for a full list of the available attributes
    and methods.
    """

    def __init__(self, rho, ados, ado_state):
        self.rho = rho
        self._ado_state = ado_state
        self._ados = ados

    def __getattr__(self, name):
        return getattr(self._ados, name)

    def extract(self, idx_or_label):
        """
        Extract a Qobj representing the specified ADO from a full
        representation of the ADO states.

        Parameters
        ----------
        idx : int or label
            The index of the ADO to extract. If an ADO label, e.g.
            ``(0, 1, 0, ...)`` is supplied instead, then the ADO
            is extracted by label instead.

        Returns
        -------
        Qobj
            A :obj:`.Qobj` representing the state of the specified ADO.
        """
        if isinstance(idx_or_label, int):
            idx = idx_or_label
        else:
            idx = self._ados.idx(idx_or_label)
        return Qobj(self._ado_state[idx, :].T, dims=self.rho.dims)


class HEOMResult(Result):
    def _post_init(self):
        super()._post_init()

        self.store_ados = self.options["store_ados"]
        if self.store_ados:
            self._final_ado_state = None
            self.ado_states = []

    def _e_op_func(self, e_op):
        """ Convert an e_op into a function ``f(t, ado_state)``. """
        if isinstance(e_op, Qobj):
            return lambda t, ado_state: (ado_state.rho * e_op).tr()
        elif isinstance(e_op, QobjEvo):
            return lambda t, ado_state: e_op.expect(t, ado_state.rho)
        elif callable(e_op):
            return e_op
        raise TypeError(f"{e_op!r} has unsupported type {type(e_op)!r}.")

    def _pre_copy(self, state):
        return state

    def _store_state(self, t, ado_state):
        self.states.append(ado_state.rho)
        if self.store_ados:
            self.ado_states.append(ado_state)

    def _store_final_state(self, t, ado_state):
        self._final_state = ado_state.rho
        if self.store_ados:
            self._final_ado_state = ado_state

    @property
    def final_ado_state(self):
        if self._final_ado_state is not None:
            return self._final_state
        if self.ado_states:
            return self.ado_states[-1]
        return None


def heomsolve(
    H, bath, max_depth, state0, tlist, *, e_ops=None, args=None, options=None,
):
    """
    Hierarchical Equations of Motion (HEOM) solver that supports multiple
    baths.

    Each bath must be either bosonic or fermionic, but bosonic and fermionic
    baths can be mixed.

    If you need to run many evolutions of the same system and bath, consider
    using :class:`HEOMSolver` directly to avoid having to continually
    reconstruct the equation hierarchy for every evolution.

    Parameters
    ----------
    H : :obj:`.Qobj`, :obj:`.QobjEvo`
        Possibly time-dependent system Liouvillian or Hamiltonian as a Qobj or
        QobjEvo. List of [:obj:`.Qobj`, :obj:`.Coefficient`], or callable that
        can be made into :obj:`.QobjEvo` are also accepted.

    bath : Bath specification or list of Bath specifications
        A bath containing the exponents of the expansion of the
        bath correlation funcion and their associated coefficients
        and coupling operators, or a list of baths.

        Each bath can be specified as *either* an object of type
        :class:`.Bath`, :class:`.BosonicBath`, :class:`.FermionicBath`, or
        their subtypes, *or* as a tuple ``(env, Q)``, where ``env`` is an
        :class:`.ExponentialBosonicEnvironment` or an
        :class:`.ExponentialFermionicEnvironment` and ``Q`` the system coupling
        operator.

    max_depth : int
        The maximum depth of the heirarchy (i.e. the maximum number of bath
        exponent "excitations" to retain).

    state0 : :obj:`.Qobj` or :class:`~HierarchyADOsState` or array-like
        If ``rho0`` is a :obj:`.Qobj` the it is the initial state
        of the system (i.e. a :obj:`.Qobj` density matrix).

        If it is a :class:`~HierarchyADOsState` or array-like, then
        ``rho0`` gives the initial state of all ADOs.

        Usually the state of the ADOs would be determine from a previous
        call to ``.run(...)`` with the solver results option ``store_ados``
        set to True. For example, ``result = solver.run(...)`` could be
        followed by ``solver.run(result.ado_states[-1], tlist)``.

        If a numpy array-like is passed its shape must be
        ``(number_of_ados, n, n)`` where ``(n, n)`` is the system shape
        (i.e. shape of the system density matrix) and the ADOs must
        be in the same order as in ``.ados.labels``.

    tlist : list
        An ordered list of times at which to return the value of the state.

    e_ops : Qobj / QobjEvo / callable / list / dict / None, optional
        A list or dictionary of operators as :obj:`.Qobj`,
        :obj:`.QobjEvo` and/or callable functions (they can be mixed) or
        a single operator or callable function. For an operator ``op``, the
        result will be computed using ``(state * op).tr()`` and the state
        at each time ``t``. For callable functions, ``f``, the result is
        computed using ``f(t, ado_state)``. The values are stored in the
        ``expect`` and ``e_data`` attributes of the result (see the return
        section below).

    args : dict, optional
        Change the ``args`` of the RHS for the evolution.

    options : dict, optional
        Generic solver options.

        - | store_final_state : bool
          | Whether or not to store the final state of the evolution in the
            result class.
        - | store_states : bool, None
          | Whether or not to store the state vectors or density matrices.
            On `None` the states will be saved if no expectation operators are
            given.
        - | store_ados : bool
          | Whether or not to store the HEOM ADOs.
        - | normalize_output : bool
          | Normalize output state to hide ODE numerical errors. Only normalize
            the state if the initial state is already normalized.
        - | progress_bar : str {'text', 'enhanced', 'tqdm', ''}
          | How to present the solver progress.
            'tqdm' uses the python module of the same name and raise an error
            if not installed. Empty string or False will disable the bar.
        - | progress_kwargs : dict
          | kwargs to pass to the progress_bar. Qutip's bars use `chunk_size`.
        - | state_data_type: str {'dense', 'CSR', 'Dia', }
          | Name of the data type of the state used during the ODE evolution.
            Use an empty string to keep the input state type. Many integrator
            can only work with `Dense`.
        - | method : str ["adams", "bdf", "lsoda", "dop853", "vern9", etc.]
          | Which differential equation integration method to use.
        - | atol, rtol : float
          | Absolute and relative tolerance of the ODE integrator.
        - | nsteps : int
          | Maximum number of (internally defined) steps allowed in one
            ``tlist`` step.
        - | max_step : float,
          | Maximum lenght of one internal step. When using pulses, it should
            be less than half the width of the thinnest pulse.

    Returns
    -------
    :class:`~HEOMResult`
        The results of the simulation run, with the following important
        attributes:

        * ``times``: the times ``t`` (i.e. the ``tlist``).

        * ``states``: the system state at each time ``t`` (only available
          if ``e_ops`` was ``None`` or if the solver option
          ``store_states`` was set to ``True``).

        * ``ado_states``: the full ADO state at each time (only available
          if the results option ``ado_return`` was set to ``True``).
          Each element is an instance of :class:`.HierarchyADOsState`.
          The state of a particular ADO may be extracted from
          ``result.ado_states[i]`` by calling
          :meth:`extract <.HierarchyADOsState.extract>`.

        * ``expect``: a list containing the values of each ``e_ops`` at
          time ``t``.

        * ``e_data``: a dictionary containing the values of each ``e_ops``
          at tme ``t``. The keys are those given by ``e_ops`` if it was
          a dict, otherwise they are the indexes of the supplied ``e_ops``.

        See :class:`~HEOMResult` and :class:`.Result` for the complete
        list of attributes.
    """
    H = QobjEvo(H, args=args, tlist=tlist)
    solver = HEOMSolver(H, bath=bath, max_depth=max_depth, options=options)
    return solver.run(state0, tlist, e_ops=e_ops)


class HEOMSolver(Solver):
    """
    HEOM solver that supports multiple baths.

    Each bath must be either bosonic or fermionic, but bosonic and fermionic
    baths can be mixed.

    Parameters
    ----------
    H : :obj:`.Qobj`, :obj:`.QobjEvo`
        Possibly time-dependent system Liouvillian or Hamiltonian as a Qobj or
        QobjEvo. list of [:obj:`.Qobj`, :obj:`.Coefficient`] or callable that
        can be made into :obj:`.QobjEvo` are also accepted.

    bath : Bath specification or list of Bath specifications
        A bath containing the exponents of the expansion of the
        bath correlation funcion and their associated coefficients
        and coupling operators, or a list of baths.

        Each bath can be specified as *either* an object of type
        :class:`.Bath`, :class:`.BosonicBath`, :class:`.FermionicBath`, or
        their subtypes, *or* as a tuple ``(env, Q)``, where ``env`` is an
        :class:`.ExponentialBosonicEnvironment` or an
        :class:`.ExponentialFermionicEnvironment` and ``Q`` the system coupling
        operator.

    odd_parity : Bool
        For fermionic baths only. Default is "False". "Parity" refers to the
        parity of the initial system state used with the HEOM. An example of
        an odd parity state is one made from applying an odd number of
        fermionic creation operators to a physical density operator.
        Physical systems have even parity, but allowing the generalization
        to odd-parity states allows one to calculate useful physical quantities
        like the system power spectrum or density of states.
        The form of the HEOM differs depending on the parity of the initial
        system state, so if this option is set to "True", a different RHS is
        constructed, which can then be used with a system state of odd parity.

    max_depth : int
        The maximum depth of the hierarchy (i.e. the maximum number of bath
        exponent "excitations" to retain).

    options : dict, optional
        Generic solver options.
        If set to None the default options will be used. Keyword only.
        Default: None.

    Attributes
    ----------
    ados : :obj:`HierarchyADOs`
        The description of the hierarchy constructed from the given bath
        and maximum depth.

    rhs : :obj:`.QobjEvo`
        The right-hand side (RHS) of the hierarchy evolution ODE. Internally
        the system and bath coupling operators are converted to
        :class:`qutip.data.CSR` instances during construction of the RHS,
        so the operators in the ``rhs`` will all be sparse.
    """

    name = "heomsolver"
    _resultclass = HEOMResult
    _avail_integrators = {}
    solver_options = {
        "progress_bar": "text",
        "progress_kwargs": {"chunk_size": 10},
        "store_final_state": False,
        "store_states": None,
        "normalize_output": False,
        "method": "adams",
        "store_ados": False,
        "state_data_type": "dense",
    }

    def __init__(self, H, bath, max_depth, *, odd_parity=False, options=None):
        _time_start = time()
        # we call bool here because odd_parity will be used in arithmetic
        self.odd_parity = bool(odd_parity)
        if not isinstance(H, (Qobj, QobjEvo)):
            raise TypeError("The Hamiltonian (H) must be a Qobj or QobjEvo")

        H = QobjEvo(H)
        self.L_sys = (
            liouvillian(H) if H.type == "oper"  # hamiltonian
            else H  # already a liouvillian
        )

        self._sys_shape = int(np.sqrt(self.L_sys.shape[0]))
        self._sup_shape = self.L_sys.shape[0]
        self._sys_dims = self.L_sys.dims[0]

        self.ados = HierarchyADOs(
            self._combine_bath_exponents(bath), max_depth,
        )
        self._n_ados = len(self.ados.labels)
        self._n_exponents = len(self.ados.exponents)

        self._init_ados_time = time() - _time_start
        _time_start = time()

        with CoreOptions(default_dtype="csr"):
            # pre-calculate identity matrix required by _grad_n
            self._sId = _data.identity(self._sup_shape, dtype="csr")

            # pre-calculate superoperators required by _grad_prev and
            # _grad_next:
            Qs = [exp.Q.to("csr") for exp in self.ados.exponents]
            self._spreQ = [spre(op).data for op in Qs]
            self._spostQ = [spost(op).data for op in Qs]
            self._s_pre_minus_post_Q = [
                _data.sub(self._spreQ[k], self._spostQ[k])
                for k in range(self._n_exponents)
            ]
            self._s_pre_plus_post_Q = [
                _data.add(self._spreQ[k], self._spostQ[k])
                for k in range(self._n_exponents)
            ]
            self._spreQdag = [spre(op.dag()).data for op in Qs]
            self._spostQdag = [spost(op.dag()).data for op in Qs]
            self._s_pre_minus_post_Qdag = [
                _data.sub(self._spreQdag[k], self._spostQdag[k])
                for k in range(self._n_exponents)
            ]
            self._s_pre_plus_post_Qdag = [
                _data.add(self._spreQdag[k], self._spostQdag[k])
                for k in range(self._n_exponents)
            ]

            self._init_superop_cache_time = time() - _time_start
            _time_start = time()

            rhs = self._calculate_rhs()

        self._init_rhs_time = time() - _time_start

        super().__init__(rhs, options=options)

    @property
    def sys_dims(self):
        """
        Dimensions of the space that the system use, excluding any environment:

        ``qutip.basis(sovler.dims)`` will create a state with proper dimensions
        for this solver.
        """
        return self._sys_dims

    def _initialize_stats(self):
        stats = super()._initialize_stats()
        stats.update({
            "init time": sum([
                stats["init time"], self._init_ados_time,
                self._init_superop_cache_time, self._init_rhs_time,
            ]),
            "init ados time": self._init_ados_time,
            "init superop cache time": self._init_superop_cache_time,
            "init rhs time": self._init_rhs_time,
            "solver": "Hierarchical Equations of Motion Solver",
            "max_depth": self.ados.max_depth,
        })
        return stats

    def _combine_bath_exponents(self, bath):
        """ Combine the exponents for the specified baths. """
        # Only one bath provided, not a list of baths
        if (not isinstance(bath, (list, tuple))
                or self._is_environment_api(bath)):
            bath = [bath]
        bath = [self._to_bath(b) for b in bath]
        exponents = []
        for b in bath:
            exponents.extend(b.exponents)

        if not all(exp.Q.dims == exponents[0].Q.dims for exp in exponents):
            raise ValueError(
                "All bath exponents must have system coupling operators"
                " with the same dimensions but a mixture of dimensions"
                " was given."
            )
        return exponents

    def _is_environment_api(self, bath_spec):
        if not isinstance(bath_spec, (list, tuple)) or len(bath_spec) < 2:
            return False
        env, Q, *args = bath_spec

        if not isinstance(env, (BosonicEnvironment, FermionicEnvironment)):
            return False

        if not isinstance(Q, (Qobj, QobjEvo)):
            return False

        return True

    def _to_bath(self, bath_spec):
        if isinstance(bath_spec, (Bath, BosonicBath, FermionicBath)):
            return bath_spec

        if not self._is_environment_api(bath_spec):
            raise ValueError(
                "Environments must be passed as either Bath instances or"
                " as a tuple or list corresponding to an environment and a"
                " coupling operator, (env, Q)"
            )
        env, Q, *args = bath_spec

        if isinstance(env, ExponentialBosonicEnvironment):
            return BosonicBath.from_environment(env, Q, *args)
        if isinstance(env, ExponentialFermionicEnvironment):
            return FermionicBath.from_environment(env, Q, *args)
        raise ValueError("The HEOM solver requires the environment to have"
                        " a multi-exponential correlation function. Use"
                        " the `approximate` function to generate a"
                        " multi-exponential approximation.")

    def _grad_n(self, he_n):
        """ Get the gradient for the hierarchy ADO at level n. """
        vk = self.ados.vk
        vk_sum = sum(he_n[i] * vk[i] for i in range(len(vk)))
        op = _data.mul(self._sId, -vk_sum)
        return op

    def _grad_prev(self, he_n, k):
        """ Get the previous gradient. """
        if self.ados.exponents[k].fermionic:
            return self._grad_prev_fermionic(he_n, k)
        else:
            return self._grad_prev_bosonic(he_n, k)

    def _grad_prev_bosonic(self, he_n, k):
        if self.ados.exponents[k].type == BathExponent.types.R:
            op = _data.mul(
                self._s_pre_minus_post_Q[k],
                -1j * he_n[k] * self.ados.ck[k],
            )
        elif self.ados.exponents[k].type == BathExponent.types.I:
            op = _data.mul(
                self._s_pre_plus_post_Q[k],
                -1j * he_n[k] * 1j * self.ados.ck[k],
            )
        elif self.ados.exponents[k].type == BathExponent.types.RI:
            term1 = _data.mul(
                self._s_pre_minus_post_Q[k],
                he_n[k] * -1j * self.ados.ck[k],
            )
            term2 = _data.mul(
                self._s_pre_plus_post_Q[k],
                he_n[k] * self.ados.ck2[k],
            )
            op = _data.add(term1, term2)
        else:
            raise ValueError(
                f"Unsupported type {self.ados.exponents[k].type}"
                f" for exponent {k}"
            )
        return op

    def _grad_prev_fermionic(self, he_n, k):
        ck = self.ados.ck
        he_fermionic_n = [
            i * int(exp.fermionic)
            for i, exp in zip(he_n, self.ados.exponents)
        ]

        n_excite = sum(he_fermionic_n)
        sign1 = (-1) ** (n_excite + 1 - self.odd_parity)

        n_excite_before_m = sum(he_fermionic_n[:k])
        sign2 = (-1) ** (n_excite_before_m + self.odd_parity)

        sigma_bar_k = k + self.ados.sigma_bar_k_offset[k]

        if self.ados.exponents[k].type == BathExponent.types["+"]:
            op = _data.sub(
                _data.mul(self._spreQdag[k], -1j * sign2 * ck[k]),
                _data.mul(
                    self._spostQdag[k],
                    -1j * sign2 * sign1 * np.conj(ck[sigma_bar_k]),
                ),
            )
        elif self.ados.exponents[k].type == BathExponent.types["-"]:
            op = _data.sub(
                _data.mul(self._spreQ[k], -1j * sign2 * ck[k]),
                _data.mul(
                    self._spostQ[k],
                    -1j * sign2 * sign1 * np.conj(ck[sigma_bar_k]),
                ),
            )
        else:
            raise ValueError(
                f"Unsupported type {self.ados.exponents[k].type}"
                f" for exponent {k}"
            )
        return op

    def _grad_next(self, he_n, k):
        """ Get the previous gradient. """
        if self.ados.exponents[k].fermionic:
            return self._grad_next_fermionic(he_n, k)
        else:
            return self._grad_next_bosonic(he_n, k)

    def _grad_next_bosonic(self, he_n, k):
        op = _data.mul(self._s_pre_minus_post_Q[k], -1j)
        return op

    def _grad_next_fermionic(self, he_n, k):
        he_fermionic_n = [
            i * int(exp.fermionic)
            for i, exp in zip(he_n, self.ados.exponents)
        ]
        n_excite = sum(he_fermionic_n)
        sign1 = (-1) ** (n_excite + 1 - self.odd_parity)

        n_excite_before_m = sum(he_fermionic_n[:k])
        sign2 = (-1) ** (n_excite_before_m + self.odd_parity)

        if self.ados.exponents[k].type == BathExponent.types["+"]:
            if sign1 == -1:
                op = _data.mul(self._s_pre_minus_post_Q[k], -1j * sign2)
            else:
                op = _data.mul(self._s_pre_plus_post_Q[k], -1j * sign2)
        elif self.ados.exponents[k].type == BathExponent.types["-"]:
            if sign1 == -1:
                op = _data.mul(self._s_pre_minus_post_Qdag[k], -1j * sign2)
            else:
                op = _data.mul(self._s_pre_plus_post_Qdag[k], -1j * sign2)
        else:
            raise ValueError(
                f"Unsupported type {self.ados.exponents[k].type}"
                f" for exponent {k}"
            )
        return op

    def _rhs(self):
        """ Make the RHS for the HEOM. """
        ops = _GatherHEOMRHS(
            self.ados.idx, block=self._sup_shape, nhe=self._n_ados
        )

        for he_n in self.ados.labels:
            op = self._grad_n(he_n)
            ops.add_op(he_n, he_n, op)
            for k in range(len(self.ados.dims)):
                next_he = self.ados.next(he_n, k)
                if next_he is not None:
                    op = self._grad_next(he_n, k)
                    ops.add_op(he_n, next_he, op)
                prev_he = self.ados.prev(he_n, k)
                if prev_he is not None:
                    op = self._grad_prev(he_n, k)
                    ops.add_op(he_n, prev_he, op)

        return ops.gather()

    def _calculate_rhs(self):
        """ Make the full RHS required by the solver. """
        rhs_mat = self._rhs()
        rhs_dims = [
            [self._sup_shape * self._n_ados], [self._sup_shape * self._n_ados]
        ]
        h_identity = _data.identity(self._n_ados, dtype="csr")

        if self.L_sys.isconstant:
            # For the constant case, we just add the Liouvillian to the
            # diagonal blocks of the RHS matrix.
            rhs_mat += _data.kron(h_identity, self.L_sys(0).to("csr").data)
            rhs = QobjEvo(Qobj(rhs_mat, dims=rhs_dims))
        else:
            # In the time dependent case, we construct the parameters
            # for the ODE gradient function under the assumption that
            #
            # RHSmat(t) = RHSmat + time dependent terms that only affect the
            # diagonal blocks of the RHS matrix.
            #
            # This assumption holds because only _grad_n dependents on
            # the system Liouvillian (and not _grad_prev or _grad_next) and
            # the bath coupling operators are not time-dependent.
            rhs = QobjEvo(Qobj(rhs_mat, dims=rhs_dims))

            def _kron(x):
                return Qobj(
                    _data.kron(h_identity, x.data),
                    dims=rhs_dims,
                ).to("csr")

            rhs += self.L_sys.linear_map(_kron)

        # The assertion that rhs_mat has data type CSR is just a sanity
        # check on the RHS creation. The base solver class will still
        # convert the RHS to the type required by the ODE integrator if
        # needed.
        assert isinstance(rhs_mat, _csr.CSR)
        assert isinstance(rhs, QobjEvo)
        assert rhs.dims == rhs_dims

        return rhs

    def steady_state(
        self,
        use_mkl=True, mkl_max_iter_refine=100, mkl_weighted_matching=False
    ):
        """
        Compute the steady state of the system.

        Parameters
        ----------
        use_mkl : bool, default=False
            Whether to use mkl or not. If mkl is not installed or if
            this is false, use the scipy splu solver instead.

        mkl_max_iter_refine : int
            Specifies the the maximum number of iterative refinement steps that
            the MKL PARDISO solver performs.

            For a complete description, see iparm(7) in
            https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/pardiso-iparm-parameter.html

        mkl_weighted_matching : bool
            MKL PARDISO can use a maximum weighted matching algorithm to
            permute large elements close the diagonal. This strategy adds an
            additional level of reliability to the factorization methods.

            For a complete description, see iparm(12) in
            https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/pardiso-iparm-parameter.html

        Returns
        -------
        steady_state : Qobj
            The steady state density matrix of the system.

        steady_ados : :class:`HierarchyADOsState`
            The steady state of the full ADO hierarchy. A particular ADO may be
            extracted from the full state by calling
            :meth:`extract`.
        """
        if not self.L_sys.isconstant:
            raise ValueError(
                "A steady state cannot be determined for a time-dependent"
                " system"
            )
        n = self._sys_shape

        b_mat = np.zeros(n ** 2 * self._n_ados, dtype=complex)
        b_mat[0] = 1.0

        L = self.rhs(0).to("CSR").data.copy().as_scipy()
        L = L.tolil()
        L[0, 0: n ** 2 * self._n_ados] = 0.0
        L = L.tocsr()
        L += sp.csr_matrix((
            np.ones(n),
            (np.zeros(n), [num * (n + 1) for num in range(n)])
        ), shape=(n ** 2 * self._n_ados, n ** 2 * self._n_ados))

        if mkl_spsolve is not None and use_mkl:
            L.sort_indices()
            solution = mkl_spsolve(
                L,
                b_mat,
                perm=None,
                verbose=False,
                max_iter_refine=mkl_max_iter_refine,
                scaling_vectors=True,
                weighted_matching=mkl_weighted_matching,
            )
        else:
            L = L.tocsc()
            solution = spsolve(L, b_mat)

        data = _data.Dense(solution[:n ** 2].reshape((n, n), order='F'))
        data = _data.mul(_data.add(data, data.adjoint()), 0.5)
        steady_state = Qobj(data, dims=self._sys_dims)

        solution = solution.reshape((self._n_ados, n, n))
        steady_ados = HierarchyADOsState(steady_state, self.ados, solution)

        return steady_state, steady_ados

    def run(self, state0, tlist, *, args=None, e_ops=None):
        """
        Solve for the time evolution of the system.

        Parameters
        ----------
        state0 : :obj:`.Qobj` or :class:`~HierarchyADOsState` or array-like
            If ``rho0`` is a :obj:`.Qobj` the it is the initial state
            of the system (i.e. a :obj:`.Qobj` density matrix).

            If it is a :class:`~HierarchyADOsState` or array-like, then
            ``rho0`` gives the initial state of all ADOs.

            Usually the state of the ADOs would be determine from a previous
            call to ``.run(...)`` with the solver results option ``store_ados``
            set to True. For example, ``result = solver.run(...)`` could be
            followed by ``solver.run(result.ado_states[-1], tlist)``.

            If a numpy array-like is passed its shape must be
            ``(number_of_ados, n, n)`` where ``(n, n)`` is the system shape
            (i.e. shape of the system density matrix) and the ADOs must
            be in the same order as in ``.ados.labels``.

        tlist : list
            An ordered list of times at which to return the value of the state.

        args : dict, optional {None}
            Change the ``args`` of the RHS for the evolution.

        e_ops : Qobj / QobjEvo / callable / list / dict / None, optional
            A list or dictionary of operators as :obj:`.Qobj`,
            :obj:`.QobjEvo` and/or callable functions (they can be mixed) or
            a single operator or callable function. For an operator ``op``, the
            result will be computed using ``(state * op).tr()`` and the state
            at each time ``t``. For callable functions, ``f``, the result is
            computed using ``f(t, ado_state)``. The values are stored in the
            ``expect`` and ``e_data`` attributes of the result (see the return
            section below).

        Returns
        -------
        :class:`~HEOMResult`
            The results of the simulation run, with the following important
            attributes:

            * ``times``: the times ``t`` (i.e. the ``tlist``).

            * ``states``: the system state at each time ``t`` (only available
              if ``e_ops`` was ``None`` or if the solver option
              ``store_states`` was set to ``True``).

            * ``ado_states``: the full ADO state at each time (only available
              if the results option ``ado_return`` was set to ``True``).
              Each element is an instance of :class:`HierarchyADOsState`.
              The state of a particular ADO may be extracted from
              ``result.ado_states[i]`` by calling :meth:`extract`.

            * ``expect``: a list containing the values of each ``e_ops`` at
              time ``t``.

            * ``e_data``: a dictionary containing the values of each ``e_ops``
              at tme ``t``. The keys are those given by ``e_ops`` if it was
              a dict, otherwise they are the indexes of the supplied ``e_ops``.

            See :class:`~HEOMResult` and :class:`.Result` for the complete
            list of attributes.
        """
        return super().run(state0, tlist, args=args, e_ops=e_ops)

    def _prepare_state(self, state):
        n = self._sys_shape
        rho_dims = self._sys_dims
        hierarchy_shape = (self._n_ados, n, n)

        rho0 = state
        ado_init = not isinstance(rho0, Qobj)

        if ado_init:
            if isinstance(rho0, HierarchyADOsState):
                rho0_he = rho0._ado_state
            elif hasattr(rho0, "shape"):  # array-like
                rho0_he = rho0
            else:
                raise TypeError(
                    f"Initial ADOs passed have type {type(rho0)}"
                    " but a HierarchyADOsState or a numpy array-like instance"
                    " was expected"
                )
            if rho0_he.shape != hierarchy_shape:
                raise ValueError(
                    f"Initial ADOs passed have shape {rho0_he.shape}"
                    f" but the solver hierarchy shape is {hierarchy_shape}"
                )
            rho0_he = rho0_he.reshape(n ** 2 * self._n_ados)
            rho0_he = _data.create(rho0_he)
        else:
            if rho0.dims != rho_dims:
                raise ValueError(
                    f"Initial state rho has dims {rho0.dims}"
                    f" but the system dims are {rho_dims}"
                )
            rho0_he = np.zeros([n ** 2 * self._n_ados], dtype=complex)
            rho0_he[:n ** 2] = rho0.full().ravel('F')
            rho0_he = _data.create(rho0_he)

        if self.options["state_data_type"]:
            rho0_he = _data.to(self.options["state_data_type"], rho0_he)

        return rho0_he

    def _restore_state(self, state, *, copy=True):
        n = self._sys_shape
        rho_shape = (n, n)
        rho_dims = self._sys_dims
        hierarchy_shape = (self._n_ados, n, n)

        rho = Qobj(
            state.to_array()[:n ** 2].reshape(rho_shape, order='F'),
            dims=rho_dims,
        )
        ado_state = HierarchyADOsState(
            rho, self.ados, state.to_array().reshape(hierarchy_shape)
        )
        return ado_state

    def start(self, state0, t0):
        """
        Set the initial state and time for a step evolution.

        Parameters
        ----------
        state0 : :obj:`.Qobj`
            Initial state of the evolution. This may provide either just the
            initial density matrix of the system, or the full set of ADOs
            for the hierarchy. See the documentation for ``rho0`` in the
            ``.run(...)`` method for details.

        t0 : double
            Initial time of the evolution.
        """
        super().start(state0, t0)

    @property
    def options(self):
        """
        Options for HEOMSolver:

        store_final_state: bool, default: False
            Whether or not to store the final state of the evolution in the
            result class.

        store_states: bool, default: None
            Whether or not to store the state vectors or density matrices.
            On `None` the states will be saved if no expectation operators are
            given.

        normalize_output: bool, default: False
            Normalize output state to hide ODE numerical errors.

        progress_bar: str {'text', 'enhanced', 'tqdm', ''}, default: "text"
            How to present the solver progress.
            'tqdm' uses the python module of the same name and raise an error
            if not installed. Empty string or False will disable the bar.

        progress_kwargs: dict, default: {"chunk_size": 10}
            Arguments to pass to the progress_bar. Qutip's bars use
            ``chunk_size``.

        method: str, default: "adams"
            Which ordinary differential equation integration method to use.

        state_data_type: str, default: "dense"
            Name of the data type of the state used during the ODE evolution.
            Use an empty string to keep the input state type. Many integrators
            support only work with `Dense`.

        store_ados : bool, default: False
            Whether or not to store the HEOM ADOs. Only relevant when using
            the HEOM solver.
        """
        return self._options

    @options.setter
    def options(self, new_options):
        Solver.options.fset(self, new_options)


class HSolverDL(HEOMSolver):
    """
    A helper class for creating an :class:`HEOMSolver` that is backwards
    compatible with the ``HSolverDL`` provided in ``qutip.nonmarkov.heom``
    in QuTiP 4.6 and below.

    See :class:`HEOMSolver` and :class:`DrudeLorentzBath` for more
    descriptions of the underlying solver and bath construction.

    .. note::

        Unlike the version of ``HSolverDL`` in QuTiP 4.6, this solver
        supports supplying a time-dependent or Liouvillian ``H_sys``.

    .. note::

        For compatibility with ``HSolverDL`` in QuTiP 4.6 and below, the
        parameter ``N_exp`` specifying the number of exponents to keep in
        the expansion of the bath correlation function is one more than
        the equivalent ``Nk`` used in the :class:`DrudeLorentzBath`. I.e.,
        ``Nk = N_exp - 1``. The ``Nk`` parameter in the
        :class:`DrudeLorentzBath` does not count the zeroeth exponent in
        order to better match common usage in the literature.

    .. note::

        The ``stats`` and ``renorm`` arguments accepted in QuTiP 4.6 and below
        are no longer supported.

    Parameters
    ----------
    H_sys : Qobj or QobjEvo or list
        The system Hamiltonian or Liouvillian. See :class:`HEOMSolver` for
        a complete description.

    coup_op : Qobj
        Operator describing the coupling between system and bath.
        See parameter ``Q`` in :class:`BosonicBath` for a complete description.

    coup_strength : float
        Coupling strength. Referred to as ``lam`` in
        :class:`.DrudeLorentzEnvironment`.

    temperature : float
        Bath temperature. Referred to as ``T`` in
        :class:`.DrudeLorentzEnvironment`.

    N_cut : int
        The maximum depth of the hierarchy. See ``max_depth`` in
        :class:`HEOMSolver` for a full description.

    N_exp : int
        Number of exponential terms used to approximate the bath correlation
        functions. The equivalent ``Nk`` in :class:`DrudeLorentzBath` is one
        less than ``N_exp`` (see note above).

    cut_freq : float
        Bath spectral density cutoff frequency. Referred to as ``gamma`` in
        :class:`.DrudeLorentzEnvironment`.

    bnd_cut_approx : bool
        Use boundary cut off approximation. If true, the Matsubara
        terminator is added to the system Liouvillian (and H_sys is
        promoted to a Liouvillian if it was a Hamiltonian). Keyword only.
        Default: False.

    options : dict, optional
        Generic solver options.
        If set to None the default options will be used. Keyword only.
        Default: None.

    combine : bool, default: True
        Whether to combine exponents with the same frequency (and coupling
        operator). See :meth:`.ExponentialBosonicEnvironment.combine` for
        details.
        Keyword only. Default: True.
    """

    def __init__(
        self, H_sys, coup_op, coup_strength, temperature,
        N_cut, N_exp, cut_freq, *, bnd_cut_approx=False, options=None,
        combine=True,
    ):
        H_sys = QobjEvo(H_sys)
        bath = DrudeLorentzBath(
            Q=coup_op,
            lam=coup_strength,
            gamma=cut_freq,
            Nk=N_exp - 1,
            T=temperature,
            combine=combine,
        )

        if bnd_cut_approx:
            # upgrade H_sys to a Liouvillian if needed and add the
            # bath terminator
            if H_sys.type == "oper":  # H_sys is a Hamiltonian
                H_sys = liouvillian(H_sys)
            _, terminator = bath.terminator()
            H_sys = H_sys + terminator

        super().__init__(H_sys, bath=bath, max_depth=N_cut, options=options)

        # store input parameters as attributes for politeness and compatibility
        # with HSolverDL in QuTiP 4.6 and below.
        self.coup_strength = coup_strength
        self.cut_freq = cut_freq
        self.temperature = temperature
        self.N_exp = N_exp
        self.bnd_cut_approx = bnd_cut_approx


class _GatherHEOMRHS:
    """ A class for collecting elements of the right-hand side matrix
        of the HEOM.

        Parameters
        ----------
        f_idx: function(he_state) -> he_idx
            A function that returns the index of a hierarchy state
            (i.e. an ADO label).
        block : int
            The size of a single ADO Liovillian operator in the hierarchy.
        nhe : int
            The number of ADOs in the hierarchy.
    """

    def __init__(self, f_idx, block, nhe):
        self._block_size = block
        self._n_blocks = nhe
        self._f_idx = f_idx
        self._ops = []

    def add_op(self, row_he, col_he, op):
        """ Add an block operator to the list. """
        self._ops.append(
            (self._f_idx(row_he), self._f_idx(col_he), op)
        )

    def gather(self):
        """ Create the HEOM liouvillian from a sorted list of smaller sparse
            matrices.

            .. note::

                The list of operators contains tuples of the form
                ``(row_idx, col_idx, op)``. The row_idx and col_idx give the
                *block* row and column for each op. An operator with
                block indices ``(N, M)`` is placed at position
                ``[N * block: (N + 1) * block, M * block: (M + 1) * block]``
                in the output matrix.

            Returns
            -------
            rhs : :obj:`Data`
                A combined matrix of shape ``(block * nhe, block * ne)``.
        """
        self._ops.sort()
        ops = np.array(self._ops, dtype=[
            ("row", _data.base.idxint_dtype),
            ("col", _data.base.idxint_dtype),
            ("op", _data.CSR),
        ])
        return _csr._from_csr_blocks(
            ops["row"], ops["col"], ops["op"],
            self._n_blocks, self._block_size,
        )
