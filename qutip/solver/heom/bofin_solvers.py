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
from scipy.sparse.linalg import spsolve

from qutip.settings import settings
from qutip import qeye, qeye_like, state_number_enumerate, CoreOptions
from qutip.core import Qobj, QobjEvo
from qutip.core import data as _data
from qutip.core.dimensions import Dimensions, Field, SumSpace, SuperSpace
from qutip.core.direct_sum import direct_component, direct_sum_sparse
from qutip.core.environment import (
    BosonicEnvironment, FermionicEnvironment,
    ExponentialBosonicEnvironment, ExponentialFermionicEnvironment
)
from qutip.core.superoperator import (
    liouvillian, operator_to_vector, spre, spost, vector_to_operator
)
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
    ado_state : :class:`.Qobj`
        The full state of the hierarchy. This is a direct sum :class:`.Qobj`
        comprising the operator-kets for the state and all ADOs.

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
        # initialize from numpy array ado_state
        if not isinstance(ado_state, Qobj) and hasattr(ado_state, "shape"):
            n_ados = len(ados.labels)
            ado_space = SuperSpace(rho._dims)
            ado_state = Qobj(
                ado_state.reshape(ado_space.size * n_ados),
                dims=Dimensions(Field(), SumSpace(ado_space, repeat=n_ados))
            )

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
        return vector_to_operator(direct_component(self._ado_state, idx, 0))


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
        operator. The system coupling operator may also be time-dependent.

    max_depth : int
        The maximum depth of the heirarchy (i.e. the maximum number of bath
        exponent "excitations" to retain).

    state0 : :obj:`.Qobj` or :class:`~HierarchyADOsState` or array-like
        If ``rho0`` is a :obj:`.Qobj` with the same dimensions as ``H``, then
        it is the initial state of the system (i.e. a :obj:`.Qobj` density
        matrix).

        If it is a :class:`~HierarchyADOsState`, direct sum :obj:`.Qobj`, or
        array-like, then ``rho0`` gives the initial state of all ADOs.

        Usually the state of the ADOs would be determine from a previous
        call to ``.run(...)`` with the solver results option ``store_ados``
        set to True. For example, ``result = solver.run(...)`` could be
        followed by ``solver.run(result.ado_states[-1], tlist)``.

        If a direct sum :obj:`.Qobj` is passed, it must have one component for
        each ADO, and the components are the ADOs in operator-ket form in the
        same order as in ``.ados.labels``.

        If a numpy array-like is passed, its shape must be
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
        Dictionary of options for the solver.

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
        - | rhs_data_type: str {'dense', 'CSR', 'Dia', }
          | Name of the data type used to store the generator. For typical
            applications, it is strongly recommended to use the default 'CSR'.
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

    Notes
    -----
    If the Hamiltonian or any coupling operators are time-dependent, for
    performance reasons, avoid constructing those :class:`.QobjEvo` from
    callables, and prefer instead the list format
    ``[[Qobj, coefficient], ...]``.
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
        operator. The system coupling operator may also be time-dependent.

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
        Options for the solver, see :obj:`HEOMSolver.options` for a list of all
        options. If set to None the default options will be used. Keyword only.
        Default: None.

    Attributes
    ----------
    ados : :obj:`HierarchyADOs`
        The description of the hierarchy constructed from the given bath
        and maximum depth.

    rhs : :obj:`.QobjEvo`
        The right-hand side (RHS) of the hierarchy evolution ODE.

    Notes
    -----
    If the Hamiltonian or any coupling operators are time-dependent, for
    performance reasons, avoid constructing those :class:`.QobjEvo` from
    callables, and prefer instead the list format
    ``[[Qobj, coefficient], ...]``.
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
        "rhs_data_type": "CSR",
        "state_data_type": "dense",
    }

    def __init__(self, H, bath, max_depth, *, odd_parity=False, options=None):
        _time_start = time()
        # we call bool here because odd_parity will be used in arithmetic
        self.odd_parity = bool(odd_parity)
        if not isinstance(H, (Qobj, QobjEvo)):
            raise TypeError("The Hamiltonian (H) must be a Qobj or QobjEvo")

        H = QobjEvo(H)
        if H.isconstant:
            H = H(0)
        self.L_sys = (
            liouvillian(H) if H.type == "oper"  # hamiltonian
            else H  # already a liouvillian
        )
        if isinstance(self.L_sys, QobjEvo):
            self.L_sys.compress(_skip_coeff=True)

        # Dimensions that L_sys acts on
        self._sys_dims_ = self.L_sys._dims[0].oper
        # Underlying Hilbert space dimension
        self._sys_shape = self._sys_dims_[0].size

        self.ados = HierarchyADOs(
            self._combine_bath_exponents(bath), max_depth,
        )
        self._n_ados = len(self.ados.labels)
        self._n_exponents = len(self.ados.exponents)

        # Direct sum space for all ADOs
        sum_space = SumSpace(self.L_sys._dims[0], repeat=self._n_ados)
        self._rhs_dims = Dimensions(sum_space, sum_space)
        self._ado_dims = Dimensions(Field(), sum_space)

        self._init_ados_time = time() - _time_start
        _time_start = time()

        rhs_dtype = _data._parse_default_dtype(
            None if options is None else options.get("rhs_data_type", None),
            "sparse"
        )
        with CoreOptions(default_dtype=rhs_dtype):
            # pre-calculate identity matrix required by _grad_n
            self._sId = qeye_like(self.L_sys)

            # pre-calculate superoperators required by _grad_prev and
            # _grad_next:
            Qs = [exp.Q.to(rhs_dtype) for exp in self.ados.exponents]
            self._spreQ = [spre(op) for op in Qs]
            self._spostQ = [spost(op) for op in Qs]
            self._s_pre_minus_post_Q = [
                self._spreQ[k] - self._spostQ[k]
                for k in range(self._n_exponents)
            ]
            self._s_pre_plus_post_Q = [
                self._spreQ[k] + self._spostQ[k]
                for k in range(self._n_exponents)
            ]
            self._spreQdag = [spre(op.dag()) for op in Qs]
            self._spostQdag = [spost(op.dag()) for op in Qs]
            self._s_pre_minus_post_Qdag = [
                self._spreQdag[k] - self._spostQdag[k]
                for k in range(self._n_exponents)
            ]
            self._s_pre_plus_post_Qdag = [
                self._spreQdag[k] + self._spostQdag[k]
                for k in range(self._n_exponents)
            ]

            self._init_superop_cache_time = time() - _time_start
            _time_start = time()

            rhs = self._calculate_rhs()

        self._init_rhs_time = time() - _time_start

        super().__init__(rhs, options=options)

    @property
    def _sys_dims(self):
        return self._sys_dims_

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

        if not all(exp.Q._dims == exponents[0].Q._dims for exp in exponents):
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
        return -vk_sum * self._sId

    def _grad_prev(self, he_n, k):
        """ Get the previous gradient. """
        if self.ados.exponents[k].fermionic:
            op = self._grad_prev_fermionic(he_n, k)
        else:
            op = self._grad_prev_bosonic(he_n, k)
        if isinstance(op, QobjEvo):
            op.compress(_skip_coeff=True)
        return op

    def _grad_prev_bosonic(self, he_n, k):
        if self.ados.exponents[k].type == BathExponent.types.R:
            return (
                -1j * he_n[k] * self.ados.ck[k]
                * self._s_pre_minus_post_Q[k]
            )
        if self.ados.exponents[k].type == BathExponent.types.I:
            return (
                 he_n[k] * self.ados.ck[k]
                 * self._s_pre_plus_post_Q[k]
            )
        if self.ados.exponents[k].type == BathExponent.types.RI:
            term1 = (
                -1j * he_n[k] * self.ados.ck[k]
                * self._s_pre_minus_post_Q[k]
            )
            term2 = (
                he_n[k] * self.ados.ck2[k]
                * self._s_pre_plus_post_Q[k]
            )
            return term1 + term2
        raise ValueError(
            f"Unsupported type {self.ados.exponents[k].type}"
            f" for exponent {k}"
        )

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
            term1 = -1j * sign2 * ck[k] * self._spreQdag[k]
            term2 = (
                -1j * sign2 * sign1 * np.conj(ck[sigma_bar_k])
                * self._spostQdag[k]
            )
            return term1 - term2
        elif self.ados.exponents[k].type == BathExponent.types["-"]:
            term1 = -1j * sign2 * ck[k] * self._spreQ[k]
            term2 = (
                -1j * sign2 * sign1 * np.conj(ck[sigma_bar_k])
                * self._spostQ[k]
            )
            return term1 - term2
        raise ValueError(
            f"Unsupported type {self.ados.exponents[k].type}"
            f" for exponent {k}"
        )

    def _grad_next(self, he_n, k):
        """ Get the previous gradient. """
        if self.ados.exponents[k].fermionic:
            op = self._grad_next_fermionic(he_n, k)
        else:
            op = self._grad_next_bosonic(he_n, k)
        if isinstance(op, QobjEvo):
            op.compress(_skip_coeff=True)
        return op

    def _grad_next_bosonic(self, he_n, k):
        return -1j * self._s_pre_minus_post_Q[k]

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
                return -1j * sign2 * self._s_pre_minus_post_Q[k]
            return -1j * sign2 * self._s_pre_plus_post_Q[k]
        elif self.ados.exponents[k].type == BathExponent.types["-"]:
            if sign1 == -1:
                return -1j * sign2 * self._s_pre_minus_post_Qdag[k]
            return -1j * sign2 * self._s_pre_plus_post_Qdag[k]
        raise ValueError(
            f"Unsupported type {self.ados.exponents[k].type}"
            f" for exponent {k}"
        )

    def _calculate_rhs(self):
        """ Make the full RHS required by the solver. """
        idx = self.ados.idx

        ops = {}
        for he_n in self.ados.labels:
            ops[(idx(he_n), idx(he_n))] = self._grad_n(he_n)
            for k in range(len(self.ados.dims)):
                next_he = self.ados.next(he_n, k)
                if next_he is not None:
                    ops[(idx(he_n), idx(next_he))] = self._grad_next(he_n, k)
                prev_he = self.ados.prev(he_n, k)
                if prev_he is not None:
                    ops[(idx(he_n), idx(prev_he))] = self._grad_prev(he_n, k)

        if isinstance(self.L_sys, QobjEvo):
            liouvillian_part = self.L_sys.linear_map(self._liouvillian_part)
        else:
            liouvillian_part = self._liouvillian_part(self.L_sys)

        return liouvillian_part + direct_sum_sparse(ops, self._rhs_dims)

    def _liouvillian_part(self, L):
        return direct_sum_sparse(
            {(idx, idx): L for idx in range(self._n_ados)}, self._rhs_dims
        )

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
        if not self.rhs.isconstant:
            raise ValueError(
                "A steady state cannot be determined for a time-dependent"
                " system"
            )

        L = self.rhs(0).to("CSR").data
        trace = direct_sum_sparse(
            {(0, 0): operator_to_vector(qeye(self._sys_dims[0]))},
            self._ado_dims, dtype="CSR"
        ).dag().data
        L = _data.block_overwrite_csr(L, trace, 0, 0).as_scipy()

        b_mat = np.zeros(self._ado_dims[0].size, dtype=complex)
        b_mat[0] = 1.0

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

        all_ados = Qobj(solution, dims=self._ado_dims, dtype='dense')
        steady_state = vector_to_operator(direct_component(all_ados, 0, 0))
        steady_state = (steady_state + steady_state.dag()) / 2
        steady_ados = HierarchyADOsState(steady_state, self.ados, all_ados)

        return steady_state, steady_ados

    def run(self, state0, tlist, *, args=None, e_ops=None):
        """
        Solve for the time evolution of the system.

        Parameters
        ----------
        state0 : :obj:`.Qobj` or :class:`~HierarchyADOsState` or array-like
            If ``rho0`` is a :obj:`.Qobj` with the same dimensions as ``H``, then
            it is the initial state of the system (i.e. a :obj:`.Qobj` density
            matrix).

            If it is a :class:`~HierarchyADOsState`, direct sum :obj:`.Qobj`, or
            array-like, then ``rho0`` gives the initial state of all ADOs.

            Usually the state of the ADOs would be determine from a previous
            call to ``.run(...)`` with the solver results option ``store_ados``
            set to True. For example, ``result = solver.run(...)`` could be
            followed by ``solver.run(result.ado_states[-1], tlist)``.

            If a direct sum :obj:`.Qobj` is passed, it must have one component for
            each ADO, and the components are the ADOs in operator-ket form in the
            same order as in ``.ados.labels``.

            If a numpy array-like is passed, its shape must be
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
        ado_init = not (
            isinstance(state, Qobj) and state._dims == self._sys_dims_
        )

        if ado_init:
            if isinstance(state, HierarchyADOsState):
                rho0_he = state._ado_state
            elif isinstance(state, Qobj):
                if state._dims == self._ado_dims:
                    rho0_he = state
                else:
                    raise ValueError(
                        f"Initial state rho has dims {state.dims}"
                        f" but the system dims are {self._sys_dims_}"
                        f" and the ADO dims are {self._ado_dims}")
            elif hasattr(state, "shape"):  # array-like
                hierarchy_shape = (
                    self._n_ados, self._sys_shape, self._sys_shape)
                if state.shape != hierarchy_shape:
                    raise ValueError(
                        f"Initial ADOs passed have shape {state.shape}"
                        f" but the solver hierarchy shape is {hierarchy_shape}"
                    )
                rho0_he = HierarchyADOsState(
                    # Reuse code for numpy ADOs -> QObj ADOs conversion.
                    # The first argument here doesn't matter,
                    # as long as it has the correct _dims.
                    qeye(self._sys_dims_[0]), self.ados, state
                )._ado_state
            else:
                raise TypeError(
                    f"Initial ADOs passed have type {type(state)}"
                    " but a Qobj, HierarchyADOsState or a numpy array-like"
                    " instance was expected"
                )
        else:
            rho0_he = direct_sum_sparse(
                {(0, 0): operator_to_vector(state)}, self._ado_dims
            )

        rho0_he = rho0_he.data
        if self.options["state_data_type"]:
            rho0_he = _data.to(self.options["state_data_type"], rho0_he)

        return rho0_he

    def _restore_state(self, state, *, copy=True):
        ado_state = Qobj(state, self._ado_dims, copy=copy)
        rho = vector_to_operator(direct_component(ado_state, 0, 0))
        return HierarchyADOsState(rho, self.ados, ado_state)

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

        rhs_data_type: str {'dense', 'CSR', 'Dia'}, default: 'CSR'
            Name of the data type used to store the generator. For typical
            applications, it is strongly recommended to use the default 'CSR'.

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

        if self.options["rhs_data_type"]:
            self.rhs.to(self.options["rhs_data_type"])


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
