"""
This module provides solvers for system-bath evoluation using the
HEOM (hierarchy equations of motion).

See https://en.wikipedia.org/wiki/Hierarchical_equations_of_motion for a very
basic introduction to the technique.

The implementation is derived from the BoFiN library (see
https://github.com/tehruhn/bofin) which was itself derived from an earlier
implementation in QuTiP itself.
"""

from copy import deepcopy

import numpy as np
import scipy.sparse as sp
import scipy.integrate
from scipy.sparse.linalg import spsolve

from qutip import settings
from qutip import state_number_enumerate
from qutip.qobj import Qobj
from qutip.qobjevo import QobjEvo
from qutip.superoperator import liouvillian, spre, spost, vec2mat
from qutip.cy.spmatfuncs import cy_ode_rhs
from qutip.solver import Options, Result
from qutip.cy.spconvert import dense2D_to_fastcsr_fmode
from qutip.ui.progressbar import BaseProgressBar, TextProgressBar
from qutip.fastsparse import fast_identity, fast_csr_matrix
from qutip.nonmarkov.bofin_baths import (
    BathExponent, DrudeLorentzBath,
)

# Load MKL spsolve if avaiable
if settings.has_mkl:
    from qutip._mkl.spsolve import mkl_spsolve
else:
    mkl_spsolve = None


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
    exponents : list of BathExponent
        The exponents of the correlation function describing the bath or
        baths.

    max_depth : int
        The maximum depth of the hierarchy (i.e. the maximum sum of
        "excitations" in the hierarchy ADO labels or maximum ADO level).

    Attributes
    ----------
    exponents : list of BathExponent
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
        tuple of BathExponent
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
            instead of by the actual type (e.g. ``BathExponent.types.R``).

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
    rho : :class:`Qobj`
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
        Extract a Qobj representing specified ADO from a full representation of
        the ADO states.

        Parameters
        ----------
        idx : int or label
            The index of the ADO to extract. If an ADO label, e.g.
            ``(0, 1, 0, ...)`` is supplied instead, then the ADO
            is extracted by label instead.

        Returns
        -------
        Qobj
            A :obj:`Qobj` representing the state of the specified ADO.
        """
        if isinstance(idx_or_label, int):
            idx = idx_or_label
        else:
            idx = self._ados.idx(idx_or_label)
        return Qobj(self._ado_state[idx, :].T, dims=self.rho.dims)


class HEOMSolver:
    """
    HEOM solver that supports multiple baths.

    The baths must be all either bosonic or fermionic baths.

    Parameters
    ----------
    H_sys : QObj, QobjEvo or a list
        The system Hamiltonian or Liouvillian specified as either a
        :obj:`Qobj`, a :obj:`QobjEvo`, or a list of elements that may
        be converted to a :obj:`ObjEvo`.

    bath : Bath or list of Bath
        A :obj:`Bath` containing the exponents of the expansion of the
        bath correlation funcion and their associated coefficients
        and coupling operators, or a list of baths.

        If multiple baths are given, they must all be either fermionic
        or bosonic baths.

    max_depth : int
        The maximum depth of the heirarchy (i.e. the maximum number of bath
        exponent "excitations" to retain).

    options : :class:`qutip.solver.Options`
        Generic solver options. If set to None the default options will be
        used.

    progress_bar : None, True or :class:`BaseProgressBar`
        Optional instance of BaseProgressBar, or a subclass thereof, for
        showing the progress of the solver. If True, an instance of
        :class:`TextProgressBar` is used instead.

    Attributes
    ----------
    ados : :obj:`HierarchyADOs`
        The description of the hierarchy constructed from the given bath
        and maximum depth.
    """
    def __init__(
        self, H_sys, bath, max_depth, options=None, progress_bar=None,
    ):
        self.H_sys = self._convert_h_sys(H_sys)
        self.options = Options() if options is None else options
        self._is_timedep = isinstance(self.H_sys, QobjEvo)
        self._H0 = self.H_sys.to_list()[0] if self._is_timedep else self.H_sys
        self._is_hamiltonian = self._H0.type == "oper"
        self._L0 = liouvillian(self._H0) if self._is_hamiltonian else self._H0

        self._sys_shape = (
            self._H0.shape[0] if self._is_hamiltonian
            else int(np.sqrt(self._H0.shape[0]))
        )
        self._sup_shape = self._L0.shape[0]
        self._sys_dims = (
            self._H0.dims if self._is_hamiltonian
            else self._H0.dims[0]
        )

        self.ados = HierarchyADOs(
            self._combine_bath_exponents(bath), max_depth,
        )
        self._n_ados = len(self.ados.labels)
        self._n_exponents = len(self.ados.exponents)

        # pre-calculate identity matrix required by _grad_n
        self._sId = fast_identity(self._sup_shape)

        # pre-calculate superoperators required by _grad_prev and _grad_next:
        Qs = [exp.Q for exp in self.ados.exponents]
        self._spreQ = [spre(op).data for op in Qs]
        self._spostQ = [spost(op).data for op in Qs]
        self._s_pre_minus_post_Q = [
            self._spreQ[k] - self._spostQ[k] for k in range(self._n_exponents)
        ]
        self._s_pre_plus_post_Q = [
            self._spreQ[k] + self._spostQ[k] for k in range(self._n_exponents)
        ]
        self._spreQdag = [spre(op.dag()).data for op in Qs]
        self._spostQdag = [spost(op.dag()).data for op in Qs]
        self._s_pre_minus_post_Qdag = [
            self._spreQdag[k] - self._spostQdag[k]
            for k in range(self._n_exponents)
        ]
        self._s_pre_plus_post_Qdag = [
            self._spreQdag[k] + self._spostQdag[k]
            for k in range(self._n_exponents)
        ]

        if progress_bar is None:
            self.progress_bar = BaseProgressBar()
        if progress_bar is True:
            self.progress_bar = TextProgressBar()

        self._configure_solver()

    def _convert_h_sys(self, H_sys):
        """ Process input system Hamiltonian, converting and raising as needed.
        """
        if isinstance(H_sys, (Qobj, QobjEvo)):
            pass
        elif isinstance(H_sys, list):
            try:
                H_sys = QobjEvo(H_sys)
            except Exception as err:
                raise ValueError(
                    "Hamiltonian (H_sys) of type list cannot be converted to"
                    " QObjEvo"
                ) from err
        else:
            raise TypeError(
                f"Hamiltonian (H_sys) has unsupported type: {type(H_sys)!r}")
        return H_sys

    def _combine_bath_exponents(self, bath):
        """ Combine the exponents for the specified baths. """
        if not isinstance(bath, (list, tuple)):
            exponents = bath.exponents
        else:
            exponents = []
            for b in bath:
                exponents.extend(b.exponents)
        all_bosonic = all(
            exp.type in (exp.types.R, exp.types.I, exp.types.RI)
            for exp in exponents
        )
        all_fermionic = all(
            exp.type in (exp.types["+"], exp.types["-"])
            for exp in exponents
        )
        if not (all_bosonic or all_fermionic):
            raise ValueError(
                "Bath exponents are currently restricted to being either"
                " all bosonic or all fermionic, but a mixture of bath"
                " exponents was given."
            )
        if not all(exp.Q.dims == exponents[0].Q.dims for exp in exponents):
            raise ValueError(
                "All bath exponents must have system coupling operators"
                " with the same dimensions but a mixture of dimensions"
                " was given."
            )
        return exponents

    def _dsuper_list_td(self, t, y, L_list):
        """ Auxiliary function for the time-dependent integration. Called every
            time step.
        """
        L = L_list[0][0]
        for n in range(1, len(L_list)):
            L = L + L_list[n][0] * L_list[n][1](t)
        return L * y

    def _grad_n(self, L, he_n):
        """ Get the gradient for the hierarchy ADO at level n. """
        vk = self.ados.vk
        vk_sum = sum(he_n[i] * vk[i] for i in range(len(vk)))
        op = L - vk_sum * self._sId
        return op

    def _grad_prev(self, he_n, k):
        """ Get the previous gradient. """
        if self.ados.exponents[k].type in (
                BathExponent.types.R, BathExponent.types.I,
                BathExponent.types.RI
        ):
            return self._grad_prev_bosonic(he_n, k)
        elif self.ados.exponents[k].type in (
                BathExponent.types["+"], BathExponent.types["-"]
        ):
            return self._grad_prev_fermionic(he_n, k)
        else:
            raise ValueError(
                f"Mode {k} has unsupported type {self.ados.exponents[k].type}")

    def _grad_prev_bosonic(self, he_n, k):
        if self.ados.exponents[k].type == BathExponent.types.R:
            op = (-1j * he_n[k] * self.ados.ck[k]) * (
                self._s_pre_minus_post_Q[k]
            )
        elif self.ados.exponents[k].type == BathExponent.types.I:
            op = (-1j * he_n[k] * 1j * self.ados.ck[k]) * (
                self._s_pre_plus_post_Q[k]
            )
        elif self.ados.exponents[k].type == BathExponent.types.RI:
            term1 = (he_n[k] * -1j * self.ados.ck[k]) * (
                self._s_pre_minus_post_Q[k]
            )
            term2 = (he_n[k] * self.ados.ck2[k]) * self._s_pre_plus_post_Q[k]
            op = term1 + term2
        else:
            raise ValueError(
                f"Unsupported type {self.ados.exponents[k].type}"
                f" for exponent {k}"
            )
        return op

    def _grad_prev_fermionic(self, he_n, k):
        ck = self.ados.ck

        n_excite = sum(he_n)
        sign1 = (-1) ** (n_excite + 1)

        n_excite_before_m = sum(he_n[:k])
        sign2 = (-1) ** (n_excite_before_m)

        sigma_bar_k = k + self.ados.sigma_bar_k_offset[k]

        if self.ados.exponents[k].type == BathExponent.types["+"]:
            op = -1j * sign2 * (
                (ck[k] * self._spreQdag[k]) -
                (sign1 * np.conj(ck[sigma_bar_k]) * self._spostQdag[k])
            )
        elif self.ados.exponents[k].type == BathExponent.types["-"]:
            op = -1j * sign2 * (
                (ck[k] * self._spreQ[k]) -
                (sign1 * np.conj(ck[sigma_bar_k]) * self._spostQ[k])
            )
        else:
            raise ValueError(
                f"Unsupported type {self.ados.exponents[k].type}"
                f" for exponent {k}"
            )
        return op

    def _grad_next(self, he_n, k):
        """ Get the previous gradient. """
        if self.ados.exponents[k].type in (
                BathExponent.types.R, BathExponent.types.I,
                BathExponent.types.RI
        ):
            return self._grad_next_bosonic(he_n, k)
        elif self.ados.exponents[k].type in (
                BathExponent.types["+"], BathExponent.types["-"]
        ):
            return self._grad_next_fermionic(he_n, k)
        else:
            raise ValueError(
                f"Mode {k} has unsupported type {self.ados.exponents[k].type}")

    def _grad_next_bosonic(self, he_n, k):
        op = -1j * self._s_pre_minus_post_Q[k]
        return op

    def _grad_next_fermionic(self, he_n, k):
        n_excite = sum(he_n)
        sign1 = (-1) ** (n_excite + 1)

        n_excite_before_m = sum(he_n[:k])
        sign2 = (-1) ** (n_excite_before_m)

        if self.ados.exponents[k].type == BathExponent.types["+"]:
            if sign1 == -1:
                op = (-1j * sign2) * self._s_pre_minus_post_Q[k]
            else:
                op = (-1j * sign2) * self._s_pre_plus_post_Q[k]
        elif self.ados.exponents[k].type == BathExponent.types["-"]:
            if sign1 == -1:
                op = (-1j * sign2) * self._s_pre_minus_post_Qdag[k]
            else:
                op = (-1j * sign2) * self._s_pre_plus_post_Qdag[k]
        else:
            raise ValueError(
                f"Unsupported type {self.ados.exponents[k].type}"
                f" for exponent {k}"
            )
        return op

    def _rhs(self, L):
        """ Make the RHS for the HEOM. """
        ops = _GatherHEOMRHS(self.ados.idx, block=L.shape[0], nhe=self._n_ados)

        for he_n in self.ados.labels:
            op = self._grad_n(L, he_n)
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

    def _configure_solver(self):
        """ Set up the solver. """
        RHSmat = self._rhs(self._L0.data)
        assert isinstance(RHSmat, sp.csr_matrix)

        if self._is_timedep:
            # In the time dependent case, we construct the parameters
            # for the ODE gradient function _dsuper_list_td under the
            # assumption that RHSmat(t) = RHSmat + time dependent terms
            # that only affect the diagonal blocks of the RHS matrix.
            # This assumption holds because only _grad_n dependents on
            # the system Liovillian (and not _grad_prev or _grad_next).

            h_identity_mat = sp.identity(self._n_ados, format="csr")
            H_list = self.H_sys.to_list()

            solver_params = [[RHSmat]]
            for idx in range(1, len(H_list)):
                temp_mat = sp.kron(
                    h_identity_mat, liouvillian(H_list[idx][0])
                )
                solver_params.append([temp_mat, H_list[idx][1]])

            solver = scipy.integrate.ode(self._dsuper_list_td)
            solver.set_f_params(solver_params)
        else:
            solver = scipy.integrate.ode(cy_ode_rhs)
            solver.set_f_params(RHSmat.data, RHSmat.indices, RHSmat.indptr)

        solver.set_integrator(
            "zvode",
            method=self.options.method,
            order=self.options.order,
            atol=self.options.atol,
            rtol=self.options.rtol,
            nsteps=self.options.nsteps,
            first_step=self.options.first_step,
            min_step=self.options.min_step,
            max_step=self.options.max_step,
        )

        self._ode = solver
        self.RHSmat = RHSmat

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

            For a complete description, see iparm(8) in
            http://cali2.unilim.fr/intel-xe/mkl/mklman/GUID-264E311E-ACED-4D56-AC31-E9D3B11D1CBF.htm.

        mkl_weighted_matching : bool
            MKL PARDISO can use a maximum weighted matching algorithm to
            permute large elements close the diagonal. This strategy adds an
            additional level of reliability to the factorization methods.

            For a complete description, see iparm(13) in
            http://cali2.unilim.fr/intel-xe/mkl/mklman/GUID-264E311E-ACED-4D56-AC31-E9D3B11D1CBF.htm.

        Returns
        -------
        steady_state : Qobj
            The steady state density matrix of the system.

        steady_ados : :class:`HierarchyADOsState`
            The steady state of the full ADO hierarchy. A particular ADO may be
            extracted from the full state by calling :meth:`.extract`.
        """
        n = self._sys_shape

        b_mat = np.zeros(n ** 2 * self._n_ados, dtype=complex)
        b_mat[0] = 1.0

        L = deepcopy(self.RHSmat)
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

        data = dense2D_to_fastcsr_fmode(vec2mat(solution[:n ** 2]), n, n)
        data = 0.5 * (data + data.H)
        steady_state = Qobj(data, dims=self._sys_dims)

        solution = solution.reshape((self._n_ados, n, n))
        steady_ados = HierarchyADOsState(steady_state, self.ados, solution)

        return steady_state, steady_ados

    def _convert_e_ops(self, e_ops):
        """
        Parse and convert a dictionary or list of e_ops.

        Returns
        -------
        e_ops, expected : tuple
            If the input ``e_ops`` was a list or scalar, ``expected`` is a list
            with one item for each element of the original e_ops.

            If the input ``e_ops`` was a dictionary, ``expected`` is a
            dictionary with the same keys.

            The output ``e_ops`` is always a dictionary. Its keys are the
            keys or indexes for ``expected`` and its elements are the e_ops
            functions or callables.
        """
        if isinstance(e_ops, (list, dict)):
            pass
        elif e_ops is None:
            e_ops = []
        elif isinstance(e_ops, Qobj):
            e_ops = [e_ops]
        elif callable(e_ops):
            e_ops = [e_ops]
        else:
            try:
                e_ops = list(e_ops)
            except Exception as err:
                raise TypeError(
                    "e_ops must be an iterable, Qobj or function"
                ) from err

        if isinstance(e_ops, dict):
            expected = {k: [] for k in e_ops}
        else:
            expected = [[] for _ in e_ops]
            e_ops = {i: op for i, op in enumerate(e_ops)}

        if not all(
            callable(op) or isinstance(op, Qobj) for op in e_ops.values()
        ):
            raise TypeError("e_ops must only contain Qobj or functions")

        return e_ops, expected

    def run(self, rho0, tlist, e_ops=None, ado_init=False, ado_return=False):
        """
        Solve for the time evolution of the system.

        Parameters
        ----------
        rho0 : Qobj or HierarchyADOsState or numpy.array
            Initial state (:obj:`~Qobj` density matrix) of the system
            if ``ado_init`` is ``False``.

            If ``ado_init`` is ``True``, then ``rho0`` should be an
            instance of :obj:`~HierarchyADOsState` or a numpy array
            giving the initial state of all ADOs. Usually
            the state of the ADOs would be determine from a previous call
            to ``.run(..., ado_return=True)``. For example,
            ``result = solver.run(..., ado_return=True)`` could be followed
            by ``solver.run(result.ado_states[-1], tlist, ado_init=True)``.

            If a numpy array is passed its shape must be
            ``(number_of_ados, n, n)`` where ``(n, n)`` is the system shape
            (i.e. shape of the system density matrix) and the ADOs must be
            in the same order as in ``.ados.labels``.

        tlist : list
            An ordered list of times at which to return the value of the state.

        e_ops : Qobj / callable / list / dict / None, optional
            A list or dictionary of operators as `Qobj` and/or callable
            functions (they can be mixed) or a single operator or callable
            function. For an operator ``op``, the result will be computed
            using ``(state * op).tr()`` and the state at each time ``t``. For
            callable functions, ``f``, the result is computed using
            ``f(t, ado_state)``. The values are stored in ``expect`` on
            (see the return section below).

        ado_init: bool, default False
            Indicates if initial condition is just the system state, or a
            numpy array including all ADOs.

        ado_return: bool, default True
            Whether to also return as output the full state of all ADOs.

        Returns
        -------
        :class:`qutip.solver.Result`
            The results of the simulation run, with the following attributes:

            * ``times``: the times ``t`` (i.e. the ``tlist``).

            * ``states``: the system state at each time ``t`` (only available
              if ``e_ops`` was ``None`` or if the solver option
              ``store_states`` was set to ``True``).

            * ``ado_states``: the full ADO state at each time (only available
              if ``ado_return`` was set to ``True``). Each element is an
              instance of :class:`HierarchyADOsState`.            .
              The state of a particular ADO may be extracted from
              ``result.ado_states[i]`` by calling :meth:`.extract`.

            * ``expect``: the value of each ``e_ops`` at time ``t`` (only
              available if ``e_ops`` were given). If ``e_ops`` was passed
              as a dictionary, then ``expect`` will be a dictionary with
              the same keys as ``e_ops`` and values giving the list of
              outcomes for the corresponding key.
        """
        e_ops, expected = self._convert_e_ops(e_ops)
        e_ops_callables = any(
            not isinstance(op, Qobj) for op in e_ops.values()
        )

        n = self._sys_shape
        rho_shape = (n, n)
        rho_dims = self._sys_dims
        hierarchy_shape = (self._n_ados, n, n)

        output = Result()
        output.solver = "HEOMSolver"
        output.times = tlist
        if e_ops:
            output.expect = expected
        if not e_ops or self.options.store_states:
            output.states = []

        if ado_init:
            if isinstance(rho0, HierarchyADOsState):
                rho0_he = rho0._ado_state
            else:
                rho0_he = rho0
            if rho0_he.shape != hierarchy_shape:
                raise ValueError(
                    f"ADOs passed with ado_init have shape {rho0_he.shape}"
                    f"but the solver hierarchy shape is {hierarchy_shape}"
                )
            rho0_he = rho0_he.reshape(n ** 2 * self._n_ados)
        else:
            rho0_he = np.zeros([n ** 2 * self._n_ados], dtype=complex)
            rho0_he[:n ** 2] = rho0.full().ravel('F')

        if ado_return:
            output.ado_states = []

        solver = self._ode
        solver.set_initial_value(rho0_he, tlist[0])

        self.progress_bar.start(len(tlist))
        for t_idx, t in enumerate(tlist):
            self.progress_bar.update(t_idx)
            if t_idx != 0:
                solver.integrate(t)
                if not solver.successful():
                    raise RuntimeError(
                        "HEOMSolver ODE integration error. Try increasing"
                        " the nsteps given in the HEOMSolver options"
                        " (which increases the allowed substeps in each"
                        " step between times given in tlist).")

            rho = Qobj(
                solver.y[:n ** 2].reshape(rho_shape, order='F'),
                dims=rho_dims,
            )
            if self.options.store_states:
                output.states.append(rho)
            if ado_return or e_ops_callables:
                ado_state = HierarchyADOsState(
                    rho, self.ados, solver.y.reshape(hierarchy_shape)
                )
            if ado_return:
                output.ado_states.append(ado_state)
            for e_key, e_op in e_ops.items():
                if isinstance(e_op, Qobj):
                    e_result = (rho * e_op).tr()
                else:
                    e_result = e_op(t, ado_state)
                output.expect[e_key].append(e_result)

        self.progress_bar.finished()
        return output


class HSolverDL(HEOMSolver):
    """
    A helper class for creating an :class:`HEOMSolver` that is backwards
    compatible with the ``HSolverDL`` provided in ``qutip.nonmarkov.heom``
    in QuTiP 4.6 and below.

    See :class:`HEOMSolver` and :class:`DrudeLorentzBath` for more
    descriptions of the underlying solver and bath construction.

    An exact copy of the QuTiP 4.6 HSolverDL is provided in
    ``qutip.nonmarkov.dlheom_solver`` for cases where the functionality of
    the older solver is required. The older solver will be completely
    removed in QuTiP 5.

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
        Coupling strength. Referred to as ``lam`` in :class:`DrudeLorentzBath`.

    temperature : float
        Bath temperature. Referred to as ``T`` in :class:`DrudeLorentzBath`.

    N_cut : int
        The maximum depth of the hierarchy. See ``max_depth`` in
        :class:`HEOMSolver` for a full description.

    N_exp : int
        Number of exponential terms used to approximate the bath correlation
        functions. The equivalent ``Nk`` in :class:`DrudeLorentzBath` is one
        less than ``N_exp`` (see note above).

    cut_freq : float
        Bath spectral density cutoff frequency. Referred to as ``gamma`` in
        :class:`DrudeLorentzBath`.

    bnd_cut_approx : bool
        Use boundary cut off approximation. If true, the Matsubara
        terminator is added to the system Liouvillian (and H_sys is
        promoted to a Liouvillian if it was a Hamiltonian).

    progress_bar : None, True or :class:`BaseProgressBar`
        Optional instance of BaseProgressBar, or a subclass thereof, for
        showing the progress of the solver. If True, an instance of
        :class:`TextProgressBar` is used instead.

    options : :class:`qutip.solver.Options`
        Generic solver options.
        If set to None the default options will be used.

    progress_bar : None, True or :class:`BaseProgressBar`
        Optional instance of BaseProgressBar, or a subclass thereof, for
        showing the progress of the solver. If True, an instance of
        :class:`TextProgressBar` is used instead.

    combine : bool, default True
        Whether to combine exponents with the same frequency (and coupling
        operator). See :meth:`BosonicBath.combine` for details.
    """
    def __init__(
        self, H_sys, coup_op, coup_strength, temperature,
        N_cut, N_exp, cut_freq, bnd_cut_approx=False, options=None,
        progress_bar=None, combine=True,
    ):
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
            H_sys = self._convert_h_sys(H_sys)
            is_timedep = isinstance(H_sys, QobjEvo)
            H0 = H_sys.to_list()[0] if is_timedep else H_sys
            is_hamiltonian = H0.type == "oper"
            if is_hamiltonian:
                H_sys = liouvillian(H_sys)
            _, terminator = bath.terminator()
            H_sys = H_sys + terminator

        super().__init__(
            H_sys, bath=bath, max_depth=N_cut, options=options,
            progress_bar=progress_bar,
        )

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
        self._block = block
        self._nhe = nhe
        self._f_idx = f_idx
        self._ops = []

    def add_op(self, row_he, col_he, op):
        """ Add an block operator to the list. """
        self._ops.append(
            (self._f_idx(row_he), self._f_idx(col_he), op)
        )

    def gather(self):
        """ Create the HEOM liouvillian from a sorted list of smaller (fast) CSR
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
            rhs : :obj:`qutip.fastsparse.fast_csr_matrix`
                A combined matrix of shape ``(block * nhe, block * ne)``.
        """
        block = self._block
        nhe = self._nhe
        ops = self._ops
        shape = (block * nhe, block * nhe)
        if not ops:
            return sp.csr_matrix(shape, dtype=np.complex128)
        ops.sort()
        nnz = sum(op.nnz for _, _, op in ops)
        indptr = np.zeros(shape[0] + 1, dtype=np.int32)
        indices = np.zeros(nnz, dtype=np.int32)
        data = np.zeros(nnz, dtype=np.complex128)
        end = 0
        op_idx = 0
        op_len = len(ops)

        for row_idx in range(nhe):
            prev_op_idx = op_idx
            while op_idx < op_len:
                if ops[op_idx][0] != row_idx:
                    break
                op_idx += 1

            row_ops = ops[prev_op_idx: op_idx]
            rowpos = row_idx * block
            for op_row in range(block):
                for _, col_idx, op in row_ops:
                    colpos = col_idx * block
                    op_row_start = op.indptr[op_row]
                    op_row_end = op.indptr[op_row + 1]
                    op_row_len = op_row_end - op_row_start
                    if op_row_len == 0:
                        continue
                    indices[end: end + op_row_len] = (
                        op.indices[op_row_start: op_row_end] + colpos
                    )
                    data[end: end + op_row_len] = (
                        op.data[op_row_start: op_row_end]
                    )
                    end += op_row_len
                indptr[rowpos + op_row + 1] = end

        return fast_csr_matrix(
            (data, indices, indptr), shape=shape, dtype=np.complex128,
        )
