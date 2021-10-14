"""
This module provides exact solvers for a system-bath setup using the
hierarchy equations of motion (HEOM).
"""

# Authors: Neill Lambert, Tarun Raheja, Shahnawaz Ahmed
# Contact: nwlambert@gmail.com

import enum
import warnings
from copy import deepcopy

import numpy as np
import scipy.sparse as sp
import scipy.integrate
from scipy.sparse.linalg import splu

from qutip import settings
from qutip import state_number_enumerate
from qutip.qobj import Qobj
from qutip.qobjevo import QobjEvo
from qutip.superoperator import liouvillian, spre, spost, vec2mat
from qutip.cy.heom import cy_pad_csr
from qutip.cy.spmatfuncs import cy_ode_rhs
from qutip.solver import Options, Result
from qutip.cy.spconvert import dense2D_to_fastcsr_fmode
from qutip.ui.progressbar import BaseProgressBar
from qutip.fastsparse import fast_identity, fast_csr_matrix

# Load MKL spsolve if avaiable
if settings.has_mkl:
    from qutip._mkl.spsolve import mkl_spsolve
else:
    mkl_spsolve = None


class BathExponent:
    """
    Represents a single exponent (naively, an excitation mode) within the
    decomposition of the correlation functions of a bath.

    Parameters
    ----------
    type : {"R", "I", "RI", "+", "-"} or BathExponent.ExponentType
        The type of bath exponent.

        "R" and "I" are bosonic bath exponents that appear in the real and
        imaginary parts of the correlation expansion.

        "RI" is combined bosonic bath exponent that appears in both the real
        and imaginary parts of the correlation expansion. The combined exponent
        has a single ``vk``. The ``ck`` is the coefficient in the real
        expansion and ``ck2`` is the coefficient in the imaginary expansion.

        "+" and "-" are fermionic bath exponents. These fermionic bath
        exponents must specify ``sigma_bar_k_offset`` which specifies
        the amount to add to ``k`` (the exponent index within the bath of this
        exponent) to determine the ``k`` of the corresponding exponent with
        the opposite sign (i.e. "-" or "+").

    dim : int or None
        The dimension (i.e. maximum number of excitations for this exponent).
        Usually ``2`` for fermionic exponents or ``None`` (i.e. unlimited) for
        bosonic exponents.

    Q : Qobj
        The coupling operator for this excitation mode.

    vk : complex
        The frequency of the exponent of the excitation term.

    ck : complex
        The coefficient of the excitation term.

    ck2 : optional, complex
        For exponents of type "RI" this is the coefficient of the term in the
        imaginary expansion (and ``ck`` is the coefficient in the real
        expansion).

    sigma_bar_k_offset : optional, int
        For exponents of type "+" this gives the offset (within the list of
        exponents within the bath) of the corresponding "-" bath exponent.
        For exponents of type "-" it gives the offset of the corresponding
        "+" exponent.

    Attributes
    ----------

    All of the parameters are available as attributes.
    """

    types = enum.Enum("ExponentType", ["R", "I", "RI", "+", "-"])

    def _check_ck2(self, type, ck2):
        if type == self.types["RI"]:
            if ck2 is None:
                raise ValueError("RI bath modes require ck2")
        else:
            if ck2 is not None:
                raise ValueError(
                    "Second co-efficient (ck2) should only be specified for RI"
                    " bath exponents"
                )

    def _check_sigma_bar_k_offset(self, type, offset):
        if type in (self.types["+"], self.types["-"]):
            if offset is None:
                raise ValueError("+ and - bath modes require sigma_bar_k")
        else:
            if offset is not None:
                raise ValueError(
                    "Offset of sigma bar (sigma_bar_k_offset) should only be"
                    " specified for + and - bath exponents"
                )

    def __init__(
            self, type, dim, Q, ck, vk, ck2=None, sigma_bar_k_offset=None):
        if not isinstance(type, self.types):
            type = self.types[type]
        self._check_ck2(type, ck2)
        self._check_sigma_bar_k_offset(type, sigma_bar_k_offset)
        self.type = type
        self.dim = dim
        self.Q = Q
        self.ck = ck
        self.vk = vk
        self.ck2 = ck2
        self.sigma_bar_k_offset = sigma_bar_k_offset


class Bath:
    """
    Represents a list of bath expansion exponents.

    Parameter
    ---------
    modes : list of BathExponent
        The exponents of the correlation function describing the bath.

    Attributes
    ----------

    All of the parameters are available as attributes.
    """
    def __init__(self, modes):
        self.modes = modes


class BosonicBath(Bath):
    def __init__(self, Q, ckAR, vkAR, ckAI, vkAI, combine=True):
        Q = _convert_coup_op(Q, len(ckAR) + len(ckAI))
        ckAR, ckAI, vkAR, vkAI = _convert_bath_exponents_bosonic(
            ckAR, ckAI, vkAR, vkAI)

        if combine:
            Q, ck, vk, NR, NI = (
                _mangle_bath_exponents_bosonic(Q, ckAR, ckAI, vkAR, vkAI)
            )
        else:
            ck = ckAR + ckAI
            vk = vkAR + vkAI
            NR = len(ckAR)
            NI = len(ckAI)

        modes = []
        modes.extend(
            BathExponent("R", None, Q[i], ck[i], vk[i])
            for i in range(0, NR))
        modes.extend(
            BathExponent("I", None, Q[i], ck[i], vk[i])
            for i in range(NR, NR + NI))
        modes.extend(
            BathExponent("RI", None, Q[i], ck[i], vk[i], ck2=ck[i+1])
            for i in range(NR + NI, len(ck), 2))
        super().__init__(modes)


class FermionicBath(Bath):
    def __init__(self, Q, ck, vk):
        ck, vk = _convert_bath_exponents_fermionic(ck, vk)
        Q = _convert_coup_op(Q, len(ck))

        modes = []
        for i in range(len(ck)):
            # currently "-" modes are generated by adding extra
            # baths outside with Q == Q.dag() when calling
            # FermionicHEOMSolver
            if i % 2 == 0:
                type = "+"
                sbk_offset = len(ck[i])
            else:
                type = "-"
                sbk_offset = -len(ck[i - 1])
            modes.extend(
                BathExponent(
                    type, 2, Q[i], ck[i][j], vk[i][j],
                    sigma_bar_k_offset=sbk_offset
                )
                for j in range(len(ck[i]))
            )

        super().__init__(modes)


class BathStates:
    """
    A description of bath states coupled to quantum system in the hierarchical
    equations of motion formulation.

    Parameters
    ----------
    modes : list of BathExponent
        The exponents of the correlation function describing the bath.

    cutoff : int
        The maximum number of excitations.

    Attributes
    ----------
    modes : list of BathExponent
        The exponents of the correlation function describing the bath.

    cutoff : int
        The maximum number of excitations.

    dims : list of int
        The dimensions of each compontent system within the bath.

    vk : list of complex
        The frequency of each exponent within the bath.

    ck : list of complex
        The coefficient of each exponent within the bath.

    ck2: list of complex
        For exponents of type "RI", the coefficient of the exponent within
        the imaginary expansion. For other exponent types, the entry is None.

    sigma_bar_k_offset: list of int
        For exponents of type "+" or "-" the offset within the list of modes
        of the corresponding "-" or "+" exponent. For other exponent types,
        the entry is None.

    states: list of tuples
        A list of the state vectors within the bath.

    n_states: int
        Total number of states. Equivalent to ``len(bath.states)``.
    """
    def __init__(self, modes, cutoff):
        self.modes = modes
        self.cutoff = cutoff

        self.dims = [mode.dim or (cutoff + 1) for mode in self.modes]
        self.vk = [mode.vk for mode in self.modes]
        self.ck = [mode.ck for mode in self.modes]
        self.ck2 = [mode.ck2 for mode in self.modes]
        self.sigma_bar_k_offset = [
            mode.sigma_bar_k_offset for mode in self.modes
        ]

        self.states = list(state_number_enumerate(self.dims, cutoff))
        self.n_states = len(self.states)
        self._state_idx = {s: i for i, s in enumerate(self.states)}

    def idx(self, state):
        """
        Return the index of the state within the list of bath states,
        i.e. within ``self.states``.

        Parameters
        ----------
        state : tuple
            The state to look up.

        Returns
        -------
        int
            The index of the state within the list of bath states.
        """
        return self._state_idx[state]

    def next(self, state, k):
        """
        Return the state with one more excitation in the k'th bath dimension
        or ``None`` if adding the excitation would exceed the dimension or
        bath cutoff.

        Parameters
        ----------
        state : tuple
            The state to add an excitation to.
        k : int
            The bath dimension to add the excitation to.

        Returns
        -------
        tuple or None
            The next state.
        """
        if state[k] >= self.dims[k] - 1:
            return None
        if sum(state) >= self.cutoff:
            return None
        return state[:k] + (state[k] + 1,) + state[k + 1:]

    def prev(self, state, k):
        """
        Return the state with one fewer excitation in the k'th bath dimension
        or ``None`` if the state has no exciations in the k'th bath dimension.

        Parameters
        ----------
        state : tuple
            The state to remove the excitation from.
        k : int
            The bath dimension to remove the excitation from.

        Returns
        -------
        tuple or None
            The previous state.
        """
        if state[k] <= 0:
            return None
        return state[:k] + (state[k] - 1,) + state[k + 1:]


def _convert_h_sys(H_sys):
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


def _convert_coup_op(coup_op, coup_op_len):
    """ Convert coup_op to a list of the appropriate length. """
    if isinstance(coup_op, Qobj):
        coup_op = [coup_op] * coup_op_len
    elif (isinstance(coup_op, list)
            and all(isinstance(x, Qobj) for x in coup_op)):
        if len(coup_op) != coup_op_len:
            raise ValueError(
                f"Expected {coup_op_len} coupling operators")
    else:
        raise TypeError(
            "Coupling operator (coup_op) must be a Qobj or a list of Qobjs"
        )
    return coup_op


def _convert_bath_exponents_bosonic(ckAI, ckAR, vkAI, vkAR):
    all_k = (ckAI, ckAR, vkAI, vkAR)
    if any(not isinstance(k, list) for k in all_k):
        raise TypeError(
            "The bath exponents ckAI, ckAR, vkAI and vkAR must all be lists")
    if len(ckAI) != len(vkAI) or len(ckAR) != len(vkAR):
        raise ValueError(
            "The bath exponent lists ckAI and vkAI, and ckAR and vkAR must"
            " be the same length"
        )
    if any(isinstance(x, list) for k in all_k for x in k):
        raise ValueError(
            "The bath exponent lists ckAI, ckAR, vkAI and vkAR should not"
            " themselves contain lists"
        )
    # warn if any of the vkAR's are close
    for i in range(len(vkAR)):
        for j in range(i + 1, len(vkAR)):
            if np.isclose(vkAR[i], vkAR[j], rtol=1e-5, atol=1e-7):
                warnings.warn(
                    "Expected simplified input. "
                    "Consider collating equal frequency parameters."
                )
    # warn if any of the vkAR's are close
    for i in range(len(vkAI)):
        for j in range(i + 1, len(vkAI)):
            if np.isclose(vkAI[i], vkAI[j], rtol=1e-5, atol=1e-7):
                warnings.warn(
                    "Expected simplified input.  "
                    "Consider collating equal frequency parameters."
                )
    return ckAI, ckAR, vkAI, vkAR


def _mangle_bath_exponents_bosonic(coup_op, ckAR, ckAI, vkAR, vkAI):
    """ Mangle bath exponents by combining similar vkAR and vkAI. """

    common_ck = []
    real_indices = []
    common_vk = []
    img_indices = []
    common_coup_op = []
    coup_op = deepcopy(coup_op)
    nr = len(ckAR)

    for i in range(len(vkAR)):
        for j in range(len(vkAI)):
            if (
                np.isclose(vkAR[i], vkAI[j], rtol=1e-5, atol=1e-7) and
                np.allclose(coup_op[i], coup_op[nr + j], rtol=1e-5, atol=1e-7)
            ):
                warnings.warn(
                    "Two similar real and imag exponents have been "
                    "collated automatically."
                )
                common_ck.append(ckAR[i])
                common_ck.append(ckAI[j])
                common_vk.append(vkAR[i])
                common_vk.append(vkAI[j])
                real_indices.append(i)
                img_indices.append(j)
                common_coup_op.append(coup_op[i])

    for i in sorted(real_indices, reverse=True):
        ckAR.pop(i)
        vkAR.pop(i)

    for i in sorted(img_indices, reverse=True):
        ckAI.pop(i)
        vkAI.pop(i)

    img_coup_ops = [x + nr for x in img_indices]
    coup_op_indices = real_indices + sorted(img_coup_ops)
    for i in sorted(coup_op_indices, reverse=True):
        coup_op.pop(i)

    coup_op += common_coup_op

    ck = np.array(ckAR + ckAI + common_ck).astype(complex)
    vk = np.array(vkAR + vkAI + common_vk).astype(complex)
    NR = len(ckAR)
    NI = len(ckAI)

    return coup_op, ck, vk, NR, NI


def _convert_bath_exponents_fermionic(ck, vk):
    """ Check the bath exponents for the fermionic solver. """
    if (type(ck) != list or not all(isinstance(x, list) for x in ck)):
        raise TypeError("The bath exponents ck must be a list or lists.")
    if (type(vk) != list or not all(isinstance(x, list) for x in vk)):
        raise TypeError("The bath exponents vk must be a list or lists.")
    if (len(ck) != len(vk)
            or any(len(ck[i]) != len(vk[i]) for i in range(len(ck)))):
        raise ValueError("Exponents ck and vk must be the same length.")
    return ck, vk


class HEOMSolver:
    """
    HEOM solver that supports a single bath which may be either bosonic or
    fermionic.
    """

    def __init__(self, H_sys, bath, N_cut, options=None):
        self.H_sys = _convert_h_sys(H_sys)
        self.options = Options() if options is None else options
        self.is_timedep = isinstance(self.H_sys, QobjEvo)
        self.H0 = self.H_sys.to_list()[0] if self.is_timedep else self.H_sys
        self.is_hamiltonian = self.H0.type == "oper"
        self.L0 = liouvillian(self.H0) if self.is_hamiltonian else self.H0

        self._sys_shape = (
            self.H0.shape[0] if self.is_hamiltonian
            else int(np.sqrt(self.H0.shape[0]))
        )
        self._sup_shape = self.L0.shape[0]

        self.bath = BathStates(bath.modes, N_cut)

        self.coup_op = [mode.Q for mode in self.bath.modes]
        self.spreQ = [spre(op).data for op in self.coup_op]
        self.spostQ = [spost(op).data for op in self.coup_op]
        self.spreQdag = [spre(op.dag()).data for op in self.coup_op]
        self.spostQdag = [spost(op.dag()).data for op in self.coup_op]

        self.sId = fast_identity(self._sup_shape)
        self.s_pre_minus_post_Q = [
            self.spreQ[k] - self.spostQ[k] for k in range(len(self.coup_op))
        ]
        self.s_pre_plus_post_Q = [
            self.spreQ[k] + self.spostQ[k] for k in range(len(self.coup_op))
        ]

        self.progress_bar = BaseProgressBar()

        self._mode = "gather-ops"  # pad-op, inplace-add-op, gather-ops

        self._configure_solver()

    def _pad_op(self, op, row_he, col_he):
        """
        Pad op into its correct position within the larger HEOM liouvillian
        for the given row and column bath states.
        """
        nhe = self.bath.n_states
        rowidx = self.bath.idx(row_he)
        colidx = self.bath.idx(col_he)
        return cy_pad_csr(op, nhe, nhe, rowidx, colidx)

    def _inplace_add_op(self, L, op, row_he, col_he):
        """
        Add the operation ``op`` to its correct position within the
        larger HEOM liouvillian for the given row and column bath states.
        """
        block = self._sup_shape
        rowpos = self.bath.idx(row_he) * block
        colpos = self.bath.idx(col_he) * block
        L[rowpos: rowpos + block, colpos: colpos + block] += op

    def _gather_ops(self, ops, block, nhe):
        """ Create the HEOM liouvillian from a list of smaller CSRs.

            Parameters
            ----------
            ops : list of (row_idx, col_idx, op)
                Operators to combine.
            block : int
                The size of a single Liovillian operator in the hierarchy.
            nhe : int
                The number of ADOs in the hierarchy.
        """
        shape = (block * nhe, block * nhe)
        if not ops:
            return sp.csr_matrix(shape, dtype=np.complex128)
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

    def _dsuper_list_td(self, t, y, L_list):
        """ Auxiliary function for the integration. Called every time step. """
        L = L_list[0][0]
        for n in range(1, len(L_list)):
            L = L + L_list[n][0] * L_list[n][1](t)
        return L * y

    def _grad_n(self, L, he_n):
        """ Get the gradient for the hierarchy ADO at level n. """
        vk = self.bath.vk
        vk_sum = sum(he_n[i] * vk[i] for i in range(len(vk)))
        op = L - vk_sum * self.sId
        return op

    def _grad_prev(self, he_n, k):
        """ Get the previous gradient. """
        if self.bath.modes[k].type in (
                BathExponent.types.R, BathExponent.types.I,
                BathExponent.types.RI
        ):
            return self._grad_prev_bosonic(he_n, k)
        elif self.bath.modes[k].type in (
                BathExponent.types["+"], BathExponent.types["-"]
        ):
            return self._grad_prev_fermionic(he_n, k)
        else:
            raise ValueError(
                f"Mode {k} has unsupported type {self.bath.modes[k].type}")

    def _grad_prev_bosonic(self, he_n, k):
        if self.bath.modes[k].type == BathExponent.types.R:
            op = (-1j * he_n[k] * self.bath.ck[k]) * self.s_pre_minus_post_Q[k]
        elif self.bath.modes[k].type == BathExponent.types.I:
            op = (-1j * he_n[k] * 1j * self.bath.ck[k]) * (
                    self.s_pre_plus_post_Q[k]
                )
        elif self.bath.modes[k].type == BathExponent.types.RI:
            term1 = (he_n[k] * -1j * self.bath.ck[k]) * (
                self.s_pre_minus_post_Q[k]
            )
            term2 = (he_n[k] * self.bath.ck2[k]) * self.s_pre_plus_post_Q[k]
            op = term1 + term2
        else:
            raise ValueError(
                f"Unsupported type {self.bath.modes[k].type} for mode {k}"
            )
        return op

    def _grad_prev_fermionic(self, he_n, k):
        ck = self.bath.ck

        n_excite = sum(he_n)
        sign1 = (-1) ** (n_excite + 1)

        n_excite_before_m = sum(he_n[:k])
        sign2 = (-1) ** (n_excite_before_m)

        sigma_bar_k = k + self.bath.sigma_bar_k_offset[k]

        op = -1j * sign2 * (
            (ck[k] * self.spreQ[k]) -
            (sign1 * np.conj(ck[sigma_bar_k] * self.spostQ[k]))
        )

        return op

    def _grad_next(self, he_n, k):
        """ Get the previous gradient. """
        if self.bath.modes[k].type in (
                BathExponent.types.R, BathExponent.types.I,
                BathExponent.types.RI
        ):
            return self._grad_next_bosonic(he_n, k)
        elif self.bath.modes[k].type in (
                BathExponent.types["+"], BathExponent.types["-"]
        ):
            return self._grad_next_fermionic(he_n, k)
        else:
            raise ValueError(
                f"Mode {k} has unsupported type {self.bath.modes[k].type}")

    def _grad_next_bosonic(self, he_n, k):
        op = -1j * self.s_pre_minus_post_Q[k]
        return op

    def _grad_next_fermionic(self, he_n, k):
        n_excite = sum(he_n)
        sign1 = (-1) ** (n_excite + 1)

        n_excite_before_m = sum(he_n[:k])
        sign2 = (-1) ** (n_excite_before_m)

        if sign1 == -1:
            op = (-1j * sign2) * self.s_pre_minus_post_Q[k]
        else:
            op = (-1j * sign2) * self.s_pre_plus_post_Q[k]

        return op

    def _rhs(self, L):
        """ Make the RHS for the HEOM. """
        nhe = len(self.bath.states)
        L_shape = L.shape[0]

        # Temporary _mode option to experiment with different methods for
        # assembling the RHS while we determine which is best:
        #
        # pad-op: RHS += cy_pad_csr(op)
        # inplace-add-op: RHS[r: r + block, c: c + block] = op
        # gather-ops: store ops in a list and then assemble the RHS at the end
        if self._mode == "pad-op":
            RHS = fast_csr_matrix(
                shape=(nhe * L_shape, nhe * L_shape),
                dtype=np.complex128,
            )

            def _add_rhs(row_he, col_he, op):
                nonlocal RHS
                RHS += self._pad_op(op.copy(), row_he, col_he)
        elif self._mode == "inplace-add-op":
            RHS = sp.lil_matrix(
                (nhe * L_shape, nhe * L_shape),
                dtype=np.complex128,
            )

            def _add_rhs(row_he, col_he, op):
                nonlocal RHS
                self._inplace_add_op(RHS, op, row_he, col_he)
        elif self._mode == "gather-ops":
            ALL_OPS = []

            def _add_rhs(row_he, col_he, op):
                ALL_OPS.append(
                    (self.bath.idx(row_he), self.bath.idx(col_he), op)
                )
        else:
            raise ValueError(f"Unknown RHS construction _mode: {self._mode!r}")

        for he_n in self.bath.states:
            op = self._grad_n(L, he_n)
            _add_rhs(he_n, he_n, op)
            for k in range(len(self.bath.dims)):
                next_he = self.bath.next(he_n, k)
                if next_he is not None:
                    op = self._grad_next(he_n, k)
                    _add_rhs(he_n, next_he, op)
                prev_he = self.bath.prev(he_n, k)
                if prev_he is not None:
                    op = self._grad_prev(he_n, k)
                    _add_rhs(he_n, prev_he, op)

        if self._mode == "inplace-add-op":
            RHS = RHS.tocsr()
        elif self._mode == "gather-ops":
            ALL_OPS.sort()
            RHS = self._gather_ops(ALL_OPS, block=L_shape, nhe=nhe)

        return RHS

    def _configure_solver(self):
        """ Set up the solver. """
        RHSmat = self._rhs(self.L0.data)
        assert isinstance(RHSmat, sp.csr_matrix)

        if self.is_timedep:
            h_identity_mat = sp.identity(self.bath.n_states, format="csr")
            H_list = self.H_sys.to_list()

            # store each time dependent component
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
        use_mkl=False, mkl_max_iter_refine=100, mkl_weighted_matching=False
    ):
        """
        Compute the steady state of the system.

        Parameters
        ----------
        use_mkl : bool, default=False
            Whether to use mkl or not. If mkl is not installed or if
            this is false, use the scipy splu solver instead.

        mkl_max_iter_refine : Int
            Parameter for the mkl LU solver. If pardiso errors are returned
            this should be increased.

        mkl_weighted_matching : Boolean
            Setting this true may increase run time, but reduce stability
            (pardisio may not converge).

        Returns
        -------
        steady_state : Qobj
            The steady state density matrix of the system.

        solution : Numpy array
            Array of the the steady-state and all ADOs.
            Further processing of this can be done with functions provided in
            example notebooks.
        """
        nstates = self.bath.n_states
        n = self._sys_shape

        b_mat = np.zeros(n ** 2 * nstates, dtype=complex)
        b_mat[0] = 1.0

        L = deepcopy(self.RHSmat)
        L = L.tolil()
        L[0, 0: n ** 2 * nstates] = 0.0
        L = L.tocsr()
        L += sp.csr_matrix((
            np.ones(n),
            (np.zeros(n), [num * (n + 1) for num in range(n)])
        ), shape=(n ** 2 * nstates, n ** 2 * nstates))

        if mkl_spsolve is not None and use_mkl:
            L.sort_indices()
            solution = mkl_spsolve(
                L,
                b_mat,
                perm=None,
                verbose=True,
                max_iter_refine=mkl_max_iter_refine,
                scaling_vectors=True,
                weighted_matching=mkl_weighted_matching,
            )
        else:
            L = L.tocsc()
            LU = splu(L)
            solution = LU.solve(b_mat)

        data = dense2D_to_fastcsr_fmode(vec2mat(solution[:n ** 2]), n, n)
        data = 0.5 * (data + data.H)

        solution = solution.reshape((nstates, n ** 2))

        return Qobj(data, dims=self.H0.dims), solution

    def run(self, rho0, tlist, full_init=False, full_return=False):
        """
        Function to solve for an open quantum system using the
        HEOM model.

        Parameters
        ----------
        rho0 : Qobj
            Initial state (density matrix) of the system
            (if full_init==False).
            If full_init = True, then rho0 should be a numpy array of
            initial state and all ADOs.

        tlist : list
            Time over which system evolves.

        full_init: Boolean
            Indicates if initial condition is just the system Qobj, or a
            numpy array including all ADOs.

        full_return: Boolean
            Whether to also return as output the full state of all ADOs.

        Returns
        -------
        :class:`qutip.solver.Result`
            The results of the simulation run.
            The times (tlist) are stored in ``result.times``.
            The state at each time is stored in ``result.states``.
            If ``full_return`` is ``True``, then the ADOs at each
            time are stored in ``result.ados``.
        """
        nstates = self.bath.n_states
        n = self._sys_shape
        rho_shape = (n, n)
        rho_dims = self.coup_op[0].dims
        hierarchy_shape = (self.bath.n_states, n ** 2)

        output = Result()
        output.solver = "HEOMSolver"
        output.times = tlist
        output.states = []

        if full_init:
            rho0_he = rho0
        else:
            rho0_he = np.zeros([n ** 2 * nstates], dtype=complex)
            rho0_he[:n ** 2] = rho0.full().ravel('F')

        if full_return:
            output.ados = []

        solver = self._ode
        solver.set_initial_value(rho0_he, tlist[0])

        self.progress_bar.start(len(tlist))
        for t_idx, t in enumerate(tlist):
            self.progress_bar.update(t_idx)
            if t_idx != 0:
                solver.integrate(t)
            rho = Qobj(
                solver.y[:n ** 2].reshape(rho_shape, order='F'),
                dims=rho_dims,
            )
            output.states.append(rho)
            if full_return:
                output.ados.append(solver.y.reshape(hierarchy_shape))
        self.progress_bar.finished()
        return output


class BosonicHEOMSolver(HEOMSolver):
    """
    This is a class for solvers that use the HEOM method for
    calculating the dynamics evolution.
    The method can compute open system dynamics without using any Markovian
    or rotating wave approximations (RWA) for systems where the bath
    correlations can be approximated to a sum of complex exponentials.
    The method builds a matrix of linked differential equations, which are
    then solved used the same ODE solvers as other qutip solvers
    (e.g. mesolve)

    Parameters
    ----------
    H_sys : Qobj or QobjEvo or list
        System Hamiltonian
        Or
        Liouvillian
        Or
        QobjEvo
        Or
        list of Hamiltonians with time dependence

        Format for input (if list):
        [time_independent_part, [H1, time_dep_function1],
        [H2, time_dep_function2]]

    coup_op : Qobj or list
        Operator describing the coupling between system and bath.
        Could also be a list of operators, which needs to be the same length
        as ck's and vk's.

    ckAR, ckAI, vkAR, vkAI : lists
        Lists containing coefficients for fitting spectral density correlation

    N_cut : int
        The maximum depth of the heirarchy (i.e. the maximum number of bath
        excitations to retain).

    options : :class:`qutip.solver.Options`
        Generic solver options.
        If set to None the default options will be used
    """

    def __init__(
        self, H_sys, coup_op, ckAR, ckAI, vkAR, vkAI, N_cut, options=None
    ):
        bath = BosonicBath(coup_op, ckAR, vkAR, ckAI, vkAI)
        super().__init__(H_sys, bath, N_cut, options=options)


class HSolverDL(HEOMSolver):
    """
    HEOM solver based on the Drude-Lorentz model for spectral density.
    Drude-Lorentz bath the correlation functions can be exactly analytically
    expressed as a sum of exponentials.

    This sub-class is included to give backwards compatability with the older
    implentation in qutip.nonmarkov.heom.

    Parameters
    ----------
    H_sys : Qobj or QobjEvo or list
        System Hamiltonian
        Or
        Liouvillian
        Or
        QobjEvo
        Or
        list of Hamiltonians with time dependence

        Format for input (if list):
        [time_independent_part, [H1, time_dep_function1],
        [H2, time_dep_function2]]

    coup_op : Qobj or list
        Operator describing the coupling between system and bath.
        Could also be a list of operators, which needs to be the same length
        as ck's and vk's.

    coup_strength : float
        Coupling strength.

    temperature : float
        Bath temperature.

    N_cut : int
        Cutoff parameter for the bath

    N_exp : int
        Number of exponential terms used to approximate the bath correlation
        functions

    cut_freq : float
        Bath spectral density cutoff frequency.

    bnd_cut_approx : bool
        Use boundary cut off approximation

    options : :class:`qutip.solver.Options`
        Generic solver options.
        If set to None the default options will be used
    """

    def __init__(
        self, H_sys, coup_op, coup_strength, temperature,
        N_cut, N_exp, cut_freq, bnd_cut_approx=False, options=None,
    ):
        ckAR, ckAI, vkAR, vkAI = self._calc_matsubara_params(
            lam=coup_strength,
            gamma=cut_freq,
            Nk=N_exp,
            T=temperature
        )

        if bnd_cut_approx:
            L_bnd = self._calc_bound_cut_liouvillian(
                Q=coup_op,
                lam=coup_strength,
                gamma=cut_freq,
                Nk=N_exp,
                T=temperature
            )
            H_sys = _convert_h_sys(H_sys)
            H_sys = liouvillian(H_sys) + L_bnd

        bath = BosonicBath(coup_op, ckAR, vkAR, ckAI, vkAI)
        super().__init__(H_sys, bath, N_cut, options=options)

        # store input parameters as attributes for politeness
        self.coup_strength = coup_strength
        self.cut_freq = cut_freq
        self.temperature = temperature
        self.N_exp = N_exp
        self.bnd_cut_approx = bnd_cut_approx

    def _calc_bound_cut_liouvillian(self, Q, lam, gamma, Nk, T):
        """ Calculate the hierarchy terminator term for the Liouvillian. """
        beta = 1 / T

        op = -2*spre(Q)*spost(Q.dag()) + spre(Q.dag()*Q) + spost(Q.dag()*Q)
        approx_factr = ((2 * lam / (beta * gamma)) - 1j * lam)
        approx_factr -= (
            lam * gamma * (-1.0j + 1 / np.tan(gamma / (2 * T))) / gamma
        )

        for k in range(1, Nk):
            vk = 2 * np.pi * k * T
            approx_factr -= (
                (4 * lam * gamma * T * vk / (vk**2 - gamma**2)) / vk
            )

        L_bnd = -approx_factr * op
        return L_bnd

    def _calc_matsubara_params(self, lam, gamma, Nk, T):
        """ Calculate the Matsubara coefficents and frequencies. """
        ckAR = [lam * gamma * (1/np.tan(gamma / (2 * T)))]
        ckAR.extend([
            (4 * lam * gamma * T * 2 * np.pi * k * T /
                ((2 * np.pi * k * T)**2 - gamma**2))
            for k in range(1, Nk)
        ])
        vkAR = [gamma]
        vkAR.extend([2 * np.pi * k * T for k in range(1, Nk)])

        ckAI = [lam * gamma * (-1.0)]
        vkAI = [gamma]

        return ckAR, ckAI, vkAR, vkAI


class FermionicHEOMSolver(HEOMSolver):
    """
    Same as BosonicHEOMSolver, but with Fermionic baths.

    Attributes
    ----------
    H_sys : Qobj or QobjEvo or list
        System Hamiltonian
        Or
        Liouvillian
        Or
        QobjEvo
        Or
        list of Hamiltonians with time dependence

        Format for input (if list):
        [time_independent_part, [H1, time_dep_function1],
        [H2, time_dep_function2]]

    coup_op : Qobj or list
        Operator describing the coupling between system and bath.
        Could also be a list of operators, which needs to be the
        same length as ck's and vk's.

    ck, vk : lists
        Lists containing spectral density correlation

    N_cut : int
        Cutoff parameter for the bath

    options : :class:`qutip.solver.Options`
        Generic solver options.
        If set to None the default options will be used
    """

    def __init__(self, H_sys, coup_op, ck, vk, N_cut, options=None):
        bath = FermionicBath(coup_op, ck, vk)
        super().__init__(H_sys, bath, N_cut, options=options)
