"""
This module provides solvers for system-bath evoluation using the
HEOM (hierarchy equations of motion).

See https://en.wikipedia.org/wiki/Hierarchical_equations_of_motion for a very
basic introduction to the technique.

The implementation is derived from the BoFiN library (see
https://github.com/tehruhn/bofin) which was itself derived from an earlier
implementation in QuTiP itself.
"""

import enum
from copy import deepcopy

import numpy as np
import scipy.sparse as sp
import scipy.integrate
from scipy.sparse.linalg import splu
from scipy.linalg import eigvalsh

from qutip import settings
from qutip import state_number_enumerate
from qutip.qobj import Qobj
from qutip.qobjevo import QobjEvo
from qutip.superoperator import liouvillian, spre, spost, vec2mat
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
                raise ValueError("RI bath exponents require ck2")
        else:
            if ck2 is not None:
                raise ValueError(
                    "Second co-efficient (ck2) should only be specified for RI"
                    " bath exponents"
                )

    def _check_sigma_bar_k_offset(self, type, offset):
        if type in (self.types["+"], self.types["-"]):
            if offset is None:
                raise ValueError("+ and - bath exponents require sigma_bar_k")
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

    Parameters
    ----------
    exponents : list of BathExponent
        The exponents of the correlation function describing the bath.

    Attributes
    ----------

    All of the parameters are available as attributes.
    """
    def __init__(self, exponents):
        self.exponents = exponents


class BosonicBath(Bath):
    """
    A helper class for constructing a bosonic bath from the expansion
    coefficients and frequencies for the real and imaginary parts of
    the bath correlation function.

    Parameters
    ----------
    Q : Qobj or list of Qobj
        The coupling operator for the bath. If a list is provided, it
        represents the coupling operators of the ckAR exponents, followed
        by the operators for the ckAI exponents.

        FIXME: Separate the real and imaginary coupling operators.

    ckAR : list of complex
        The coefficients of the expansion terms for the real part of the
        correlation function. The corresponding frequencies are passed as
        vkAR.

    vkAR : list of complex
        The frequencies (exponents) of the expansion terms for the real part of
        the correlation function. The corresponding ceofficients are passed as
        ckAR.

    ckAI : list of complex
        The coefficients of the expansion terms in the imaginary part of the
        correlation function. The corresponding frequencies are passed as
        vkAI.

    vkAI : list of complex
        The frequencies (exponents) of the expansion terms for the imaginary
        part of the correlation function. The corresponding ceofficients are
        passed as ckAI.

    combine : bool, default True
        Whether to combine exponents with the same frequency (and coupling
        operator). See :meth:`combine` for details.
    """
    def _check_cks_and_vks(self, ckAR, vkAR, ckAI, vkAI):
        if len(ckAI) != len(vkAI) or len(ckAR) != len(vkAR):
            raise ValueError(
                "The bath exponent lists ckAI and vkAI, and ckAR and vkAR must"
                " be the same length."
            )

    def __init__(self, Q, ckAR, vkAR, ckAI, vkAI, combine=True):
        self._check_cks_and_vks(ckAR, vkAR, ckAI, vkAI)
        Q = _convert_coup_op(Q, len(ckAR) + len(ckAI))

        exponents = []
        exponents.extend(
            BathExponent("R", None, Qk, ck, vk)
            for Qk, ck, vk in zip(Q[:len(ckAR)], ckAR, vkAR))
        exponents.extend(
            BathExponent("I", None, Qk, ck, vk)
            for Qk, ck, vk in zip(Q[len(ckAR):], ckAI, vkAI))

        if combine:
            exponents = self.combine(exponents)

        super().__init__(exponents)

    @classmethod
    def combine(cls, exponents, rtol=1e-5, atol=1e-7):
        """
        Group bosonic exponents with the same frequency and return a
        single exponent for each frequency present.

        Exponents with the same frequency are only combined if they share the
        same coupling operator ``.Q``.

        Parameters
        ----------
        exponents : list of BathExponent
            The list of exponents to combine.

        rtol : float, default 1e-5
            The relative tolerance to use to when comparing frequencies and
            coupling operators.

        atol : float, default 1e-7
            The absolute tolerance to use to when comparing frequencies and
            coupling operators.

        Return
        ------
        list of BathExponent
            The new reduced list of exponents.
        """
        groups = []
        remaining = exponents[:]

        while remaining:
            e1 = remaining.pop(0)
            group = [e1]
            for e2 in remaining[:]:
                if (
                    np.isclose(e1.vk, e2.vk, rtol=rtol, atol=atol) and
                    np.allclose(e1.Q, e2.Q, rtol=rtol, atol=atol)
                ):
                    group.append(e2)
                    remaining.remove(e2)
            groups.append(group)

        new_exponents = []
        for combine in groups:
            exp1 = combine[0]
            if len(combine) == 1:
                new_exponents.append(exp1)
            elif all(exp2.type == exp1.type for exp2 in combine):
                ck = sum(exp.ck for exp in combine)
                new_exponents.append(
                    BathExponent(e1.type, None, e1.Q, ck, e1.vk)
                )
            else:
                ck_R = (
                    sum(exp.ck for exp in combine if exp.type == exp.types.R) +
                    sum(exp.ck for exp in combine if exp.type == exp.types.RI)
                )
                ck_I = (
                    sum(exp.ck for exp in combine if exp.type == exp.types.I) +
                    sum(exp.ck2 for exp in combine if exp.type == exp.types.RI)
                )
                new_exponents.append(
                    BathExponent("RI", None, exp1.Q, ck_R, exp1.vk, ck2=ck_I)
                )

        return new_exponents


class DrudeLorentzBath(BosonicBath):
    """
    A helper class for constructing a Drude-Lorentz bosonic bath from the
    bath parameters (see parameters below).

    Parameters
    ----------
    Q : Qobj
        Operator describing the coupling between system and bath.

    lam : float
        Coupling strength.

    T : float
        Bath temperature.

    Nk : int
        Number of exponential terms used to approximate the bath correlation
        functions.

    gamma : float
        Bath spectral density cutoff frequency.

    terminator : bool, default False
        Whether to calculate the Matsubara terminator for the bath. If
        true, the calculate terminator is provided via the ``.terminator``
        attibute. Otherwise, ``.terminator`` is set to ``None``. The
        terminator is a Liouvillian term, so it's dimensions are those of
        a superoperator of ``Q``.

    Attributes
    ----------
    terminator : Qobj
        The Matsubara terminator -- i.e. a liouvillian term representing the
        contribution to the system-bath dynamics of all exponential expansion
        terms beyond Nk. It should be used by adding it to the system
        liouvillian (i.e. ``liouvillian(H_sys)``).
    """
    def __init__(
        self, Q, lam, T, Nk, gamma, terminator=False,
    ):
        ckAR, vkAR, ckAI, vkAI = self._matsubara_params(
            lam=lam,
            gamma=gamma,
            Nk=Nk,
            T=T,
        )
        if terminator:
            self.terminator = self._matsubara_terminator(
                Q=Q, lam=lam, gamma=gamma, Nk=Nk, T=T
            )
        else:
            self.terminator = None
        super().__init__(Q, ckAR, vkAR, ckAI, vkAI)

    def _matsubara_terminator(self, Q, lam, gamma, Nk, T):
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

    def _matsubara_params(self, lam, gamma, Nk, T):
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

        return ckAR, vkAR, ckAI, vkAI


class DrudeLorentzPadeBath(BosonicBath):
    """
    A helper class for constructing a Padé expansion for a Drude-Lorentz
    bosonic bath from the bath parameters (see parameters below).

    This is an alternative to the :class:`DrudeLorentzBath` which constructs
    a simpler exponential expansion.

    Parameters
    ----------
    Q : Qobj
        Operator describing the coupling between system and bath.

    lam : float
        Coupling strength.

    T : float
        Bath temperature.

    Nk : int
        Number of exponential terms used to approximate the bath correlation
        functions

    gamma : float
        Bath spectral density cutoff frequency.

    lmax : int, default: 2
        FIXME: Some kind of cut off. Default to 2?

    Attributes
    ----------
    eta_p : list of complex
        FIXME: Some parameters of some sort.

    gamma_p: list of complex
        FIXME: Another list of paramaters of some sort.
    """
    def __init__(self, Q, lam, T, Nk, gamma, lmax):
        eta_p, gamma_p = self._pade_corr(lam=lam, gamma=gamma, T=T, lmax=lmax)

        ckAR = [np.real(eta) for eta in eta_p]
        vkAR = [gam for gam in gamma_p]
        ckAI = [np.imag(eta_p[0])]
        vkAI = [gamma_p[0]]

        super().__init__(Q, ckAR, vkAR, ckAI, vkAI)
        self.eta_p = eta_p
        self.gamma_p = gamma_p

    def _delta(self, i, j):
        return 1.0 if i == j else 0.0

    def _cot(self, x):
        return 1. / np.tan(x)

    def _corr(self, lam, gamma, T, lmax):
        beta = 1. / T
        kappa, epsilon = self._kappa_epsilon(lmax)

        eta_p = [lam * gamma * (self._cot(gamma * beta / 2.0) - 1.0j)]
        gamma_p = [gamma]

        for ll in range(1, lmax + 1):
            eta_p.append(
                (kappa[ll] / beta) * 4 * lam * gamma * (epsilon[ll] / beta)
                / ((epsilon[ll]**2 / beta**2) - gamma**2)
            )
            gamma_p.append(epsilon[ll] / beta)

        return eta_p, gamma_p

    def _kappa_epsilon(self, lmax):
        eps = self._calc_eps(lmax)
        chi = self._calc_chi(lmax)

        kappa = [0]
        prefactor = 0.5 * lmax * (2 * (lmax + 1) + 1)
        for j in range(lmax):
            term = prefactor
            for k in range(lmax - 1):
                term *= (
                    (chi[k]**2 - eps[j]**2) /
                    (eps[k]**2 - eps[j]**2 + self._delta(j, k))
                )
            for k in range(lmax - 1, lmax):
                term /= (eps[k]**2 - eps[j]**2 + self._delta(j, k))
            kappa.append(term)

        epsilon = [0] + eps

        return kappa, epsilon

    def _calc_eps(self, lmax):
        alpha = np.zeros((2 * lmax, 2 * lmax))
        for j in range(2 * lmax):
            for k in range(2 * lmax):
                alpha[j][k] = (
                    self._delta(j, k + 1) + self._delta(j, k - 1)
                ) / np.sqrt((2 * (j + 1) + 1) * (2 * (k + 1) + 1))
        evals = eigvalsh(alpha)
        eps = [-2. / val for val in evals[0: lmax]]
        return eps

    def _calc_chi(self, lmax):
        alpha_p = np.zeros((2 * lmax - 1, 2 * lmax - 1))
        for j in range(2 * lmax - 1):
            for k in range(2 * lmax - 1):
                alpha_p[j][k] = (
                    self._delta(j, k + 1) + self._delta(j, k - 1)
                ) / np.sqrt((2 * (j + 1) + 3) * (2 * (k + 1) + 3))
        evals = eigvalsh(alpha_p)
        chi = [-2. / val for val in evals[0: lmax - 1]]
        return chi


class FermionicBath(Bath):
    """
    A helper class for constructing a fermionic bath from the expansion
    coefficients and frequencies for the + and - modes of
    the bath correlation function.

    Parameters
    ----------
    Q : Qobj or list of Qobj
        The coupling operator for the bath. If a list is provided, it
        represents the coupling operators for each exponent in the expansion
        and the list should contain one operator per element of ``ck`` /
        ``vk``. If the operator for a ``+`` mode term is ``Q``, the
        operator for the corresponding ``-`` mode term is typically
        ``Q.dag()``.

    ck : list of complex
        The coefficients of the expansion terms. The even elements of the
        list are coefficients for ``+`` modes and the odd elements are
        coefficients for the ``-`` modes. The corresponding frequencies
        are passed as ``vk``.

        FIXME: Move the + and - modes into separate lists.

    vk : list of complex
        The frequencies (exponents) of the expansion terms. The even elements
        of the list are frequencies for ``+`` modes and the odd elements are
        frequencies for the ``-`` modes. The corresponding coefficients
        are passed as ``ck``.
    """

    def _check_ck_and_vk(self, ck, vk):
        if (len(ck) != len(vk)
                or any(len(ck[i]) != len(vk[i]) for i in range(len(ck)))):
            raise ValueError("Exponents ck and vk must be the same length.")

    def __init__(self, Q, ck, vk):
        self._check_ck_and_vk(ck, vk)
        Q = _convert_coup_op(Q, len(ck))

        exponents = []
        for i in range(len(ck)):
            # FIXME: currently "-" modes are generated by adding extra
            # baths outside with Q == Q.dag() when calling
            # FermionicHEOMSolver
            if i % 2 == 0:
                type = "+"
                sbk_offset = len(ck[i])
            else:
                type = "-"
                sbk_offset = -len(ck[i - 1])
            exponents.extend(
                BathExponent(
                    type, 2, Q[i], ck[i][j], vk[i][j],
                    sigma_bar_k_offset=sbk_offset
                )
                for j in range(len(ck[i]))
            )

        super().__init__(exponents)


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

    cutoff : int
        The maximum depth of the hierarchy (i.e. the maximum sum of
        "excitations" in the hierarchy ADO labels).

    Attributes
    ----------
    exponents : list of BathExponent
        The exponents of the correlation function describing the bath or
        baths.

    cutoff : int
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
        A list of the state labels within the bath.
    """
    def __init__(self, exponents, cutoff):
        self.exponents = exponents
        self.cutoff = cutoff

        self.dims = [exp.dim or (cutoff + 1) for exp in self.exponents]
        self.vk = [exp.vk for exp in self.exponents]
        self.ck = [exp.ck for exp in self.exponents]
        self.ck2 = [exp.ck2 for exp in self.exponents]
        self.sigma_bar_k_offset = [
            exp.sigma_bar_k_offset for exp in self.exponents
        ]

        self.labels = list(state_number_enumerate(self.dims, cutoff))
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
        dimension or bath cutoff.

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
        if sum(label) >= self.cutoff:
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

        self.ados = HierarchyADOs(bath.exponents, N_cut)
        self.n_ados = len(self.ados.labels)

        self.coup_op = [mode.Q for mode in self.ados.exponents]
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

        self._configure_solver()

    def _dsuper_list_td(self, t, y, L_list):
        """ Auxiliary function for the integration. Called every time step. """
        L = L_list[0][0]
        for n in range(1, len(L_list)):
            L = L + L_list[n][0] * L_list[n][1](t)
        return L * y

    def _grad_n(self, L, he_n):
        """ Get the gradient for the hierarchy ADO at level n. """
        vk = self.ados.vk
        vk_sum = sum(he_n[i] * vk[i] for i in range(len(vk)))
        op = L - vk_sum * self.sId
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
            op = (-1j * he_n[k] * self.ados.ck[k]) * self.s_pre_minus_post_Q[k]
        elif self.ados.exponents[k].type == BathExponent.types.I:
            op = (-1j * he_n[k] * 1j * self.ados.ck[k]) * (
                    self.s_pre_plus_post_Q[k]
                )
        elif self.ados.exponents[k].type == BathExponent.types.RI:
            term1 = (he_n[k] * -1j * self.ados.ck[k]) * (
                self.s_pre_minus_post_Q[k]
            )
            term2 = (he_n[k] * self.ados.ck2[k]) * self.s_pre_plus_post_Q[k]
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

        op = -1j * sign2 * (
            (ck[k] * self.spreQ[k]) -
            (sign1 * np.conj(ck[sigma_bar_k] * self.spostQ[k]))
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
        nhe = len(self.ados.labels)
        ops = _GatherHEOMRHS(self.ados.idx, block=L.shape[0], nhe=nhe)

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
        RHSmat = self._rhs(self.L0.data)
        assert isinstance(RHSmat, sp.csr_matrix)

        if self.is_timedep:
            h_identity_mat = sp.identity(self.n_ados, format="csr")
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
        n = self._sys_shape

        b_mat = np.zeros(n ** 2 * self.n_ados, dtype=complex)
        b_mat[0] = 1.0

        L = deepcopy(self.RHSmat)
        L = L.tolil()
        L[0, 0: n ** 2 * self.n_ados] = 0.0
        L = L.tocsr()
        L += sp.csr_matrix((
            np.ones(n),
            (np.zeros(n), [num * (n + 1) for num in range(n)])
        ), shape=(n ** 2 * self.n_ados, n ** 2 * self.n_ados))

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

        solution = solution.reshape((self.n_ados, n ** 2))

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
        n = self._sys_shape
        rho_shape = (n, n)
        rho_dims = self.coup_op[0].dims
        hierarchy_shape = (self.n_ados, n ** 2)

        output = Result()
        output.solver = "HEOMSolver"
        output.times = tlist
        output.states = []

        if full_init:
            rho0_he = rho0
        else:
            rho0_he = np.zeros([n ** 2 * self.n_ados], dtype=complex)
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
        bath = DrudeLorentzBath(
            Q=coup_op,
            lam=coup_strength,
            gamma=cut_freq,
            Nk=N_exp,
            T=temperature,
            terminator=bnd_cut_approx,
        )

        if bnd_cut_approx:
            H_sys = _convert_h_sys(H_sys)
            H_sys = liouvillian(H_sys) + bath.terminator

        super().__init__(H_sys, bath, N_cut, options=options)

        # store input parameters as attributes for politeness and compatibility
        # with HSolverDL in QuTiP 4.6 and below.
        self.coup_strength = coup_strength
        self.cut_freq = cut_freq
        self.temperature = temperature
        self.N_exp = N_exp
        self.bnd_cut_approx = bnd_cut_approx


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
