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
    Q : Qobj
        The coupling operator for the bath.

    ck_real : list of complex
        The coefficients of the expansion terms for the real part of the
        correlation function. The corresponding frequencies are passed as
        vk_real.

    vk_real : list of complex
        The frequencies (exponents) of the expansion terms for the real part of
        the correlation function. The corresponding ceofficients are passed as
        ck_real.

    ck_imag : list of complex
        The coefficients of the expansion terms in the imaginary part of the
        correlation function. The corresponding frequencies are passed as
        vk_imag.

    vk_imag : list of complex
        The frequencies (exponents) of the expansion terms for the imaginary
        part of the correlation function. The corresponding ceofficients are
        passed as ck_imag.

    combine : bool, default True
        Whether to combine exponents with the same frequency (and coupling
        operator). See :meth:`combine` for details.
    """
    def _check_cks_and_vks(self, ck_real, vk_real, ck_imag, vk_imag):
        if len(ck_real) != len(vk_real) or len(ck_imag) != len(vk_imag):
            raise ValueError(
                "The bath exponent lists ck_real and vk_real, and ck_imag and"
                " vk_imag must be the same length."
            )

    def _check_coup_op(self, Q):
        if not isinstance(Q, Qobj):
            raise ValueError("The coupling operator Q must be a Qobj.")

    def __init__(self, Q, ck_real, vk_real, ck_imag, vk_imag, combine=True):
        self._check_cks_and_vks(ck_real, vk_real, ck_imag, vk_imag)
        self._check_coup_op(Q)

        exponents = []
        exponents.extend(
            BathExponent("R", None, Q, ck, vk)
            for ck, vk in zip(ck_real, vk_real)
        )
        exponents.extend(
            BathExponent("I", None, Q, ck, vk)
            for ck, vk in zip(ck_imag, vk_imag)
        )

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
            if (exp1.type != exp1.types.RI) and all(
                exp2.type == exp1.type for exp2 in combine
            ):
                # the group is either type I or R
                ck = sum(exp.ck for exp in combine)
                new_exponents.append(
                    BathExponent(exp1.type, None, exp1.Q, ck, exp1.vk)
                )
            else:
                # the group includes both type I and R exponents
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
        ck_real, vk_real, ck_imag, vk_imag = self._matsubara_params(
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
        super().__init__(Q, ck_real, vk_real, ck_imag, vk_imag)

    def _matsubara_terminator(self, Q, lam, gamma, Nk, T):
        """ Calculate the hierarchy terminator term for the Liouvillian. """
        beta = 1 / T

        op = -2*spre(Q)*spost(Q.dag()) + spre(Q.dag()*Q) + spost(Q.dag()*Q)
        approx_factr = ((2 * lam / (beta * gamma)) - 1j * lam)
        approx_factr -= (
            lam * gamma * (-1.0j + 1 / np.tan(gamma / (2 * T))) / gamma
        )

        for k in range(1, Nk + 1):
            vk = 2 * np.pi * k * T
            approx_factr -= (
                (4 * lam * gamma * T * vk / (vk**2 - gamma**2)) / vk
            )

        L_bnd = -approx_factr * op
        return L_bnd

    def _matsubara_params(self, lam, gamma, Nk, T):
        """ Calculate the Matsubara coefficents and frequencies. """
        ck_real = [lam * gamma * (1/np.tan(gamma / (2 * T)))]
        ck_real.extend([
            (4 * lam * gamma * T * 2 * np.pi * k * T /
                ((2 * np.pi * k * T)**2 - gamma**2))
            for k in range(1, Nk + 1)
        ])
        vk_real = [gamma]
        vk_real.extend([2 * np.pi * k * T for k in range(1, Nk + 1)])

        ck_imag = [lam * gamma * (-1.0)]
        vk_imag = [gamma]

        return ck_real, vk_real, ck_imag, vk_imag


class DrudeLorentzPadeBath(BosonicBath):
    """
    A helper class for constructing a Padé expansion for a Drude-Lorentz
    bosonic bath from the bath parameters (see parameters below).

    A Padé approximant is a sum-over-poles expansion (
    see `https://en.wikipedia.org/wiki/Pad%C3%A9_approximant`_).

    The application of the Padé method to spectrum decompoisitions is described
    in "Padé spectrum decompositions of quantum distribution functions and
    optimal hierarchical equations of motion construction for quantum open
    systems" [1].

    The implementation here follows the approach in the paper.

    [1] J. Chem. Phys. 134, 244106 (2011); https://doi.org/10.1063/1.3602466

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
        Number of Padé exponentials terms used to approximate the bath
        correlation functions.

    gamma : float
        Bath spectral density cutoff frequency.
    """
    def __init__(self, Q, lam, T, Nk, gamma):
        eta_p, gamma_p = self._corr(lam=lam, gamma=gamma, T=T, Nk=Nk)

        ck_real = [np.real(eta) for eta in eta_p]
        vk_real = [gam for gam in gamma_p]
        # There is only one term in the expansion of the imaginary part of the
        # Drude-Lorentz correlation function.
        ck_imag = [np.imag(eta_p[0])]
        vk_imag = [gamma_p[0]]

        super().__init__(Q, ck_real, vk_real, ck_imag, vk_imag)

    def _delta(self, i, j):
        return 1.0 if i == j else 0.0

    def _cot(self, x):
        return 1. / np.tan(x)

    def _corr(self, lam, gamma, T, Nk):
        beta = 1. / T
        kappa, epsilon = self._kappa_epsilon(Nk)

        eta_p = [lam * gamma * (self._cot(gamma * beta / 2.0) - 1.0j)]
        gamma_p = [gamma]

        for ll in range(1, Nk + 1):
            eta_p.append(
                (kappa[ll] / beta) * 4 * lam * gamma * (epsilon[ll] / beta)
                / ((epsilon[ll]**2 / beta**2) - gamma**2)
            )
            gamma_p.append(epsilon[ll] / beta)

        return eta_p, gamma_p

    def _kappa_epsilon(self, Nk):
        eps = self._calc_eps(Nk)
        chi = self._calc_chi(Nk)

        kappa = [0]
        prefactor = 0.5 * Nk * (2 * (Nk + 1) + 1)
        for j in range(Nk):
            term = prefactor
            for k in range(Nk - 1):
                term *= (
                    (chi[k]**2 - eps[j]**2) /
                    (eps[k]**2 - eps[j]**2 + self._delta(j, k))
                )
            for k in range(Nk - 1, Nk):
                term /= (eps[k]**2 - eps[j]**2 + self._delta(j, k))
            kappa.append(term)

        epsilon = [0] + eps

        return kappa, epsilon

    def _calc_eps(self, Nk):
        alpha = np.zeros((2 * Nk, 2 * Nk))
        for j in range(2 * Nk):
            for k in range(2 * Nk):
                alpha[j][k] = (
                    self._delta(j, k + 1) + self._delta(j, k - 1)
                ) / np.sqrt((2 * (j + 1) + 1) * (2 * (k + 1) + 1))
        evals = eigvalsh(alpha)
        eps = [-2. / val for val in evals[0: Nk]]
        return eps

    def _calc_chi(self, Nk):
        alpha_p = np.zeros((2 * Nk - 1, 2 * Nk - 1))
        for j in range(2 * Nk - 1):
            for k in range(2 * Nk - 1):
                alpha_p[j][k] = (
                    self._delta(j, k + 1) + self._delta(j, k - 1)
                ) / np.sqrt((2 * (j + 1) + 3) * (2 * (k + 1) + 3))
        evals = eigvalsh(alpha_p)
        chi = [-2. / val for val in evals[0: Nk - 1]]
        return chi


class FermionicBath(Bath):
    """
    A helper class for constructing a fermionic bath from the expansion
    coefficients and frequencies for the ``+`` and ``-`` modes of
    the bath correlation function.

    There must be the same number of ``+`` and ``-`` modes and their
    coefficients must be specified in the same order so that ``ck_plus[i],
    vk_plus[i]`` are the plus coefficient and frequency corresponding
    to the minus mode ``ck_minus[i], vk_minus[i]``.

    Parameters
    ----------
    Q : Qobj
        The coupling operator for the bath. ``Q.dag()`` is used as the coupling
        operator for ``+`` mode terms and ``Q`` for the ``-`` mode terms.

    ck_plus : list of complex
        The coefficients of the expansion terms for the ``+`` part of the
        correlation function. The corresponding frequencies are passed as
        vk_plus.

    vk_plus : list of complex
        The frequencies (exponents) of the expansion terms for the ``+`` part
        of the correlation function. The corresponding ceofficients are passed
        as ck_plus.

    ck_minus : list of complex
        The coefficients of the expansion terms for the ``-`` part of the
        correlation function. The corresponding frequencies are passed as
        vk_minus.

    vk_minus : list of complex
        The frequencies (exponents) of the expansion terms for the ``-`` part
        of the correlation function. The corresponding ceofficients are passed
        as ck_minus.
    """

    def _check_cks_and_vks(self, ck_plus, vk_plus, ck_minus, vk_minus):
        if len(ck_plus) != len(vk_plus) or len(ck_minus) != len(vk_minus):
            raise ValueError(
                "The bath exponent lists ck_plus and vk_plus, and ck_minus and"
                " vk_minus must be the same length."
            )
        if len(ck_plus) != len(ck_minus):
            raise ValueError(
                "The must be the same number of plus and minus exponents"
                " in the bath, and elements of plus and minus arrays"
                " should be arranged so that ck_plus[i] is the plus mode"
                " corresponding to ck_minus[i]."
            )

    def _check_coup_op(self, Q):
        if not isinstance(Q, Qobj):
            raise ValueError("The coupling operator Q must be a Qobj.")

    def __init__(self, Q, ck_plus, vk_plus, ck_minus, vk_minus):
        self._check_cks_and_vks(ck_plus, vk_plus, ck_minus, vk_minus)
        self._check_coup_op(Q)
        Qdag = Q.dag()

        exponents = []
        for ckp, vkp, ckm, vkm in zip(ck_plus, vk_plus, ck_minus, vk_minus):
            exponents.append(BathExponent(
                "+", 2, Qdag, ckp, vkp, sigma_bar_k_offset=1,
            ))
            exponents.append(BathExponent(
                "-", 2, Q, ckm, vkm, sigma_bar_k_offset=-1,
            ))
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
        "excitations" in the hierarchy ADO labels or maximum ADO level).

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
        A list of the ADO labels within the hierarchy.
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

    def filter_by_level(self, level):
        """
        Return a list of ADO indexes and labels for ADOs at the
        given level. An ADO is at a particular level if
        the sum of the "excitations" in its label equals the level.

        Parameters
        ----------
        level : int
            The hierarchy depth to return ADOs from.

        Returns
        -------
        list of (idx, label)
            The ADO index and lable for each ADO.
        """
        results = []
        for idx, label in enumerate(self.labels):
            if sum(label) == level:
                results.append((idx, label))
        return results


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

    baths : Bath or list of Bath
        A :obj:`Bath` containing the exponents of the expansion of the
        bath correlation funcion and their associated coefficients
        and coupling operators.

        If multiple baths are given, they must all be either fermionic
        or bosonic baths.

    max_depth : int
        The maximum depth of the heirarchy (i.e. the maximum number of bath
        exponent "excitations" to retain).

    options : :class:`qutip.solver.Options`
        Generic solver options. If set to None the default options will be
        used.

    Attributes
    ----------
    ados : :obj:`HierarchyADOs`
        The description of the hierarchy constructed from the given bath
        and maximum depth.
    """
    def __init__(self, H_sys, bath, max_depth, options=None):
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

        self.progress_bar = BaseProgressBar()

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

        op = -1j * sign2 * (
            (ck[k] * self._spreQ[k]) -
            (sign1 * np.conj(ck[sigma_bar_k] * self._spostQ[k]))
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

        if sign1 == -1:
            op = (-1j * sign2) * self._s_pre_minus_post_Q[k]
        else:
            op = (-1j * sign2) * self._s_pre_plus_post_Q[k]

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

    def extract_ado(self, ado_state, idx):
        """
        Extract a Qobj representing specified ADO from a full representation of
        the ADO states, as returned by :meth:`.steady_state` or :meth:`.run`.

        Parameters
        ----------
        ado_state : numpy array
            A full representation of the ADO state.
        idx : int
            The index of the ADO to extract.

        Returns
        -------
        Qobj
            A :obj:`Qobj` representing the state of the specified ADO.
        """
        return Qobj(ado_state[idx, :].T, dims=self._sys_dims)

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

        full_ado_state : Numpy array
            Array of the the steady-state and all ADOs.
            The state of a particular ADO may be extracted from the
            full state by calling :meth:`.extract_ado`.
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

        solution = solution.reshape((self._n_ados, n, n))

        return Qobj(data, dims=self._H0.dims), solution

    def run(self, rho0, tlist, e_ops=None, ado_init=False, ado_return=False):
        """
        Solve for the time evolution of the system.

        Parameters
        ----------
        rho0 : Qobj or numpy.array
            Initial state (density matrix) of the system
            if ``full_init`` is ``False``.
            If ``full_init`` is ``True``, then ``rho0`` should be a numpy array
            of the initial state and the initial state of all ADOs. Usually
            the state of the ADOs would be determine from a previous call
            to ``.run(..., full_return=True)``.

        tlist : list
            An ordered list of times at which to return the value of the state.

        e_ops : Qobj / callable / list / None, optional
            A list of operators as `Qobj` and/or callable functions (can be
            mixed) or a single operator or callable function. For operators op,
            the result's expect will be computed by ``(state * op).tr()``. For
            callable functions, they are called as ``f(solver, t, state)`` and
            return the expectation value.

        ado_init: bool, default False
            Indicates if initial condition is just the system state, or a
            numpy array including all ADOs.

        ado_return: bool, default True
            Whether to also return as output the full state of all ADOs.

        Returns
        -------
        :class:`qutip.solver.Result`
            The results of the simulation run.
            The times (tlist) are stored in ``result.times``.
            The state at each time is stored in ``result.states``.
            If ``ado_return`` is ``True``, then the full ADO state at each
            time is stored in ``result.ado_states``.
            The state of a particular ADO may be extracted from
            ``result.ado_states[i]`` by calling :meth:`.extract_ado`.
        """

        if e_ops is None:
            e_ops = []

        if isinstance(e_ops, dict):
            e_ops_dict = e_ops
            e_ops = [e for e in e_ops.values()]
        else:
            e_ops_dict = None

        if isinstance(e_ops, Qobj):
            e_ops = [e_ops]
        try:
            _ = iter(e_ops)
        except TypeError:
            e_ops = [e_ops]

        ado_states_required = ado_return
        for e_op in e_ops:
            if not callable(e_op):
                raise ValueError("The e_ops list must only contain QObj objects or callback functions")
            if not isinstance(e_op, Qobj): # callable but not QObj -> callback function
                ado_states_required = True

        n = self._sys_shape
        rho_shape = (n, n)
        rho_dims = self._sys_dims
        hierarchy_shape = (self._n_ados, n, n)

        output = Result()
        output.solver = "HEOMSolver"
        output.times = tlist
        output.num_expect = len(e_ops)
        if output.num_expect > 0:
            output.expect = [[] for _ in range(output.num_expect)]
        if self.options.store_states:
            output.states = []

        if ado_init:
            rho0_he = rho0
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
            rho = Qobj(
                solver.y[:n ** 2].reshape(rho_shape, order='F'),
                dims=rho_dims,
            )
            if self.options.store_states:
                output.states.append(rho)
            if ado_states_required:
                ado_state = solver.y.reshape(hierarchy_shape)
            if ado_return:
                output.ado_states.append(ado_state)

            for cnt, e_op in enumerate(e_ops):
                if isinstance(e_op, Qobj):
                    output.expect[cnt].append((rho * e_op).tr())
                else:
                    output.expect[cnt].append(e_op(self, t, ado_state))

        if e_ops_dict:
            output.expect = {e: output.expect[n]
                             for n, e in enumerate(e_ops_dict.keys())}

        self.progress_bar.finished()
        return output


class BosonicHEOMSolver(HEOMSolver):
    """
    A helper class for creating an :class:`HEOMSolver` for a single
    bosonic bath.

    See :class:`HEOMSolver` and :class:`BosonicBath` for more detailed
    descriptions of the solver and bath parameters.

    Parameters
    ----------
    H_sys : Qobj or QobjEvo or list
        The system Hamiltonian or Liouvillian. See :class:`HEOMSolver` for
        a complete description.

    Q : Qobj
        Operator describing the coupling between system and bath.
        See :class:`BosonicBath` for a complete description.

    ck_real, vk_real, ck_imag, vk_imag : lists
        Lists containing coefficients of the fitted bath correlation
        functions. See :class:`BosonicBath` for a complete description.

    max_depth : int
        The maximum depth of the heirarchy (i.e. the maximum number of bath
        exponent "excitations" to retain). See :class:`HEOMSolver` for
        a complete description.

    options : :class:`qutip.solver.Options`
        Generic solver options. If set to None the default options will be
        used. See :class:`HEOMSolver` for a complete description.
    """
    def __init__(
        self, H_sys, Q, ck_real, vk_real, ck_imag, vk_imag, max_depth,
        options=None,
    ):
        bath = BosonicBath(Q, ck_real, vk_real, ck_imag, vk_imag)
        super().__init__(
            H_sys=H_sys, bath=bath, max_depth=max_depth, options=options,
        )


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

    options : :class:`qutip.solver.Options`
        Generic solver options.
        If set to None the default options will be used.
    """
    def __init__(
        self, H_sys, coup_op, coup_strength, temperature,
        N_cut, N_exp, cut_freq, bnd_cut_approx=False, options=None,
    ):
        bath = DrudeLorentzBath(
            Q=coup_op,
            lam=coup_strength,
            gamma=cut_freq,
            Nk=N_exp - 1,
            T=temperature,
            terminator=bnd_cut_approx,
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
            H_sys = H_sys + bath.terminator

        super().__init__(H_sys, bath=bath, max_depth=N_cut, options=options)

        # store input parameters as attributes for politeness and compatibility
        # with HSolverDL in QuTiP 4.6 and below.
        self.coup_strength = coup_strength
        self.cut_freq = cut_freq
        self.temperature = temperature
        self.N_exp = N_exp
        self.bnd_cut_approx = bnd_cut_approx


class FermionicHEOMSolver(HEOMSolver):
    """
    A helper class for creating an :class:`HEOMSolver` for a single
    fermionic bath.

    See :class:`HEOMSolver` and :class:`FermionicBath` for more detailed
    descriptions of the solver and bath parameters.

    Attributes
    ----------
    H_sys : Qobj or QobjEvo or list
        The system Hamiltonian or Liouvillian. See :class:`HEOMSolver` for
        a complete description.

    Q : Qobj
        Operator describing the coupling between system and bath.
        See :class:`FermionicBath` for a complete description.

    ck_plus, vk_plus, ck_minus, vk_minus : lists
        Lists containing coefficients of the fitted bath correlation
        functions. See :class:`FermionicBath` for a complete description.

    max_depth : int
        The maximum depth of the heirarchy (i.e. the maximum number of bath
        exponent "excitations" to retain). See :class:`HEOMSolver` for
        a complete description.

    options : :class:`qutip.solver.Options`
        Generic solver options. If set to None the default options will be
        used. See :class:`HEOMSolver` for a complete description.
    """
    def __init__(
        self, H_sys, Q, ck_plus, vk_plus, ck_minus, vk_minus, max_depth,
        options=None,
    ):
        bath = FermionicBath(Q, ck_plus, vk_plus, ck_minus, vk_minus)
        super().__init__(
            H_sys, bath=bath, max_depth=max_depth, options=options
        )


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
