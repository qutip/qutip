"""
This module provides utilities for describing baths when using the
HEOM (hierarchy equations of motion) to model system-bath interactions.

See the ``qutip.nonmarkov.bofin_solvers`` module for the associated solver.

The implementation is derived from the BoFiN library (see
https://github.com/tehruhn/bofin) which was itself derived from an earlier
implementation in QuTiP itself.
"""
from time import time
import enum
import numpy as np
from mpmath import mp
from scipy.linalg import eigvalsh
from scipy.optimize import curve_fit
from qutip.core import data as _data
from qutip.core.qobj import Qobj
from qutip.core.superoperator import spre, spost
from qutip.visualization import gen_spectral_plots, gen_corr_plots

__all__ = [
    "BathExponent",
    "Bath",
    "BosonicBath",
    "DrudeLorentzBath",
    "DrudeLorentzPadeBath",
    "UnderDampedBath",
    "FermionicBath",
    "LorentzianBath",
    "LorentzianPadeBath",
    "FitSpectral",
    "FirCorr",
    "OhmicBath",
]


def _isequal(Q1, Q2, tol):
    """ Return true if Q1 and Q2 are equal to within the given tolerance. """
    return _data.iszero(_data.sub(Q1.data, Q2.data), tol=tol)


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

    tag : optional, str, tuple or any other object
        A label for the exponent (often the name of the bath). It
        defaults to None.

    Attributes
    ----------
    fermionic : bool
        True if the type of the exponent is a Fermionic type (i.e. either
        "+" or "-") and False otherwise.

    All of the parameters are also available as attributes.
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
                raise ValueError(
                    "+ and - bath exponents require sigma_bar_k_offset"
                )
        else:
            if offset is not None:
                raise ValueError(
                    "Offset of sigma bar (sigma_bar_k_offset) should only be"
                    " specified for + and - bath exponents"
                )

    def _type_is_fermionic(self, type):
        return type in (self.types["+"], self.types["-"])

    def __init__(
            self, type, dim, Q, ck, vk, ck2=None,
            sigma_bar_k_offset=None, tag=None,
    ):
        if not isinstance(type, self.types):
            type = self.types[type]
        self._check_ck2(type, ck2)
        self._check_sigma_bar_k_offset(type, sigma_bar_k_offset)
        self.type = type
        self.fermionic = self._type_is_fermionic(type)
        self.dim = dim
        self.Q = Q
        self.ck = ck
        self.vk = vk
        self.ck2 = ck2
        self.sigma_bar_k_offset = sigma_bar_k_offset
        self.tag = tag

    def __repr__(self):
        dims = getattr(self.Q, "dims", None)
        return (
            f"<{self.__class__.__name__} type={self.type.name}"
            f" dim={self.dim!r}"
            f" Q.dims={dims!r}"
            f" ck={self.ck!r} vk={self.vk!r} ck2={self.ck2!r}"
            f" sigma_bar_k_offset={self.sigma_bar_k_offset!r}"
            f" fermionic={self.fermionic!r}"
            f" tag={self.tag!r}>"
        )


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

    If the correlation functions ``C(t)`` is split into real and imaginary
    parts::

        C(t) = C_real(t) + i * C_imag(t)

    then::

        C_real(t) = sum(ck_real * exp(- vk_real * t))
        C_imag(t) = sum(ck_imag * exp(- vk_imag * t))

    Defines the coefficients ``ck`` and the frequencies ``vk``.

    Note that the ``ck`` and ``vk`` may be complex, even through ``C_real(t)``
    and ``C_imag(t)`` (i.e. the sum) is real.

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

    tag : optional, str, tuple or any other object
        A label for the bath exponents (for example, the name of the
        bath). It defaults to None but can be set to help identify which
        bath an exponent is from.
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

    def __init__(
            self, Q, ck_real, vk_real, ck_imag, vk_imag, combine=True,
            tag=None,
    ):
        self._check_cks_and_vks(ck_real, vk_real, ck_imag, vk_imag)
        self._check_coup_op(Q)

        exponents = []
        exponents.extend(
            BathExponent("R", None, Q, ck, vk, tag=tag)
            for ck, vk in zip(ck_real, vk_real)
        )
        exponents.extend(
            BathExponent("I", None, Q, ck, vk, tag=tag)
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

        Note that combined exponents take their tag from the first
        exponent in the group being combined (i.e. the one that occurs first
        in the given exponents list).

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

        Returns
        -------
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
                    _isequal(e1.Q, e2.Q, tol=atol)
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
                new_exponents.append(BathExponent(
                    exp1.type, None, exp1.Q, ck, exp1.vk, tag=exp1.tag,
                ))
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
                new_exponents.append(BathExponent(
                    "RI", None, exp1.Q, ck_R, exp1.vk, ck2=ck_I,
                    tag=exp1.tag,
                ))

        return new_exponents

    @classmethod
    def pack(self, a, b, c):
        """Pack parameter lists for fitting."""
        return np.concatenate((a, b, c))

    @classmethod
    def unpack(self, params):
        """Unpack parameter lists for fitting."""
        N = len(params) // 3
        a = params[:N]
        b = params[N: 2 * N]
        c = params[2 * N:]
        return a, b, c

    @classmethod
    def _leastsq(self, func, y, x, guesses=None, lower=None, upper=None, sigma=None):
        """
        Performs nonlinear least squares  to fit the function func to x and y

        Parameters
        ----------
        func : function
            The function we wish to fit.
        x: np.array
            a numpy array containing the independent variable we use for the fit
        y: np.array
            a numpy array containing the dependent variable we use for the fit
        guesses : list
            Initial guess for the parameters.
        lower : list
            lower bounds on the parameters for the fit.
        upper: list
            upper bounds on the parameters for the fit
        sigma:
            uncertainty in the data considered for the fit
        Returns
        -------
        params: list
            It returns the fitted parameters.
        """
        try:
            sigma = [sigma] * len(x)
            if None in [guesses, lower, upper, sigma]:
                raise ValueError(
                    "Fit parameters were not provided"
                )  # maybe unnecessary and can let it go to scipy defaults
        except:
            pass
        params, _ = curve_fit(
            lambda x, *params: func(x, *self.unpack(params)),
            x,
            y,
            p0=guesses,
            bounds=(lower, upper),
            sigma=sigma,
            maxfev=int(1e9),
            method="trf",
        )

        return self.unpack(params)

    @classmethod
    def _rmse(self, func, x, y, lam, gamma, w0):
        """
        Calculates the normalized root Mean squared error for fits, from the fitted parameters lam,gamma,w0

        Parameters
        ----------
        func : function
            The approximated function for which we want to compute the rmse.
        x: np.array
            a numpy array containing the independent variable we used for the fit
        y: np.array
            a numpy array containing the dependent variable we used for the fit
        lam : list
            a listed containing fitted couplings strength.
        gamma : list
            a list containing fitted cutoff frequencies.
        w0s:
            a list containing fitted resonance frequecies
        Returns
        -------
        rmse: float
            The normalized root mean squared error for the fit, the closer to zero the better the fit.
        """
        yhat = func(x, lam, gamma, w0)
        rmse = np.sqrt(np.mean((yhat - y) ** 2) / len(y)) / \
            (np.max(y) - np.min(y))
        return rmse

    @classmethod
    def _fit(
        self, func, C, t, N=4, label="correlation_real",
        guesses=None, lower=None, upper=None, sigma=None
    ):
        """
        Performs a fit the function func to t and C, with N number of terms in func, the guesses,bounds and uncertainty can be determined by the user.
        If none is provided it constructs default ones according to the label.

        Parameters
        ----------
        func : function
            The function we wish to fit.
        x: np.array
            a numpy array containing the independent variable we use for the fit
        y: np.array
            a numpy array containing the dependent variable we use for the fit
        guesses : list
            Initial guess for the parameters.
        lower : list
            lower bounds on the parameters for the fit.
        upper: list
            upper bounds on the parameters for the fit
        sigma: float
            uncertainty in the data considered for the fit

        Returns
        -------
        params:
            It returns the fitted parameters as a list.
        rmse:
            It returns the normalized mean squared error from the fit
        """
        try:
            if None in [guesses, lower, upper, sigma]:
                raise Exception(
                    "No parameters for the fit provided, using default ones"
                )
        except:
            sigma = 1e-4
            C_max = abs(max(C, key=abs))
            wc = t[np.argmax(C)]
            if label == "correlation_real":
                guesses = self.pack([C_max] * N, [-wc] * N, [wc] * N)
                lower = self.pack([-20 * C_max] * N, [-np.inf] * N, [0.0] * N)
                upper = self.pack([20 * C_max] * N, [0.1] * N, [np.inf] * N)
            elif label == "correlation_imag":
                guesses = self.pack([-C_max] * N, [-2] * N, [1] * N)
                lower = self.pack([-5 * C_max] * N, [-100] * N, [0.0] * N)
                upper = self.pack([5 * C_max] * N, [0.01] * N, [100] * N)

            else:
                guesses = self.pack([C_max] * N, [wc] * N, [wc] * N)
                lower = self.pack([-100 * C_max] * N,
                                  [0.1 * wc] * N, [0.1 * wc] * N)
                upper = self.pack([100 * C_max] * N,
                                  [100 * wc] * N, [100 * wc] * N)

        lam, gamma, w0 = self._leastsq(
            func,
            C,
            t,
            sigma=sigma,
            guesses=guesses,
            lower=lower,
            upper=upper,
        )
        rmse = self._rmse(func, t, C, lam=lam, gamma=gamma, w0=w0)
        params = [lam, gamma, w0]
        return rmse, params


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

    gamma : float
        Bath spectral density cutoff frequency.

    T : float
        Bath temperature.

    Nk : int
        Number of exponential terms used to approximate the bath correlation
        functions.

    combine : bool, default True
        Whether to combine exponents with the same frequency (and coupling
        operator). See :meth:`BosonicBath.combine` for details.

    tag : optional, str, tuple or any other object
        A label for the bath exponents (for example, the name of the
        bath). It defaults to None but can be set to help identify which
        bath an exponent is from.
    """

    def __init__(
        self, Q, lam, gamma, T, Nk, combine=True, tag=None,
    ):
        ck_real, vk_real, ck_imag, vk_imag = self._matsubara_params(
            lam=lam,
            gamma=gamma,
            T=T,
            Nk=Nk,
        )

        super().__init__(
            Q, ck_real, vk_real, ck_imag, vk_imag, combine=combine, tag=tag,
        )

        self._dl_terminator = _DrudeLorentzTerminator(
            Q=Q, lam=lam, gamma=gamma, T=T,
        )

    def terminator(self):
        """
        Return the Matsubara terminator for the bath and the calculated
        approximation discrepancy.

        Returns
        -------
        delta: float

            The approximation discrepancy. That is, the difference between the
            true correlation function of the Drude-Lorentz bath and the sum of
            the ``Nk`` exponential terms is approximately ``2 * delta *
            dirac(t)``, where ``dirac(t)`` denotes the Dirac delta function.

        terminator : Qobj

            The Matsubara terminator -- i.e. a liouvillian term representing
            the contribution to the system-bath dynamics of all exponential
            expansion terms beyond ``Nk``. It should be used by adding it to
            the system liouvillian (i.e. ``liouvillian(H_sys)``).
        """
        delta, L = self._dl_terminator.terminator(self.exponents)
        return delta, L

    def _matsubara_params(self, lam, gamma, T, Nk):
        """ Calculate the Matsubara coefficents and frequencies. """
        ck_real = [lam * gamma / np.tan(gamma / (2 * T))]
        ck_real.extend([
            (8 * lam * gamma * T * np.pi * k * T /
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
    see https://en.wikipedia.org/wiki/Pad%C3%A9_approximant).

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

    gamma : float
        Bath spectral density cutoff frequency.

    T : float
        Bath temperature.

    Nk : int
        Number of Padé exponentials terms used to approximate the bath
        correlation functions.

    combine : bool, default True
        Whether to combine exponents with the same frequency (and coupling
        operator). See :meth:`BosonicBath.combine` for details.

    tag : optional, str, tuple or any other object
        A label for the bath exponents (for example, the name of the
        bath). It defaults to None but can be set to help identify which
        bath an exponent is from.
    """

    def __init__(
        self, Q, lam, gamma, T, Nk, combine=True, tag=None
    ):
        eta_p, gamma_p = self._corr(lam=lam, gamma=gamma, T=T, Nk=Nk)

        ck_real = [np.real(eta) for eta in eta_p]
        vk_real = [gam for gam in gamma_p]
        # There is only one term in the expansion of the imaginary part of the
        # Drude-Lorentz correlation function.
        ck_imag = [np.imag(eta_p[0])]
        vk_imag = [gamma_p[0]]

        super().__init__(
            Q, ck_real, vk_real, ck_imag, vk_imag, combine=combine, tag=tag,
        )

        self._dl_terminator = _DrudeLorentzTerminator(
            Q=Q, lam=lam, gamma=gamma, T=T,
        )

    def terminator(self):
        """
        Return the Padé terminator for the bath and the calculated
        approximation discrepancy.

        Returns
        -------
        delta: float

            The approximation discrepancy. That is, the difference between the
            true correlation function of the Drude-Lorentz bath and the sum of
            the ``Nk`` exponential terms is approximately ``2 * delta *
            dirac(t)``, where ``dirac(t)`` denotes the Dirac delta function.

        terminator : Qobj

            The Padé terminator -- i.e. a liouvillian term representing
            the contribution to the system-bath dynamics of all exponential
            expansion terms beyond ``Nk``. It should be used by adding it to
            the system liouvillian (i.e. ``liouvillian(H_sys)``).
        """
        delta, L = self._dl_terminator.terminator(self.exponents)
        return delta, L

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

    def _cot(self, x):
        return 1. / np.tan(x)

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
            for k in [Nk - 1]:
                term /= (eps[k]**2 - eps[j]**2 + self._delta(j, k))
            kappa.append(term)

        epsilon = [0] + eps

        return kappa, epsilon

    def _delta(self, i, j):
        return 1.0 if i == j else 0.0

    def _calc_eps(self, Nk):
        alpha = np.diag([
            1. / np.sqrt((2 * k + 5) * (2 * k + 3))
            for k in range(2 * Nk - 1)
        ], k=1)
        alpha += alpha.transpose()
        evals = eigvalsh(alpha)
        eps = [-2. / val for val in evals[0: Nk]]
        return eps

    def _calc_chi(self, Nk):
        alpha_p = np.diag([
            1. / np.sqrt((2 * k + 7) * (2 * k + 5))
            for k in range(2 * Nk - 2)
        ], k=1)
        alpha_p += alpha_p.transpose()
        evals = eigvalsh(alpha_p)
        chi = [-2. / val for val in evals[0: Nk - 1]]
        return chi


class _DrudeLorentzTerminator:
    """ A class for calculating the terminator of a Drude-Lorentz bath
        expansion.
    """

    def __init__(self, Q, lam, gamma, T):
        self.Q = Q
        self.lam = lam
        self.gamma = gamma
        self.T = T

    def terminator(self, exponents):
        """ Calculate the terminator for a Drude-Lorentz bath. """
        Q = self.Q
        lam = self.lam
        gamma = self.gamma
        beta = 1 / self.T

        delta = 2 * lam / (beta * gamma) - 1j * lam

        for exp in exponents:
            if exp.type == BathExponent.types["R"]:
                delta -= exp.ck / exp.vk
            elif exp.type == BathExponent.types["RI"]:
                delta -= (exp.ck + 1j * exp.ck2) / exp.vk
            else:
                delta -= 1j * exp.ck / exp.vk

        op = -2*spre(Q)*spost(Q.dag()) + spre(Q.dag()*Q) + spost(Q.dag()*Q)
        L_bnd = -delta * op

        return delta, L_bnd


class UnderDampedBath(BosonicBath):
    """
    A helper class for constructing an under-damped bosonic bath from the
    bath parameters (see parameters below).

    Parameters
    ----------
    Q : Qobj
        Operator describing the coupling between system and bath.

    lam : float
        Coupling strength.

    gamma : float
        Bath spectral density cutoff frequency.

    w0 : float
        Bath spectral density resonance frequency.

    T : float
        Bath temperature.

    Nk : int
        Number of exponential terms used to approximate the bath correlation
        functions.

    combine : bool, default True
        Whether to combine exponents with the same frequency (and coupling
        operator). See :meth:`BosonicBath.combine` for details.

    tag : optional, str, tuple or any other object
        A label for the bath exponents (for example, the name of the
        bath). It defaults to None but can be set to help identify which
        bath an exponent is from.
    """

    def __init__(
        self, Q, lam, gamma, w0, T, Nk, combine=True, tag=None,
    ):
        ck_real, vk_real, ck_imag, vk_imag = self._matsubara_params(
            lam=lam,
            gamma=gamma,
            w0=w0,
            T=T,
            Nk=Nk,
        )

        super().__init__(
            Q, ck_real, vk_real, ck_imag, vk_imag, combine=combine, tag=tag,
        )

    def _matsubara_params(self, lam, gamma, w0, T, Nk):
        """ Calculate the Matsubara coefficents and frequencies. """
        beta = 1/T
        Om = np.sqrt(w0**2 - (gamma/2)**2)
        Gamma = gamma/2.

        ck_real = ([
            (lam**2 / (4 * Om))
            * (1 / np.tanh(beta * (Om + 1.0j * Gamma) / 2)),
            (lam**2 / (4*Om))
            * (1 / np.tanh(beta * (Om - 1.0j * Gamma) / 2)),
        ])

        ck_real.extend([
            (-2 * lam**2 * gamma / beta) * (2 * np.pi * k / beta)
            / (
                ((Om + 1.0j * Gamma)**2 + (2 * np.pi * k/beta)**2)
                * ((Om - 1.0j * Gamma)**2 + (2 * np.pi * k / beta)**2)
            )
            for k in range(1, Nk + 1)
        ])

        vk_real = [-1.0j * Om + Gamma, 1.0j * Om + Gamma]
        vk_real.extend([
            2 * np.pi * k * T
            for k in range(1, Nk + 1)
        ])

        ck_imag = [
            1.0j * lam**2 / (4 * Om),
            -1.0j * lam**2 / (4 * Om),
        ]

        vk_imag = [-1.0j * Om + Gamma, 1.0j * Om + Gamma]

        return ck_real, vk_real, ck_imag, vk_imag


class FermionicBath(Bath):
    """
    A helper class for constructing a fermionic bath from the expansion
    coefficients and frequencies for the ``+`` and ``-`` modes of
    the bath correlation function.

    There must be the same number of ``+`` and ``-`` modes and their
    coefficients must be specified in the same order so that ``ck_plus[i],
    vk_plus[i]`` are the plus coefficient and frequency corresponding
    to the minus mode ``ck_minus[i], vk_minus[i]``.

    In the fermionic case the order in which excitations are created or
    destroyed is important, resulting in two different correlation functions
    labelled ``C_plus(t)`` and ``C_plus(t)``::

        C_plus(t) = sum(ck_plus * exp(- vk_plus * t))
        C_minus(t) = sum(ck_minus * exp(- vk_minus * t))

    where the expansions above define the coeffiients ``ck`` and the
    frequencies ``vk``.

    Parameters
    ----------
    Q : Qobj
        The coupling operator for the bath.

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

    tag : optional, str, tuple or any other object
        A label for the bath exponents (for example, the name of the
        bath). It defaults to None but can be set to help identify which
        bath an exponent is from.
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

    def __init__(self, Q, ck_plus, vk_plus, ck_minus, vk_minus, tag=None):
        self._check_cks_and_vks(ck_plus, vk_plus, ck_minus, vk_minus)
        self._check_coup_op(Q)

        exponents = []
        for ckp, vkp, ckm, vkm in zip(ck_plus, vk_plus, ck_minus, vk_minus):
            exponents.append(BathExponent(
                "+", 2, Q, ckp, vkp, sigma_bar_k_offset=1, tag=tag,
            ))
            exponents.append(BathExponent(
                "-", 2, Q, ckm, vkm, sigma_bar_k_offset=-1, tag=tag,
            ))
        super().__init__(exponents)


class LorentzianBath(FermionicBath):
    """
    A helper class for constructing a Lorentzian fermionic bath from the
    bath parameters (see parameters below).

    .. note::

        This Matsubara expansion used in this bath converges very slowly
        and ``Nk > 20`` may be required to get good convergence. The
        Padé expansion used by :class:`LorentzianPadeBath` converges much
        more quickly.

    Parameters
    ----------
    Q : Qobj
        Operator describing the coupling between system and bath.

    gamma : float
        The coupling strength between the system and the bath.

    w : float
        The width of the environment.

    mu : float
        The chemical potential of the bath.

    T : float
        Bath temperature.

    Nk : int
        Number of exponential terms used to approximate the bath correlation
        functions.

    tag : optional, str, tuple or any other object
        A label for the bath exponents (for example, the name of the
        bath). It defaults to None but can be set to help identify which
        bath an exponent is from.
    """

    def __init__(self, Q, gamma, w, mu, T, Nk, tag=None):
        ck_plus, vk_plus = self._corr(gamma, w, mu, T, Nk, sigma=1.0)
        ck_minus, vk_minus = self._corr(gamma, w, mu, T, Nk, sigma=-1.0)

        super().__init__(
            Q, ck_plus, vk_plus, ck_minus, vk_minus, tag=tag,
        )

    def _corr(self, gamma, w, mu, T, Nk, sigma):
        beta = 1. / T
        kappa = [0.]
        kappa.extend([1. for _ in range(1, Nk + 1)])
        epsilon = [0]
        epsilon.extend([(2 * ll - 1) * np.pi for ll in range(1, Nk + 1)])

        def f(x):
            return 1 / (np.exp(x) + 1)

        eta_list = [0.5 * gamma * w * f(1.0j * beta * w)]
        gamma_list = [w - sigma * 1.0j * mu]

        for ll in range(1, Nk + 1):
            eta_list.append(
                -1.0j * (kappa[ll] / beta) * gamma * w**2 /
                (-(epsilon[ll]**2 / beta**2) + w**2)
            )
            gamma_list.append(epsilon[ll] / beta - sigma * 1.0j * mu)

        return eta_list, gamma_list


class LorentzianPadeBath(FermionicBath):
    """
    A helper class for constructing a Padé expansion for Lorentzian fermionic
    bath from the bath parameters (see parameters below).

    A Padé approximant is a sum-over-poles expansion (
    see https://en.wikipedia.org/wiki/Pad%C3%A9_approximant).

    The application of the Padé method to spectrum decompoisitions is described
    in "Padé spectrum decompositions of quantum distribution functions and
    optimal hierarchical equations of motion construction for quantum open
    systems" [1].

    The implementation here follows the approach in the paper.

    [1] J. Chem. Phys. 134, 244106 (2011); https://doi.org/10.1063/1.3602466

    This is an alternative to the :class:`LorentzianBath` which constructs
    a simpler exponential expansion that converges much more slowly in
    this particular case.

    Parameters
    ----------
    Q : Qobj
        Operator describing the coupling between system and bath.

    gamma : float
        The coupling strength between the system and the bath.

    w : float
        The width of the environment.

    mu : float
        The chemical potential of the bath.

    T : float
        Bath temperature.

    Nk : int
        Number of exponential terms used to approximate the bath correlation
        functions.

    tag : optional, str, tuple or any other object
        A label for the bath exponents (for example, the name of the
        bath). It defaults to None but can be set to help identify which
        bath an exponent is from.
    """

    def __init__(self, Q, gamma, w, mu, T, Nk, tag=None):
        ck_plus, vk_plus = self._corr(gamma, w, mu, T, Nk, sigma=1.0)
        ck_minus, vk_minus = self._corr(gamma, w, mu, T, Nk, sigma=-1.0)

        super().__init__(
            Q, ck_plus, vk_plus, ck_minus, vk_minus, tag=tag,
        )

    def _corr(self, gamma, w, mu, T, Nk, sigma):
        beta = 1. / T
        kappa, epsilon = self._kappa_epsilon(Nk)

        def f_approx(x):
            f = 0.5
            for ll in range(1, Nk + 1):
                f = f - 2 * kappa[ll] * x / (x**2 + epsilon[ll]**2)
            return f

        eta_list = [0.5 * gamma * w * f_approx(1.0j * beta * w)]
        gamma_list = [w - sigma * 1.0j * mu]

        for ll in range(1, Nk + 1):
            eta_list.append(
                -1.0j * (kappa[ll] / beta) * gamma * w**2
                / (-(epsilon[ll]**2 / beta**2) + w**2)
            )
            gamma_list.append(epsilon[ll] / beta - sigma * 1.0j * mu)

        return eta_list, gamma_list

    def _kappa_epsilon(self, Nk):
        eps = self._calc_eps(Nk)
        chi = self._calc_chi(Nk)

        kappa = [0]
        prefactor = 0.5 * Nk * (2 * (Nk + 1) - 1)
        for j in range(Nk):
            term = prefactor
            for k in range(Nk - 1):
                term *= (
                    (chi[k]**2 - eps[j]**2) /
                    (eps[k]**2 - eps[j]**2 + self._delta(j, k))
                )
            for k in [Nk - 1]:
                term /= (eps[k]**2 - eps[j]**2 + self._delta(j, k))
            kappa.append(term)

        epsilon = [0] + eps

        return kappa, epsilon

    def _delta(self, i, j):
        return 1.0 if i == j else 0.0

    def _calc_eps(self, Nk):
        alpha = np.diag([
            1. / np.sqrt((2 * k + 3) * (2 * k + 1))
            for k in range(2 * Nk - 1)
        ], k=1)
        alpha += alpha.transpose()

        evals = eigvalsh(alpha)
        eps = [-2. / val for val in evals[0: Nk]]
        return eps

    def _calc_chi(self, Nk):
        alpha_p = np.diag([
            1. / np.sqrt((2 * k + 5) * (2 * k + 3))
            for k in range(2 * Nk - 2)
        ], k=1)
        alpha_p += alpha_p.transpose()
        evals = eigvalsh(alpha_p)
        chi = [-2. / val for val in evals[0: Nk - 1]]
        return chi


class FitSpectral(BosonicBath):
    """
    A helper class for constructing a Bosonic bath from a spectral density fit.

    Parameters
    ----------
    Q : Qobj
        Operator describing the coupling between system and bath.

    T : float
        Bath temperature.

    Nk : int
        Number of exponential terms used to approximate the bath correlation
        functions.
    """

    def __init__(self, T, Q, Nk=14):
        self.Q = Q
        self.T = T
        self.beta = 1 / self.T
        self.Nk = Nk

    def __str__(self):
        try:
            lam, gamma, w0 = self.params_spec
            summary = f"Results of the fitting the Spectral density with {self.spec_n} terms: \n \n {'Parameters' : <10}|{'lam' : ^10}|{'gamma' : ^10}|{'w0' : >5} \n "
            for i in range(len(lam)):
                summary += (
                    f"{ i+1 : <10}|{lam[i]: ^10.2e}|{gamma[i]:^10.2e}|{w0[i]:>5.2e} \n "
                )
            summary += f"\nA  normalized RMSE of {self._rmse: .2e} was obtained for the Spectral density \n"
            summary += f" The current fit took {self.fit_time : 2f} seconds"
            return summary
        except:
            return "Fit correlation instance: \n No fit has been performed yet"

    def summary(self):
        print(self.__str__())

    def spectral_density_approx(self, w, a, b, c):
        """Calculate the fitted value of the function for the given parameters."""
        tot = 0
        for i in range(len(a)):
            tot += (
                2
                * a[i]
                * b[i]
                * w
                / (((w + c[i]) ** 2 + b[i] ** 2) * ((w - c[i]) ** 2 + b[i] ** 2))
            )
        return tot

    def spec_spectrum_approx(self, w):
        """Calculates the approximate power spectrum w and the fitted parameters"""
        lam, gamma, w0 = self.params_spec
        s_fit = (
            self.spectral_density_approx(w, lam, gamma, w0)
            * ((1 / (np.e ** (w / self.T) - 1)) + 1)
            * 2
        )
        return s_fit

    def get_fit(
        self,
        J,
        w,
        N=None,
        final_rmse=5e-6,
        lower=None,
        upper=None,
        sigma=None,
        guesses=None,
    ):
        """
        Provides a fit to the spectral density with N underdamped oscillators baths, This function gets the number of harmonic oscillators based
        on reducing the normalized root mean squared error below a certain threshold
        Parameters
        ----------
        J : np.array
            Spectral density to be fit.
        w : np.array
            range of frequencies for the fit.

        N : optional,int
            Number of underdamped oscillators to use, if set to False it is found automaticlly.
        final_rmse : float
            Desired normalized root mean squared error .
        lower : list
            lower bounds on the parameters for the fit.
        upper: list
            upper bounds on the parameters for the fit
        sigma: float
            uncertainty in the data considered for the fit
        guesses : list
            Initial guess for the parameters.
        """
        start = time()
        if N == None:
            N = 1
            rmse = 8
            while rmse > final_rmse:
                rmse, params = self._fit(
                    self.spectral_density_approx,
                    J,
                    w,
                    N,
                    label="Spectral Density",
                    sigma=sigma,
                    guesses=guesses,
                    lower=lower,
                    upper=upper,
                )
                N += 1
        else:
            rmse, params = self._fit(
                self.spectral_density_approx,
                J,
                w,
                N,
                label="Spectral Density",
                sigma=sigma,
                guesses=guesses,
                lower=lower,
                upper=upper,
            )
        self.params_spec = params
        self.spec_n = N - 1
        self._rmse = rmse
        self._matsubara_spectral_fit()
        end = time()
        self.fit_time = end - start

    def correlation_approx_matsubara(self, t, real=True):
        """ Calculate the approximate real or imaginary part of the
            correlation function from the matsubara expansion co-efficients.
        """
        if real:
            ck, vk = self.cvars[0], self.cvars[1]
        else:
            ck, vk = self.cvars[2], self.cvars[3]
        ck = np.array(ck)
        vk = np.array(vk)
        return np.sum(ck[:, None] * np.exp(-vk[:, None] * t), axis=0)

    def fit_plots(self, w, J, t, C, w2, S):
        gen_spectral_plots(self, w, J, t, C, w2, S)

    def _matsubara_spectral_fit(self):
        """
        Obtains the bath exponents from the list of fit parameters
        """
        lam, gamma, w0 = self.params_spec
        w0 = np.array(
            [
                np.sqrt((w0[i] + 0j) ** 2 + (gamma[i] + 0j / 2) ** 2)
                for i in range(len(w0))
            ]
        )
        lam = np.sqrt(
            lam + 0j
        )  # both w0, and lam are needed to input the right value of the fit into the Underdamped bath
        ckAR = []
        vkAR = []
        ckAI = []
        vkAI = []
        UD = UnderDampedBath(self.Q, lam[0], gamma[0], w0[0], self.T, self.Nk)
        terminator = 0. * spre(self.Q)
        # the number of matsubara expansion terms to include in the terminator:
        terminator_max_k = 1000

        for lamt, Gamma, Om in zip(lam, gamma, w0):
            ck_real, vk_real, ck_imag, vk_imag = UD._matsubara_params(
                lamt, 2 * Gamma, Om + 0j, self.T, self.Nk
            )
            ckAR.append(ck_real)
            vkAR.append(vk_real)
            vkAI.append(vk_imag)
            ckAI.append(ck_imag)
            terminator_factor = 0
            for k in range(self.Nk + 1, terminator_max_k):
                ek = 2 * np.pi * k / self.beta
                ck = (
                    (-2 * lamt * 2 * Gamma / self.beta) * ek /
                    (
                        ((Om + 1.0j * Gamma)**2 + ek**2) *
                        ((Om - 1.0j * Gamma)**2 + ek**2)
                    )
                )
                terminator_factor += ck / ek
            terminator += terminator_factor * (
                2 * spre(self.Q) * spost(self.Q.dag())
                - spre(self.Q.dag() * self.Q)
                - spost(self.Q.dag() * self.Q)
            )
        self.cvars = [
            np.array(ckAR).flatten(),
            np.array(vkAR).flatten(),
            np.array(ckAI).flatten(),
            np.array(vkAI).flatten(),
        ]
        self.terminator = terminator
        self.Bath_spec = BosonicBath(
            self.Q,
            np.array(ckAR).flatten(),
            np.array(vkAR).flatten(),
            np.array(ckAI).flatten(),
            np.array(vkAI).flatten(),
        )


class FitCorr(BosonicBath):
    """
    A helper class for constructing a Bosonic bath from a Correlation function fit.

    Parameters
    ----------
    Q : Qobj
        Operator describing the coupling between system and bath.
    """

    def __init__(self, Q):
        self.Q = Q

    def corr_approx(self, t, a, b, c):
        """Calculate the fitted value of the function for the given parameters."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        return np.sum(
            a[:, None] * np.exp(b[:, None] * t) * np.exp(1j * c[:, None] * t),
            axis=0,
        )

    def __str__(self):
        try:
            lam, gamma, w0 = self.params_real
            lam2, gamma2, w02 = self.params_imag
            summary = f"Results of the fitting the Real Part with {self.Nr} terms: \n \n {'Parameters' : <10}|{'lam' : ^10}|{'gamma' : ^10}|{'w0' : >5} \n "
            summary2 = f"\t Results of the fitting the Imaginary Part with {self.Ni} terms: \n \n \t {'Parameters' : <10}|{'lam' : ^10}|{'gamma' : ^10}|{'w0' : >5} \n"
            for i in range(len(lam)):
                summary += (
                    f"{ i+1 : <10}|{lam[i]: ^10.2e}|{gamma[i]:^10.2e}|{w0[i]:>5.2e} \n "
                )
            for i in range(len(lam2)):
                summary2 += f"\t {i+1 : <10}|{lam2[i]: ^10.2e}|{gamma2[i]: ^10.2e}|{w02[i] :>5.2e} \n "
            summary += f"\n  A  normalized RMSE of {self.rmse_real: .2e} was obtained for the real part \n"
            summary2 += f"\n \t A  normalized RMSE of {self.rmse_imag :.2e} was obtained for the imaginary part \n"
            time = f" The current fit took {self.fit_time : 2f} seconds"
            return summary, summary2, time

        except:
            return "Fit correlation instance: \n No fit has been performed yet"

    def summary(self):
        print("Fit correlation class instance: \n \n")
        string1, string2, time = self.__str__()
        lines1 = string1.splitlines()
        lines2 = string2.splitlines()
        max_lines = max(len(lines1), len(lines2))
        # Fill the shorter string with blank lines
        lines1 = lines1[:-1] + (max_lines - len(lines1)) * [""] + [lines1[-1]]
        lines2 = lines2[:-1] + (max_lines - len(lines2)) * [""] + [lines2[-1]]
        # Find the maximum line length in each column
        max_length1 = max(len(line) for line in lines1)
        max_length2 = max(len(line) for line in lines2)

        # Print the strings side by side with a vertical bar separator
        for line1, line2 in zip(lines1, lines2):
            formatted_line1 = f"{line1:<{max_length1}} |"
            formatted_line2 = f"{line2:<{max_length2}}"
            print(formatted_line1 + formatted_line2)
        print("\t " * 6 + time)

    def fit_plots(self, w, J, t, C, w2, S, beta):
        gen_corr_plots(self, w, J, t, C, w2, S, beta)

    def fit_correlation(
        self,
        t,
        C,
        Nr=None,
        Ni=None,
        final_rmse=2e-5,
        lower=None,
        upper=None,
        sigma=None,
        guesses=None,
    ):
        """Fit the correlation function with Ni underdamped oscillators baths for the imaginary part of the correlation function and Nr for the real.
        Provides a fit to the with N underdamped oscillators. If no number of terms is provided This function gets the number of harmonic oscillators based
        on reducing the normalized root mean squared error below a certain threshold.
        Parameters
        ----------
        t : np.array
            range of frequencies for the fit.
        C : np.array
            Correlation function to be fit.

        Nr : optional,int
            Number of underdamped oscillators to use for the real part, if set to False it is found automaticlly.
        Ni : optional,int
            Number of underdamped oscillators to use for the imaginary part, if set to False it is found automaticlly.
        final_rmse : float
            Desired normalized root mean squared error .
        lower : list
            lower bounds on the parameters for the fit.
        upper: list
            upper bounds on the parameters for the fit
        sigma: float
            uncertainty in the data considered for the fit
        guesses : list
            Initial guess for the parameters.
        """
        start = time()
        rmse1, rmse2 = [8, 8]
        flag = False
        if (Nr is not None) & (Ni is not None):
            flag = True
            Nr -= 1
            Ni -= 1

        if Nr is None:
            Nr = 0
        if Ni is None:
            Ni = 0
        while rmse1 > final_rmse:
            Nr += 1
            rmse1, params_real = self._fit(
                lambda *args: np.real(self.corr_approx(*args)),
                np.real(C),
                t,
                N=Nr,
                sigma=sigma,
                guesses=guesses,
                lower=lower,
                upper=upper,
                label="correlation_real",
            )
            if flag == True:
                break
        while rmse2 > final_rmse:
            Ni += 1
            rmse2, params_imag = self._fit(
                lambda *args: np.imag(self.corr_approx(*args)),
                np.imag(C),
                t,
                N=Ni,
                sigma=sigma,
                guesses=guesses,
                lower=lower,
                upper=upper,
                label="correlation_imag",
            )
            if flag == True:
                break
        self.Nr = Nr
        self.Ni = Ni
        self.rmse_real = rmse1
        self.rmse_imag = rmse2
        self.params_real = params_real
        self.params_imag = params_imag
        self._matsubara_coefficients()
        end = time()
        self.fit_time = end - start

    def corr_spectrum_approx(self, w):
        """Calculates the approximate power spectrum from ck and vk."""
        S = np.zeros(len(w), dtype=np.complex128)
        self._matsubara_coefficients()
        ckAR, vkAR, ckAI, vkAI = self.matsubara_coeff
        for ck, vk in zip(ckAR, vkAR):
            S += 2 * ck * np.real(vk) / ((w - np.imag(vk))
                                         ** 2 + (np.real(vk) ** 2))
        for ck, vk in zip(ckAI, vkAI):
            S += (
                2
                * 1.0j
                * ck
                * np.real(vk)
                / ((w - np.imag(vk)) ** 2 + (np.real(vk) ** 2))
            )
        return S

    def corr_spectral_approx(self, w, beta):
        J = np.real(
            self.corr_spectrum_approx(w) /
            (((1 / (np.e**(w * beta) - 1)) + 1) * 2)
        )
        return J

    def _matsubara_coefficients(self):
        lam, gamma, w0 = self.params_real
        lam2, gamma2, w02 = self.params_imag
        ckAR = [0.5 * x + 0j for x in lam]  # the 0.5 is from the cosine
        # extend the list with the complex conjugates:
        ckAR.extend(np.conjugate(ckAR))
        vkAR = [-x - 1.0j * y for x, y in zip(gamma, w0)]
        vkAR.extend([-x + 1.0j * y for x, y in zip(gamma, w0)])
        ckAI = [-0.5j * x for x in lam2]  # the 0.5 is from the sine
        # extend the list with the complex conjugates:
        ckAI.extend(np.conjugate(ckAI))

        vkAI = [-x - 1.0j * y for x, y in zip(gamma2, w02)]
        vkAI.extend([-x + 1.0j * y for x, y in zip(gamma2, w02)])
        self.matsubara_coeff = [ckAR, vkAR, ckAI, vkAI]
        self.Bath_corr = BosonicBath(self.Q, ckAR, vkAR, ckAI, vkAI)


class OhmicBath(BosonicBath):
    def __init__(self, T, Q, alpha, wc, s, Nk=4, method="spectral", rmse=7e-5):
        self.alpha = alpha
        self.wc = wc
        self.s = s
        self.method = method
        self.Q = Q
        self.T = T
        self.Nk = Nk
        self.beta = 1 / T
        if method == "correlation":
            self.fit = FitCorr(self.Q)
            t = np.linspace(0, 15 / self.wc, 1000)
            C = self.ohmic_correlation(t, self.s)
            self.fit.fit_correlation(t, C, final_rmse=rmse)
            self.bath = self.fit.Bath_corr
        else:
            self.fit = FitSpectral(self.T, self.Q, self.Nk)
            w = np.linspace(0, 15 * self.wc, 20000)
            J = self.ohmic_spectral_density(w)
            self.fit.get_fit(J, w, final_rmse=rmse)
            self.bath = self.fit.Bath_spec
            self.terminator = self.fit.terminator

    def summary(self):
        self.fit.summary()

    def ohmic_spectral_density(self, w):
        """The Ohmic bath spectral density as a function of w
        (and the bath parameters).
        """
        return (
            self.alpha
            * w ** (self.s)
            / (self.wc ** (1 - self.s))
            * np.e ** (-abs(w) / self.wc)
        )

    def ohmic_correlation(self, t, s=1):
        """The Ohmic bath correlation function as a function of t
        (and the bath parameters).
        """
        corr = (
            (1 / np.pi)
            * self.alpha
            * self.wc ** (1 - s)
            * self.beta ** (-(s + 1))
            * mp.gamma(s + 1)
        )
        z1_u = (1 + self.beta * self.wc - 1.0j *
                self.wc * t) / (self.beta * self.wc)
        z2_u = (1 + 1.0j * self.wc * t) / (self.beta * self.wc)
        # Note: the arguments to zeta should be in as high precision as possible.
        # See http://mpmath.org/doc/current/basics.html#providing-correct-input
        return np.array(
            [
                complex(corr * (mp.zeta(s + 1, u1) + mp.zeta(s + 1, u2)))
                for u1, u2 in zip(z1_u, z2_u)
            ],
            dtype=np.complex128,
        )

    def ohmic_power_spectrum(self, w):
        """The Ohmic bath power spectrum as a function of w
        (and the bath parameters).
        """
        return (
            self.ohmic_spectral_density(w)
            * ((1 / (np.e ** (w * self.beta) - 1)) + 1)
            * 2
        )
