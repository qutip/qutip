"""
This is a legacy module, kept for backwards compatibility. It provides
utilities for describing baths when using the HEOM (hierarchy equations of
motion) to model system-bath interactions.

See the ``qutip.nonmarkov.bofin_solvers`` module for the associated solver.

The implementation is derived from the BoFiN library (see
https://github.com/tehruhn/bofin) which was itself derived from an earlier
implementation in QuTiP itself.

Instead of this module, use `qutip.environment`, which provides the full
functionality of this module and is compatible with other QuTiP solvers.
This module might be removed in a future version of QuTiP.
"""

import numpy as np
from scipy.linalg import eigvalsh

from qutip.core import data as _data
from qutip.core import environment
from qutip.core.qobj import Qobj

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
]


def _isequal(Q1, Q2, tol):
    """ Return true if Q1 and Q2 are equal to within the given tolerance. """
    return _data.iszero(_data.sub(Q1.data, Q2.data), tol=tol)


class BathExponent(environment.CFExponent):
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
    types = environment.CFExponent.types

    def __init__(
            self, type, dim, Q, ck, vk, ck2=None,
            sigma_bar_k_offset=None, tag=None,
    ):
        super().__init__(type, ck, vk, ck2, sigma_bar_k_offset)

        self.dim = dim
        self.Q = Q
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

    def _can_combine(self, other, rtol, atol):
        if not super()._can_combine(other, rtol, atol):
            return False
        if not _isequal(self.Q, other.Q, tol=atol):
            return False
        return True

    def _combine(self, other):
        # Assumes can combine was checked
        combined_cf = super()._combine(other)
        return BathExponent(
            combined_cf.type, None, self.Q, combined_cf.ck, combined_cf.vk,
            combined_cf.ck2, combined_cf.sigma_bar_k_offset, self.tag
        )


class Bath:
    """
    Represents a list of bath expansion exponents.

    Parameters
    ----------
    exponents : list of BathExponent
        The exponents of the correlation function describing the bath.
    """

    def __init__(self, exponents):
        self.exponents = exponents


class BosonicBath(environment.ExponentialBosonicEnvironment):
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
    T: optional, float
        The temperature of the bath.
    """

    def _check_coup_op(self, Q):
        if not isinstance(Q, Qobj):
            raise ValueError("The coupling operator Q must be a Qobj.")

    def __init__(
            self, Q, ck_real, vk_real, ck_imag, vk_imag, combine=True,
            tag=None, T=None
    ):
        self._check_cks_and_vks(ck_real, vk_real, ck_imag, vk_imag)
        self._check_coup_op(Q)

        # Create `BathExponent`s instead of `CFExponent`s
        exponents = []
        exponents.extend(
            BathExponent("R", None, Q, ck, vk, tag=tag)
            for ck, vk in zip(ck_real, vk_real)
        )
        exponents.extend(
            BathExponent("I", None, Q, ck, vk, tag=tag)
            for ck, vk in zip(ck_imag, vk_imag)
        )

        super().__init__(
            exponents=exponents, combine=combine, T=T, tag=tag
        )

    @staticmethod
    def _from_env(env, Q, tag=None, dim=None):
        bath_exponents = []

        for exponent in env.exponents:
            new_exponent = BathExponent(
                exponent.type, dim, Q, exponent.ck, exponent.vk,
                exponent.ck2, exponent.sigma_bar_k_offset, tag
            )
            bath_exponents.append(new_exponent)

        result = BosonicBath(Q, [], [], [], [], tag=env.tag, T=env.T)
        result.exponents = bath_exponents
        return result


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

    def __new__(
        mcs, Q, lam, gamma, T, Nk, combine=True, tag=None,
    ):
        env = environment.DrudeLorentzEnvironment(T, lam, gamma, tag=tag)
        matsubara_approx = env.exponential_approximation(
            'matsubara', Nk=Nk, combine=combine
        )
        # TODO terminator stuff
        return BosonicBath._from_env(matsubara_approx, Q, tag=tag)


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

    def __new__(
        mcs, Q, lam, gamma, T, Nk, combine=True, tag=None,
    ):
        env = environment.DrudeLorentzEnvironment(T, lam, gamma, tag=tag)
        pade_approx = env.exponential_approximation(
            'pade', Nk=Nk, combine=combine
        )
        # TODO terminator stuff
        return BosonicBath._from_env(pade_approx, Q, tag=tag)


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

    def __new__(
        mcs, Q, lam, gamma, w0, T, Nk, combine=True, tag=None,
    ):
        env = environment.UnderDampedEnvironment(T, lam, gamma, w0, tag=tag)
        matsubara_approx = env.exponential_approximation(
            'matsubara', Nk=Nk, combine=combine
        )
        print('helo')
        for exp in matsubara_approx.exponents:
            print(exp)
            print()
        # TODO terminator stuff
        return BosonicBath._from_env(matsubara_approx, Q, tag=tag)


# TODO fermionic environments

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
