"""
This module provides utilities for describing baths when using the HEOM
(hierarchy equations of motion) to model system-bath interactions. See the
``qutip.nonmarkov.bofin_solvers`` module for the associated solver.

The implementation is derived from the BoFiN library (see
https://github.com/tehruhn/bofin) which was itself derived from an earlier
implementation in QuTiP itself.

The "bath" classes in this module are closely related to the "environment"
classes in the `qutip.core.environment` module. The bath classes were
implemented first, specifically for the HEOM solver. The environment classes,
added later provide additional functionality and are designed to work also with
other QuTiP solvers.

The bath classes are kept partly for backwards compatibility, partly to use as
a "shortcut" when one only wants to use the HEOM solver.

Note that this module also contains the `BathExponent` class, which is used by
the HEOM solver internally and in the result object describing its output.
"""

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

    This class extends the
    :class:`CFExponent <qutip.core.environment.CFExponent>` of the
    environment-module for use with the HEOM solver. In addition to the
    exponent itself, the `BathExponent` keeps track of the corresponding system
    coupling operator ``Q``, as well as the parameters ``dim`` and
    ``sigma_bar_k_offset``.

    Parameters
    ----------
    type : {"R", "I", "RI", "+", "-"} or ``CFExponent.ExponentType``
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

    def _check_sigma_bar_k_offset(self, type, offset):
        if type in (self.types["+"], self.types["-"]):
            if offset is None:
                raise ValueError(
                    "+ and - type exponents require sigma_bar_k_offset"
                )
        else:
            if offset is not None:
                raise ValueError(
                    "Offset of sigma bar (sigma_bar_k_offset) should only be"
                    " specified for + and - type exponents"
                )

    def __init__(
            self, type, dim, Q, ck, vk, ck2=None,
            sigma_bar_k_offset=None, tag=None,
    ):
        super().__init__(type, ck, vk, ck2, tag)

        self.dim = dim
        self.Q = Q

        self._check_sigma_bar_k_offset(self.type, sigma_bar_k_offset)
        self.sigma_bar_k_offset = sigma_bar_k_offset

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
        return super()._combine(other, dim=None, Q=self.Q)


class Bath:
    """
    Represents a list of bath expansion exponents.

    Parameters
    ----------
    exponents : list of :class:`.BathExponent`
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
        operator). See :meth:`.ExponentialBosonicEnvironment.combine` for
        details.

    tag : optional, str, tuple or any other object
        A label for the bath exponents (for example, the name of the
        bath). It defaults to None but can be set to help identify which
        bath an exponent is from.

    Notes
    -----
    This class is part of the "bath" API, which is now mirrored by the newer
    "environment" API. The bath classes are kept in QuTiP for reasons of
    backwards compatibility and convenience. This class is an extended version
    of the :class:`.ExponentialBosonicEnvironment`, adding the parameter ``Q``,
    which is not included in the newer "environment" API.
    """

    def _make_exponent(self, type, ck, vk, ck2=None, tag=None):
        return BathExponent(type, None, self._Q, ck, vk, ck2, tag=tag)

    def _check_coup_op(self, Q):
        if not isinstance(Q, Qobj):
            raise ValueError("The coupling operator Q must be a Qobj.")

    def __init__(
        self, Q, ck_real, vk_real, ck_imag, vk_imag, combine=True, tag=None
    ):
        self._check_coup_op(Q)
        self._Q = Q

        super().__init__(ck_real, vk_real, ck_imag, vk_imag,
                         combine=combine, tag=tag)

    @classmethod
    def from_environment(cls, env, Q, dim=None):
        """
        Converts from the "environment" API to the "bath" API. A `BosonicBath`
        combines the information from an `ExponentialBosonicEnvironment` and a
        coupling operator.

        Parameters
        ----------
        env : :class:`.ExponentialBosonicEnvironment`
            The bath.
        Q : Qobj
            The coupling operator for the bath.
        dim : optional, int or ``None`` (default ``None``)
            The maximum number of excitations for each exponent. Usually
            ``None`` (i.e. unlimited).
        """
        bath_exponents = []
        for exponent in env.exponents:
            new_exponent = BathExponent(
                exponent.type, dim, Q, exponent.ck, exponent.vk,
                exponent.ck2, None, env.tag
            )
            bath_exponents.append(new_exponent)

        result = cls(Q, [], [], [], [], tag=env.tag)
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
        operator). See :meth:`.ExponentialBosonicEnvironment.combine` for
        details.

    tag : optional, str, tuple or any other object
        A label for the bath exponents (for example, the name of the
        bath). It defaults to None but can be set to help identify which
        bath an exponent is from.

    Notes
    -----
    This class is part of the "bath" API, which is now mirrored by the newer
    "environment" API. The bath classes are kept in QuTiP for reasons of
    backwards compatibility and convenience. Creating a `DrudeLorentzBath` is
    equivalent to creating a :class:`.DrudeLorentzEnvironment`, performing a
    :meth:`Matsubara <.DrudeLorentzEnvironment.approx_by_matsubara>`
    approximation, and finally bundling the result together with the coupling
    operator ``Q`` for convenient use with the HEOM solver.
    """

    def __new__(
        mcs, Q, lam, gamma, T, Nk, combine=True, tag=None,
    ):
        # Basically this makes `DrudeLorentzBath` a function
        # (Q, lam, ...) -> BosonicBath
        # but it is made to look like a class because it was a class in the
        # initial bofin release
        env = environment.DrudeLorentzEnvironment(T, lam, gamma)
        matsubara_approx, delta = env.approx_by_matsubara(
            Nk=Nk, combine=combine, compute_delta=True, tag=tag
        )

        result = BosonicBath.from_environment(matsubara_approx, Q)
        result.terminator = lambda: (
            delta, environment.system_terminator(Q, delta)
        )
        return result

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

            The Padé terminator -- i.e. a liouvillian term representing
            the contribution to the system-bath dynamics of all exponential
            expansion terms beyond ``Nk``. It should be used by adding it to
            the system liouvillian (i.e. ``liouvillian(H_sys)``).
        """
        # This is only here to keep the API doc
        ...


class DrudeLorentzPadeBath(BosonicBath):
    """
    A helper class for constructing a Padé expansion for a Drude-Lorentz
    bosonic bath from the bath parameters (see parameters below).

    A Padé approximant is a sum-over-poles expansion (see
    https://en.wikipedia.org/wiki/Pad%C3%A9_approximant).

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
        operator). See :meth:`.ExponentialBosonicEnvironment.combine` for
        details.

    tag : optional, str, tuple or any other object
        A label for the bath exponents (for example, the name of the
        bath). It defaults to None but can be set to help identify which
        bath an exponent is from.

    Notes
    -----
    This class is part of the "bath" API, which is now mirrored by the newer
    "environment" API. The bath classes are kept in QuTiP for reasons of
    backwards compatibility and convenience. Creating a `DrudeLorentzPadeBath`
    is equivalent to creating a :class:`.DrudeLorentzEnvironment`, performing a
    :meth:`Pade <.DrudeLorentzEnvironment.approx_by_pade>` approximation, and
    finally bundling the result together with the coupling operator ``Q`` for
    convenient use with the HEOM solver.
    """

    def __new__(
        mcs, Q, lam, gamma, T, Nk, combine=True, tag=None,
    ):
        # See DrudeLorentzBath comment
        env = environment.DrudeLorentzEnvironment(T, lam, gamma)
        pade_approx, delta = env.approx_by_pade(
            Nk=Nk, combine=combine, compute_delta=True, tag=tag
        )

        result = BosonicBath.from_environment(pade_approx, Q)
        result.terminator = lambda: (
            delta, environment.system_terminator(Q, delta)
        )
        return result

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
        # This is only here to keep the API doc
        ...


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
        operator). See :meth:`.ExponentialBosonicEnvironment.combine` for
        details.

    tag : optional, str, tuple or any other object
        A label for the bath exponents (for example, the name of the
        bath). It defaults to None but can be set to help identify which
        bath an exponent is from.

    Notes
    -----
    This class is part of the "bath" API, which is now mirrored by the newer
    "environment" API. The bath classes are kept in QuTiP for reasons of
    backwards compatibility and convenience. Creating an `UnderDampedBath` is
    equivalent to creating an :class:`.UnderDampedEnvironment`, performing a
    :meth:`Matsubara <.UnderDampedEnvironment.approx_by_matsubara>`
    approximation, and finally bundling the result together with the coupling
    operator ``Q`` for convenient use with the HEOM solver.
    """

    def __new__(
        mcs, Q, lam, gamma, w0, T, Nk, combine=True, tag=None,
    ):
        # See DrudeLorentzBath comment
        env = environment.UnderDampedEnvironment(T, lam, gamma, w0)
        matsubara_approx = env.approx_by_matsubara(
            Nk=Nk, combine=combine, tag=tag
        )
        return BosonicBath.from_environment(matsubara_approx, Q)


class FermionicBath(environment.ExponentialFermionicEnvironment):
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

    Notes
    -----
    This class is part of the "bath" API, which is now mirrored by the newer
    "environment" API. The bath classes are kept in QuTiP for reasons of
    backwards compatibility and convenience. This class is an extended version
    of the :class:`.ExponentialFermionicEnvironment`, adding the parameter
    ``Q``, which is not included in the newer "environment" API.
    """

    def _check_cks_and_vks(self, ck_plus, vk_plus, ck_minus, vk_minus):
        lists_provided = super()._check_cks_and_vks(
            ck_plus, vk_plus, ck_minus, vk_minus)
        if lists_provided and len(ck_plus) != len(ck_minus):
            raise ValueError(
                "The must be the same number of plus and minus exponents"
                " in the bath, and elements of plus and minus arrays"
                " should be arranged so that ck_plus[i] is the plus mode"
                " corresponding to ck_minus[i]."
            )
        return lists_provided

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
        super().__init__(exponents=exponents, tag=tag)

    @classmethod
    def from_environment(cls, env, Q, dim=2):
        """
        Converts from the "environment" API to the "bath" API. A
        `FermionicBath` combines the information from an
        `ExponentialFermionicEnvironment` and a coupling operator.

        Note that the "environment" API does not require fermionic exponents to
        come in pairs. This method will add additional exponents with
        zero coefficient in order to achieve the same amount of ``+`` and ``-``
        exponents.

        Parameters
        ----------
        env : :class:`.ExponentialFermionicEnvironment`
            The bath.
        Q : Qobj
            The coupling operator for the bath.
        dim : optional, int or ``None`` (default ``2``)
            The maximum number of excitations for each exponent. Usually ``2``.
        """

        # make same amount of plus and minus exponents by adding zeros
        plus_exponents = []
        minus_exponents = []
        for exponent in env.exponents:
            if exponent.type == BathExponent.types['+']:
                plus_exponents.append(exponent)
            else:
                minus_exponents.append(exponent)
        while len(plus_exponents) > len(minus_exponents):
            minus_exponents.append(environment.CFExponent('-', 0, 0))
        while len(minus_exponents) > len(plus_exponents):
            plus_exponents.append(environment.CFExponent('+', 0, 0))

        bath_exponents = []
        for expp, expm in zip(plus_exponents, minus_exponents):
            bath_exponents.append(BathExponent(
                '+', dim, Q, expp.ck, expp.vk, None, 1, env.tag
            ))
            bath_exponents.append(BathExponent(
                '-', dim, Q, expm.ck, expm.vk, None, -1, env.tag
            ))

        result = cls(Q, [], [], [], [], tag=env.tag)
        result.exponents = bath_exponents
        return result


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

    Notes
    -----
    This class is part of the "bath" API, which is now mirrored by the newer
    "environment" API. The bath classes are kept in QuTiP for reasons of
    backwards compatibility and convenience. Creating a `LorentzianBath` is
    equivalent to creating a :class:`.LorentzianEnvironment`, performing a
    :meth:`Matsubara <.LorentzianEnvironment.approx_by_matsubara>`
    approximation, and finally bundling the result together with the coupling
    operator ``Q`` for convenient use with the HEOM solver.
    """

    def __new__(self, Q, gamma, w, mu, T, Nk, tag=None):
        # See DrudeLorentzBath comment
        env = environment.LorentzianEnvironment(T, mu, gamma, w)
        mats_approx = env.approx_by_matsubara(Nk, tag)
        return FermionicBath.from_environment(mats_approx, Q)


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

    Notes
    -----
    This class is part of the "bath" API, which is now mirrored by the newer
    "environment" API. The bath classes are kept in QuTiP for reasons of
    backwards compatibility and convenience. Creating a `LorentzianPadeBath` is
    equivalent to creating a :class:`.LorentzianEnvironment`, performing a
    :meth:`Pade <.LorentzianEnvironment.approx_by_pade>` approximation, and
    finally bundling the result together with the coupling operator ``Q`` for
    convenient use with the HEOM solver.
    """

    def __new__(self, Q, gamma, w, mu, T, Nk, tag=None):
        # See DrudeLorentzBath comment
        env = environment.LorentzianEnvironment(T, mu, gamma, w)
        mats_approx = env.approx_by_pade(Nk, tag)
        return FermionicBath.from_environment(mats_approx, Q)
