#cython: language_level=3

import numpy as np

from qutip.core.cy.coefficient cimport Coefficient

cdef extern from "<complex>" namespace "std" nogil:
    double real(double complex x)


cdef class RateSet:
    """
    A set of possibly negative rate coefficients for the Lindblad operators
    used by the :class:`NonMarkovianMCSolver`.

    :class:`Rates` exists to solve a few different problems:

    - Calculating the shifted rate is slow in Python because it requires
      continually looping over the full set of coefficients to calculate
      the minimum rate.

    - Coefficients return complex numbers, but we have to take the minimum,
      which requires converting them to floats, and then squaring rooting
      them. Doing this in Python adds at least two Python function calls
      to every coefficient call.

    - It's not safe for a solver to expose its internal coefficient instances
      directly to the user because the coefficients may be replaced by
      new instances when their arguments are updated. Usually this is solved
      by the existence of :class:`QobjEvo`, but that solution is awkward in
      this case because we have mutliple coefficients shared across the
      :class:`QobjEvo` for each of the collapse operators. We solve this
      here by returning :class:`SqrtShiftedRateCoefficient` instances
      which are immutable references to an instance :class:`RateSet`. The
      small cost is that one must also call ``.arguments()`` on the
      :class:`RateSet` in addition to the usual call to the ``.arguments()``
      on all the :class:`QobjEvo` instances.

    """
    def __init__(self, Coefficient[:] rate_coeffs):
        self.rate_coeffs = rate_coeffs

    def arguments(self, dict _args=None, **kwargs):
        """
        Update the arguments for the rate coefficients.

        This method behaves as the method ``.arguments`` of :class:`QobjEvo`.

        Parameters
        ----------
        _args : dict [optional]
            New arguments as a dict. Update args with ``arguments(new_args)``.

        **kwargs :
            New arguments as a keywors. Update args with
            ``arguments(**new_args)``.

        .. note::
            If both the positional ``_args`` and keywords are passed new values
            from both will be used. If a key is present with both, the ``_args``
            dict value will take priority.
        """
        if _args is not None:
            kwargs.update(_args)

        cdef Coefficient coeff
        cache = []
        for i in range(len(self.rate_coeffs)):
            coeff = self.rate_coeffs[i].replace_arguments(kwargs, cache=cache)
            self.rate_coeffs[i] = coeff

    def __reduce__(self):
        return (RateSet, (self.rate_coeffs,))

    def sqrt_shifted_rate(self, i):
        """
        Return a :obj:`Coefficient` to calculate the shifted rate of the i'th
        Lindblad operator.

        The returned coefficient references this :class:`RateSet` instance.
        """
        return SqrtShiftedRateCoefficient(self, i)

    cpdef double rate_shift(RateSet self, double t) except *:
        """
        Calculate the rate shift required to make the shifted rates
        positive.

        Returns twice the absolute value of the minimum unshifted rates or
        zero if all rates are positive.
        """
        cdef int N = len(self.rate_coeffs)
        cdef int i
        cdef double min_rate = 0
        cdef Coefficient coeff

        for i in range(N):
            coeff = self.rate_coeffs[i]
            min_rate = min(min_rate, real(coeff._call(t)))

        return 2 * abs(min_rate)

    cpdef double rate(RateSet self, double t, int i) except *:
        """
        Return the unshifted value of rate ``i`` at time ``t``.
        """
        cdef Coefficient coeff = self.rate_coeffs[i]
        return real(coeff._call(t))


cdef class SqrtShiftedRateCoefficient(Coefficient):
    """
    A coefficient representing the square root of the shifted
    rates used by NonMarkovianMCSolver.

    Note that a :class:`SqrtShiftedRateCoefficient` is an immutable
    reference to a particular rate in a :class:`RateSet`.
    """
    def __init__(self, RateSet rates, int rate_idx):
        self.rates = rates
        self.rate_idx = rate_idx

    def replace_arguments(self, _args=None, **kwargs):
        """
        Replace the arguments (``args``) of a coefficient.

        Returns a new :obj:`Coefficient` if the coefficient has arguments, or
        the original coefficient if it does not. Arguments to replace may be
        supplied either in a dictionary as the first position argument, or
        passed as keywords, or as a combination of the two. Arguments not
        replaced retain their previous values.

        Parameters
        ----------
        _args : dict
            Dictionary of arguments to replace.

        **kwargs
            Arguments to replace.
        """
        return self

    cdef complex _call(self, double t) except *:
        """Return the shifted rate."""
        cdef Coefficient coeff = self.rates.rate_coeffs[self.rate_idx]
        return np.sqrt(
            real(coeff._call(t)) +
            self.rates.rate_shift(t)
        )

    cpdef Coefficient copy(self):
        """Return a copy of the :obj:`Coefficient`."""
        return SqrtShiftedRateCoefficient(
            self.rates,
            self.rate_idx,
        )

    def __reduce__(self):
        return (SqrtShiftedRateCoefficient, (self.rates, self.rate_idx))
