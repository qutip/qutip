#cython: language_level=3

import numpy as np

from qutip.core.cy.coefficient cimport Coefficient

cdef extern from "<complex>" namespace "std" nogil:
    double real(double complex x)


cdef class RateShiftCoefficient(Coefficient):
    """
    A coefficient representing the rate shift required by
    NonMarkovianMCSolver.
    """
    def __init__(self, Coefficient[:] rate_coeffs):
        self.rate_coeffs = rate_coeffs

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
        if _args:
            kwargs.update(_args)
        if not kwargs:
            return self
        cdef Coefficient[:] new_coeffs = np.array([
            coeff.replace_arguments(_args, **kwargs)
            for coeff in self.rate_coeffs
        ], dtype=Coefficient)
        return RateShiftCoefficient(new_coeffs)

    cdef complex _call(self, double t) except *:
        return self.rate_shift(t)

    cpdef Coefficient copy(self):
        """Return a copy of the :obj:`Coefficient`."""
        cdef Coefficient[:] new_coeffs = np.array([
            coeff.copy()
            for coeff in self.rate_coeffs
        ], dtype=Coefficient)
        return RateShiftCoefficient(new_coeffs)

    def __reduce__(self):
        return (RateShiftCoefficient, (self.rates,))

    cpdef double rate_shift(self, double t) except *:
        """ Calculate the rate shift required to make the shifted rates
            positive.

            Returns twice the absoluate value of the minimum unshifted rates.
        """
        cdef int N = len(self.rate_coeffs)
        cdef min_rate = 0
        cdef Coefficient coeff

        for i in range(N):
            coeff = self.rate_coeffs[i]
            min_rate = min(min_rate, real(coeff._call(t)))

        return 2 * abs(min_rate)

    cpdef Coefficient sqrt_shifted_rate(self, int i):
        """ Return a :obj:`Coefficient` to calculate the shifted rate of the i'th
            Lindblad operator.
        """
        return SqrtShiftedRateCoefficient(i, self)

    cpdef double discrete_martingale(self, double[:] collapse_times, int[:] collapse_idx) except *:
        """
        Discrete part of the martingale evolution. The collapses that have
        happened until now must be provided as a list of (t_k, i_k).
        """
        cdef int N = len(collapse_times)
        cdef double martingale = 1.0
        cdef double t_i
        cdef double rate_i
        cdef double shift_i
        cdef Coefficient coeff

        for i in range(N):
            t_i = collapse_times[i]
            coeff = self.rate_coeffs[collapse_idx[i]]
            shift_i = self.rate_shift(t_i)
            rate_i = real(coeff._call(t_i))
            martingale *= rate_i / (rate_i + shift_i)

        return martingale


cdef class SqrtShiftedRateCoefficient(Coefficient):
    """
    A coefficient representing the square root of the shifted
    rates used by NonMarkovianMCSolver.
    """
    def __init__(self, int rate_idx, RateShiftCoefficient rate_shift):
        self.rate_idx = rate_idx
        self.rate_shift = rate_shift

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
        if _args:
            kwargs.update(_args)
        if not kwargs:
            return self
        return SqrtShiftedRateCoefficient(
            self.rate_idx,
            self.rate_shift.replace_arguments(_args, **kwargs),
        )

    cdef complex _call(self, double t) except *:
        """Return the shifted rate."""
        cdef Coefficient coeff = self.rate_shift.rate_coeffs[self.rate_idx]
        return np.sqrt(
            real(coeff._call(t)) +
            self.rate_shift.rate_shift(t)
        )

    cpdef Coefficient copy(self):
        """Return a copy of the :obj:`Coefficient`."""
        return SqrtShiftedRateCoefficient(
            self.rate_idx,
            self.rate_shift.copy(),
        )

    def __reduce__(self):
        return (SqrtShiftedRateCoefficient, (self.rate_idx, self.rate_shift))
