#cython: language_level=3

import numpy as np

cimport cython

from qutip.core.cy.coefficient cimport Coefficient

cdef extern from "<math.h>" namespace "std" nogil:
    double sqrt(double x)

cdef extern from "<complex>" namespace "std" nogil:
    double real(double complex x)


cdef class RateShiftCoefficient(Coefficient):
    """
    A coefficient representing the rate shift of a list of coefficients.

    The rate shift is ``2 * abs(min([0, coeff_1(t), coeff_2(t), ...]))``.

    Parameters
    ----------
    coeffs : list of :obj:`.Coefficient`
        The list of coefficients to determine the rate shift of.
    """
    def __init__(self, list coeffs):
        self.coeffs = np.array(coeffs, dtype=Coefficient)

    def __reduce__(self):
        return (RateShiftCoefficient, (list(self.coeffs),))

    def replace_arguments(self, _args=None, **kwargs):
        """
        Replace the arguments (``args``) of a coefficient.

        Returns a new :obj:`.Coefficient` if the coefficient has arguments, or
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
        return RateShiftCoefficient(
            [coeff.replace_arguments(_args, **kwargs) for coeff in self.coeffs],
        )

    cdef complex _call(self, double t) except *:
        """ Return the rate shift. """
        cdef int N = len(self.coeffs)
        cdef int i
        cdef double min_rate = 0
        cdef Coefficient coeff

        for i in range(N):
            coeff = self.coeffs[i]
            min_rate = min(min_rate, real(coeff._call(t)))

        return 2 * abs(min_rate)

    cpdef double as_double(self, double t) except *:
        """ Return the rate shift as a float. """
        return real(self._call(t))

    cpdef Coefficient copy(self):
        """Return a copy of the :obj:`.Coefficient`."""
        return RateShiftCoefficient(
            [coeff.copy() for coeff in self.coeffs],
        )


@cython.auto_pickle(True)
cdef class SqrtRealCoefficient(Coefficient):
    """
    A coefficient representing the positive square root of the real part of
    another coefficient.
    """
    def __init__(self, Coefficient base):
        self.base = base

    def replace_arguments(self, _args=None, **kwargs):
        """
        Replace the arguments (``args``) of a coefficient.

        Returns a new :obj:`.Coefficient` if the coefficient has arguments, or
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
        return SqrtRealCoefficient(
            self.base.replace_arguments(_args, **kwargs)
        )

    cdef complex _call(self, double t) except *:
        """Return the shifted rate."""
        return sqrt(real(self.base._call(t)))

    cpdef Coefficient copy(self):
        """Return a copy of the :obj:`.Coefficient`."""
        return SqrtRealCoefficient(self.base.copy())
