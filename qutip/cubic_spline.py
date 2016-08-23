import numpy as np
import scipy.linalg as la
from qutip.cy.interpolate import (_interpolate, _array_interpolate,
                                 _interpolate_complex,
                                 _array_interpolate_complex)

__all__ = ['Cubic_Spline']


class Cubic_Spline(object):
    '''
    Calculates coefficients for a cubic spline
    interpolation of a given data set.
    
    This function assumes that the data is sampled
    uniformly over a given interval.

    Parameters
    ----------
    a : float
        Lower bound of the interval.
    b : float
        Upper bound of the interval.
    y : ndarray
        Function values at interval points.
    alpha : float
        Second-order derivative at a. Default is 0.
    beta : float
        Second-order derivative at b. Default is 0.
    
    Attributes
    ----------
    a : float
        Lower bound of the interval.
    b : float
        Upper bound of the interval.
    coeffs : ndarray
        Array of coeffcients defining cubic spline.
    
    Notes
    -----
    This object can be called like a normal function with a
    single or array of input points at which to evaluate
    the interplating function.
        
    
    '''
    
    def __init__(self, a, b, y, alpha=0, beta=0):
        y = np.asarray(y)
        n = y.shape[0] - 1
        h = (b - a)/n

        coeff = np.zeros(n + 3, dtype=y.dtype)
        coeff[1] = 1/6. * (y[0] - (alpha * h**2)/6)
        coeff[n + 1] = 1/6. * (y[n] - (beta * h**2)/6)

        ab = np.ones((3, n - 1), dtype=float)
        ab[0, 0] = 0
        ab[1, :] = 4
        ab[-1, -1] = 0

        B = y[1:-1].copy()
        B[0] -= coeff[1]
        B[-1] -=  coeff[n + 1]

        coeff[2:-2] = la.solve_banded((1, 1), ab, B, overwrite_ab=True, 
                        overwrite_b=True, check_finite=False)

        coeff[0] = alpha * h**2/6. + 2 * coeff[1] - coeff[2]
        coeff[-1] = beta * h**2/6. + 2 * coeff[-2] - coeff[-3]

        self.a = a          # Lower-bound of domain
        self.b = b          # Uppser-bound of domain
        self.coeffs = coeff # Spline coefficients
        self.is_complex = (y.dtype == complex)
        
    def __call__(self, pnts):
        #If requesting a single return value
        if isinstance(pnts, (int, float, complex)):
            if self.is_complex:
                return _interpolate_complex(pnts, self.a, self.b, self.coeffs)
            else:
                return _interpolate(pnts, self.a, self.b, self.coeffs)
        #If requesting multiple return values from array_like
        elif isinstance(pnts, (np.ndarray,list)):
            pnts = np.asarray(pnts)
            if self.is_complex:
                return _array_interpolate_complex(pnts, self.a, self.b, self.coeffs)
            else:
                return _array_interpolate(pnts, self.a, self.b, self.coeffs)
    
    