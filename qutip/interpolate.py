# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################
import numpy as np
import scipy.linalg as la
from qutip.cy.interpolate import (interp, arr_interp,
                                 zinterp, arr_zinterp)

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
    
    Habermann & Kindermann, "Multidimensional Spline Interpolation: 
    Theory and Applications", Comput Econ 30, 153 (2007).  
    
    '''
    
    def __init__(self, a, b, y, alpha=0, beta=0):
        y = np.asarray(y)
        n = y.shape[0] - 1
        h = (b - a)/n

        coeff = np.zeros(n + 3, dtype=y.dtype)
        # Solutions to boundary coeffcients of spline
        coeff[1] = 1/6. * (y[0] - (alpha * h**2)/6) #C2 in paper
        coeff[n + 1] = 1/6. * (y[n] - (beta * h**2)/6) #cn+2 in paper

        # Compressed tridiagonal matrix 
        ab = np.ones((3, n - 1), dtype=float)
        ab[0,0] = 0 # Because top row is upper diag with one less elem
        ab[1, :] = 4
        ab[-1,-1] = 0 # Because bottom row is lower diag with one less elem
        
        B = y[1:-1].copy() #grabs elements y[1] - > y[n-2] for reduced array
        B[0] -= coeff[1]
        B[-1] -=  coeff[n + 1]

        coeff[2:-2] = la.solve_banded((1, 1), ab, B, overwrite_ab=True, 
                        overwrite_b=True, check_finite=False)

        coeff[0] = alpha * h**2/6. + 2 * coeff[1] - coeff[2]
        coeff[-1] = beta * h**2/6. + 2 * coeff[-2] - coeff[-3]

        self.a = a          # Lower-bound of domain
        self.b = b          # Uppser-bound of domain
        self.coeffs = coeff # Spline coefficients
        self.is_complex = (y.dtype == complex) #Tells which dtype solver to use
        
    def __call__(self, pnts, *args):
        #If requesting a single return value
        if isinstance(pnts, (int, float, complex)):
            if self.is_complex:
                return zinterp(pnts, self.a, 
                                    self.b, self.coeffs)
            else:
                return interp(pnts, self.a, self.b, self.coeffs)
        #If requesting multiple return values from array_like
        elif isinstance(pnts, (np.ndarray,list)):
            pnts = np.asarray(pnts)
            if self.is_complex:
                return arr_zinterp(pnts, self.a, 
                                                self.b, self.coeffs)
            else:
                return arr_interp(pnts, self.a, self.b, self.coeffs)
    
    