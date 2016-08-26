# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
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

__all__ = ['wigner', 'qfunc', 'spin_q_function', 'spin_wigner']

import numpy as np
from scipy import (zeros, array, arange, exp, real, conj, pi,
                   copy, sqrt, meshgrid, size, polyval, fliplr, conjugate,
                   cos, sin)
import scipy.sparse as sp
import scipy.fftpack as ft
import scipy.linalg as la
from scipy.special import genlaguerre
from scipy.special import binom
from scipy.special import sph_harm

from qutip.qobj import Qobj, isket, isoper
from qutip.states import ket2dm
from qutip.parallel import parfor
from qutip.utilities import clebsch
from scipy.misc import factorial
from qutip.cy.sparse_utils import _csr_get_diag


def wigner(psi, xvec, yvec, method='clenshaw', g=sqrt(2), 
            sparse=False, parfor=False):
    """Wigner function for a state vector or density matrix at points
    `xvec + i * yvec`.

    Parameters
    ----------

    state : qobj
        A state vector or density matrix.

    xvec : array_like
        x-coordinates at which to calculate the Wigner function.

    yvec : array_like
        y-coordinates at which to calculate the Wigner function.  Does not
        apply to the 'fft' method.

    g : float
        Scaling factor for `a = 0.5 * g * (x + iy)`, default `g = sqrt(2)`.

    method : string {'clenshaw', 'iterative', 'laguerre', 'fft'}
        Select method 'clenshaw' 'iterative', 'laguerre', or 'fft', where 'clenshaw' 
        and 'iterative' use an iterative method to evaluate the Wigner functions for density
        matrices :math:`|m><n|`, while 'laguerre' uses the Laguerre polynomials
        in scipy for the same task. The 'fft' method evaluates the Fourier
        transform of the density matrix. The 'iterative' method is default, and
        in general recommended, but the 'laguerre' method is more efficient for
        very sparse density matrices (e.g., superpositions of Fock states in a
        large Hilbert space). The 'clenshaw' method is the preferred method for
        dealing with density matrices that have a large number of excitations
        (>~50). 'clenshaw' is a fast and numerically stable method.

    sparse : bool {False, True}
        Tells the default solver whether or not to keep the input density
        matrix in sparse format.  As the dimensions of the density matrix
        grow, setthing this flag can result in increased performance.
    
    parfor : bool {False, True}
        Flag for calculating the Laguerre polynomial based Wigner function
        method='laguerre' in parallel using the parfor function.


    Returns
    -------

    W : array
        Values representing the Wigner function calculated over the specified
        range [xvec,yvec].

    yvex : array
        FFT ONLY. Returns the y-coordinate values calculated via the Fourier
        transform.

    Notes
    -----
    The 'fft' method accepts only an xvec input for the x-coordinate.
    The y-coordinates are calculated internally.

    References
    ----------

    Ulf Leonhardt,
    Measuring the Quantum State of Light, (Cambridge University Press, 1997)

    """

    if not (psi.type == 'ket' or psi.type == 'oper' or psi.type == 'bra'):
        raise TypeError('Input state is not a valid operator.')

    if method == 'fft':
        return _wigner_fourier(psi, xvec, g)

    if psi.type == 'ket' or psi.type == 'bra':
        rho = ket2dm(psi)
    else:
        rho = psi

    if method == 'iterative':
        return _wigner_iterative(rho, xvec, yvec, g)

    elif method == 'laguerre':
        return _wigner_laguerre(rho, xvec, yvec, g, parfor)
        
    elif method == 'clenshaw':
        return _wigner_clenshaw(rho, xvec, yvec, g, sparse=sparse)

    else:
        raise TypeError(
            "method must be either 'iterative', 'laguerre', or 'fft'.")


def _wigner_iterative(rho, xvec, yvec, g=sqrt(2)):
    """
    Using an iterative method to evaluate the wigner functions for the Fock
    state :math:`|m><n|`.

    The Wigner function is calculated as
    :math:`W = \sum_{mn} \\rho_{mn} W_{mn}` where :math:`W_{mn}` is the Wigner
    function for the density matrix :math:`|m><n|`.

    In this implementation, for each row m, Wlist contains the Wigner functions
    Wlist = [0, ..., W_mm, ..., W_mN]. As soon as one W_mn Wigner function is
    calculated, the corresponding contribution is added to the total Wigner
    function, weighted by the corresponding element in the density matrix
    :math:`rho_{mn}`.
    """

    M = np.prod(rho.shape[0])
    X, Y = meshgrid(xvec, yvec)
    A = 0.5 * g * (X + 1.0j * Y)

    Wlist = array([zeros(np.shape(A), dtype=complex) for k in range(M)])
    Wlist[0] = exp(-2.0 * abs(A) ** 2) / pi

    W = real(rho[0, 0]) * real(Wlist[0])
    for n in range(1, M):
        Wlist[n] = (2.0 * A * Wlist[n - 1]) / sqrt(n)
        W += 2 * real(rho[0, n] * Wlist[n])

    for m in range(1, M):
        temp = copy(Wlist[m])
        Wlist[m] = (2 * conj(A) * temp - sqrt(m) * Wlist[m - 1]) / sqrt(m)

        # Wlist[m] = Wigner function for |m><m|
        W += real(rho[m, m] * Wlist[m])

        for n in range(m + 1, M):
            temp2 = (2 * A * Wlist[n - 1] - sqrt(m) * temp) / sqrt(n)
            temp = copy(Wlist[n])
            Wlist[n] = temp2

            # Wlist[n] = Wigner function for |m><n|
            W += 2 * real(rho[m, n] * Wlist[n])

    return 0.5 * W * g ** 2


def _wigner_laguerre(rho, xvec, yvec, g, parallel):
    """
    Using Laguerre polynomials from scipy to evaluate the Wigner function for
    the density matrices :math:`|m><n|`, :math:`W_{mn}`. The total Wigner
    function is calculated as :math:`W = \sum_{mn} \\rho_{mn} W_{mn}`.
    """

    M = np.prod(rho.shape[0])
    X, Y = meshgrid(xvec, yvec)
    A = 0.5 * g * (X + 1.0j * Y)
    W = zeros(np.shape(A))

    # compute wigner functions for density matrices |m><n| and
    # weight by all the elements in the density matrix
    B = 4 * abs(A) ** 2
    if sp.isspmatrix_csr(rho.data):
        # for compress sparse row matrices
        if parallel:
            iterator = (
                (m, rho, A, B) for m in range(len(rho.data.indptr) - 1))
            W1_out = parfor(_par_wig_eval, iterator)
            W += sum(W1_out)
        else:
            for m in range(len(rho.data.indptr) - 1):
                for jj in range(rho.data.indptr[m], rho.data.indptr[m + 1]):
                    n = rho.data.indices[jj]

                    if m == n:
                        W += real(rho[m, m] * (-1) ** m * genlaguerre(m, 0)(B))

                    elif n > m:
                        W += 2.0 * real(rho[m, n] * (-1) ** m *
                                        (2 * A) ** (n - m) *
                                        sqrt(factorial(m) / factorial(n)) *
                                        genlaguerre(m, n - m)(B))
    else:
        # for dense density matrices
        B = 4 * abs(A) ** 2
        for m in range(M):
            if abs(rho[m, m]) > 0.0:
                W += real(rho[m, m] * (-1) ** m * genlaguerre(m, 0)(B))
            for n in range(m + 1, M):
                if abs(rho[m, n]) > 0.0:
                    W += 2.0 * real(rho[m, n] * (-1) ** m *
                                    (2 * A) ** (n - m) *
                                    sqrt(factorial(m) / factorial(n)) *
                                    genlaguerre(m, n - m)(B))

    return 0.5 * W * g ** 2 * np.exp(-B / 2) / pi


def _par_wig_eval(args):
    """
    Private function for calculating terms of Laguerre Wigner function
    using parfor.
    """
    m, rho, A, B = args
    W1 = zeros(np.shape(A))
    for jj in range(rho.data.indptr[m], rho.data.indptr[m + 1]):
        n = rho.data.indices[jj]

        if m == n:
            W1 += real(rho[m, m] * (-1) ** m * genlaguerre(m, 0)(B))

        elif n > m:
            W1 += 2.0 * real(rho[m, n] * (-1) ** m *
                             (2 * A) ** (n - m) *
                             sqrt(factorial(m) / factorial(n)) *
                             genlaguerre(m, n - m)(B))
    return W1


def _wigner_fourier(psi, xvec, g=np.sqrt(2)):
    """
    Evaluate the Wigner function via the Fourier transform.
    """
    if psi.type == 'bra':
        psi = psi.dag()
    if psi.type == 'ket':
        return _psi_wigner_fft(psi.full(), xvec, g)
    elif psi.type == 'oper':
        eig_vals, eig_vecs = la.eigh(psi.full())
        W = 0
        for ii in range(psi.shape[0]):
            W1, yvec = _psi_wigner_fft(
                np.reshape(eig_vecs[:, ii], (psi.shape[0], 1)), xvec, g)
            W += eig_vals[ii] * W1
        return W, yvec


def _psi_wigner_fft(psi, xvec, g=sqrt(2)):
    """
    FFT method for a single state vector.  Called multiple times when the
    input is a density matrix.
    """
    n = len(psi)
    A = _osc_eigen(n, xvec * g / np.sqrt(2))
    xpsi = np.dot(psi.T, A)
    W, yvec = _wigner_fft(xpsi, xvec * g / np.sqrt(2))
    return (0.5 * g ** 2) * np.real(W.T), yvec * np.sqrt(2) / g


def _wigner_fft(psi, xvec):
    """
    Evaluates the Fourier transformation of a given state vector.
    Returns the corresponding density matrix and range
    """
    n = 2*len(psi.T)
    r1 = np.concatenate((np.array([[0]]),
                        np.fliplr(psi.conj()),
                        np.zeros((1, n//2 - 1))), axis=1)
    r2 = np.concatenate((np.array([[0]]), psi,
                        np.zeros((1, n//2 - 1))), axis=1)
    w = la.toeplitz(np.zeros((n//2, 1)), r1) * \
        np.flipud(la.toeplitz(np.zeros((n//2, 1)), r2))
    w = np.concatenate((w[:, n//2:n], w[:, 0:n//2]), axis=1)
    w = ft.fft(w)
    w = np.real(np.concatenate((w[:, 3*n//4:n+1], w[:, 0:n//4]), axis=1))
    p = np.arange(-n/4, n/4)*np.pi / (n*(xvec[1] - xvec[0]))
    w = w / (p[1] - p[0]) / n
    return w, p


def _osc_eigen(N, pnts):
    """
    Vector of and N-dim oscillator eigenfunctions evaluated
    at the points in pnts.
    """
    pnts = np.asarray(pnts)
    lpnts = len(pnts)
    A = np.zeros((N, lpnts))
    A[0, :] = np.exp(-pnts ** 2 / 2.0) / pi ** 0.25
    if N == 1:
        return A
    else:
        A[1, :] = np.sqrt(2) * pnts * A[0, :]
        for k in range(2, N):
            A[k, :] = np.sqrt(2.0 / k) * pnts * A[k - 1, :] - \
                np.sqrt((k - 1.0) / k) * A[k - 2, :]
        return A


def _wigner_clenshaw(rho, xvec, yvec, g=sqrt(2), sparse=False):
    """
    Using Clenshaw summation - numerically stable and efficient
    iterative algorithm to evaluate polynomial series.
    
    The Wigner function is calculated as
    :math:`W = e^(-0.5*x^2)/pi * \sum_{L} c_L (2x)^L / sqrt(L!)` where 
    :math:`c_L = \sum_n \\rho_{n,L+n} LL_n^L` where
    :math:`LL_n^L = (-1)^n sqrt(L!n!/(L+n)!) LaguerreL[n,L,x]`
    
    """

    M = np.prod(rho.shape[0])
    X,Y = np.meshgrid(xvec, yvec)
    #A = 0.5 * g * (X + 1.0j * Y)
    A2 = g * (X + 1.0j * Y) #this is A2 = 2*A
    
    B = np.abs(A2)
    B *= B
    w0 = (2*rho.data[0,-1])*np.ones_like(A2)
    L = M-1
    #calculation of \sum_{L} c_L (2x)^L / sqrt(L!)
    #using Horner's method
    if not sparse:
        rho = rho.full() * (2*np.ones((M,M)) - np.diag(np.ones(M)))
        while L > 0:
            L -= 1
            #here c_L = _wig_laguerre_val(L, B, np.diag(rho, L))
            w0 = _wig_laguerre_val(L, B, np.diag(rho, L)) + w0 * A2 * (L+1)**-0.5
    else:
        while L > 0:
            L -= 1
            diag = _csr_get_diag(rho.data.data,rho.data.indices,
                                rho.data.indptr,L)
            if L != 0:
                diag *= 2
            #here c_L = _wig_laguerre_val(L, B, np.diag(rho, L))
            w0 = _wig_laguerre_val(L, B, diag) + w0 * A2 * (L+1)**-0.5
        
    return w0.real * np.exp(-B*0.5) * (g*g*0.5 / pi)


def _wig_laguerre_val(L, x, c):
    """
    this is evaluation of polynomial series inspired by hermval from numpy.    
    Returns polynomial series
    \sum_n b_n LL_n^L,
    where
    LL_n^L = (-1)^n sqrt(L!n!/(L+n)!) LaguerreL[n,L,x]    
    The evaluation uses Clenshaw recursion
    """

    if len(c) == 1:
        y0 = c[0]
        y1 = 0
    elif len(c) == 2:
        y0 = c[0]
        y1 = c[1]
    else:
        k = len(c)
        y0 = c[-2]
        y1 = c[-1]
        for i in range(3, len(c) + 1):
            k -= 1
            y0,    y1 = c[-i] - y1 * (float((k - 1)*(L + k - 1))/((L+k)*k))**0.5, \
            y0 - y1 * ((L + 2*k -1) - x) * ((L+k)*k)**-0.5
            
    return y0 - y1 * ((L + 1) - x) * (L + 1)**-0.5
    
    

# -----------------------------------------------------------------------------
# Q FUNCTION
#
def qfunc(state, xvec, yvec, g=sqrt(2)):
    """Q-function of a given state vector or density matrix
    at points `xvec + i * yvec`.

    Parameters
    ----------
    state : qobj
        A state vector or density matrix.

    xvec : array_like
        x-coordinates at which to calculate the Wigner function.

    yvec : array_like
        y-coordinates at which to calculate the Wigner function.

    g : float
        Scaling factor for `a = 0.5 * g * (x + iy)`, default `g = sqrt(2)`.

    Returns
    --------
    Q : array
        Values representing the Q-function calculated over the specified range
        [xvec,yvec].

    """
    X, Y = meshgrid(xvec, yvec)
    amat = 0.5 * g * (X + Y * 1j)

    if not (isoper(state) or isket(state)):
        raise TypeError('Invalid state operand to qfunc.')

    qmat = zeros(size(amat))

    if isket(state):
        qmat = _qfunc_pure(state, amat)
    elif isoper(state):
        d, v = la.eig(state.full())
        # d[i]   = eigenvalue i
        # v[:,i] = eigenvector i

        qmat = zeros(np.shape(amat))
        for k in arange(0, len(d)):
            qmat1 = _qfunc_pure(v[:, k], amat)
            qmat += real(d[k] * qmat1)

    qmat = 0.25 * qmat * g ** 2
    return qmat


#
# Q-function for a pure state: Q = |<alpha|psi>|^2 / pi
#
# |psi>   = the state in fock basis
# |alpha> = the coherent state with amplitude alpha
#
def _qfunc_pure(psi, alpha_mat):
    """
    Calculate the Q-function for a pure state.
    """
    n = np.prod(psi.shape)
    if isinstance(psi, Qobj):
        psi = psi.full().flatten()
    else:
        psi = psi.T

    qmat = abs(polyval(fliplr([psi / sqrt(factorial(arange(n)))])[0],
                       conjugate(alpha_mat))) ** 2

    return real(qmat) * exp(-abs(alpha_mat) ** 2) / pi


# -----------------------------------------------------------------------------
# PSEUDO DISTRIBUTION FUNCTIONS FOR SPINS
#
def spin_q_function(rho, theta, phi):
    """Husimi Q-function for spins.

    Parameters
    ----------
    state : qobj
        A state vector or density matrix for a spin-j quantum system.
    theta : array_like
        theta-coordinates at which to calculate the Q function.
    phi : array_like
        phi-coordinates at which to calculate the Q function.

    Returns
    -------
    Q, THETA, PHI : 2d-array
        Values representing the spin Q function at the values specified
        by THETA and PHI.

    """

    if rho.type == 'bra':
        rho = rho.dag()

    if rho.type == 'ket':
        rho = ket2dm(rho)

    J = rho.shape[0]
    j = (J - 1) / 2

    THETA, PHI = meshgrid(theta, phi)

    Q = np.zeros_like(THETA, dtype=complex)

    for m1 in arange(-j, j+1):

        Q += binom(2*j, j+m1) * cos(THETA/2) ** (2*(j-m1)) * sin(THETA/2) ** (2*(j+m1)) * \
            rho.data[int(j-m1), int(j-m1)]

        for m2 in arange(m1+1, j+1):

            Q += (sqrt(binom(2*j, j+m1)) * sqrt(binom(2*j, j+m2)) *
                  cos(THETA/2) ** (2*j-m1-m2) * sin(THETA/2) ** (2*j+m1+m2)) * \
                (exp(1j * (m2-m1) * PHI) * rho.data[int(j-m1), int(j-m2)] +
                 exp(1j * (m1-m2) * PHI) * rho.data[int(j-m2), int(j-m1)])

    return Q.real, THETA, PHI


def _rho_kq(rho, j, k, q):
    v = 0j

    for m1 in arange(-j, j+1):
        for m2 in arange(-j, j+1):
            v += (-1)**(j - m1 - q) * clebsch(j, j, k, m1, -m2,
                                              q) * rho.data[m1 + j, m2 + j]

    return v


def spin_wigner(rho, theta, phi):
    """Wigner function for spins on the Bloch sphere.

    Parameters
    ----------
    state : qobj
        A state vector or density matrix for a spin-j quantum system.
    theta : array_like
        theta-coordinates at which to calculate the Q function.
    phi : array_like
        phi-coordinates at which to calculate the Q function.

    Returns
    -------
    W, THETA, PHI : 2d-array
        Values representing the spin Wigner function at the values specified
        by THETA and PHI.

    Notes
    -----
    Experimental.

    """

    if rho.type == 'bra':
        rho = rho.dag()

    if rho.type == 'ket':
        rho = ket2dm(rho)

    J = rho.shape[0]
    j = (J - 1) / 2

    THETA, PHI = meshgrid(theta, phi)

    W = np.zeros_like(THETA, dtype=complex)

    for k in range(int(2 * j)+1):
        for q in range(-k, k+1):
            W += _rho_kq(rho, j, k, q) * sph_harm(q, k, PHI, THETA)

    return W, THETA, PHI
