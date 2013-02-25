# This file is part of QuTiP.
#
#    QuTiP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    QuTiP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTiP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011-2013, Paul D. Nation & Robert J. Johansson
#
###########################################################################
import numpy as np
from scipy import (zeros, array, arange, exp, real, imag, conj, pi,
                   copy, sqrt, meshgrid, size, polyval, fliplr, conjugate)
import scipy.sparse as sp
import scipy.linalg as la
from scipy.special import genlaguerre
from qutip.tensor import tensor
from qutip.qobj import Qobj, isket, isoper, issuper
from qutip.states import *
from qutip.parfor import parfor

try:  # for scipy v <= 0.90
    from scipy import factorial
except:  # for scipy v >= 0.10
    from scipy.misc import factorial


def wigner(psi, xvec, yvec, g=sqrt(2), method='iterative', parfor=False):
    """Wigner function for a state vector or density matrix at points
    `xvec + i * yvec`.

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

    method : string {'iterative', 'laguerre'}
        Select method 'iterative' or 'laguerre', where 'iterative' use a
        iterative method to evaluate the Wigner functions for density matrices
        :math:`|m><n|`, while 'laguerre' uses the Laguerre polynomials in scipy
        for the same task. The 'iterative' method is default, and in general
        recommended, but the 'laguerre' method is more efficient for very
        sparse density matrices (e.g., superpositions of Fock states in a large
        Hilbert space).

    parfor : bool {False, True}
        Flag for calculating the Laguerre polynomial based Wigner function
        method='laguerre' in parallel using the parfor function.


    Returns
    -------

    W : array
        Values representing the Wigner function calculated over the specified
        range [xvec,yvec].


    References
    ----------

    Ulf Leonhardt,
    Measuring the Quantum State of Light, (Cambridge University Press, 1997)

    """

    if not (psi.type == 'ket' or psi.type == 'oper' or psi.type == 'bra'):
        raise TypeError('Input state is not a valid operator.')

    if psi.type == 'ket' or psi.type == 'bra':
        rho = ket2dm(psi)
    else:
        rho = psi

    if method == 'iterative':
        return _wigner_iterative(rho, xvec, yvec, g)

    elif method == 'laguerre':
        return _wigner_laguerre(rho, xvec, yvec, g, parfor)

    else:
        raise TypeError("method must be either 'iterative' or 'laguerre'")


def _wigner_iterative(rho, xvec, yvec, g=sqrt(2)):
    """
    Using an iterative method to evaluate the wigner functions for the Fock
    state :math:`|m><n|`.

    The wigner function is calculated as
    :math:`W = \sum_{mn} \\rho_{mn} W_{mn}` where :math:`W_{mn}` is the wigner
    function for the density matrix :math:`|m><n|`.

    In this implementation, for each row m, Wlist contains the wigner functions
    Wlist = [0, ..., W_mm, ..., W_mN]. As soon as one W_mn wigner function is
    calculated, the corresponding contribution is added to the total wigner
    function, weighted by the corresponding element in the density matrix
    :math:`rho_{mn}`.
    """

    M = prod(rho.shape[0])
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

    M = prod(rho.shape[0])
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
    Private function for calculating terms of Laguerre Wigner function in
    parfor.
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


#------------------------------------------------------------------------------
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
    n = prod(psi.shape)
    if isinstance(psi, Qobj):
        psi = array(psi.trans().full())[0, :]
    else:
        psi = psi.T

    qmat1 = abs(polyval(fliplr(
        [psi / sqrt(factorial(arange(0, n)))])[0], conjugate(alpha_mat))) ** 2
    qmat1 = real(qmat1) * exp(-abs(alpha_mat) ** 2) / pi

    return qmat1
