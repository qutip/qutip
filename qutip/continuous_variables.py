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

"""
This module contains a collection functions for calculating continous variable
quantities from fock-basis representation of the state of multi-mode fields.
"""

from qutip.expect import expect
import numpy as np


def correlation_matrix(basis, rho=None):
    """
    Given a basis set of operators :math:`\\{a\\}_n`, calculate the correlation
    matrix:

    .. math::

        C_{mn} = \\langle a_m a_n \\rangle

    Parameters
    ----------

    basis : list of :class:`qutip.qobj.Qobj`
        List of operators that defines the basis for the correlation matrix.

    rho : :class:`qutip.qobj.Qobj`
        Density matrix for which to calculate the correlation matrix. If
        `rho` is `None`, then a matrix of correlation matrix operators is
        returned instead of expectation values of those operators.

    Returns
    -------

    corr_mat: *array*
        A 2-dimensional *array* of correlation values or operators.


    """

    if rho is None:
        # return array of operators
        return np.array([[op1 * op2 for op1 in basis] for op2 in basis])
    else:
        # return array of expectation values
        return np.array([[expect(op1 * op2, rho)
                          for op1 in basis] for op2 in basis])


def covariance_matrix(basis, rho, symmetrized=True):
    """
    Given a basis set of operators :math:`\{a\}_n`, calculate the covariance
    matrix:

    .. math::

        V_{mn} = \\frac{1}{2}\\langle a_m a_n + a_n a_m \\rangle -
        \\langle a_m \\rangle \\langle a_n\\rangle

    or, if of the optional argument `symmetrized=False`,

    .. math::

        V_{mn} = \\langle a_m a_n\\rangle -
        \\langle a_m \\rangle \\langle a_n\\rangle


    Parameters
    ----------

    basis : list of :class:`qutip.qobj.Qobj`
        List of operators that defines the basis for the covariance matrix.

    rho : :class:`qutip.qobj.Qobj`
        Density matrix for which to calculate the covariance matrix.

    symmetrized : *bool*
        Flag indicating whether the symmetrized (default) or non-symmetrized
        correlation matrix is to be calculated.

    Returns
    -------

    corr_mat: *array*
        A 2-dimensional *array* of covariance values.

    """
    if symmetrized:
        return np.array([[0.5 * expect(op1 * op2 + op2 * op1, rho) -
                          expect(op1, rho) * expect(op2, rho)
                          for op1 in basis] for op2 in basis])
    else:
        return np.array([[expect(op1 * op2, rho) -
                          expect(op1, rho) * expect(op2, rho)
                          for op1 in basis] for op2 in basis])


def correlation_matrix_field(a1, a2, rho=None):
    """
    Calculate the correlation matrix for given field operators :math:`a_1` and
    :math:`a_2`. If a density matrix is given the expectation values are
    calculated, otherwise a matrix with operators is returned.

    Parameters
    ----------

    a1 : :class:`qutip.qobj.Qobj`
        Field operator for mode 1.

    a2 : :class:`qutip.qobj.Qobj`
        Field operator for mode 2.

    rho : :class:`qutip.qobj.Qobj`
        Density matrix for which to calculate the covariance matrix.

    Returns
    -------

    cov_mat: *array* of complex numbers or :class:`qutip.qobj.Qobj`
        A 2-dimensional *array* of covariance values, or, if rho=0, a matrix
        of operators.
    """

    basis = [a1, a1.dag(), a2, a2.dag()]

    return correlation_matrix(basis, rho)


def correlation_matrix_quadrature(a1, a2, rho=None):
    """
    Calculate the quadrature correlation matrix with given field operators
    :math:`a_1` and :math:`a_2`. If a density matrix is given the expectation
    values are calculated, otherwise a matrix with operators is returned.

    Parameters
    ----------

    a1 : :class:`qutip.qobj.Qobj`
        Field operator for mode 1.

    a2 : :class:`qutip.qobj.Qobj`
        Field operator for mode 2.

    rho : :class:`qutip.qobj.Qobj`
        Density matrix for which to calculate the covariance matrix.

    Returns
    -------

    corr_mat: *array* of complex numbers or :class:`qutip.qobj.Qobj`
        A 2-dimensional *array* of covariance values for the field quadratures,
        or, if rho=0, a matrix of operators.

    """
    x1 = (a1 + a1.dag()) / np.sqrt(2)
    p1 = -1j * (a1 - a1.dag()) / np.sqrt(2)
    x2 = (a2 + a2.dag()) / np.sqrt(2)
    p2 = -1j * (a2 - a2.dag()) / np.sqrt(2)

    basis = [x1, p1, x2, p2]

    return correlation_matrix(basis, rho)


def wigner_covariance_matrix(a1=None, a2=None, R=None, rho=None):
    """
    Calculate the wigner covariance matrix
    :math:`V_{ij} = \\frac{1}{2}(R_{ij} + R_{ji})`, given
    the quadrature correlation matrix
    :math:`R_{ij} = \\langle R_{i} R_{j}\\rangle -
    \\langle R_{i}\\rangle \\langle R_{j}\\rangle`, where
    :math:`R = (q_1, p_1, q_2, p_2)^T` is the vector with quadrature operators
    for the two modes.

    Alternatively, if `R = None`, and if annilation operators `a1` and `a2`
    for the two modes are supplied instead, the quadature correlation matrix
    is constructed from the annihilation operators before then the covariance
    matrix is calculated.

    Parameters
    ----------

    a1 : :class:`qutip.qobj.Qobj`
        Field operator for mode 1.

    a2 : :class:`qutip.qobj.Qobj`
        Field operator for mode 2.

    R : *array*
        The quadrature correlation matrix.

    rho : :class:`qutip.qobj.Qobj`
        Density matrix for which to calculate the covariance matrix.

    Returns
    -------

    cov_mat: *array*
        A 2-dimensional *array* of covariance values.

    """
    if R is not None:

        if rho is None:
            return np.array([[0.5 * np.real(R[i, j] + R[j, i])
                              for i in range(4)]
                             for j in range(4)])
        else:
            return np.array([[0.5 * np.real(expect(R[i, j] + R[j, i], rho))
                              for i in range(4)]
                             for j in range(4)])

    elif a1 is not None and a2 is not None:

        if rho is not None:
            x1 = (a1 + a1.dag()) / np.sqrt(2)
            p1 = -1j * (a1 - a1.dag()) / np.sqrt(2)
            x2 = (a2 + a2.dag()) / np.sqrt(2)
            p2 = -1j * (a2 - a2.dag()) / np.sqrt(2)
            return covariance_matrix([x1, p1, x2, p2], rho)
        else:
            raise ValueError("Must give rho if using field operators " +
                             "(a1 and a2)")

    else:
        raise ValueError("Must give either field operators (a1 and a2) " +
                         "or a precomputed correlation matrix (R)")


def logarithmic_negativity(V):
    """
    Calculate the logarithmic negativity given the symmetrized covariance
    matrix, see :func:`qutip.continous_variables.covariance_matrix`. Note that
    the two-mode field state that is described by `V` must be Gaussian for this
    function to applicable.

    Parameters
    ----------

    V : *2d array*
        The covariance matrix.

    Returns
    -------

    N: *float*, the logarithmic negativity for the two-mode Guassian state
    that is described by the the Wigner covariance matrix V.

    """

    A = V[0:2, 0:2]
    B = V[2:4, 2:4]
    C = V[0:2, 2:4]

    sigma = np.linalg.det(A) + np.linalg.det(B) - 2 * np.linalg.det(C)
    nu_ = sigma / 2 - np.sqrt(sigma ** 2 - 4 * np.linalg.det(V)) / 2
    if nu_ < 0.0:
        return 0.0
    nu = np.sqrt(nu_)
    lognu = -np.log(2 * nu)
    logneg = max(0, lognu)

    return logneg
