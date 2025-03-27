"""
This module contains a collection functions for calculating continuous variable
quantities from fock-basis representation of the state of multi-mode fields.
"""

__all__ = ['correlation_matrix', 'covariance_matrix',
           'correlation_matrix_field', 'correlation_matrix_quadrature',
           'wigner_covariance_matrix', 'logarithmic_negativity']

from . import expect
import numpy as np


def correlation_matrix(basis, rho=None):
    r"""
    Given a basis set of operators :math:`\{a\}_n`, calculate the correlation
    matrix:

    .. math::

        C_{mn} = \langle a_m a_n \rangle

    Parameters
    ----------
    basis : list
        List of operators that defines the basis for the correlation matrix.
    rho : Qobj, optional
        Density matrix for which to calculate the correlation matrix. If
        `rho` is `None`, then a matrix of correlation matrix operators is
        returned instead of expectation values of those operators.

    Returns
    -------
    corr_mat : ndarray
        A 2-dimensional *array* of correlation values or operators.


    """
    if rho is None:
        # return array of operators
        out = np.empty((len(basis), len(basis)), dtype=object)
        for i, op2 in enumerate(basis):
            out[i, :] = [op1 * op2 for op1 in basis]
        return out
    else:
        # return array of expectation values
        return np.array([[expect(op1 * op2, rho)
                          for op1 in basis] for op2 in basis])


def covariance_matrix(basis, rho, symmetrized=True):
    r"""
    Given a basis set of operators :math:`\{a\}_n`, calculate the covariance
    matrix:

    .. math::

        V_{mn} = \frac{1}{2}\langle a_m a_n + a_n a_m \rangle -
        \langle a_m \rangle \langle a_n\rangle

    or, if of the optional argument `symmetrized=False`,

    .. math::

        V_{mn} = \langle a_m a_n\rangle -
        \langle a_m \rangle \langle a_n\rangle

    Parameters
    ----------
    basis : list
        List of operators that defines the basis for the covariance matrix.
    rho : Qobj
        Density matrix for which to calculate the covariance matrix.
    symmetrized : bool, default: True
        Flag indicating whether the symmetrized (default) or non-symmetrized
        correlation matrix is to be calculated.

    Returns
    -------
    corr_mat : ndarray
        A 2-dimensional array of covariance values.

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
    Calculates the correlation matrix for given field operators :math:`a_1` and
    :math:`a_2`. If a density matrix is given the expectation values are
    calculated, otherwise a matrix with operators is returned.

    Parameters
    ----------
    a1 : Qobj
        Field operator for mode 1.
    a2 : Qobj
        Field operator for mode 2.
    rho : Qobj, optional
        Density matrix for which to calculate the covariance matrix.

    Returns
    -------
    cov_mat : ndarray
        Array of complex numbers or Qobj's
        A 2-dimensional *array* of covariance values, or, if rho=0, a matrix
        of operators.
    """
    basis = [a1, a1.dag(), a2, a2.dag()]
    return correlation_matrix(basis, rho)


def correlation_matrix_quadrature(a1, a2, rho=None, g=np.sqrt(2)):
    """
    Calculate the quadrature correlation matrix with given field operators
    :math:`a_1` and :math:`a_2`. If a density matrix is given the expectation
    values are calculated, otherwise a matrix with operators is returned.

    Parameters
    ----------
    a1 : Qobj
        Field operator for mode 1.
    a2 : Qobj
        Field operator for mode 2.
    rho : Qobj, optional
        Density matrix for which to calculate the covariance matrix.
    g : float, default: sqrt(2)
        Scaling factor for ``a = 0.5 * g * (x + iy)``, default ``g = sqrt(2)``.
        The value of ``g`` is related to the value of ``hbar`` in the
        commutation relation ``[x, y] = i * hbar`` via ``hbar=2/g ** 2`` giving
        the default value ``hbar=1``.

    Returns
    -------
    corr_mat : ndarray
        Array of complex numbers or Qobj's
        A 2-dimensional *array* of covariance values for the field quadratures,
        or, if rho=0, a matrix of operators.

    """
    x1 = (a1 + a1.dag()) / g
    p1 = -1j * (a1 - a1.dag()) / g
    x2 = (a2 + a2.dag()) / g
    p2 = -1j * (a2 - a2.dag()) / g
    basis = [x1, p1, x2, p2]
    return correlation_matrix(basis, rho)


def wigner_covariance_matrix(a1=None, a2=None, R=None, rho=None, g=np.sqrt(2)):
    r"""
    Calculates the Wigner covariance matrix
    :math:`V_{ij} = \frac{1}{2}(R_{ij} + R_{ji})`, given
    the quadrature correlation matrix
    :math:`R_{ij} = \langle R_{i} R_{j}\rangle -
    \langle R_{i}\rangle \langle R_{j}\rangle`, where
    :math:`R = (q_1, p_1, q_2, p_2)^T` is the vector with quadrature operators
    for the two modes.

    Alternatively, if ``R = None``, and if annihilation operators ``a1`` and
    ``a2`` for the two modes are supplied instead, the quadrature correlation
    matrix is constructed from the annihilation operators before then the
    covariance matrix is calculated.

    Parameters
    ----------

    a1 : Qobj, optional
        Field operator for mode 1.

    a2 : Qobj, optional
        Field operator for mode 2.

    R : ndarray, optional
        The quadrature correlation matrix.

    rho : Qobj, optional
        Density matrix for which to calculate the covariance matrix.

    g : float, default: sqrt(2)
        Scaling factor for ``a = 0.5 * g * (x + iy)``, default ``g = sqrt(2)``.
        The value of ``g`` is related to the value of ``hbar`` in the
        commutation relation ``[x, y] = i * hbar`` via ``hbar=2/g ** 2`` giving
        the default value ``hbar=1``.

    Returns
    -------
    cov_mat : ndarray
        A 2-dimensional array of covariance values.

    """
    if R is not None:
        if rho is None:
            return np.array([[0.5 * np.real(R[i, j] + R[j, i])
                              for i in range(4)]
                             for j in range(4)], dtype=np.float64)
        else:
            return np.array([[0.5 * np.real(expect(R[i, j] + R[j, i], rho))
                              for i in range(4)]
                             for j in range(4)], dtype=np.float64)
    elif a1 is not None and a2 is not None:
        if rho is not None:
            x1 = (a1 + a1.dag()) / g
            p1 = -1j * (a1 - a1.dag()) / g
            x2 = (a2 + a2.dag()) / g
            p2 = -1j * (a2 - a2.dag()) / g
            return covariance_matrix([x1, p1, x2, p2], rho)
        else:
            raise ValueError("Must give rho if using field operators " +
                             "(a1 and a2)")
    else:
        raise ValueError("Must give either field operators (a1 and a2) " +
                         "or a precomputed correlation matrix (R)")


def logarithmic_negativity(V, g=np.sqrt(2)):
    """
    Calculates the logarithmic negativity given a symmetrized covariance
    matrix, see :func:`qutip.continuous_variables.covariance_matrix`. Note that
    the two-mode field state that is described by `V` must be Gaussian for this
    function to applicable.

    Parameters
    ----------

    V : ndarray
        The covariance matrix.

    g : float, default: sqrt(2)
        Scaling factor for ``a = 0.5 * g * (x + iy)``, default ``g = sqrt(2)``.
        The value of ``g`` is related to the value of ``hbar`` in the
        commutation relation ``[x, y] = i * hbar`` via ``hbar=2/g ** 2`` giving
        the default value ``hbar=1``.

    Returns
    -------

    N : float
        The logarithmic negativity for the two-mode Gaussian state
        that is described by the the Wigner covariance matrix V.

    """
    A = 0.5 * V[0:2, 0:2] * g ** 2
    B = 0.5 * V[2:4, 2:4] * g ** 2
    C = 0.5 * V[0:2, 2:4] * g ** 2
    sigma = np.linalg.det(A) + np.linalg.det(B) - 2 * np.linalg.det(C)
    nu_ = sigma / 2 - np.sqrt(sigma ** 2 - 4 * np.linalg.det(V)) / 2
    if nu_ < 0.0:
        return 0.0
    nu = np.sqrt(nu_)
    lognu = -np.log(2 * nu)
    logneg = max(0, lognu)
    return logneg
