# -*- coding: utf-8 -*-

"""
This module contains a collection of functions for calculating metrics
(distance measures) between states and operators.
"""

__all__ = ['fidelity', 'tracedist', 'bures_dist', 'bures_angle',
           'hellinger_dist', 'hilbert_dist', 'average_gate_fidelity',
           'process_fidelity', 'unitarity', 'dnorm']

import numpy as np
from scipy import linalg as la
import scipy.sparse as sp
from qutip.sparse import sp_eigs
from qutip.states import ket2dm
from qutip.superop_reps import to_kraus, to_stinespring, to_choi, _super_to_superpauli, to_super
from qutip.superoperator import operator_to_vector, vector_to_operator
from qutip.operators import qeye
from qutip.semidefinite import dnorm_problem, dnorm_sparse_problem
import qutip.settings as settings

import qutip.logging_utils as logging
logger = logging.get_logger('qutip.metrics')

try:
    import cvxpy
except ImportError:
    cvxpy = None


def fidelity(A, B):
    """
    Calculates the fidelity (pseudo-metric) between two density matrices.
    See: Nielsen & Chuang, "Quantum Computation and Quantum Information"

    Parameters
    ----------
    A : qobj
        Density matrix or state vector.
    B : qobj
        Density matrix or state vector with same dimensions as A.

    Returns
    -------
    fid : float
        Fidelity pseudo-metric between A and B.

    Examples
    --------
    >>> x = fock_dm(5,3)
    >>> y = coherent_dm(5,1)
    >>> np.testing.assert_almost_equal(fidelity(x,y), 0.24104350624628332)
    """
    if A.isket or A.isbra:
        if B.isket or B.isbra:
            # The fidelity for pure states reduces to the modulus of their
            # inner product.
            return np.abs(A.overlap(B))
        # Take advantage of the fact that the density operator for A
        # is a projector to avoid a sqrtm call.
        sqrtmA = ket2dm(A)
    else:
        if B.isket or B.isbra:
            # Swap the order so that we can take a more numerically
            # stable square root of B.
            return fidelity(B, A)
        # If we made it here, both A and B are operators, so
        # we have to take the sqrtm of one of them.
        sqrtmA = A.sqrtm()

    if sqrtmA.dims != B.dims:
        raise TypeError('Density matrices do not have same dimensions.')

    # We don't actually need the whole matrix here, just the trace
    # of its square root, so let's just get its eigenenergies instead.
    # We also truncate negative eigenvalues to avoid nan propagation;
    # even for positive semidefinite matrices, small negative eigenvalues
    # can be reported.
    eig_vals = (sqrtmA * B * sqrtmA).eigenenergies()
    return float(np.real(np.sqrt(eig_vals[eig_vals > 0]).sum()))


def process_fidelity(U1, U2, normalize=True):
    """
    Calculate the process fidelity given two process operators.
    """
    if normalize:
        return (U1 * U2).tr() / (U1.tr() * U2.tr())
    else:
        return (U1 * U2).tr()


def average_gate_fidelity(oper, target=None):
    """
    Given a Qobj representing the supermatrix form of a map, returns the
    average gate fidelity (pseudo-metric) of that map.

    Parameters
    ----------
    A : Qobj
        Quantum object representing a superoperator.
    target : Qobj
        Quantum object representing the target unitary; the inverse
        is applied before evaluating the fidelity.

    Returns
    -------
    fid : float
        Fidelity pseudo-metric between A and the identity superoperator,
        or between A and the target superunitary.
    """
    kraus_form = to_kraus(oper)
    d = kraus_form[0].shape[0]

    if kraus_form[0].shape[1] != d:
        return TypeError("Average gate fidelity only implemented for square "
                         "superoperators.")

    if target is None:
        return (d + np.sum([np.abs(A_k.tr())**2
                        for A_k in kraus_form])) / (d**2 + d)
    else:
        return (d + np.sum([np.abs((A_k * target.dag()).tr())**2
                        for A_k in kraus_form])) / (d**2 + d)


def tracedist(A, B, sparse=False, tol=0):
    """
    Calculates the trace distance between two density matrices..
    See: Nielsen & Chuang, "Quantum Computation and Quantum Information"

    Parameters
    ----------!=
    A : qobj
        Density matrix or state vector.
    B : qobj
        Density matrix or state vector with same dimensions as A.
    tol : float
        Tolerance used by sparse eigensolver, if used. (0=Machine precision)
    sparse : {False, True}
        Use sparse eigensolver.

    Returns
    -------
    tracedist : float
        Trace distance between A and B.

    Examples
    --------
    >>> x=fock_dm(5,3)
    >>> y=coherent_dm(5,1)
    >>> np.testing.assert_almost_equal(tracedist(x,y), 0.9705143161472971)
    """
    if A.isket or A.isbra:
        A = ket2dm(A)
    if B.isket or B.isbra:
        B = ket2dm(B)

    if A.dims != B.dims:
        raise TypeError("A and B do not have same dimensions.")

    diff = A - B
    diff = diff.dag() * diff
    vals = sp_eigs(diff.data, diff.isherm, vecs=False, sparse=sparse, tol=tol)
    return float(np.real(0.5 * np.sum(np.sqrt(np.abs(vals)))))


def hilbert_dist(A, B):
    """
    Returns the Hilbert-Schmidt distance between two density matrices A & B.

    Parameters
    ----------
    A : qobj
        Density matrix or state vector.
    B : qobj
        Density matrix or state vector with same dimensions as A.

    Returns
    -------
    dist : float
        Hilbert-Schmidt distance between density matrices.

    Notes
    -----
    See V. Vedral and M. B. Plenio, Phys. Rev. A 57, 1619 (1998).

    """
    if A.isket or A.isbra:
        A = ket2dm(A)
    if B.isket or B.isbra:
        B = ket2dm(B)

    if A.dims != B.dims:
        raise TypeError('A and B do not have same dimensions.')

    return ((A - B)**2).tr()


def bures_dist(A, B):
    """
    Returns the Bures distance between two density matrices A & B.

    The Bures distance ranges from 0, for states with unit fidelity,
    to sqrt(2).

    Parameters
    ----------
    A : qobj
        Density matrix or state vector.
    B : qobj
        Density matrix or state vector with same dimensions as A.

    Returns
    -------
    dist : float
        Bures distance between density matrices.
    """
    if A.isket or A.isbra:
        A = ket2dm(A)
    if B.isket or B.isbra:
        B = ket2dm(B)

    if A.dims != B.dims:
        raise TypeError('A and B do not have same dimensions.')

    dist = np.sqrt(2.0 * (1.0 - fidelity(A, B)))
    return dist


def bures_angle(A, B):
    """
    Returns the Bures Angle between two density matrices A & B.

    The Bures angle ranges from 0, for states with unit fidelity, to pi/2.

    Parameters
    ----------
    A : qobj
        Density matrix or state vector.
    B : qobj
        Density matrix or state vector with same dimensions as A.

    Returns
    -------
    angle : float
        Bures angle between density matrices.
    """
    if A.isket or A.isbra:
        A = ket2dm(A)
    if B.isket or B.isbra:
        B = ket2dm(B)

    if A.dims != B.dims:
        raise TypeError('A and B do not have same dimensions.')

    return np.arccos(fidelity(A, B))


def hellinger_dist(A, B, sparse=False, tol=0):
    """
    Calculates the quantum Hellinger distance between two density matrices.

    Formula:
    hellinger_dist(A, B) = sqrt(2-2*Tr(sqrt(A)*sqrt(B)))

    See: D. Spehner, F. Illuminati, M. Orszag, and W. Roga, "Geometric
    measures of quantum correlations with Bures and Hellinger distances"
    arXiv:1611.03449

    Parameters
    ----------
    A : :class:`qutip.Qobj`
        Density matrix or state vector.
    B : :class:`qutip.Qobj`
        Density matrix or state vector with same dimensions as A.
    tol : float
        Tolerance used by sparse eigensolver, if used. (0=Machine precision)
    sparse : {False, True}
        Use sparse eigensolver.

    Returns
    -------
    hellinger_dist : float
        Quantum Hellinger distance between A and B. Ranges from 0 to sqrt(2).

    Examples
    --------
    >>> x=fock_dm(5,3)
    >>> y=coherent_dm(5,1)
    >>> np.testing.assert_almost_equal(hellinger_dist(x,y), 1.3725145002591095)
    """
    if A.isket or A.isbra:
        sqrtmA = ket2dm(A)
    else:
        sqrtmA = A.sqrtm(sparse=sparse, tol=tol)
    if B.isket or B.isbra:
        sqrtmB = ket2dm(B)
    else:
        sqrtmB = B.sqrtm(sparse=sparse, tol=tol)

    if sqrtmA.dims != sqrtmB.dims:
        raise TypeError("A and B do not have compatible dimensions.")

    product = sqrtmA*sqrtmB
    eigs = sp_eigs(product.data,
                   isherm=product.isherm, vecs=False, sparse=sparse, tol=tol)
    # np.maximum() is to avoid nan appearing sometimes due to numerical
    # instabilities causing np.sum(eigs) slightly (~1e-8) larger than 1
    # when hellinger_dist(A, B) is called for A=B
    return np.sqrt(2.0 * np.maximum(0., (1.0 - np.real(np.sum(eigs)))))


def dnorm(A, B=None, solver="CVXOPT", verbose=False, force_solve=False,
          sparse=True):
    """
    Calculates the diamond norm of the quantum map q_oper, using
    the simplified semidefinite program of [Wat13]_.

    The diamond norm SDP is solved by using CVXPY_.

    Parameters
    ----------
    A : Qobj
        Quantum map to take the diamond norm of.
    B : Qobj or None
        If provided, the diamond norm of :math:`A - B` is taken instead.
    solver : str
        Solver to use with CVXPY. One of "CVXOPT" (default) or "SCS". The
        latter tends to be significantly faster, but somewhat less accurate.
    verbose : bool
        If True, prints additional information about the solution.
    force_solve : bool
        If True, forces dnorm to solve the associated SDP, even if a special
        case is known for the argument.
    sparse : bool
        Whether to use sparse matrices in the convex optimisation problem.
        Default True.

    Returns
    -------
    dn : float
        Diamond norm of q_oper.

    Raises
    ------
    ImportError
        If CVXPY cannot be imported.

    .. _cvxpy: https://www.cvxpy.org/en/latest/
    """
    if cvxpy is None:  # pragma: no cover
        raise ImportError("dnorm() requires CVXPY to be installed.")

    # We follow the strategy of using Watrous' simpler semidefinite
    # program in its primal form. This is the same strategy used,
    # for instance, by both pyGSTi and SchattenNorms.jl. (By contrast,
    # QETLAB uses the dual problem.)

    # Check if A and B are both unitaries. If so, then we can without
    # loss of generality choose B to be the identity by using the
    # unitary invariance of the diamond norm,
    #     || A - B ||_♢ = || A B⁺ - I ||_♢.
    # Then, using the technique mentioned by each of Johnston and
    # da Silva,
    #     || A B⁺ - I ||_♢ = max_{i, j} | \lambda_i(A B⁺) - \lambda_j(A B⁺) |,
    # where \lambda_i(U) is the ith eigenvalue of U.

    if (
        # There's a lot of conditions to check for this path.
        not force_solve and B is not None and
        # Only check if they aren't superoperators.
        A.type == "oper" and B.type == "oper" and
        # The difference of unitaries optimization is currently
        # only implemented for d == 2. Much of the code below is more general,
        # though, in anticipation of generalizing the optimization.
        A.shape[0] == 2
    ):
        # Make an identity the same size as A and B to
        # compare against.
        I = qeye(A.dims[0])
        # Compare to B first, so that an error is raised
        # as soon as possible.
        Bd = B.dag()
        if (
            (B * Bd - I).norm() < 1e-6 and
            (A * A.dag() - I).norm() < 1e-6
        ):
            # Now we are on the fast path, so let's compute the
            # eigenvalues, then find the diameter of the smallest circle
            # containing all of them.
            #
            # For now, this is only implemented for dim = 2, such that
            # generalizing here will allow for generalizing the optimization.
            # A reasonable approach would probably be to use Welzl's algorithm
            # (https://en.wikipedia.org/wiki/Smallest-circle_problem).
            U = A * B.dag()
            eigs = U.eigenenergies()
            eig_distances = np.abs(eigs[:, None] - eigs[None, :])
            return np.max(eig_distances)

    # Force the input superoperator to be a Choi matrix.
    J = to_choi(A)

    if B is not None:
        J -= to_choi(B)

    # Watrous 2012 also points out that the diamond norm of Lambda
    # is the same as the completely-bounded operator-norm (∞-norm)
    # of the dual map of Lambda. We can evaluate that norm much more
    # easily if Lambda is completely positive, since then the largest
    # eigenvalue is the same as the largest singular value.

    if not force_solve and J.iscp:
        S_dual = to_super(J.dual_chan())
        vec_eye = operator_to_vector(qeye(S_dual.dims[1][1]))
        op = vector_to_operator(S_dual * vec_eye)
        # The 2-norm was not implemented for sparse matrices as of the time
        # of this writing. Thus, we must yet again go dense.
        return la.norm(op.full(), 2)

    # If we're still here, we need to actually solve the problem.

    # Assume square...
    dim = np.prod(J.dims[0][0])

    J_dat = J.data

    if not sparse:
        # The parameters and constraints only depend on the dimension, so
        # we can cache them efficiently.
        problem, Jr, Ji = dnorm_problem(dim)

        # Load the parameters with the Choi matrix passed in.
        Jr.value = sp.csr_matrix((J_dat.data.real, J_dat.indices,
                                  J_dat.indptr),
                                 shape=J_dat.shape).toarray()

        Ji.value = sp.csr_matrix((J_dat.data.imag, J_dat.indices,
                                  J_dat.indptr),
                                 shape=J_dat.shape).toarray()
    else:

        # The parameters do not depend solely on the dimension,
        # so we can not cache them efficiently.
        problem = dnorm_sparse_problem(dim, J_dat)

    problem.solve(solver=solver, verbose=verbose)

    return problem.value


def unitarity(oper):
    """
    Returns the unitarity of a quantum map, defined as the Frobenius norm
    of the unital block of that map's superoperator representation.

    Parameters
    ----------
    oper : Qobj
        Quantum map under consideration.

    Returns
    -------
    u : float
        Unitarity of ``oper``.
    """
    Eu = _super_to_superpauli(oper).full()[1:, 1:]
    #return np.real(np.trace(np.dot(Eu, Eu.conj().T))) / len(Eu)
    return np.linalg.norm(Eu, 'fro')**2 / len(Eu)
