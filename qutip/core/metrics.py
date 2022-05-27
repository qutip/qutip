# -*- coding: utf-8 -*-

"""
This module contains a collection of functions for calculating metrics
(distance measures) between states and operators.
"""

__all__ = ['fidelity', 'tracedist', 'bures_dist', 'bures_angle',
           'hellinger_dist', 'hilbert_dist', 'average_gate_fidelity',
           'process_fidelity', 'unitarity', 'dnorm']

import warnings

import numpy as np
from scipy import linalg as la
import scipy.sparse as sp
from .superop_reps import (to_kraus, to_choi, _to_superpauli, to_super,
                           kraus_to_choi)
from .superoperator import operator_to_vector, vector_to_operator
from .operators import qeye
from .states import ket2dm
from .semidefinite import dnorm_problem, dnorm_sparse_problem
from . import data as _data

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
        # Take advantage of the fact that the density operator for A
        # is a projector to avoid a sqrtm call.
        sqrtmA = A.proj()
        # Check whether we have to turn B into a density operator, too.
        if B.isket or B.isbra:
            B = B.proj()
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


def _hilbert_space_dims(oper):
    """
    For a quantum channel `oper`, return the dimensions `[dims_out, dims_in]`
    of the output Hilbert space and the input Hilbert space.
    - If oper is a unitary, then `oper.dims == [dims_out, dims_in]`.
    - If oper is a list of Kraus operators, then
     `oper[0].dims == [dims_out, dims_in]`.
    - If oper is a superoperator with `oper.superrep == 'super'`:
     `oper.dims == [[dims_out, dims_out], [dims_in, dims_in]]`
    - If oper is a superoperator with `oper.superrep == 'choi'`:
     `oper.dims == [[dims_in, dims_out], [dims_in, dims_out]]`
    - If oper is a superoperator with `oper.superrep == 'chi', then
      `dims_out == dims_in` and
      `oper.dims == [[dims_out, dims_out], [dims_out, dims_out]]`.
    :param oper: A quantum channel, represented by a unitary, a list of Kraus
    operators, or a superoperator
    :return: `[dims_out, dims_in]`, where `dims_out` and `dims_in` are lists
     of integers
    """
    if isinstance(oper, list):
        return oper[0].dims
    elif oper.type == 'oper':  # interpret as unitary quantum channel
        return oper.dims
    elif oper.type == 'super' and oper.superrep in ['choi', 'chi', 'super']:
        return [oper.dims[0][1], oper.dims[1][0]]
    else:
        raise TypeError('oper is not a valid quantum channel!')


def _process_fidelity_to_id(oper):
    """
    Internal function returning the process fidelity of a quantum channel
    to the identity quantum channel.
    Parameters
    ----------
    oper : :class:`qutip.Qobj`/list
        A unitary operator, or a superoperator in supermatrix, Choi or
        chi-matrix form, or a list of Kraus operators
    Returns
    -------
    fid : float
    """
    dims_out, dims_in = _hilbert_space_dims(oper)
    if dims_out != dims_in:
        raise TypeError('The process fidelity to identity is only defined '
                        'for dimension preserving channels.')
    d = np.prod(dims_in)
    if isinstance(oper, list):  # oper is a list of Kraus operators
        return np.sum([np.abs(k.tr()) ** 2 for k in oper]) / d ** 2
    elif oper.type == 'oper':  # interpret as unitary
        return np.abs(oper.tr()) ** 2 / d ** 2
    elif oper.type == 'super':
        if oper.superrep == 'chi':
            return oper[0, 0].real / d ** 2
        else:  # oper.superrep is either 'super' or 'choi':
            return to_super(oper).tr().real / d ** 2


def _kraus_or_qobj_to_choi(oper):
    if isinstance(oper, list):
        return kraus_to_choi(oper)
    else:
        return to_choi(oper)


def process_fidelity(oper, target=None):
    """
    Returns the process fidelity of a quantum channel to the target
    channel, or to the identity channel if no target is given.
    The process fidelity between two channels is defined as the state
    fidelity between their normalized Choi matrices.

    Parameters
    ----------
    oper : :class:`qutip.Qobj`/list
        A unitary operator, or a superoperator in supermatrix, Choi or
        chi-matrix form, or a list of Kraus operators
    target : :class:`qutip.Qobj`/list
        A unitary operator, or a superoperator in supermatrix, Choi or
        chi-matrix form, or a list of Kraus operators

    Returns
    -------
    fid : float
        Process fidelity between oper and target, or between oper and identity.

    Notes
    -----
    Since Qutip 5.0, this function computes the process fidelity as defined
    for example in: A. Gilchrist, N.K. Langford, M.A. Nielsen,
    Phys. Rev. A 71, 062310 (2005). Previously, it computed a function
    that is now implemented in
    :func:`control.fidcomp.FidCompUnitary.get_fidelity`.
    The definition of state fidelity that the process fidelity is based on
    is the one from R. Jozsa, Journal of Modern Optics, 41:12, 2315 (1994).
    It is the square of the one implemented in
    :func:`qutip.metrics.fidelity` which follows Nielsen & Chuang,
    "Quantum Computation and Quantum Information"

    """
    if target is None:
        return _process_fidelity_to_id(oper)

    dims_out, dims_in = _hilbert_space_dims(oper)
    dims_out_target, dims_in_target = _hilbert_space_dims(target)
    if [dims_out, dims_in] != [dims_out_target, dims_in_target]:
        raise TypeError('Dimensions of oper and target do not match')

    if not isinstance(target, list) and target.type == 'oper':
        # interpret target as unitary.
        if isinstance(oper, list):  # oper is a list of Kraus operators
            return _process_fidelity_to_id([k * target.dag() for k in oper])
        elif oper.type == 'oper':
            return _process_fidelity_to_id(oper*target.dag())
        elif oper.type == 'super':
            oper_super = to_super(oper)
            target_dag_super = to_super(target.dag())
            return _process_fidelity_to_id(oper_super * target_dag_super)
    else:  # target is a list of Kraus operators or a superoperator
        if not isinstance(oper, list) and oper.type == 'oper':
            return process_fidelity(target, oper)  # reverse order
        oper_choi = _kraus_or_qobj_to_choi(oper)
        target_choi = _kraus_or_qobj_to_choi(target)
        d = np.prod(dims_in)
        return (fidelity(oper_choi, target_choi)/d)**2


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
    kraus = to_kraus(oper)
    d = kraus[0].shape[0]

    if kraus[0].shape[1] != d:
        return TypeError("Average gate fidelity only implemented for square "
                         "superoperators.")

    if target is None:
        return (d + np.sum([np.abs(A_k.tr())**2 for A_k in kraus])) / (d*d + d)
    return (
        (d + np.sum([np.abs((A_k*target.dag()).tr())**2 for A_k in kraus]))
        / (d*d + d)
    )


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
        A = A.proj()
    if B.isket or B.isbra:
        B = B.proj()
    if A.dims != B.dims:
        raise TypeError("A and B do not have same dimensions.")
    diff = A - B
    diff = diff.dag() * diff
    vals = diff.eigenenergies(sparse=sparse, tol=tol)
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
        A = A.proj()
    if B.isket or B.isbra:
        B = B.proj()
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
        A = A.proj()
    if B.isket or B.isbra:
        B = B.proj()
    if A.dims != B.dims:
        raise TypeError('A and B do not have same dimensions.')
    dist = np.sqrt(2 * (1 - fidelity(A, B)))
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
        A = A.proj()
    if B.isket or B.isbra:
        B = B.proj()
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
    eigs = product.eigenenergies(sparse=sparse, tol=tol)
    # np.maximum() is to avoid nan appearing sometimes due to numerical
    # instabilities causing np.sum(eigs) slightly (~1e-8) larger than 1 when
    # hellinger_dist(A, B) is called for A=B
    return np.sqrt(2.0 * np.maximum(0, 1 - np.real(np.sum(eigs))))


def dnorm(A, B=None, solver="CVXOPT", verbose=False, force_solve=False,
          sparse=True):
    """
    Calculates the diamond norm of the quantum map q_oper, using
    the simplified semidefinite program of [Wat12]_.

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

    # There's a lot of conditions to check for this path.  Only check if they
    # aren't superoperators.  The difference of unitaries optimization is
    # currently only implemented for d == 2. Much of the code below is more
    # general, though, in anticipation of generalizing the optimization.
    if (
        not force_solve
        and B is not None
        and A.isoper and B.isoper
        and A.shape[0] == 2
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

    # The constraints only depend on the dimension, so
    # we can cache them efficiently.
    problem, Jr, Ji, *_ = dnorm_problem(dim)

    # Load the parameters with the Choi matrix passed in.
    J_dat = _data.to('csr', J.data).as_scipy()

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
    Eu = _to_superpauli(oper).full()[1:, 1:]
    return np.linalg.norm(Eu, 'fro')**2 / len(Eu)
