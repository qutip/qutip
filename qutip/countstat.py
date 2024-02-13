"""
This module contains functions for calculating current and current noise using
the counting statistics formalism.
"""
__all__ = ['countstat_current', 'countstat_current_noise']

import numpy as np
import scipy.sparse as sp

from qutip.expect import expect_rho_vec
from qutip.steadystate import pseudo_inverse, steadystate
from qutip.superoperator import mat2vec, sprepost, spre
from qutip import operator_to_vector, identity, tensor
import qutip.settings as settings
from qutip.qobj import Qobj, issuper, isoper
# Load MKL spsolve if avaiable
if settings.has_mkl:
    from qutip._mkl.spsolve import (mkl_splu, mkl_spsolve)


def countstat_current(L, c_ops=None, rhoss=None, J_ops=None):
    """
    Calculate the current corresponding a system Liouvillian `L` and a list of
    current collapse operators `c_ops` or current superoperators `J_ops`
    (either must be specified). Optionally the steadystate density matrix
    `rhoss` and a list of current superoperators `J_ops` can be specified. If
    either of these are omitted they are computed internally.

    Parameters
    ----------

    L : :class:`qutip.Qobj`
        Qobj representing the system Liouvillian.

    c_ops : array / list (optional)
        List of current collapse operators.

    rhoss : :class:`qutip.Qobj` (optional)
        The steadystate density matrix corresponding the system Liouvillian
        `L`.

    J_ops : array / list (optional)
        List of current superoperators.

    Returns
    --------
    I : array
        The currents `I` corresponding to each current collapse operator
        `c_ops` (or, equivalently, each current superopeator `J_ops`).
    """

    if J_ops is None:
        if c_ops is None:
            raise ValueError("c_ops must be given if J_ops is not")
        J_ops = [sprepost(c, c.dag()) for c in c_ops]

    if rhoss is None:
        if c_ops is None:
            raise ValueError("c_ops must be given if rhoss is not")
        rhoss = steadystate(L, c_ops)

    rhoss_vec = mat2vec(rhoss.full()).ravel()

    N = len(J_ops)
    I = np.zeros(N)

    for i, Ji in enumerate(J_ops):
        I[i] = expect_rho_vec(Ji.data, rhoss_vec, 1)

    return I


def countstat_current_noise(L, c_ops, wlist=None, rhoss=None, J_ops=None,
                            sparse=True, method='direct'):
    """
    Compute the cross-current noise spectrum for a list of collapse operators
    `c_ops` corresponding to monitored currents, given the system
    Liouvillian `L`. The current collapse operators `c_ops` should be part
    of the dissipative processes in `L`, but the `c_ops` given here does not
    necessarily need to be all collapse operators contributing to dissipation
    in the Liouvillian. Optionally, the steadystate density matrix `rhoss`
    and the current operators `J_ops` correpsonding to the current collapse
    operators `c_ops` can also be specified. If either of
    `rhoss` and `J_ops` are omitted, they will be computed internally.
    'wlist' is an optional list of frequencies at which to evaluate the noise
    spectrum.

    Note:
    The default method is a direct solution using dense matrices, as sparse
    matrix methods fail for some examples of small systems.
    For larger systems it is reccomended to use the sparse solver
    with the direct method, as it avoids explicit calculation of the
    pseudo-inverse, as described in page 67 of "Electrons in nanostructures"
    C. Flindt, PhD Thesis, available online:
    https://orbit.dtu.dk/fedora/objects/orbit:82314/datastreams/file_4732600/content

    Parameters
    ----------

    L : :class:`qutip.Qobj`
        Qobj representing the system Liouvillian.

    c_ops : array / list
        List of current collapse operators.

    rhoss : :class:`qutip.Qobj` (optional)
        The steadystate density matrix corresponding the system Liouvillian
        `L`.

    wlist : array / list (optional)
        List of frequencies at which to evaluate (if none are given, evaluates
        at zero frequency)

    J_ops : array / list (optional)
        List of current superoperators.

    sparse : bool
        Flag that indicates whether to use sparse or dense matrix methods when
        computing the pseudo inverse. Default is false, as sparse solvers
        can fail for small systems. For larger systems the sparse solvers
        are reccomended.


    Returns
    --------
    I, S : tuple of arrays
        The currents `I` corresponding to each current collapse operator
        `c_ops` (or, equivalently, each current superopeator `J_ops`) and the
        zero-frequency cross-current correlation `S`.
    """

    if rhoss is None:
        rhoss = steadystate(L, c_ops)

    if J_ops is None:
        J_ops = [sprepost(c, c.dag()) for c in c_ops]



    N = len(J_ops)
    I = np.zeros(N)

    if wlist is None:
        S = np.zeros((N, N,1))
        wlist=[0.]
    else:
        S = np.zeros((N, N,len(wlist)))

    if sparse == False:
        rhoss_vec = mat2vec(rhoss.full()).ravel()
        for k,w in enumerate(wlist):
            R = pseudo_inverse(L, rhoss=rhoss, w= w, sparse = sparse, method=method)
            for i, Ji in enumerate(J_ops):
                for j, Jj in enumerate(J_ops):
                    if i == j:
                        I[i] = expect_rho_vec(Ji.data, rhoss_vec, 1)
                        S[i, j,k] = I[i]
                    S[i, j,k] -= expect_rho_vec((Ji * R * Jj
                                                + Jj * R * Ji).data,
                                                rhoss_vec, 1)
    else:
        if method == "direct":
            N = np.prod(L.dims[0][0])

            rhoss_vec = operator_to_vector(rhoss)

            tr_op = tensor([identity(n) for n in L.dims[0][0]])
            tr_op_vec = operator_to_vector(tr_op)

            Pop = sp.kron(rhoss_vec.data, tr_op_vec.data.T, format='csr')
            Iop = sp.eye(N*N, N*N, format='csr')
            Q = Iop - Pop

            for k,w in enumerate(wlist):

                if w != 0.0:
                    L_temp = 1.0j*w*spre(tr_op) + L
                else:
                    # At zero frequency some solvers fail for small systems.
                    # Adding a small finite frequency of order 1e-12
                    # helps prevent the solvers from returning wrong results.
                    # Anything under 1e-12 is removed by the auto_tidyup.
                    L_temp =  1.0j*(1e-12)*spre(tr_op) + L

                if not settings.has_mkl:
                    A = L_temp.data.tocsc()
                else:
                    A = L_temp.data.tocsr()
                    A.sort_indices()

                rhoss_vec = mat2vec(rhoss.full()).ravel()

                for j, Jj in enumerate(J_ops):
                    Qj = Q.dot( Jj.data.dot( rhoss_vec))
                    try:
                        if settings.has_mkl:
                            X_rho_vec_j = mkl_spsolve(A,Qj)
                        else:
                            X_rho_vec_j = sp.linalg.splu(A, permc_spec
                                                 ='COLAMD').solve(Qj)
                    except:
                        X_rho_vec_j = sp.linalg.lsqr(A,Qj)[0]
                    for i, Ji in enumerate(J_ops):
                        Qi = Q.dot( Ji.data.dot(rhoss_vec))
                        try:
                            if settings.has_mkl:
                                X_rho_vec_i = mkl_spsolve(A,Qi)
                            else:
                                X_rho_vec_i = sp.linalg.splu(A, permc_spec
                                                     ='COLAMD').solve(Qi)
                        except:
                             X_rho_vec_i = sp.linalg.lsqr(A,Qi)[0]
                        if i == j:
                            I[i] = expect_rho_vec(Ji.data,
                                                 rhoss_vec, 1)
                            S[j, i, k] = I[i]

                        S[j, i, k] -= (expect_rho_vec(Jj.data * Q,
                                        X_rho_vec_i, 1)
                                        + expect_rho_vec(Ji.data * Q,
                                        X_rho_vec_j, 1))

        else:
            rhoss_vec = mat2vec(rhoss.full()).ravel()
            for k,w in enumerate(wlist):

                R = pseudo_inverse(L,rhoss=rhoss, w= w, sparse = sparse,
                                   method=method)

                for i, Ji in enumerate(J_ops):
                    for j, Jj in enumerate(J_ops):
                        if i == j:
                            I[i] = expect_rho_vec(Ji.data, rhoss_vec, 1)
                            S[i, j, k] = I[i]
                        S[i, j, k] -= expect_rho_vec((Ji * R * Jj
                                                     + Jj * R * Ji).data,
                                                     rhoss_vec, 1)
    return I, S
