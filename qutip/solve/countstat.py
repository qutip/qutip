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
"""
This module contains functions for calculating current and current noise using
the counting statistics formalism.
"""
__all__ = ['countstat_current', 'countstat_current_noise']

import numpy as np
import scipy.sparse as sp

from ..core import (
    sprepost, spre, operator_to_vector, identity, tensor,
)
from ..core import data as _data
from ..core.data import matmul_csr_dense_dense as matmul
from ..core.data import expect_super_csr_dense as expect
from .steadystate import pseudo_inverse, steadystate
from .. import settings

# Load MKL spsolve if avaiable
if settings.has_mkl:
    from qutip._mkl.spsolve import mkl_spsolve


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

    rhoss_vec = _data.dense.fast_from_numpy(rhoss.full())
    rhoss_vec = _data.column_stack_dense(rhoss_vec)

    N = len(J_ops)
    I = np.zeros(N)

    for i, Ji in enumerate(J_ops):
        I[i] = expect(Ji.data, rhoss_vec).real

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
    http://orbit.dtu.dk/fedora/objects/orbit:82314/datastreams/file_4732600/content

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
        wlist = [0.]
    S = np.zeros((N, N, len(wlist)))

    if sparse == False:
        rhoss_vec = _data.dense.fast_from_numpy(rhoss.full())
        rhoss_vec = _data.column_stack_dense(rhoss_vec)
        for k, w in enumerate(wlist):
            R = pseudo_inverse(L, rhoss=rhoss, w=w,
                               sparse=sparse, method=method).data
            for i, Ji in enumerate(J_ops):
                Ji = Ji.data
                for j, Jj in enumerate(J_ops):
                    Jj = Jj.data
                    if i == j:
                        I[i] = expect(Ji, rhoss_vec).real
                        S[i, j, k] = I[i]
                    S[i, j, k] -= expect(Ji@R@Jj + Jj@R@Ji, rhoss_vec).real
    else:
        if method == "direct":
            N = np.prod(L.dims[0][0])

            rhoss_vec = operator_to_vector(rhoss)

            tr_op = tensor([identity(n) for n in L.dims[0][0]])
            tr_op_vec = operator_to_vector(tr_op)

            Pop = _data.kron_csr(rhoss_vec.data, tr_op_vec.data.transpose())
            Iop = _data.csr.identity(N*N)
            Q = Iop - Pop

            for k, w in enumerate(wlist):

                if w != 0.0:
                    L_temp = 1.0j*w*spre(tr_op) + L
                else:
                    # At zero frequency some solvers fail for small systems.
                    # Adding a small finite frequency of order 1e-15
                    # helps prevent the solvers from throwing an exception.
                    L_temp = 1e-15j*spre(tr_op) + L

                if not settings.has_mkl:
                    A = L_temp.data.as_scipy().tocsc()
                else:
                    A = L_temp.data.as_scipy()
                    A.sort_indices()

                rhoss_vec = _data.dense.fast_from_numpy(rhoss.full())
                rhoss_vec = _data.column_stack_dense(rhoss_vec)

                for j, Jj in enumerate(J_ops):
                    Jj = Jj.data
                    Qj = matmul(Q, matmul(Jj, rhoss_vec)).as_ndarray()
                    try:
                        if settings.has_mkl:
                            X_rho_vec_j = mkl_spsolve(A, Qj)
                        else:
                            X_rho_vec_j =\
                                sp.linalg.splu(A,
                                               permc_spec='COLAMD').solve(Qj)
                    except:
                        X_rho_vec_j = sp.linalg.lsqr(A, Qj)[0]
                    X_rho_vec_j = _data.dense.fast_from_numpy(X_rho_vec_j)
                    for i, Ji in enumerate(J_ops):
                        Ji = Ji.data
                        Qi = matmul(Q, matmul(Ji, rhoss_vec)).as_ndarray()
                        try:
                            if settings.has_mkl:
                                X_rho_vec_i = mkl_spsolve(A, Qi)
                            else:
                                X_rho_vec_i = (
                                    sp.linalg.splu(A, permc_spec='COLAMD')
                                    .solve(Qi)
                                )
                        except:
                            X_rho_vec_i = sp.linalg.lsqr(A, Qi)[0]
                        if i == j:
                            I[i] = expect(Ji, rhoss_vec).real
                            S[j, i, k] = I[i]

                        X_rho_vec_i = _data.dense.fast_from_numpy(X_rho_vec_i)
                        S[j, i, k] -= (expect(Jj @ Q, X_rho_vec_i)
                                       + expect(Ji @ Q, X_rho_vec_j)).real

        else:
            rhoss_vec = _data.dense.fast_from_numpy(rhoss.full())
            rhoss_vec = _data.column_stack_dense(rhoss_vec)
            for k, w in enumerate(wlist):

                R = pseudo_inverse(L, rhoss=rhoss, w=w, sparse=sparse,
                                   method=method)

                for i, Ji in enumerate(J_ops):
                    Ji = Ji.data
                    for j, Jj in enumerate(J_ops):
                        Jj = Jj.data
                        if i == j:
                            I[i] = expect(Ji, rhoss_vec).real
                            S[i, j, k] = I[i]
                        S[i, j, k] -= expect(Ji@R@Jj + Jj@R@Ji, rhoss_vec).real
    return I, S
