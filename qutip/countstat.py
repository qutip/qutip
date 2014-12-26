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

from qutip.expect import expect_rho_vec
from qutip.steadystate import pseudo_inverse, steadystate
from qutip.superoperator import mat2vec, sprepost
from qutip import operator_to_vector, identity, tensor


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


def countstat_current_noise(L, c_ops, rhoss=None, J_ops=None, R=False):
    """
    Compute the cross-current noise spectrum for a list of collapse operators
    `c_ops` corresponding to monitored currents, given the system
    Liouvillian `L`. The current collapse operators `c_ops` should be part
    of the dissipative processes in `L`, but the `c_ops` given here does not
    necessarily need to be all collapse operators contributing to dissipation
    in the Liouvillian. Optionally, the steadystate density matrix `rhoss`
    and/or the pseudo inverse `R` of the Liouvillian `L`, and the current
    operators `J_ops` correpsonding to the current collapse operators `c_ops`
    can also be specified. If `R` is not given, the cross-current correlations
    will be computed directly without computing `R` explicitly. If either of
    `rhoss` and `J_ops` are omitted, they will be computed internally.

    Parameters
    ----------

    L : :class:`qutip.Qobj`
        Qobj representing the system Liouvillian.

    c_ops : array / list
        List of current collapse operators.

    rhoss : :class:`qutip.Qobj` (optional)
        The steadystate density matrix corresponding the system Liouvillian
        `L`.

    J_ops : array / list (optional)
        List of current superoperators.

    R : :class:`qutip.Qobj` (optional)
        Qobj representing the pseudo inverse of the system Liouvillian `L`.

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

    rhoss_vec = mat2vec(rhoss.full()).ravel()

    N = len(J_ops)
    I = np.zeros(N)
    S = np.zeros((N, N))

    if R:
        if R is True:
            R = pseudo_inverse(L, rhoss)

        for i, Ji in enumerate(J_ops):
            for j, Jj in enumerate(J_ops):
                if i == j:
                    I[i] = expect_rho_vec(Ji.data, rhoss_vec, 1)
                    S[i, j] = I[i]

                S[i, j] -= expect_rho_vec((Ji * R * Jj + Jj * R * Ji).data,
                                          rhoss_vec, 1)
    else:
        N = np.prod(L.dims[0][0])

        rhoss_vec = operator_to_vector(rhoss)

        tr_op = tensor([identity(n) for n in L.dims[0][0]])
        tr_op_vec = operator_to_vector(tr_op)

        Pop = sp.kron(rhoss_vec.data, tr_op_vec.data.T, format='csc')
        Iop = sp.eye(N*N, N*N, format='csc')
        Q = Iop - Pop
        A = L.data.tocsc()
        rhoss_vec = mat2vec(rhoss.full()).ravel()

        for j, Jj in enumerate(J_ops):
            Qj = Q * Jj.data * rhoss_vec
            X_rho_vec = sp.linalg.splu(A, permc_spec='COLAMD').solve(Qj)
            for i, Ji in enumerate(J_ops):
                if i == j:
                    S[i, i] = I[i] = expect_rho_vec(Ji.data, rhoss_vec, 1)

                S[i, j] -= expect_rho_vec(Ji.data * Q, X_rho_vec, 1)

    return I, S
