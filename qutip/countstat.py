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

__all__ = ['countstat_current', 'countstat_current_noise']

import numpy as np
from qutip.expect import expect_rho_vec, pseudo_inverse, mat2vec, sprepost


def countstat_current(L, rhoss, c_ops, J_ops=None):
    """
    Calculate the current 
    """

    if J_ops is None:
        J_ops = [sprepost(c, c.dag()) for c in c_ops]

    rhoss_vec = mat2vec(rhoss.full()).ravel()

    N = len(J_ops)
    I = np.zeros(N)

    for i, Ji in enumerate(J_ops):
        I[i] = expect_rho_vec(Ji.data, rhoss_vec, 1)

    return I


def countstat_current_noise(L, rhoss, c_ops, R=None):
    """
    Compute the cross-current noise spectrum.
    """

    if R is None:
        R = pseudo_inverse(L, rhoss)

    J_ops = [sprepost(c, c.dag()) for c in c_ops]

    rhoss_vec = mat2vec(rhoss.full()).ravel()

    N = len(J_ops)
    I = np.zeros(N)
    S = np.zeros(N, N)

    for i, Ji in enumerate(J_ops):
        for j, Jj in enumerate(J_ops):
            if i == j:
                I[i] = expect_rho_vec(Ji.data, rhoss_vec, 1)
                S[i, j] = I[i]

            S[i, j] -= expect_rho_vec((Ji * R * Jj + Jj * R * Ji).data,
                                      rhoss_vec, 1)

    return I, S