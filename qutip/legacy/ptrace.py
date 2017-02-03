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

__all__ = []

import numpy as np
import scipy.sparse as sp
from qutip.sparse import sp_reshape


def _ptrace(rho, sel):
    """
    Private function calculating the partial trace.
    """

    if isinstance(sel, int):
        sel = np.array([sel])
    else:
        sel = np.asarray(sel)

    if (sel < 0).any() or (sel >= len(rho.dims[0])).any():
        raise TypeError("Invalid selection index in ptrace.")

    drho = rho.dims[0]
    N = np.prod(drho)
    M = np.prod(np.asarray(drho).take(sel))

    if np.prod(rho.dims[1]) == 1:
        rho = rho * rho.dag()

    perm = sp.lil_matrix((M * M, N * N))
    # all elements in range(len(drho)) not in sel set
    rest = np.setdiff1d(np.arange(len(drho)), sel)
    ilistsel = _select(sel, drho)
    indsel = _list2ind(ilistsel, drho)
    ilistrest = _select(rest, drho)
    indrest = _list2ind(ilistrest, drho)
    irest = (indrest - 1) * N + indrest - 2
    # Possibly use parfor here if M > some value ?
    perm.rows = np.array(
        [(irest + (indsel[int(np.floor(m / M))] - 1) * N +
         indsel[int(np.mod(m, M))]).T[0]
         for m in range(M ** 2)])
    # perm.data = np.ones_like(perm.rows,dtype=int)
    perm.data = np.ones_like(perm.rows)
    perm = perm.tocsr()
    rhdata = perm * sp_reshape(rho.data, (np.prod(rho.shape), 1))
    rho1_data = sp_reshape(rhdata, (M, M))
    dims_kept0 = np.asarray(rho.dims[0]).take(sel)
    dims_kept1 = np.asarray(rho.dims[0]).take(sel)
    rho1_dims = [dims_kept0.tolist(), dims_kept1.tolist()]
    rho1_shape = [np.prod(dims_kept0), np.prod(dims_kept1)]
    return rho1_data, rho1_dims, rho1_shape


def _list2ind(ilist, dims):
    """!
    Private function returning indicies
    """
    ilist = np.asarray(ilist)
    dims = np.asarray(dims)
    irev = np.fliplr(ilist) - 1
    fact = np.append(np.array([1]), (np.cumprod(np.flipud(dims)[:-1])))
    fact = fact.reshape(len(fact), 1)
    return np.array(np.sort(np.dot(irev, fact) + 1, 0), dtype=int)


def _select(sel, dims):
    """
    Private function finding selected components
    """
    sel = np.asarray(sel)  # make sure sel is np.array
    dims = np.asarray(dims)  # make sure dims is np.array
    rlst = dims.take(sel)
    rprod = np.prod(rlst)
    ilist = np.ones((rprod, len(dims)), dtype=int)
    counter = np.arange(rprod)
    for k in range(len(sel)):
        ilist[:, sel[k]] = np.remainder(
            np.fix(counter / np.prod(dims[sel[k + 1:]])), dims[sel[k]]) + 1
    return ilist
