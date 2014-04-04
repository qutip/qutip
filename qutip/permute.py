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
import numpy as np
import scipy.sparse as sp
from qutip.ptrace import _select


def _permute(Q, order):
    if Q.type == 'ket':
        dims, perm = _perm_inds(Q.dims[0], order)
        nzs = Q.data.nonzero()[0]
        wh = np.where(perm == nzs)[0]
        data = np.ones(len(wh), dtype=int)
        cols = perm[wh].T[0]
        perm_matrix = sp.coo_matrix((data, (wh, cols)),
                                    shape=(Q.shape[0], Q.shape[0]), dtype=int)
        perm_matrix = perm_matrix.tocsr()
        return perm_matrix * Q.data, Q.dims, Q.shape
    if Q.type == 'bra':
        dims, perm = _perm_inds(A.dims[0], order)
        nzs = Q.data.nonzero()[1]
        wh = np.where(perm == nzs)[0]
        data = np.ones(len(wh), dtype=int)
        rows = perm[wh].T[0]
        perm_matrix = sp.coo_matrix((data, (rows, wh)),
                                    shape=(Q.shape[1], Q.shape[1]), dtype=int)
        perm_matrix = perm_matrix.tocsr()
        return Q.data * perm_matrix, Q.dims, Q.shape
    if Q.type == 'oper':
        dims, perm = _perm_inds(Q.dims[0], order)
        data = np.ones(Q.shape[0], dtype=int)
        rows = np.arange(Q.shape[0], dtype=int)
        perm_matrix = sp.coo_matrix((data, (rows, perm.T[0])),
                                    shape=(Q.shape[1], Q.shape[1]), dtype=int)
        perm_matrix = perm_matrix.tocsr()
        dims_part = list(dims[order])
        dims = [dims_part, dims_part]
        return (perm_matrix * Q.data) * perm_matrix.T, dims, Q.shape
    if Q.type == 'super':
        # Get the breakout of the left index into dims.
        # Since this is a super, the left index itself breaks into left
        # and right indices, each of which breaks down further.
        # The best way to deal with that here is to flatten dims.
        q_dims_left = sum(Q.dims[0], [])
        dims, perm = _perm_inds(q_dims_left, order)
        dims = dims.flatten()
        
        data = np.ones(Q.shape[0], dtype=int)
        rows = np.arange(Q.shape[0], dtype=int)
        
        perm_matrix = sp.coo_matrix((data, (rows, perm.T[0])),
                                    shape=(Q.shape[1], Q.shape[1]), dtype=int)
        
        perm_matrix = perm_matrix.tocsr()
        dims_part = list(dims[order])
        
        # Finally, we need to restructure the now-decomposed left index
        # into left and right subindices, so that the overall dims we return
        # are of the form
        # [[left_1, right_1], [left_2, right_2], ... [left_n, right_n]] * 2,
        # with each of the [left_i, right_i] pairs indicating one underlying
        # factor.
        dims_part = [
            dims_part[i:i+2] for i in range(0, len(dims_part), 2)
        ]
        
        return (perm_matrix * Q.data) * perm_matrix.T, [dims_part, dims_part], Q.shape
    else:
        raise TypeError('Invalid quantum object for permutation.')

def _perm_inds(dims, order):
    """
    Private function giving permuted indices for permute function.
    """
    dims = np.asarray(dims)
    order = np.asarray(order)
    if not np.all(np.sort(order) == np.arange(len(dims))):
        raise ValueError(
            'Requested permutation does not match tensor structure.')
    sel = _select(order, dims)
    irev = np.fliplr(sel) - 1
    fact = np.append(np.array([1]), np.cumprod(np.flipud(dims)[:-1]))
    fact = fact.reshape(len(fact), 1)
    perm_inds = np.dot(irev, fact)
    return dims, perm_inds
    
def reshuffle(q_oper):
    """
    Column-reshuffles a ``type="super"`` Qobj.
    """
    if not q_oper.type == "super":
        raise TypeError("Reshuffling is only supported on type='super'.")
    n_subsystems = len(q_oper.dims[0][0])
    perm_idxs = np.arange(2 * n_subsystems)[np.arange(n_subsystems * 2).reshape((2, n_subsystems)).T.flatten()]
    
    return q_oper.permute(perm_idxs)
