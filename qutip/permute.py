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

__all__ = ['reshuffle']

import numpy as np
import scipy.sparse as sp
from qutip.cy.ptrace import _select
from qutip.cy.spconvert import arr_coo2fast, cy_index_permute


def _chunk_dims(dims, order):
    lens_order = map(len, order)

    for chunk_len in lens_order:
        yield list(dims[:chunk_len])
        dims = dims[chunk_len:]


def _permute(Q, order):
    Qcoo = Q.data.tocoo()
    
    if Q.isket:
        cy_index_permute(Qcoo.row,
                         np.array(Q.dims[0], dtype=np.int32),
                         np.array(order, dtype=np.int32))
        
        new_dims = [[Q.dims[0][i] for i in order], Q.dims[1]]

    elif Q.isbra:
        cy_index_permute(Qcoo.col,
                         np.array(Q.dims[1], dtype=np.int32),
                         np.array(order, dtype=np.int32))
        
        new_dims = [Q.dims[0], [Q.dims[1][i] for i in order]]

    elif Q.isoper:
        cy_index_permute(Qcoo.row,
                         np.array(Q.dims[0], dtype=np.int32),
                         np.array(order, dtype=np.int32))
        cy_index_permute(Qcoo.col,
                         np.array(Q.dims[1], dtype=np.int32),
                         np.array(order, dtype=np.int32))
        
        new_dims = [[Q.dims[0][i] for i in order], [Q.dims[1][i] for i in order]]
    
    elif Q.isoperket:
    	# For superoperators, we expect order to be something like
        # [[0, 2], [1, 3]], which tells us to permute according to
        # [0, 2, 1 ,3], and then group indices according to the length
        # of each sublist.
        # As another example,
        # permuting [[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]] by
        # [[0, 3], [1, 4], [2, 5]] should give
        # [[[1, 1], [2, 2], [3, 3]], [[1, 1], [2, 2], [3, 3]]].
        #
        # Get the breakout of the left index into dims.
        # Since this is a super, the left index itself breaks into left
        # and right indices, each of which breaks down further.
        # The best way to deal with that here is to flatten dims.
        
        flat_order = np.array(sum(order, []), dtype=np.int32)
        q_dims = np.array(sum(Q.dims[0], []), dtype=np.int32)
        
        cy_index_permute(Qcoo.row, q_dims, flat_order)
        
        # Finally, we need to restructure the now-decomposed left index
        # into left and right subindices, so that the overall dims we return
        # are of the form specified by order.
        
        new_dims = [q_dims[i] for i in flat_order]
        new_dims = list(_chunk_dims(new_dims, order))
        new_dims = [new_dims, [1]]
    
    elif Q.isoperbra:
        flat_order = np.array(sum(order, []), dtype=np.int32)
        q_dims = np.array(sum(Q.dims[1], []), dtype=np.int32)
        
        cy_index_permute(Qcoo.col, q_dims, flat_order)
        
        new_dims = [q_dims[i] for i in flat_order]
        new_dims = list(_chunk_dims(new_dims, order))
        new_dims = [[1], new_dims]
    
    elif Q.issuper:
        flat_order = np.array(sum(order, []), dtype=np.int32)
        q_dims = np.array(sum(Q.dims[0], []), dtype=np.int32)
        
        cy_index_permute(Qcoo.row, q_dims, flat_order)
        cy_index_permute(Qcoo.col, q_dims, flat_order)
        
        new_dims = [q_dims[i] for i in flat_order]
        new_dims = list(_chunk_dims(new_dims, order))
        new_dims = [new_dims, new_dims]
        
    else:
        raise TypeError('Invalid quantum object for permutation.')
    
    return arr_coo2fast(Qcoo.data, Qcoo.row, Qcoo.col, Qcoo.shape[0], Qcoo.shape[1]), new_dims


def _perm_inds(dims, order):
    """
    Private function giving permuted indices for permute function.
    """
    dims = np.asarray(dims,dtype=np.int32)
    order = np.asarray(order,dtype=np.int32)
    if not np.all(np.sort(order) == np.arange(len(dims))):
        raise ValueError(
            'Requested permutation does not match tensor structure.')
    sel = _select(order, dims,np.prod(dims))
    irev = np.fliplr(sel)
    fact = np.append(np.array([1]), np.cumprod(np.flipud(dims)[:-1]))
    fact = fact.reshape(len(fact), 1)
    perm_inds = np.dot(irev, fact)
    return dims, perm_inds


def reshuffle(q_oper):
    """
    Column-reshuffles a ``type="super"`` Qobj.
    """
    if q_oper.type not in ('super', 'operator-ket'):
        raise TypeError("Reshuffling is only supported on type='super' "
                        "or type='operator-ket'.")

    # How many indices are there, and how many subsystems can we decompose
    # each index into?
    n_indices = len(q_oper.dims[0])
    n_subsystems = len(q_oper.dims[0][0])

    # Generate a list of lists (lol) that represents the permutation order we
    # need. It's easiest to do so if we make an array, then turn it into a lol
    # by using map(list, ...). That array is generated by using reshape and
    # transpose to turn an array like [a, b, a, b, ..., a, b] into one like
    # [a, a, ..., a, b, b, ..., b].
    perm_idxs = map(list,
                    np.arange(n_subsystems * n_indices)[
                        np.arange(n_subsystems * n_indices).reshape(
                            (n_indices, n_subsystems)).T.flatten()
                    ].reshape((n_subsystems, n_indices))
                    )

    return q_oper.permute(list(perm_idxs))
