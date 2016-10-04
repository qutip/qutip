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
from qutip.ptrace import _select


def _chunk_dims(dims, order):
    lens_order = map(len, order)

    for chunk_len in lens_order:
        yield list(dims[:chunk_len])
        dims = dims[chunk_len:]


def _permute(Q, order):
    if Q.isket:
        dims, perm = _perm_inds(Q.dims[0], order)
        nzs = Q.data.nonzero()[0]
        wh = np.where(perm == nzs)[0]
        data = np.ones(len(wh), dtype=int)
        cols = perm[wh].T[0]
        perm_matrix = sp.coo_matrix((data, (wh, cols)),
                                    shape=(Q.shape[0], Q.shape[0]), dtype=int)
        perm_matrix = perm_matrix.tocsr()
        return perm_matrix * Q.data, Q.dims

    elif Q.isbra:
        dims, perm = _perm_inds(Q.dims[1], order)
        nzs = Q.data.nonzero()[1]
        wh = np.where(perm == nzs)[0]
        data = np.ones(len(wh), dtype=int)
        rows = perm[wh].T[0]
        perm_matrix = sp.coo_matrix((data, (rows, wh)),
                                    shape=(Q.shape[1], Q.shape[1]), dtype=int)
        perm_matrix = perm_matrix.tocsr()
        return Q.data * perm_matrix, Q.dims

    elif Q.isoper:
        dims, perm = _perm_inds(Q.dims[0], order)
        data = np.ones(Q.shape[0], dtype=int)
        rows = np.arange(Q.shape[0], dtype=int)
        perm_matrix = sp.coo_matrix((data, (rows, perm.T[0])),
                                    shape=(Q.shape[1], Q.shape[1]), dtype=int)
        perm_matrix = perm_matrix.tocsr()
        dims_part = list(dims[order])
        dims = [dims_part, dims_part]
        return (perm_matrix * Q.data) * perm_matrix.T, dims

    elif Q.issuper or Q.isoperket:
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
        flat_order = sum(order, [])
        q_dims_left = sum(Q.dims[0], [])
        dims, perm = _perm_inds(q_dims_left, flat_order)
        dims = dims.flatten()

        data = np.ones(Q.shape[0], dtype=int)
        rows = np.arange(Q.shape[0], dtype=int)

        perm_matrix = sp.coo_matrix((data, (rows, perm.T[0])),
                                    shape=(Q.shape[0], Q.shape[0]), dtype=int)

        perm_matrix = perm_matrix.tocsr()
        dims_part = list(dims[flat_order])

        # Finally, we need to restructure the now-decomposed left index
        # into left and right subindices, so that the overall dims we return
        # are of the form specified by order.
        dims_part = list(_chunk_dims(dims_part, order))
        perm_left = (perm_matrix * Q.data)
        if Q.type == 'operator-ket':
            return perm_left, [dims_part, [1]]
        elif Q.type == 'super':
            return perm_left * perm_matrix.T, [dims_part, dims_part]
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
