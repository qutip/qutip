# This file is part of QuTiP.
#
#    QuTiP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    QuTiP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTiP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011 and later, Paul D. Nation & Robert J. Johansson
#
###########################################################################
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
        raise TypeError('Not implemented yet.')
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
