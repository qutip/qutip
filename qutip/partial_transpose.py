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
#
#    Significant parts of this code was contributed by Jonas Neergaard-Nielsen
###############################################################################

__all__ = ['partial_transpose']

import numpy as np
import scipy.sparse as sp

from qutip.qobj import Qobj
from qutip.states import (state_index_number, state_number_index,
                          state_number_enumerate)


def partial_transpose(rho, mask, method='dense'):
    """
    Return the partial transpose of a Qobj instance `rho`,
    where `mask` is an array/list with length that equals
    the number of components of `rho` (that is, the length of
    `rho.dims[0]`), and the values in `mask` indicates whether
    or not the corresponding subsystem is to be transposed.
    The elements in `mask` can be boolean or integers `0` or `1`,
    where `True`/`1` indicates that the corresponding subsystem
    should be tranposed.

    Parameters
    ----------

    rho : :class:`qutip.qobj`
        A density matrix.

    mask : *list* / *array*
        A mask that selects which subsystems should be transposed.

    method : str
        choice of method, `dense` or `sparse`. The default method
        is `dense`. The `sparse` implementation can be faster for
        large and sparse systems (hundreds of quantum states).

    Returns
    -------

    rho_pr: :class:`qutip.qobj`

        A density matrix with the selected subsystems transposed.

    """
    if method == 'sparse':
        return _partial_transpose_sparse(rho, mask)
    else:
        return _partial_transpose_dense(rho, mask)


def _partial_transpose_dense(rho, mask):
    """
    Based on Jonas' implementation using numpy.
    Very fast for dense problems.
    """
    nsys = len(mask)
    pt_dims = np.arange(2 * nsys).reshape(2, nsys).T
    pt_idx = np.concatenate([[pt_dims[n, mask[n]] for n in range(nsys)],
                            [pt_dims[n, 1 - mask[n]] for n in range(nsys)]])

    data = rho.data.toarray().reshape(
        np.array(rho.dims).flatten()).transpose(pt_idx).reshape(rho.shape)

    return Qobj(data, dims=rho.dims)


def _partial_transpose_sparse(rho, mask):
    """
    Implement the partial transpose using the CSR sparse matrix.
    """

    data = sp.lil_matrix((rho.shape[0], rho.shape[1]), dtype=complex)

    for m in range(len(rho.data.indptr) - 1):

        n1 = rho.data.indptr[m]
        n2 = rho.data.indptr[m + 1]

        psi_A = state_index_number(rho.dims[0], m)

        for idx, n in enumerate(rho.data.indices[n1:n2]):

            psi_B = state_index_number(rho.dims[1], n)

            m_pt = state_number_index(
                rho.dims[1], np.choose(mask, [psi_A, psi_B]))
            n_pt = state_number_index(
                rho.dims[0], np.choose(mask, [psi_B, psi_A]))

            data[m_pt, n_pt] = rho.data.data[n1 + idx]

    return Qobj(data.tocsr(), dims=rho.dims)


def _partial_transpose_reference(rho, mask):
    """
    This is a reference implementation that explicitly loops over
    all states and performs the transpose. It's slow but easy to
    understand and useful for testing.
    """

    A_pt = np.zeros(rho.shape, dtype=complex)

    for psi_A in state_number_enumerate(rho.dims[0]):
        m = state_number_index(rho.dims[0], psi_A)

        for psi_B in state_number_enumerate(rho.dims[1]):
            n = state_number_index(rho.dims[1], psi_B)

            m_pt = state_number_index(
                rho.dims[1], np.choose(mask, [psi_A, psi_B]))
            n_pt = state_number_index(
                rho.dims[0], np.choose(mask, [psi_B, psi_A]))

            A_pt[m_pt, n_pt] = rho.data[m, n]

    return Qobj(A_pt, dims=rho.dims)
