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

import scipy
import scipy.linalg as la
import scipy.sparse as sp
from scipy import prod, transpose, reshape
from qutip.qobj import *
from qutip.operators import destroy
from qutip.sparse import sp_reshape


def liouvillian(H, c_op_list=[]):
    """Assembles the Liouvillian superoperator from a Hamiltonian
    and a ``list`` of collapse operators.

    Parameters
    ----------
    H : qobj
        System Hamiltonian.

    c_op_list : array_like
        A ``list`` or ``array`` of collapse operators.

    Returns
    -------
    L : qobj
        Liouvillian superoperator.

    """

    L = -1.0j * (spre(H) - spost(H)) if H else 0

    for c in c_op_list:
        if issuper(c):
            L += c
        else:
            cdc = c.dag() * c
            L += spre(c) * spost(c.dag()) - 0.5 * spre(cdc) - 0.5 * spost(cdc)
    L.type = 'super'
    return L


def liouvillian_fast(H, c_op_list, data_only=False):
    """Assembles the Liouvillian superoperator from a Hamiltonian
    and a ``list`` of collapse operators. Like liouvillian, but with an
    experimental implementation which avoids creating extra Qobj instances,
    which can be advantageous for large systems.

    Parameters
    ----------
    H : qobj
        System Hamiltonian.

    c_op_list : array_like
        A ``list`` or ``array`` of collapse operators.

    Returns
    -------
    L : qobj
        Liouvillian superoperator.

    """

    if H is not None:
        if isoper(H):
            op_dims = H.dims
            op_shape = H.shape
        elif issuper(H):
            op_dims = H.dims[0]
            op_shape = [prod(op_dims[0]), prod(op_dims[0])]
        else:
            raise TypeError("Invalid type for Hamiltonian.")
    else:
        # no hamiltonian given, pick system size from a collapse operator
        if isinstance(c_op_list, list) and len(c_op_list) > 0:
            c = c_op_list[0]
            if isoper(c):
                op_dims = c.dims
                op_shape = c.shape
            elif issuper(c):
                op_dims = c.dims[0]
                op_shape = [prod(op_dims[0]), prod(op_dims[0])]
            else:
                raise TypeError("Invalid type for collapse operator.")
        else:
            raise TypeError("Either H or c_op_list must be given.")

    sop_dims = [[op_dims[0], op_dims[0]], [op_dims[1], op_dims[1]]]
    sop_shape = [prod(op_dims), prod(op_dims)]

    spI = sp.identity(op_shape[0])

    if H:
        if isoper(H):
            data = -1j * (sp.kron(spI, H.data, format='csr')
                          - sp.kron(H.data.T, spI, format='csr'))
        else:
            data = H.data
    else:
        data = sp.csr_matrix((sop_shape[0], sop_shape[1]), dtype=complex)

    for c_op in c_op_list:
        if issuper(c_op):
            data = data + c_op.data
        else:
            cd = c_op.data.T.conj()
            c = c_op.data
            data = data + sp.kron(cd.T, c, format='csr')
            cdc = cd * c
            data = data - 0.5 * sp.kron(spI, cdc, format='csr')
            data = data - 0.5 * sp.kron(cdc.T, spI, format='csr')

    if data_only:
        return data
    else:
        L = Qobj()
        L.dims = sop_dims
        L.shape = sop_shape
        L.data = data
        L.isherm = False
        L.type = 'super'
        return L


def lindblad_dissipator(c, data_only=False):
    """
    Return the Lindblad dissipator for a single collapse operator.

    TODO: optimize like liouvillian_fast
    """

    cdc = c.dag() * c
    D = spre(c) * spost(c.dag()) - 0.5 * spre(cdc) - 0.5 * spost(cdc)

    return D.data if data_only else D


def operator_to_vector(op):
    """
    Create a vector representation of a quantum operator given
    the matrix representation.
    """
    q = Qobj()
    q.shape = [prod(op.shape), 1]
    q.dims = [op.dims, [1]]
    q.data = sp_reshape(op.data.T, tuple(q.shape))
    q.type = 'operator-vector'
    return q


def vector_to_operator(op):
    """
    Create a matrix representation given a quantum operator in
    vector form.
    """
    q = Qobj()
    q.shape = [op.dims[0][0][0], op.dims[0][1][0]]
    q.dims = op.dims[0]
    q.data = sp_reshape(op.data.H, tuple(q.shape))
    q.type = 'oper'
    return q


def mat2vec(mat):
    """
    Private function reshaping matrix to vector.
    """
    return mat.T.reshape(prod(shape(mat)), 1)


def vec2mat(vec):
    """
    Private function reshaping vector to matrix.
    """
    n = int(sqrt(len(vec)))
    return vec.reshape((n, n)).T


def vec2mat_index(N, I):
    """
    Convert a vector index to a matrix index pair that is compatible with the
    vector to matrix rearrangement done by the vec2mat function.
    """
    j = int(I / N)
    i = I - N * j
    return i, j


def mat2vec_index(N, i, j):
    """
    Convert a matrix index pair to a vector index that is compatible with the
    matrix to vector rearrangement done by the mat2vec function.
    """
    return i + N * j


def spost(A):
    """Superoperator formed from post-multiplication by operator A

    Parameters
    ----------
    A : qobj
        Quantum operator for post multiplication.

    Returns
    -------
    super : qobj
        Superoperator formed from input qauntum object.
    """
    if not isoper(A):
        raise TypeError('Input is not a quantum object')

    S = Qobj(isherm=A.isherm, type='super')
    S.dims = [[A.dims[0], A.dims[1]], [A.dims[0], A.dims[1]]]
    S.shape = [prod(S.dims[0]), prod(S.dims[1])]
    S.data = sp.kron(A.data.T, sp.identity(prod(A.dims[0])), format='csr')
    return S


def spre(A):
    """Superoperator formed from pre-multiplication by operator A.

    Parameters
    ----------
    A : qobj
        Quantum operator for pre-multiplication.

    Returns
    --------
    super :qobj
        Superoperator formed from input quantum object.

    """
    if not isoper(A):
        raise TypeError('Input is not a quantum object')

    S = Qobj(isherm=A.isherm, type='super')
    S.dims = [[A.dims[0], A.dims[1]], [A.dims[0], A.dims[1]]]
    S.shape = [prod(S.dims[0]), prod(S.dims[1])]
    S.data = sp.kron(sp.identity(prod(A.dims[1])), A.data, format='csr')
    return S
