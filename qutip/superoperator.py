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

__all__ = ['liouvillian', 'liouvillian_ref', 'lindblad_dissipator',
           'operator_to_vector', 'vector_to_operator', 'mat2vec', 'vec2mat',
           'vec2mat_index', 'mat2vec_index', 'spost', 'spre', 'sprepost']

import scipy.sparse as sp
import numpy as np
from qutip.qobj import Qobj
from qutip.td_qobj import td_Qobj
from qutip.fastsparse import fast_csr_matrix, fast_identity
from qutip.sparse import sp_reshape
from qutip.cy.spmath import zcsr_kron
from functools import partial


"""def td_liouvillian(H, c_ops=[], chi=None, args={}, tlist=None, raw_str=False):
    print("Old method")
    h = td_Qobj(H, args=args, tlist=tlist, raw_str=raw_str)
    co = []
    for c in c_ops:
        co.append(td_Qobj(c, args=args, tlist=tlist, raw_str=raw_str))
    return liouvillian(h, co, chi=chi)

def td_lindblad_dissipator(a, args={}, tlist=None, raw_str=False):
    print("Old method")
    return lindblad_dissipator(td_Qobj(a, args=args, tlist=tlist, raw_str=raw_str))
"""


def liouvillian(H, c_ops=[], data_only=False, chi=None):
    """Assembles the Liouvillian superoperator from a Hamiltonian
    and a ``list`` of collapse operators. Like liouvillian, but with an
    experimental implementation which avoids creating extra Qobj instances,
    which can be advantageous for large systems.

    Parameters
    ----------
    H : qobj
        System Hamiltonian.

    c_ops : array_like
        A ``list`` or ``array`` of collapse operators.

    Returns
    -------
    L : qobj
        Liouvillian superoperator.

    """
    if isinstance(c_ops, (Qobj, td_Qobj)):
        c_ops = [c_ops]
    if chi and len(chi) != len(c_ops):
        raise ValueError('chi must be a list with same length as c_ops')

    h = None
    if H is not None:
        if isinstance(H, td_Qobj):
            h = H.cte
        else:
            h = H
        if h.isoper:
            op_dims = h.dims
            op_shape = h.shape
        elif h.issuper:
            op_dims = h.dims[0]
            op_shape = [np.prod(op_dims[0]), np.prod(op_dims[0])]
        else:
            raise TypeError("Invalid type for Hamiltonian.")
    else:
        # no hamiltonian given, pick system size from a collapse operator
        if isinstance(c_ops, list) and len(c_ops) > 0:
            if isinstance(c_ops[0], td_Qobj):
                c = c_ops[0].cte
            else:
                c = c_ops[0]
            if c.isoper:
                op_dims = c.dims
                op_shape = c.shape
            elif c.issuper:
                op_dims = c.dims[0]
                op_shape = [np.prod(op_dims[0]), np.prod(op_dims[0])]
            else:
                raise TypeError("Invalid type for collapse operator.")
        else:
            raise TypeError("Either H or c_ops must be given.")

    sop_dims = [[op_dims[0], op_dims[0]], [op_dims[1], op_dims[1]]]
    sop_shape = [np.prod(op_dims), np.prod(op_dims)]

    spI = fast_identity(op_shape[0])

    td = False
    L = None
    if isinstance(H, td_Qobj):
        td = True
        def H2L(H):
            if H.isoper:
                return -1.0j * (spre(H) - spost(H))
            else:
                return H
        L = H.apply(H2L)
        data = L.cte.data
    elif isinstance(H, Qobj):
        if H.isoper:
            Ht = H.data.T
            data = -1j * zcsr_kron(spI, H.data)
            data += 1j * zcsr_kron(Ht, spI)
        else:
            data = H.data
    else:
        data = fast_csr_matrix(shape=(sop_shape[0], sop_shape[1]))

    td_c_ops = []
    for idx, c_op in enumerate(c_ops):
        if isinstance(c_op, td_Qobj):
            td = True
            if c_op.const:
                c_ = c_op.cte
            elif chi:
                td_c_ops.append(lindblad_dissipator(c_op, chi=chi[idx]))
                continue
            else:
                td_c_ops.append(lindblad_dissipator(c_op))
                continue
        else:
            c_ = c_op

        if c_.issuper:
            data = data + c_.data
        else:
            cd = c_.data.H
            c = c_.data
            if chi:
                data = data + np.exp(1j * chi[idx]) * \
                                zcsr_kron(c.conj(), c)
            else:
                data = data + zcsr_kron(c.conj(), c)
            cdc = cd * c
            cdct = cdc.T
            data = data - 0.5 * zcsr_kron(spI, cdc)
            data = data - 0.5 * zcsr_kron(cdct, spI)

    if not td:
        if data_only:
            return data
        else:
            L = Qobj()
            L.dims = sop_dims
            L.data = data
            L.superrep = 'super'
            return L
    else:
        if not L:
            l = Qobj()
            l.dims = sop_dims
            l.data = data
            l.superrep = 'super'
            L = td_Qobj(l)
        else:
            L.cte.data = data
        for c_op in td_c_ops:
            L += c_op
        return L


def liouvillian_ref(H, c_ops=[]):
    """Assembles the Liouvillian superoperator from a Hamiltonian
    and a ``list`` of collapse operators.

    Parameters
    ----------
    H : qobj
        System Hamiltonian.

    c_ops : array_like
        A ``list`` or ``array`` of collapse operators.

    Returns
    -------
    L : qobj
        Liouvillian superoperator.
    """

    L = -1.0j * (spre(H) - spost(H)) if H else 0

    for c in c_ops:
        if c.issuper:
            L += c
        else:
            cdc = c.dag() * c
            L += spre(c) * spost(c.dag()) - 0.5 * spre(cdc) - 0.5 * spost(cdc)

    return L


def lindblad_dissipator(a, b=None, data_only=False, chi=None):
    """
    Lindblad dissipator (generalized) for a single pair of collapse operators
    (a, b), or for a single collapse operator (a) when b is not specified:

    .. math::

        \\mathcal{D}[a,b]\\rho = a \\rho b^\\dagger -
        \\frac{1}{2}a^\\dagger b\\rho - \\frac{1}{2}\\rho a^\\dagger b

    Parameters
    ----------
    a : qobj, td_qobj
        Left part of collapse operator.

    b : qobj (optional), td_qobj
        Right part of collapse operator. If not specified, b defaults to a.

    Returns
    -------
    D : qobj, td_qobj
        Lindblad dissipator superoperator.
    """
    if b is None:
        if isinstance(a, td_Qobj):
            if not a.N_obj == 1:
                raise Exception("Single collapse operator as a Td_Qobj " +\
                                "must be composed of only 1 Qobj")
            else:
                def partial_lindblad_dissipator(A):
                    return lindblad_dissipator(A, chi=chi)
                return a.apply(partial_lindblad_dissipator)._f_norm2()
        else:
            b = a

    td = False
    #colapse constant td_Qobj
    if isinstance(a, td_Qobj) and a.const:
        a = a.cte
        td = True
    if isinstance(b, td_Qobj) and b.const:
        b = b.cte
        td = True

    if isinstance(a, td_Qobj):
        if isinstance(b, td_Qobj):
            # No td coeff crossterm
            raise NotImplementedError("lindblad_dissipator do not support 2 td_Qobj")
        else:
            def partial_lindblad_dissipator(A):
                return lindblad_dissipator(A, b, chi=chi)
            return a.apply(partial_lindblad_dissipator)

    elif isinstance(b, td_Qobj):
        def partial_lindblad_dissipator(B):
            return lindblad_dissipator(a, B, chi=chi)
        return b.apply(partial_lindblad_dissipator)

    else:
        #both Qobj
        ad_b = a.dag() * b
        if chi:
            D = spre(a) * spost(b.dag()) *np.exp(1j * chi) \
                - 0.5 * spre(ad_b) - 0.5 * spost(ad_b)
        else:
            D = spre(a) * spost(b.dag()) - 0.5 * spre(ad_b) - 0.5 * spost(ad_b)
        td_Qobj(D) if td else D.data if data_only else D
        if td:
            return td_Qobj(D)
        else:
            return D.data if data_only else D


def operator_to_vector(op):
    """
    Create a vector representation of a quantum operator given
    the matrix representation.
    """
    if isinstance(op, td_Qobj):
        return op.apply(operator_to_vector)

    q = Qobj()
    q.dims = [op.dims, [1]]
    q.data = sp_reshape(op.data.T, (np.prod(op.shape), 1))
    return q


def vector_to_operator(op):
    """
    Create a matrix representation given a quantum operator in
    vector form.
    """
    if isinstance(op, td_Qobj):
        return op.apply(vector_to_operator)

    q = Qobj()
    q.dims = op.dims[0]
    n = int(np.sqrt(op.shape[0]))
    q.data = sp_reshape(op.data.T, (n, n)).T
    return q


def mat2vec(mat):
    """
    Private function reshaping matrix to vector.
    """
    return mat.T.reshape(np.prod(np.shape(mat)), 1)


def vec2mat(vec):
    """
    Private function reshaping vector to matrix.
    """
    n = int(np.sqrt(len(vec)))
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
    if isinstance(A, td_Qobj):
        return A.apply(spost)

    if not isinstance(A, Qobj):
        raise TypeError('Input is not a quantum object')

    if not A.isoper:
        raise TypeError('Input is not a quantum operator')

    S = Qobj(isherm=A.isherm, superrep='super')
    S.dims = [[A.dims[0], A.dims[1]], [A.dims[0], A.dims[1]]]
    S.data = zcsr_kron(A.data.T,
                fast_identity(np.prod(A.shape[0])))
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
    if isinstance(A, td_Qobj):
        return A.apply(spre)

    if not isinstance(A, Qobj):
        raise TypeError('Input is not a quantum object')

    if not A.isoper:
        raise TypeError('Input is not a quantum operator')

    S = Qobj(isherm=A.isherm, superrep='super')
    S.dims = [[A.dims[0], A.dims[1]], [A.dims[0], A.dims[1]]]
    S.data = zcsr_kron(fast_identity(np.prod(A.shape[1])), A.data)
    return S


def _drop_projected_dims(dims):
    """
    Eliminate subsystems that has been collapsed to only one state due to
    a projection.
    """
    return [d for d in dims if d != 1]


def sprepost(A, B):
    """Superoperator formed from pre-multiplication by operator A and post-
    multiplication of operator B.

    Parameters
    ----------
    A : Qobj
        Quantum operator for pre-multiplication.

    B : Qobj
        Quantum operator for post-multiplication.

    Returns
    --------
    super : Qobj
        Superoperator formed from input quantum objects.
    """
    # reduce false td
    td = False
    if isinstance(A, td_Qobj) and A.const:
        A = A.cte
        td = True
    if isinstance(B, td_Qobj) and B.const:
        B = B.cte
        td = True

    # A is time-dependent
    if isinstance(A, td_Qobj):
        if isinstance(B, td_Qobj):
            # No td coeff crossterm
            raise NotImplementedError("sprepost: do not support 2 td_Qobj")
        else:
            def partial_sprepost(a):
                return sprepost(a, B)
            return A.apply(partial_sprepost)

    elif isinstance(B, td_Qobj):
        def partial_sprepost(b):
            return sprepost(A, b)
        return B.apply(partial_sprepost)

    else:
        dims = [[_drop_projected_dims(A.dims[0]), _drop_projected_dims(B.dims[1])],
                [_drop_projected_dims(A.dims[1]), _drop_projected_dims(B.dims[0])]]
        data = zcsr_kron(B.data.T, A.data)
        if not td:
            return Qobj(data, dims=dims, superrep='super')
        else:
            return td_Qobj(Qobj(data, dims=dims, superrep='super'))
