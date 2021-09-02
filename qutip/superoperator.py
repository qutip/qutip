__all__ = ['liouvillian', 'liouvillian_ref', 'lindblad_dissipator',
           'operator_to_vector', 'vector_to_operator', 'mat2vec', 'vec2mat',
           'vec2mat_index', 'mat2vec_index', 'spost', 'spre', 'sprepost']

import scipy.sparse as sp
import numpy as np
from qutip.qobj import Qobj
from qutip.fastsparse import fast_csr_matrix, fast_identity
from qutip.sparse import sp_reshape
from qutip.cy.spmath import zcsr_kron
from functools import partial


def liouvillian(H, c_ops=[], data_only=False, chi=None):
    """Assembles the Liouvillian superoperator from a Hamiltonian
    and a ``list`` of collapse operators. Like liouvillian, but with an
    experimental implementation which avoids creating extra Qobj instances,
    which can be advantageous for large systems.

    Parameters
    ----------
    H : Qobj or QobjEvo
        System Hamiltonian.

    c_ops : array_like of Qobj or QobjEvo
        A ``list`` or ``array`` of collapse operators.

    Returns
    -------
    L : Qobj or QobjEvo
        Liouvillian superoperator.

    """
    if isinstance(c_ops, (Qobj, QobjEvo)):
        c_ops = [c_ops]
    if chi and len(chi) != len(c_ops):
        raise ValueError('chi must be a list with same length as c_ops')

    h = None
    if H is not None:
        if isinstance(H, QobjEvo):
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
            if isinstance(c_ops[0], QobjEvo):
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
    if isinstance(H, QobjEvo):
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
        if isinstance(c_op, QobjEvo):
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
            L = QobjEvo(l)
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
    a : Qobj or QobjEvo
        Left part of collapse operator.

    b : Qobj or QobjEvo (optional)
        Right part of collapse operator. If not specified, b defaults to a.

    Returns
    -------
    D : qobj, QobjEvo
        Lindblad dissipator superoperator.
    """
    if b is None:
        b = a
    ad_b = a.dag() * b
    if chi:
        D = spre(a) * spost(b.dag()) * np.exp(1j * chi) \
            - 0.5 * spre(ad_b) - 0.5 * spost(ad_b)
    else:
        D = spre(a) * spost(b.dag()) - 0.5 * spre(ad_b) - 0.5 * spost(ad_b)

    if isinstance(a, QobjEvo) or isinstance(b, QobjEvo):
        return D
    else:
        return D.data if data_only else D


def operator_to_vector(op):
    """
    Create a vector representation given a quantum operator in matrix form.
    The passed object should have a ``Qobj.type`` of 'oper' or 'super'; this
    function is not designed for general-purpose matrix reshaping.

    Parameters
    ----------
    op : Qobj or QobjEvo
        Quantum operator in matrix form.  This must have a type of 'oper' or
        'super'.

    Returns
    -------
    Qobj or QobjEvo
        The same object, but re-cast into a column-stacked-vector form of type
        'operator-ket'.  The output is the same type as the passed object.
    """
    if isinstance(op, QobjEvo):
        return op.apply(operator_to_vector)
    if not (op.isoper or op.issuper):
        raise ValueError("only valid for operator matrices")
    size = op.shape[0] * op.shape[1]
    return Qobj(
        sp_reshape(op.data.T, (size, 1)),
        dims=[op.dims, [1]], shape=(size, 1), type='operator-ket', copy=False,
    )


def vector_to_operator(op):
    """
    Create a matrix representation given a quantum operator in vector form.
    The passed object should have a ``Qobj.type`` of 'operator-ket'; this
    function is not designed for general-purpose matrix reshaping.

    Parameters
    ----------
    op : Qobj or QobjEvo
        Quantum operator in column-stacked-vector form.  This must have a type
        of 'operator-ket'.

    Returns
    -------
    Qobj or QobjEvo
        The same object, but re-cast into "standard" operator form.  The output
        is the same type as the passed object.
    """
    if isinstance(op, QobjEvo):
        return op.apply(vector_to_operator)
    if not op.isoperket:
        raise ValueError(
            "only valid for operators in column-stacked 'operator-ket' format"
        )
    # e.g. op.dims = [ [[rows], [cols]], [1]]
    dims = op.dims[0]
    shape = (np.prod(dims[0]), np.prod(dims[1]))
    return Qobj(
        sp_reshape(op.data.T, shape[::-1]).T,
        dims=dims, shape=shape, copy=False,
    )


def mat2vec(mat):
    """
    Private function reshaping matrix to vector.
    """
    return mat.T.reshape(np.prod(np.shape(mat)), 1)


def vec2mat(vec, shape=None):
    """
    Private function reshaping vector to matrix.
    """
    if shape is None:
        n = int(np.sqrt(len(vec)))
        shape = (n, n)
    return vec.reshape(shape[::-1]).T


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
    A : Qobj or QobjEvo
        Quantum operator for post multiplication.

    Returns
    -------
    super : Qobj or QobjEvo
        Superoperator formed from input qauntum object.
    """
    if isinstance(A, QobjEvo):
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
    A : Qobj or QobjEvo
        Quantum operator for pre-multiplication.

    Returns
    --------
    super :Qobj or QobjEvo
        Superoperator formed from input quantum object.
    """
    if isinstance(A, QobjEvo):
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
    A : Qobj or QobjEvo
        Quantum operator for pre-multiplication.

    B : Qobj or QobjEvo
        Quantum operator for post-multiplication.

    Returns
    --------
    super : Qobj or QobjEvo
        Superoperator formed from input quantum objects.
    """
    if isinstance(A, QobjEvo) or isinstance(B, QobjEvo):
        return spre(A) * spost(B)

    else:
        dims = [[_drop_projected_dims(A.dims[0]),
                 _drop_projected_dims(B.dims[1])],
                [_drop_projected_dims(A.dims[1]),
                 _drop_projected_dims(B.dims[0])]]
        data = zcsr_kron(B.data.T, A.data)
        return Qobj(data, dims=dims, superrep='super')

from qutip.qobjevo import QobjEvo
