# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Copyright 2020 United States Government as represented by the Administrator
#    of the National Aeronautics and Space Administration. All Rights Reserved.
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

from qutip import Qobj
from qutip.qip.operations.gates import gate_expand_1toN, expand_operator
from qutip.core import data


def local_multiply(targets_op, vec_mat, targets, right=False, backend=None):
    """
    Wrapper for `local_multiply_dense` and `local_multiply_sparse`.

    Parameters
    ----------
    targets_op: :class:`qutip.Qobj` or np.ndarray
        The local operator which will act on the `targets` of `vec_mat`.
        Note: if specifying a sparse backend, this must be a Qobj type.
    vec_mat: :class:`qutip.Qobj`
            Can be 'oper', 'ket', or 'bra' type.
            Best to be of same data format as backend being used (see Notes).
            Convert using Qobj().to(data.Dense) or Qobj().to(data.CSR).
    targets: int or list of int
        The target indices that targets_op acts on.
        E.g. ``targets=1`` acts on index 1.
        E.g. ``targets=[6,3,0]`` acts on indices 6, 3, 0, in that order.
    right: bool
        Left or right matrix multiplication (default is False).
        If ``right==True``, compute  vec_mat * targets_op
        (in the appropriate sense), and if ``right==False``,
        compute targets_op * vec_mat.
    backend: str or None
        Which backend to perform the computation.
        Current accepted arguments (case insensitive):
        'dense': uses `local_multiply_dense` and picks backend based on input.
        'einsum': uses NumPy's einsum in `local_multiply_dense`.
        'vectorize': uses the vectorization based protocol in
        `local_multiply_dense`.
        'sparse' or 'csr': uses `expand_operator` to build the full
        sparse operator.
        If unspecified (default of None used), this is decided based on the
        data format of vec_mat.

    Returns
    -------
    :class:`qutip.Qobj`
        Result of the matrix multiplication of `targets_op` on `vec_mat`,
        with the same data format as `vec_mat`.

    Notes
    -----
    To avoid conversion slow-downs, if using the dense routines,
    specify vec_mat data as Dense. Similarly, if using sparse, specify
    data as CSR type.

    See `local_multiply_dense`, `local_multiply_sparse` docstring for more
    information on allowed input.
    """
    backends = ['csr', 'sparse', 'einsum', 'vectorize', 'dense', None]
    if isinstance(backend, str):
        backend = backend.lower()

    if backend is None and isinstance(vec_mat.data, data.CSR):
        # use sparse routine if vec_mat is sparse
        backend = 'csr'
    elif backend == 'dense':
        # local_multiply_dense will pick the best backend given None
        backend = None
    if backend not in backends:
        raise ValueError(f'Received invalid backend: {backends}. '
                         f'Should be one of: {backends}')

    if backend in backends[0: 2]:
        return local_multiply_sparse(targets_op, vec_mat, targets, right)
    else:
        return local_multiply_dense(targets_op, vec_mat, targets,
                                    right, backend)


def local_multiply_sparse(targets_op, vec_mat, targets, right=False):
    """
    Sparse routine for local multiplication.

    Parameters
    ----------
    targets_op: :class:`qutip.Qobj`
        The local operator which will act on the `targets` of `vec_mat`.
        `expand_operator` is used to build the full operator based on targets.
    vec_mat: :class:`qutip.Qobj`
        Can be 'oper', 'ket', or 'bra' type.
        Preferred data format is `CSR`.
    targets: int or list of int
        The target indices that targets_op acts on.
        E.g. ``targets=1`` acts on index 1.
        E.g. ``targets=[6,3,0]`` acts on indices 6, 3, 0, in that order.
    right: bool
        Left or right matrix multiplication (default is False).
        If ``right==True``, compute  vec_mat * targets_op
        (in the appropriate sense), and if ``right==False``,
        compute targets_op * vec_mat.
        If ``vec_mat.type=='ket'``, can only have left multiplication (A|psi>),
        if ``vec_mat.type=='bra'``, can only have right multiplication (<psi|A).

    Returns
    -------
    :class:`qutip.Qobj`
        Result of the matrix multiplication of `targets_op` on `vec_mat`,
        with the same data format as `vec_mat`.

    Notes
    -----
    It is most efficient to specify vec_mat as CSR type, otherwise there
    is conversion overhead.

    This routine simply builds the full (sparse) matrix using `expand_operator`.
    """
    targets = [targets] if isinstance(targets, int) else list(targets)
    n = len(vec_mat.dims[0])
    if len(targets) == 1:
        # this is faster since it doesn't use Qobj().permute
        full_op = gate_expand_1toN(targets_op, n, targets[0])
    else:
        full_op = expand_operator(targets_op, n, targets)

    out = vec_mat * full_op if right else full_op * vec_mat

    if not isinstance(vec_mat.data, data.CSR):
        # convert back to Dense if given Dense input
        # for this reason best to specify as input CSR type
        out = out.to(data.Dense)
    return out


def local_multiply_dense(targets_op, vec_mat, targets, right=False,
                         backend=None):
    """
    Matrix multiplication for local operator, acting on specified targets,
    using dense matrix routines.

    Parameters
    ----------
    targets_op: :class:`qutip.Qobj` or np.ndarray
        The local operator which will act on the `targets` of `vec_mat`.
        This should be square, with shape op_dim x op_dim,
        where op_dim = product(target dims).
        E.g. if target indices have dims (2, 3, 2), op_dim must be 2.3.2 = 12.
    vec_mat: :class:`qutip.Qobj`
        Can be 'oper', 'ket', or 'bra' type.
        Preferred data format is `Dense`.
    targets: int or list of int
        The target indices that targets_op acts on.
        E.g. ``targets=1`` acts on index 1.
        E.g. ``targets=[6,3,0]`` acts on indices 6, 3, 0.
        Note, the order of the targets is important.
        Note, for vector input, targets take values [0,...,n-1] for n qubits,
        and for density input, targets can take values [0,...,2(n-1)].
    right: bool
        Left or right matrix multiplication (default is False).
        If ``right==True``, compute  vec_mat * targets_op
        (in the appropriate sense), and if ``right==False``,
        compute targets_op * vec_mat.
        If ``vec_mat.type=='ket'``, can only have left multiplication (A|psi>),
        if ``vec_mat.type=='bra'``, can only have right multiplication (<psi|A).
    backend: str or None
        Which backend to perform the computation.
        Current accepted arguments (case insensitive):
        'einsum': uses NumPy's einsum
        'vectorize': uses a vectorization based protocol,
                     with standard matrix multiplication (using NumPy).
        If unspecified (default of None used), the most efficient protocol is
        guessed based on the input.

    Returns
    -------
    :class:`qutip.Qobj`
        Result of the matrix multiplication of `targets_op` on `vec_mat`,
        with the same data format as `vec_mat`.

    Notes
    -----
    The result of ``local_multiply_dense(local_op, rho_or_psi, targets)``
    is the same as ``expand_operator(local_op, N, targets) * rho_or_psi``,
    but is expected to be more efficient in cases where `rho_or_psi`
    is fairly dense, and at larger sizes, where the
    targets are a small fraction of the total sub-systems.

    For speed reasons, it is best to specify `vec_mat` as a Dense Qobj.
    This is true, in particular, if performing multiple local operations
    sequentially (e.g. in a circuit).
    Otherwise there can be conversion slow downs, since this function
    would have to keep converting CSR to Dense format.
    `targets_op` preferred type is a np.ndarray, or Dense Qobj.

    When right multiplying bra type, <psi|A, for A acting on
    targets t1, t2..., the indices are defined as
    <..., t2, t1| A
    i.e. it is the same as (A^dag |psi>)^dag with targets [t1, t2, ...].

    If for left multiplication of `oper` type, one specifies a target larger
    than the sytem size, it will act on the relevant bra indices
    (e.g. target=n+k of an n qubit system will act from the right on qubit k).
    This is useful since one may occasionally prefer to perform a simultaneous
    left and right multiplication. This is the case, for example,
    when converting a superoperator to an operator, and
    vectorizing the density matrix [1, 2].
    Here we use the row-stacking convention |AXB>> = (A⊗B^T)|X>>, i.e.:
    |i1, ..., in><j1,..., jn| -> |i1, ..., in, j1, ..., jn>,
    where T is the transpose.
    If one wishes to left and right multiply a set of targets [t1, ...., t_nt],
    one should specify ``targets=[t1, ..., t_nt, t1 + n, ..., t_nt + n]``.
    Depending on the system, this can be more efficient than first applying
    a matrix on the left, and then separately on the right.

    References
    ----------
    .. [1] Y. Nambu, K. Nakamura,
    "On the Matrix Representation of Quantum Operations",
    arXiv:quant-ph/0504091 (2005).

    .. [2] J. Biamonte,
    "Lectures on Quantum Tensor Networks",
    arXiv:1912.10049 (2019)

    Examples
    --------
    1-local example on 10 qubits, pure state:

    >>> from qutip import *
    >>> from qutip.qip.operations.local_operations import local_multiply_dense
    >>> local_op = sigmay()
    >>> psi = tensor([ghz_state(1)] * 10)
    >>> # apply local_op on qubit 3
    >>> rho_out = local_multiply_dense(local_op, psi, targets=3)

    2-local example, unitary on density matrix:

    >>> from qutip import *
    >>> from qutip.qip.operations.local_operations import local_multiply_dense
    >>> local_op = tensor(sigmay(), sigmaz())
    >>> rho = ket2dm(tensor([ghz_state(1)] * 10))
    >>> # apply local_op on qubits 8, 3, from the left:
    >>> rho_out = local_multiply_dense(local_op, rho, targets=[8, 3])
    >>> # now apply hermitian conjugate on right
    >>> rho_out = local_multiply_dense(local_op.dag(), rho_out, targets=[8, 3], right=True)

    The same example, applying unitary via vectorization on density matrix:

    >>> from qutip import *
    >>> from qutip.qip.operations.local_operations import local_multiply_dense
    >>> n = 10
    >>> local_op = tensor(sigmay(), sigmaz())
    >>> # use convention |AXB>> = (A⊗B^T) |X>>
    >>> vector_op = tensor(local_op, local_op.conj())
    >>> rho = ket2dm(tensor([ghz_state(1)] * n))
    >>> rho_out = local_multiply_dense(vector_op, rho, targets=[8, 3, 8 + n, 3 + n])

    Dense Qobj example, using np.ndarray for local operator:

    >>> import numpy as np
    >>> from qutip import *
    >>> from qutip.qip.operations.local_operations import local_multiply_dense
    >>> from qutip.core import data
    >>> local_op = np.array([[0, -1j], [1j, 0]])
    >>> # uniform superposition on 10 qubits as Dense Qobj
    >>> rho = ket2dm(tensor([ghz_state(1)] * 10)).to(data.Dense)
    >>> rho_out = local_multiply_dense(local_op, rho, targets=3)
    >>> rho_out = local_multiply_dense(local_op.conj().T, rho_out, targets=3, right=True)

    """
    # targets as list
    targets = [targets] if isinstance(targets, int) else list(targets)
    if len(set(targets)) != len(targets):
        raise ValueError('targets must be unique')

    # get targets_op and vec_mat as NumPy ndarray
    # data.as_ndarray() is most efficient if given Dense type.
    # otherwise we just use Qobj.full()
    if isinstance(targets_op, Qobj):
        if isinstance(targets_op.data, data.Dense):
            op = targets_op.data.as_ndarray()
        else:
            op = np.asarray(targets_op.full())
    else:
        op = np.asarray(targets_op)

    if isinstance(vec_mat.data, data.Dense):
        arr_in = vec_mat.data.as_ndarray()
    else:
        arr_in = np.asarray(vec_mat.full())

    n = len(vec_mat.dims[0])  # number of tensor spaces
    # get vector dimensions of input
    if vec_mat.type == 'oper':
        vector_dims = vec_mat.dims[0] + vec_mat.dims[1]
        if right:
            # shift to act on bra indices
            targets = [t + n for t in targets]
    elif vec_mat.type == 'ket':
        vector_dims = vec_mat.dims[0]
        if right:
            raise ValueError('must specify left multiplication for ket type, '
                             'i.e. A|psi>')
    elif vec_mat.type == 'bra':
        # use the bra indices
        vector_dims = vec_mat.dims[1]
        if not right:
            raise ValueError('must specify right multiplication for bra type, '
                             'i.e. <psi|A')
    else:
        raise TypeError(f'invalid Qobj type: {vec_mat.type}. '
                        'should be oper, bra or ket')

    n_vec = len(vector_dims)  # vectorized number of spaces
    if not np.all([t in range(n_vec) for t in targets]):
        # the difference in the errors is for left multiplication,
        # one can specify targets up to the full vectorized system size
        e_l = f'Left multiplication: targets must be in [0,..., {n_vec - 1}]'
        e_r = f'Right multiplication: targets must be in [0,..., {n-1}]'
        raise ValueError(e_r if right else e_l)

    target_dims = [vector_dims[t] for t in targets]
    op_dim = targets_op.shape[0]
    protocol = _multiplication_protocol(n_vec, vec_mat.type, backend)

    # validate target operator.
    # needs to be square, with dims consistent with targets.
    if targets_op.shape != (op_dim, op_dim):
        raise ValueError(f'invalid op shape. Should be ({op_dim}, {op_dim}), '
                         f'but got {targets_op.shape}')
    if op_dim != np.prod(target_dims):
        raise ValueError(
            f'operator dim {op_dim} inconsistent with '
            f'target dim product{target_dims} = {np.prod(target_dims)}')

    arr_in = arr_in.reshape(vector_dims)  # new indexing: np_arr[i1][i2]...[jn]
    arr_out = None

    if protocol == 'einsum':
        # reshape op for use with einsum
        arr_out = _einsum_protocol(targets,
                                   op.reshape(target_dims + target_dims),
                                   arr_in, right)
    if protocol == 'vectorize' or arr_out is None:
        # None is used to catch the case where einsum can't be used
        arr_out = _reorder_vectorization_protocol(targets, op, arr_in, right)

    arr_out = arr_out.reshape(vec_mat.shape)

    if isinstance(vec_mat.data, data.Dense):
        # if given Dense, return as Dense
        return Qobj(data.dense.fast_from_numpy(arr_out), dims=vec_mat.dims,
                    type=vec_mat.type, copy=False)

    return Qobj(arr_out, dims=vec_mat.dims, type=vec_mat.type, copy=False)


def _multiplication_protocol(vector_size, qobj_type, backend) -> str:
    """
    Pick backend protocol for local_multiply.

    Parameters
    ----------
    vector_size: int
        Vecorized size of input (number tensor spaces after vectorization)
    qobj_type: str
        'oper', 'vec', 'bra' etc. as in Qobj
    backend: str or None
        String specifying desired backend, or None.
        If None, we estimate best backend from prior speed tests.
        Note, this should not be considered the most optimal for all situations.

    Returns
    -------
    str
        String specifying desired backend.

    """
    backends = ['einsum', 'vectorize', None]
    if isinstance(backend, str):
        backend = backend.lower()
    if backend not in backends:
        raise ValueError(f'invalid backend, should be one of {backends}')

    if backend is None:
        backend = 'einsum'
        if qobj_type == 'oper':
            if vector_size >= 8:
                backend = 'vectorize'
        else:
            if 15 <= vector_size <= 21:
                backend = 'vectorize'
    return backend


_EINSUM_AXES = ''.join([chr(ord('a') + i) for i in range(0, 26)])
_EINSUM_AXES += ''.join([chr(ord('A') + i) for i in range(0, 26)])


def _einsum_protocol(targets, local_op, full_op, right):
    """
    Use NumPy's einsum to compute matrix operation.

    Parameters
    ----------
    targets: list of int
        The target indices relating to full_op.
    local_op: np.ndarray
        This should have correct tensor shape for the einsum indices,
        e.g. if specifying an 1 qubit operation A=sum_{i,j} A_ij |i><j|,
        it should have shape (2, 2) (corresponding to (i, j)).
    full_op: np.ndarray
        This should have correct tensor shape for the einsum indices,
        e.g. if a 2 qubit density matrix rho=sum_{i,j,k,l} rho_ijkl |ij><kl|,
        it should have shape (2, 2, 2, 2) (corresponding to (i, j, k, l)).
    right: bool
        Specify left or right matrix multiplication, i.e., in the
        appropriate sense, local_op * full_op (right=False)
         or full_op * local_op (right=True)

    Returns
    -------
    np.ndarray or None
        Result of the matrix multiplication.
        If einsum can not be used, which is the case if there are too
        many tensor indices (> 52), None is returned.

    """

    n = len(full_op.shape)
    nt = len(targets)
    target_map = {t: i for i, t in enumerate(targets)}
    shift = 0 if right else nt
    sgn = 1 if right else -1

    if n + 2 * nt > len(_EINSUM_AXES):
        # this should actually never happen since the system would be massive,
        # nevertheless good to catch this.
        print('Not able to use einsum (system too large). '
              'Falling back to alternative protocol.')
        return None

    targ_axes = ''.join([_EINSUM_AXES[i] for i in range(2 * nt)])
    op_axes_temp = ''.join([_EINSUM_AXES[i + 2 * nt] for i in range(n)])
    op_axes = ''
    for i in range(n):
        if i in targets:
            op_axes += targ_axes[target_map[i] + shift]
        else:
            op_axes += op_axes_temp[i]

    out_axes = ''
    for c in op_axes:
        if c in targ_axes:
            out_axes += targ_axes[targ_axes.index(c) + sgn * nt]
        else:
            out_axes += c

    return np.einsum(targ_axes + ',' + op_axes + '->' + out_axes,
                     local_op, full_op)


def _reorder_vectorization_protocol(targets, local_op, full_op,
                                    right) -> np.ndarray:
    """
    Use a reorder and reshape protocol to compute matrix operation.

    Parameters
    ----------
    targets: list of int
        Indices of full_op to apply local_op on.
    local_op: np.ndarray
        The local operator to apply to full_op. Should be square matrix.
    full_op: np.ndarray
        The operator to which local_op is applied.
        This should have correct tensor shape for the subspace indices,
        e.g. if a 2 qubit density matrix rho=sum_{i,j,k,l} rho_ijkl |ij><kl|,
        it should have shape (2, 2, 2, 2) (corresponding to (i, j, k, l)).
    right: bool
        left or right matrix multiplication, i.e. in the appropriate sense,
        local_op * full_op (right=False) or full_op * local_op (right=True)

    Returns
    -------
    np.ndarray
        Result of the matrix multiplication

    Notes
    -----
    The reordering results in, e.g. for left multiplication, a vector
    |t1, t2, ..., t_nt, r_1, r_2, ...>
    where t_1, ..., t_nt are the targets, and r_1, ... the remaining indices.
    This is then reshaped to respect the shape of the local operator:
    |t1, ..., t_nt><r_1, r_2, ...|.
    Then one can just use regular matrix multiplication.
    Right multiplication is similar (swapping the bra and ket indices).

    This is a private function, and all dimension/shape checks are assumed
    to have already been performed.

    """
    vector_dims = full_op.shape
    vector_dim = np.prod(vector_dims)
    op_dim = local_op.shape[0]

    # desired shape of array to apply local op to
    # permutation moves targets to front (left) or back (right)
    if right:
        op_shape = (int(np.round(vector_dim / op_dim)), op_dim)
        perm = [n for n in range(len(vector_dims)) if
                          n not in targets] + targets
    else:
        op_shape = (op_dim, int(np.round(vector_dim / op_dim)))
        perm = targets + [n for n in range(len(vector_dims)) if
                          n not in targets]

    arr_out = np.transpose(full_op, perm).reshape(op_shape)
    arr_out = arr_out @ local_op if right else local_op @ arr_out

    # now permute back order
    arr_out = arr_out.reshape([vector_dims[p] for p in perm])
    return np.transpose(arr_out, np.argsort(perm))
