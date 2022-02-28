# -*- coding: utf-8 -*-
# The above line is so that UTF-8 comments won't break Py2.

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
"""
This module is a collection of random state and operator generators.
The sparsity of the ouput Qobj's is controlled by varing the
`density` parameter.
"""

__all__ = [
    'rand_herm',
    'rand_unitary', 'rand_unitary_haar',
    'rand_dm', 'rand_dm_ginibre', 'rand_dm_hs',
    'rand_stochastic',
    'rand_ket', 'rand_ket_haar',
    'rand_kraus_map',
    'rand_super_bcsz', 'rand_super'
]

import numbers

import numpy as np
import scipy.linalg
import scipy.sparse as sp

from . import Qobj, create, destroy, jmat, basis, to_super
from .core import data as _data
from .core.dimensions import Dimensions

_UNITS = np.array([1, 1j])


def rand_jacobi_rotation(A, *, seed=None):
    """Random Jacobi rotation of a matrix.

    Parameters
    ----------
    A : qutip.data.Data
        Matrix to rotate as a data layer object.

    seed : int32
        seed to reseed the random number generator.

    Returns
    -------
    qutip.data.Data
        Rotated sparse matrix.
    """
    if seed is not None:
        np.random.seed(seed=seed)
    if A.shape[0] != A.shape[1]:
        raise Exception('Input matrix must be square.')
    n = A.shape[0]
    angle = 2 * np.random.random() * np.pi
    a = np.sqrt(0.5) * np.exp(-1j * angle)
    b = np.conj(a)
    i = np.random.randint(n)
    j = i
    while i == j:
        j = np.random.randint(n)
    data = np.hstack((np.array([a, -b, a, b], dtype=complex),
                      np.ones(n - 2, dtype=complex)))
    diag = np.delete(np.arange(n), [i, j])
    rows = np.hstack(([i, i, j, j], diag))
    cols = np.hstack(([i, j, i, j], diag))
    R = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=complex)
    R = _data.create(R.tocsr())
    return _data.matmul(_data.matmul(R, A), R.adjoint())


def _randnz(shape, norm=np.sqrt(0.5), *, seed=None):
    """
    Returns an array of standard normal complex random variates.
    The Ginibre ensemble corresponds to setting ``norm = 1`` [Mis12]_.

    Parameters
    ----------
    shape : tuple
        Shape of the returned array of random variates.

    norm : float
        Scale of the returned random variates, or 'ginibre' to draw
        from the Ginibre ensemble.

    seed : int32
        seed to reseed the random number generator.
    """
    # This function is intended for internal use.
    if seed is not None:
        np.random.seed(seed=seed)
    if norm == 'ginibre':
        norm = 1
    return np.sum(np.random.randn(*(shape + (2,))) * _UNITS, axis=-1) * norm


def rand_herm(N, density=0.75, dims=None, pos_def=False,
              *, seed=None, dtype=_data.CSR):
    """Creates a random NxN sparse Hermitian quantum object.

    If 'N' is an integer, uses :math:`H=0.5*(X+X^{+})` where :math:`X` is
    a randomly generated quantum operator with a given `density`. Else uses
    complex Jacobi rotations when 'N' is given by an array.

    Parameters
    ----------
    N : int, list/ndarray
        If int, then shape of output operator. If list/ndarray then eigenvalues
        of generated operator.

    density : float
        Density between [0,1] of output Hermitian operator.

    dims : list
        Dimensions of quantum object.  Used for specifying
        tensor structure. Default is dims=[[N],[N]].

    pos_def : bool (default=False)
        Return a positive semi-definite matrix (by diagonal dominance).

    seed : int
        seed for the random number generator

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    oper : qobj
        NxN Hermitian quantum operator.

    Note
    ----
    If given a list/ndarray as input 'N', this function returns a
    random Hermitian object with eigenvalues given in the list/ndarray.
    This is accomplished via complex Jacobi rotations.  While this method
    is ~50% faster than the corresponding (real only) Matlab code, it should
    not be repeatedly used for generating matrices larger than ~1000x1000.

    """
    if seed is not None:
        np.random.seed(seed=seed)
    if isinstance(N, (np.ndarray, list)):
        M = _data.diag[_data.CSR](N, 0)
        N = len(N)
        if dims:
            _check_dims(dims, N, N)
        nvals = max([N**2 * density, 1])
        M = rand_jacobi_rotation(M)
        while _data.csr.nnz(M) < 0.95 * nvals:
            M = rand_jacobi_rotation(M)

    elif isinstance(N, numbers.Integral):
        N = int(N)
        if dims:
            _check_dims(dims, N, N)
        if density < 0.5:
            M = _rand_herm_sparse(N, density, pos_def)
        else:
            M = _rand_herm_dense(N, density, pos_def)

    else:
        raise TypeError('Input N must be an integer or array_like.')
    out = Qobj(M, dims=dims or [[N]]*2, type='oper',
               isherm=True, copy=False).to(dtype)
    if dtype:
        out = out.to(dtype)
    return out


def _rand_herm_sparse(N, density, pos_def):
    target = (1 - (1 - density)**0.5)
    num_elems = (N**2 - 0.666 * N) * target + 0.666 * N * density
    num_elems = max([num_elems, 1])
    num_elems = int(num_elems)
    data = (2 * np.random.rand(num_elems) - 1) + \
           (2 * np.random.rand(num_elems) - 1) * 1j
    row_idx, col_idx = zip(*[divmod(index, N) for index
                             in np.random.choice(N*N,
                                                 num_elems,
                                                 replace=False)])
    M = sp.coo_matrix((data, (row_idx,col_idx)),
                      dtype=complex, shape=(N,N)).tocsr()
    M = 0.5 * (M + M.conj().transpose())
    if pos_def:
        M.setdiag(np.abs(M.diagonal()) + np.sqrt(2) * N)
    M.sort_indices()
    return _data.create(M)


def _rand_herm_dense(N, density, pos_def):
    M = (2 * np.random.rand(N,N) - 1) + \
        (2 * np.random.rand(N,N) - 1) * 1j
    M = 0.5 * (M + M.conj().transpose())
    target = (1-(density)**0.5)
    num_remove = N * (N - 0.666) * target + 0.666 * N * (1 - density)
    num_remove = max([num_remove, 1])
    num_remove = int(num_remove)
    for row, col in [divmod(index, N)
                     for index in np.random.choice(N*N,
                                                   num_remove,
                                                   replace=False)]:
        M[col, row] = 0
        M[row, col] = 0
    if pos_def:
        np.fill_diagonal(M, np.abs(M.diagonal()) + np.sqrt(2) * N )
    return _data.create(M)



def rand_unitary(N, density=0.75, dims=None, *, seed=None, dtype=_data.Dense):
    r"""Creates a random NxN sparse unitary quantum object.

    Uses :math:`\exp(-iH)` where H is a randomly generated
    Hermitian operator.

    Parameters
    ----------
    N : int
        Shape of output quantum operator.

    density : float
        Density between [0,1] of output Unitary operator.

    dims : list
        Dimensions of quantum object.  Used for specifying
        tensor structure. Default is dims=[[N],[N]].

    seed : int
        seed for the random number generator

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    oper : qobj
        NxN Unitary quantum operator.

    """
    if dims:
        _check_dims(dims, N, N)
    return (-1.0j * rand_herm(N, density, dims=dims,
                              seed=seed, dtype=dtype)
            ).expm().to(dtype)


def rand_unitary_haar(N=2, dims=None, *, seed=None, dtype=_data.Dense):
    """
    Returns a Haar random unitary matrix of dimension
    ``dim``, using the algorithm of [Mez07]_.

    Parameters
    ----------
    N : int
        Dimension of the unitary to be returned.

    dims : list of lists of int, or None
        Dimensions of quantum object.  Used for specifying
        tensor structure. Default is dims=[[N],[N]].

    seed : int
        seed for the random number generator

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    U : Qobj
        Unitary of dims ``[[dim], [dim]]`` drawn from the Haar
        measure.
    """
    if dims is not None:
        _check_dims(dims, N, N)
    else:
        dims = [[N], [N]]

    # Mez01 STEP 1: Generate an N × N matrix Z of complex standard
    #               normal random variates.
    Z = _randnz((N, N), seed=seed)

    # Mez01 STEP 2: Find a QR decomposition Z = Q · R.
    Q, R = scipy.linalg.qr(Z)

    # Mez01 STEP 3: Create a diagonal matrix Lambda by rescaling
    #               the diagonal elements of R.
    Lambda = np.diag(R).copy()
    Lambda /= np.abs(Lambda)

    # Mez01 STEP 4: Note that R' := Λ¯¹ · R has real and
    #               strictly positive elements, such that
    #               Q' = Q · Λ is Haar distributed.
    # NOTE: Λ is a diagonal matrix, represented as a vector
    #       of the diagonal entries. Thus, the matrix dot product
    #       is represented nicely by the NumPy broadcasting of
    #       the *scalar* multiplication. In particular,
    #       Q · Λ = Q_ij Λ_jk = Q_ij δ_jk λ_k = Q_ij λ_j.
    #       As NumPy arrays, Q has shape (N, N) and
    #       Lambda has shape (N, ), such that the broadcasting
    #       represents precisely Q_ij λ_j.
    return Qobj(Q * Lambda, dims=dims,
                type='oper', isunitary=True, copy=False).to(dtype)


def rand_ket(N=0, density=1, dims=None, *, seed=None, dtype=_data.Dense):
    """Creates a random Nx1 sparse ket vector.

    Parameters
    ----------
    N : int
        Number of rows for output quantum operator.
        If None or 0, N is deduced from dims.

    density : float
        Density between [0,1] of output ket state.

    dims : list
        Dimensions of quantum object.  Used for specifying
        tensor structure. Default is dims=[[N],[1]].

    seed : int
        seed for the random number generator

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    oper : qobj
        Nx1 ket state quantum operator.

    """
    if seed is not None:
        np.random.seed(seed=seed)
    if N and dims:
        _check_dims(dims, N, 1)
    elif dims:
        N = np.prod(dims[0])
        _check_dims(dims, N, 1)
    else:
        dims = [[N], [1]]
    X = scipy.sparse.rand(N, 1, density, format='csr')
    while X.nnz == 0:
        # ensure that the ket is not all zeros.
        X = scipy.sparse.rand(N, 1, density+1/N, format='csr')
    X.data = X.data - 0.5
    Y = X.copy()
    Y.data = 1.0j * (np.random.random(len(X.data)) - 0.5)
    X = _data.csr.CSR(X + Y)
    return Qobj(_data.mul(X, 1 / _data.norm.l2(X)),
                dims=dims,
                copy=False,
                type='ket',
                isherm=False,
                isunitary=False).to(dtype)


def rand_ket_haar(N=2, dims=None, *, seed=None, dtype=_data.Dense):
    """
    Returns a Haar random pure state of dimension ``dim`` by
    applying a Haar random unitary to a fixed pure state.

    Parameters
    ----------
    N : int
        Dimension of the state vector to be returned.
        If None or 0, N is deduced from dims.

    dims : list of ints, or None
        Dimensions of the resultant quantum object.
        If None, [[N],[1]] is used.

    seed : int
        seed for the random number generator

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    psi : Qobj
        A random state vector drawn from the Haar measure.
    """
    if N and dims:
        _check_dims(dims, N, 1)
    elif dims:
        N = np.prod(dims[0])
        _check_dims(dims, N, 1)
    else:
        dims = [[N], [1]]
    U = rand_unitary_haar(N, seed=seed, dims=[dims[0], dims[0]])
    return (U @ basis(dims[0], [0]*len(dims[0]))).to(dtype)


def rand_dm(N, density=0.75, pure=False, dims=None, *,
            seed=None, dtype=_data.CSR):
    r"""Creates a random NxN density matrix.

    Parameters
    ----------
    N : int, ndarray, list
        If int, then shape of output operator. If list/ndarray then eigenvalues
        of generated density matrix.

    density : float
        Density between [0,1] of output density matrix.

    dims : list
        Dimensions of quantum object.  Used for specifying
        tensor structure. Default is dims=[[N],[N]].

    seed : int
        seed for the random number generator

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    oper : qobj
        NxN density matrix quantum operator.

    Notes
    -----
    For small density matrices., choosing a low density will result in an error
    as no diagonal elements will be generated such that :math:`Tr(\rho)=1`.
    """
    if seed is not None:
        np.random.seed(seed=seed)
    if isinstance(N, (np.ndarray, list)):
        if np.abs(np.sum(N)-1.0) > 1e-15:
            raise ValueError('Eigenvalues of a density matrix '
                             'must sum to one.')
        H = _data.diag(N, 0)
        N = len(N)
        if dims:
            _check_dims(dims, N, N)
        nvals = N**2 * density
        H = rand_jacobi_rotation(H)
        while _data.csr.nnz(H) < 0.95*nvals:
            H = rand_jacobi_rotation(H)
    elif isinstance(N, numbers.Integral):
        N = int(N)
        if dims:
            _check_dims(dims, N, N)
        if pure:
            dm_density = np.sqrt(density)
            psi = rand_ket(N, dm_density, dtype=dtype)
            H = psi.proj().data
        else:
            trace = 0
            tries = 0
            while trace == 0 and tries < 10:
                H = rand_herm(N, density, seed=seed, dtype=dtype)
                H = H.dag() @ H
                trace = H.tr()
                tries += 1
            if tries >= 10:
                raise ValueError(
                    "Requested density is too low to generate density matrix.")
            H /= trace
            H = H.data

    else:
        raise TypeError('Input N must be an integer or array_like.')
    return Qobj(H, dims=dims, type='oper', isherm=True, copy=False).to(dtype)


def rand_dm_ginibre(N=2, rank=None, dims=None, *, seed=None, dtype=_data.CSR):
    """
    Returns a Ginibre random density operator of dimension
    ``dim`` and rank ``rank`` by using the algorithm of
    [BCSZ08]_. If ``rank`` is `None`, a full-rank
    (Hilbert-Schmidt ensemble) random density operator will be
    returned.

    Parameters
    ----------
    N : int
        Dimension of the density operator to be returned.
    dims : list
        Dimensions of quantum object.  Used for specifying
        tensor structure. Default is dims=[[N],[N]].

    rank : int or None
        Rank of the sampled density operator. If None, a full-rank
        density operator is generated.

    seed : int
        seed for the random number generator

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    rho : Qobj
        An N × N density operator sampled from the Ginibre
        or Hilbert-Schmidt distribution.
    """
    if rank is None:
        rank = N
    if rank > N:
        raise ValueError("Rank cannot exceed dimension.")

    X = _randnz((N, rank), norm='ginibre', seed=seed)
    rho = np.dot(X, X.T.conj())
    rho /= np.trace(rho)

    return Qobj(rho, dims=dims, type='oper', isherm=True, copy=False).to(dtype)


def rand_dm_hs(N=2, dims=None, *, seed=None, dtype=_data.CSR):
    """
    Returns a Hilbert-Schmidt random density operator of dimension
    ``dim`` and rank ``rank`` by using the algorithm of
    [BCSZ08]_.

    Parameters
    ----------
    N : int
        Dimension of the density operator to be returned.

    dims : list
        Dimensions of quantum object.  Used for specifying
        tensor structure. Default is dims=[[N],[N]].

    seed : int
        seed for the random number generator

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    rho : Qobj
        A dim × dim density operator sampled from the Ginibre
        or Hilbert-Schmidt distribution.

    """
    return rand_dm_ginibre(N, rank=None, dims=dims, seed=seed, dtype=dtype)


def rand_kraus_map(N, dims=None, *, seed=None, dtype=_data.Dense):
    """
    Creates a random CPTP map on an N-dimensional Hilbert space in Kraus
    form.

    Parameters
    ----------
    N : int
        Length of input/output density matrix.

    dims : list
        Dimensions of quantum object.  Used for specifying
        tensor structure. Default is dims=[[N],[N]].

    seed : int
        seed for the random number generator

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    oper_list : list of qobj
        N^2 x N x N qobj operators.

    """
    if dims:
        _check_dims(dims, N, N)

    # Random unitary (Stinespring Dilation)
    big_unitary = rand_unitary(N ** 3, seed=seed, dtype=dtype).full()
    orthog_cols = np.array(big_unitary[:, :N])
    oper_list = np.reshape(orthog_cols, (N ** 2, N, N))
    return [Qobj(x, dims=dims, type='oper', copy=False).to(dtype)
            for x in oper_list]


def rand_super(N, dims=None, *, seed=None, dtype=_data.Dense):
    """
    Returns a randomly drawn superoperator acting on operators acting on
    N dimensions.

    Parameters
    ----------
    N : int
        Square root of the dimension of the superoperator to be returned.

    dims : list
        Dimensions of quantum object.  Used for specifying
        tensor structure. Default is dims=[[[N],[N]], [[N],[N]]].

    seed : int
        seed for the random number generator

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.
    """
    if dims is not None:
        # TODO: check!
        _check_dims(dims, N**2, N**2)
        pass
    else:
        dims = [[[N], [N]], [[N], [N]]]
    H = rand_herm(N, seed=seed, dtype=dtype)
    S = propagator(H, np.random.rand(), [
        create(N), destroy(N), jmat(float(N - 1) / 2.0, 'z')
    ])
    S.dims = dims
    return S.to(dtype)


def rand_super_bcsz(N=2, enforce_tp=True, rank=None, dims=None, *,
                    seed=None, dtype=_data.CSR):
    """
    Returns a random superoperator drawn from the Bruzda
    et al ensemble for CPTP maps [BCSZ08]_. Note that due to
    finite numerical precision, for ranks less than full-rank,
    zero eigenvalues may become slightly negative, such that the
    returned operator is not actually completely positive.

    Parameters
    ----------
    N : int
        Square root of the dimension of the superoperator to be returned.

    enforce_tp : bool
        If True, the trace-preserving condition of [BCSZ08]_ is enforced;
        otherwise only complete positivity is enforced.

    rank : int or None
        Rank of the sampled superoperator. If None, a full-rank
        superoperator is generated.

    dims : list
        Dimensions of quantum object.  Used for specifying
        tensor structure. Default is dims=[[[N],[N]], [[N],[N]]].

    seed : int
        seed for the random number generator

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    rho : Qobj
        A superoperator acting on vectorized dim × dim density operators,
        sampled from the BCSZ distribution.
    """
    if dims is not None:
        # TODO: check!
        pass
    else:
        dims = Dimensions([[[N], [N]], [[N], [N]]], rep='choi')

    if rank is None:
        rank = N**2
    if rank > N**2:
        raise ValueError("Rank cannot exceed superoperator dimension.")

    # We use mainly dense matrices here for speed in low
    # dimensions. In the future, it would likely be better to switch off
    # between sparse and dense matrices as the dimension grows.

    # We start with a Ginibre uniform matrix X of the appropriate rank,
    # and use it to construct a positive semidefinite matrix X X⁺.
    X = _randnz((N**2, rank), norm='ginibre', seed=seed)

    # Precompute X X⁺, as we'll need it in two different places.
    XXdag = np.dot(X, X.T.conj())
    tmp_dims = [[[N], [N]], [[N], [N]]]

    if enforce_tp:
        # We do the partial trace over the first index by using dense reshape
        # operations, so that we can avoid bouncing to a sparse representation
        # and back.
        Y = np.einsum('ijik->jk', XXdag.reshape((N, N, N, N)))

        # Now we have the matrix 𝟙 ⊗ Y^{-1/2}, which we can find by doing
        # the square root and the inverse separately. As a possible
        # improvement, iterative methods exist to find inverse square root
        # matrices directly, as this is important in statistics.
        Z = np.kron(
            np.eye(N),
            scipy.linalg.sqrtm(scipy.linalg.inv(Y))
        )

        # Finally, we dot everything together and pack it into a Qobj,
        # marking the dimensions as that of a type=super (that is,
        # with left and right compound indices, each representing
        # left and right indices on the underlying Hilbert space).
        D = Qobj(np.dot(Z, np.dot(XXdag, Z)), dims=tmp_dims, type='super')
    else:
        D = N * Qobj(XXdag / np.trace(XXdag), dims=tmp_dims, type='super')

    # Since [BCSZ08] gives a row-stacking Choi matrix, but QuTiP
    # expects a column-stacking Choi matrix, we must permute the indices.
    D = D.permute([[1], [0]])

    D.dims = dims

    return to_super(D).to(dtype)


def rand_stochastic(N, density=0.75, kind='left', dims=None,
                    *, seed=None, dtype=_data.CSR):
    """Generates a random stochastic matrix.

    Parameters
    ----------
    N : int
        Dimension of matrix.

    density : float
        Density between [0,1] of output density matrix.

    kind : str (Default = 'left')
        Generate 'left' or 'right' stochastic matrix.

    dims : list
        Dimensions of quantum object.  Used for specifying
        tensor structure. Default is dims=[[N],[N]].

    seed : int
        seed for the random number generator

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    oper : qobj
        Quantum operator form of stochastic matrix.
    """
    if seed is not None:
        np.random.seed(seed=seed)
    if dims:
        _check_dims(dims, N, N)
    num_elems = max([int(np.ceil(N*(N+1)*density)/2), N])
    data = np.random.rand(num_elems)
    # Ensure an element on every row and column
    row_idx = np.hstack([np.random.permutation(N),
                         np.random.choice(N, num_elems-N)])
    col_idx = np.hstack([np.random.permutation(N),
                         np.random.choice(N, num_elems-N)])
    M = sp.coo_matrix((data, (row_idx, col_idx)),
                      dtype=np.complex128, shape=(N, N)).tocsr()
    M = 0.5 * (M + M.conj().transpose())
    num_rows = M.indptr.shape[0] - 1
    for row in range(num_rows):
        row_start = M.indptr[row]
        row_end = M.indptr[row+1]
        row_sum = np.sum(M.data[row_start:row_end])
        M.data[row_start:row_end] /= row_sum
    if kind == 'left':
        M = M.transpose()
    if dims:
        return Qobj(M, dims=dims).to(dtype)
    else:
        return Qobj(M).to(dtype)


def _check_dims(dims, N1, N2):
    if len(dims) != 2:
        raise Exception("Qobj dimensions must be list of length 2.")
    if (not isinstance(dims[0], list)) or (not isinstance(dims[1], list)):
        raise TypeError(
            "Qobj dimension components must be lists. i.e. dims=[[N],[N]]")
    if np.prod(dims[0]) != N1 or np.prod(dims[1]) != N2:
        raise ValueError("Qobj dimensions must match matrix shape.")
    if len(dims[0]) != len(dims[1]):
        raise TypeError("Qobj dimension components must have same length.")

# TRAILING IMPORTS
# qutip.propagator depends on rand_dm, so we need to put this import last.
from qutip.solve.propagator import propagator
