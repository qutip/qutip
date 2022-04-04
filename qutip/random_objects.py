# -*- coding: utf-8 -*-
"""
This module is a collection of random state and operator generators.
The sparsity of the output Qobj's is controlled by varying the
`density` parameter.
"""

__all__ = [
    'rand_herm', 'rand_unitary', 'rand_ket', 'rand_dm',
    'rand_unitary_haar', 'rand_ket_haar', 'rand_dm_ginibre',
    'rand_dm_hs', 'rand_super_bcsz', 'rand_stochastic', 'rand_super'
]

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from qutip.qobj import Qobj
from qutip.operators import create, destroy, jmat
from qutip.states import basis
import qutip.superop_reps as sr


UNITS = np.array([1, 1j])


def rand_jacobi_rotation(A, seed=None):
    """Random Jacobi rotation of a sparse matrix.

    Parameters
    ----------
    A : spmatrix
        Input sparse matrix.

    Returns
    -------
    spmatrix
        Rotated sparse matrix.
    """
    if seed is not None:
        np.random.seed(seed=seed)
    if A.shape[0] != A.shape[1]:
        raise ValueError('Input matrix must be square.')
    n = A.shape[0]
    angle = 2*np.random.random()*np.pi
    a = 1.0/np.sqrt(2)*np.exp(-1j*angle)
    b = 1.0/np.sqrt(2)*np.exp(1j*angle)
    i = int(np.floor(np.random.random()*n))
    j = i
    while i == j:
        j = int(np.floor(np.random.random()*n))
    data = np.hstack((np.array([a, -b, a, b], dtype=complex),
                      np.ones(n-2, dtype=complex)))
    diag = np.delete(np.arange(n), [i, j])
    rows = np.hstack(([i, i, j, j], diag))
    cols = np.hstack(([i, j, i, j], diag))
    R = sp.coo_matrix(
        (data, (rows, cols)), shape=(n, n), dtype=complex,
    ).tocsr()
    A = R*A*R.conj().transpose()
    return A


def randnz(shape, norm=1 / np.sqrt(2), seed=None):
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
    """
    if seed is not None:
        np.random.seed(seed=seed)
    if norm == 'ginibre':
        norm = 1
    return np.sum(np.random.randn(*(shape + (2,))) * UNITS, axis=-1) * norm


def rand_herm(N=None, density=0.75, dims=None, pos_def=False, seed=None):
    """Creates a random NxN sparse Hermitian quantum object.

    If 'N' is an integer or None, uses :math:`H=0.5*(X+X^{+})` where :math:`X` is
    a randomly generated quantum operator with a given `density`. Else uses
    complex Jacobi rotations when 'N' is given by an array.

    Parameters
    ----------
    N : int, list/ndarray, optional
        If int, then shape of output operator. If list/ndarray then eigenvalues
        of generated operator.
    density : float (default=0.75)
        Density between [0,1] of output Hermitian operator.
    dims : list of lists of int, optional
        Dimensions of quantum object. Used for specifying
        tensor structure. Default is dims=[[N],[N]].
    pos_def : bool (default=False)
        Return a positive semi-definite matrix (by diagonal dominance).
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    oper : qobj
        NxN Hermitian quantum operator.

    Notes
    -----
    If given a list/ndarray as input 'N', this function returns a
    random Hermitian object with eigenvalues given in the list/ndarray.
    This is accomplished via complex Jacobi rotations.  While this method
    is ~50% faster than the corresponding (real only) Matlab code, it should
    not be repeatedly used for generating matrices larger than ~1000x1000.
    """
    if seed is not None:
        np.random.seed(seed=seed)
    if isinstance(N, (np.ndarray, list)):
        M = sp.diags(N, 0, dtype=complex, format='csr')
        N = len(N)
        if dims:
            _check_dims(dims, N, N)
        nvals = max([N**2 * density, 1])
        M = rand_jacobi_rotation(M)
        while M.nnz < 0.95 * nvals:
            M = rand_jacobi_rotation(M)
        M.sort_indices()
    elif isinstance(N, (int, np.int32, np.int64)):
        if dims:
            _check_dims(dims, N, N)
        else:
            dims = [[N], [N]]
        if density < 0.5:
            M = _rand_herm_sparse(N, density, pos_def)
        else:
            M = _rand_herm_dense(N, density, pos_def)
    elif N is None:
        if dims:
            N = np.prod(dims[0])
        else:
            raise ValueError('Specify either the number of rows of Hermitian'
                             ' operator (N) or dimensions of quantum object'
                             ' (dims).')
        if density < 0.5:
            M = _rand_herm_sparse(N, density, pos_def)
        else:
            M = _rand_herm_dense(N, density, pos_def)
    else:
        raise TypeError('Input N must be an integer or array_like.')
    return Qobj(M, dims=dims)


def _rand_herm_sparse(N, density, pos_def):
    target = (1-(1-density)**0.5)
    num_elems = (N**2 - 0.666 * N) * target + 0.666 * N * density
    num_elems = max([num_elems, 1])
    num_elems = int(num_elems)
    data = (2 * np.random.rand(num_elems) - 1) + \
           (2 * np.random.rand(num_elems) - 1) * 1j
    row_idx, col_idx = zip(*[
        divmod(index, N)
        for index in np.random.choice(N*N, num_elems, replace=False)
    ])
    M = sp.coo_matrix((data, (row_idx, col_idx)),
                      dtype=complex, shape=(N, N))
    M = 0.5 * (M + M.conj().transpose())
    if pos_def:
        M = M.tocoo()
        M.setdiag(np.abs(M.diagonal()) + np.sqrt(2)*N)
    M = M.tocsr()
    M.sort_indices()
    return M


def _rand_herm_dense(N, density, pos_def):
    M = (
        (2*np.random.rand(N, N) - 1)
        + 1j*(2*np.random.rand(N, N) - 1)
    )
    M = 0.5 * (M + M.conj().transpose())
    target = 1 - density**0.5
    num_remove = N * (N - 0.666) * target + 0.666 * N * (1 - density)
    num_remove = max([num_remove, 1])
    num_remove = int(num_remove)
    for index in np.random.choice(N*N, num_remove, replace=False):
        row, col = divmod(index, N)
        M[col, row] = 0
        M[row, col] = 0
    if pos_def:
        np.fill_diagonal(M, np.abs(M.diagonal()) + np.sqrt(2)*N)
    return M


def rand_unitary(N=None, density=0.75, dims=None, seed=None):
    r"""Creates a random NxN sparse unitary quantum object.

    Uses :math:`\exp(-iH)` where H is a randomly generated
    Hermitian operator.

    Parameters
    ----------
    N : int, optional
        Shape of output quantum operator.
    density : float
        Density between [0,1] of output Unitary operator.
    dims : list of lists of int, optional
        Dimensions of quantum object. Used for specifying
        tensor structure. Default is dims=[[N],[N]].
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    oper : qobj
        NxN Unitary quantum operator.

    """
    if N is None and dims is None:
        raise ValueError('Specify either the number of rows of unitary'
                         ' operator (N) or dimensions of quantum object'
                         ' (dims).')
    elif N is None and dims:
        N = np.prod(dims[0])
    elif N and dims is None:
        dims = [[N], [N]]
    elif N and dims:
        _check_dims(dims, N, N)

    U = (-1.0j * rand_herm(N, density, seed=seed)).expm()
    U.data.sort_indices()
    return Qobj(U, dims=dims, shape=[N, N])


def rand_unitary_haar(N=None, dims=None, seed=None):
    """
    Returns a Haar random unitary matrix of dimension
    ``dim``, using the algorithm of [Mez07]_.

    Parameters
    ----------
    N : int
        Dimension of the unitary to be returned.
    dims : list of lists of int, optional
        Dimensions of quantum object. Used for specifying
        tensor structure. Default is dims=[[N],[N]].
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    U : Qobj
        Unitary of dims ``[[dim], [dim]]`` drawn from the Haar
        measure.
    """
    if N is None and dims is None:
        raise ValueError('Specify either the number of rows of unitary'
                         ' operator (N) or dimensions of quantum object'
                         ' (dims).')
    elif N is None and dims:
        N = np.prod(dims[0])
    elif N and dims is None:
        dims = [[N], [N]]
    elif N and dims:
        _check_dims(dims, N, N)

    # Mez01 STEP 1: Generate an N × N matrix Z of complex standard
    #               normal random variates.
    Z = randnz((N, N), seed=seed)

    # Mez01 STEP 2: Find a QR decomposition Z = Q · R.
    Q, R = la.qr(Z)

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
    U = Qobj(Q * Lambda)
    U.dims = dims
    return U


def rand_ket(N=None, density=1, dims=None, seed=None):
    """Creates a random Nx1 sparse ket vector.

    Parameters
    ----------
    N : int, optional
        Number of rows for output state vector. If None, N is deduced
         from dims.
    density : float
        Density between [0,1] of output ket state.
    dims : list of lists of int, optional
        Dimensions of quantum object. Used for specifying tensor
        structure. If None, dims = [[N], [1]].
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    oper : qobj
        Nx1 ket quantum state vector.

    Raises
    -------
    ValueError
        If neither `N` nor `dims` are specified.
    """
    if seed is not None:
        np.random.seed(seed=seed)

    if N is None and dims is None:
        raise ValueError('Specify either the number of rows of state vector'
                         ' (N) or dimensions of quantum object (dims).')
    elif N is None and dims:
        N = np.prod(dims[0])
    elif N and dims is None:
        dims = [[N], [1]]
    elif N and dims:
        _check_dims(dims, N, 1)

    X = sp.rand(N, 1, density, format='csr')
    while X.nnz == 0:
        # ensure that the ket is not all zeros.
        X = sp.rand(N, 1, density+1/N, format='csr')
    X.data = X.data - 0.5
    Y = X.copy()
    Y.data = 1.0j * (np.random.random(len(X.data)) - 0.5)
    X = X + Y
    X.sort_indices()
    X = Qobj(X)
    return Qobj(X / X.norm(), dims=dims)


def rand_ket_haar(N=None, dims=None, seed=None):
    """
    Returns a Haar random pure state of dimension ``dim`` by
    applying a Haar random unitary to a fixed pure state.

    Parameters
    ----------
    N : int, optional
        Number of rows for output state vector. If None, N is deduced
        from dims.
    dims : list of lists of int, optional
        Dimensions of quantum object. Used for specifying tensor
        structure. If None, dims = [[N], [1]].
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    psi : Qobj
        A random state vector drawn from the Haar measure.

    Raises
    -------
    ValueError
        If neither `N` nor `dims` are specified.
    """
    if N is None and dims is None:
        raise ValueError('Specify either the number of rows of state vector'
                         ' (N) or dimensions of quantum object (dims).')
    elif N is None and dims:
        N = np.prod(dims[0])
    elif N and dims is None:
        dims = [[N], [1]]
    elif N and dims:
        _check_dims(dims, N, 1)

    psi = rand_unitary_haar(N, seed=seed) * basis(N, 0)
    psi.dims = dims
    return psi


def rand_dm(N=None, density=0.75, pure=False, dims=None, seed=None):
    r"""Creates a random NxN density matrix.

    Parameters
    ----------
    N : int, ndarray, list, optional
        If int, then shape of output operator. If list/ndarray then eigenvalues
        of generated density matrix.
    density : float (default=0.75)
        Density between [0,1] of output density matrix.
    dims : list of lists of int, optional
        Dimensions of quantum object. Used for specifying
        tensor structure. Default is dims=[[N],[N]].
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    oper : qobj
        NxN density matrix quantum operator.

    Notes
    -----
    For small density matrices., choosing a low density will result in an error
    as no diagonal elements will be generated such that :math:`Tr(\rho)=1`.
    """
    if isinstance(N, (np.ndarray, list)):
        if np.abs(np.sum(N)-1.0) > 1e-15:
            raise ValueError('Eigenvalues of a density matrix '
                             'must sum to one.')
        H = sp.diags(N, 0, dtype=complex, format='csr')
        N = len(N)
        if dims:
            _check_dims(dims, N, N)
        nvals = N**2*density
        H = rand_jacobi_rotation(H, seed=seed)
        while H.nnz < 0.95*nvals:
            H = rand_jacobi_rotation(H)
        H.sort_indices()
    elif isinstance(N, (int, np.int32, np.int64)) or N is None:
        if N is None and dims is None:
            raise ValueError('Specify either the number of rows of density'
                             ' operator (N) or dimensions of quantum object'
                             ' (dims).')
        elif dims and N is None:
            N = np.prod(dims[0])
        elif dims and N:
            _check_dims(dims, N, N)
        if pure:
            dm_density = np.sqrt(density)
            psi = rand_ket(N, dm_density, seed=seed)
            H = psi * psi.dag()
            H.data.sort_indices()
        else:
            non_zero = 0
            tries = 0
            while non_zero == 0 and tries < 10:
                H = rand_herm(N, density, seed=seed)
                H = H.dag() * H
                non_zero = H.tr()
                tries += 1
            if tries >= 10:
                raise ValueError(
                    "Requested density is too low to generate density matrix.")
            H = H / H.tr()
            H.data.sort_indices()
    else:
        raise TypeError('Input N must be an integer or array_like.')
    return Qobj(H, dims=dims)


def rand_dm_ginibre(N=None, rank=None, dims=None, seed=None):
    """
    Returns a Ginibre random density operator.

    The operator has dimension ``dim`` and rank ``rank`` and is
    obtained by using the algorithm of [BCSZ08]_. If ``rank`` is
    `None`, a full-rank (Hilbert-Schmidt ensemble) random density
    operator will be returned.

    Parameters
    ----------
    N : int, optional
        Dimension of the density operator to be returned. If None, N is
        deduced from dims.
    rank : int or None, optional
        Rank of the sampled density operator. If None, a full-rank
        density operator is generated.
    dims : list of lists of int, optional
        Dimensions of quantum object. Used for specifying tensor
        structure. If None, dims = [[N], [N]].
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    rho : Qobj
        An N × N density operator sampled from the Ginibre
        or Hilbert-Schmidt distribution.

    Raises
    -------
    ValueError
        If neither `N` nor `dims` are specified.
    """
    if N is None and dims is None:
        raise ValueError('Specify either the number of rows of density'
                         ' operator (N) or dimensions of quantum object'
                         ' (dims).')
    elif N is None and dims:
        N = np.prod(dims[0])
    elif N and dims is None:
        dims = [[N], [N]]
    elif N and dims:
        _check_dims(dims, N, N)

    if rank is None:
        rank = N
    if rank > N:
        raise ValueError("Rank cannot exceed dimension.")

    X = randnz((N, rank), norm='ginibre', seed=seed)
    rho = np.dot(X, X.T.conj())
    rho /= np.trace(rho)

    return Qobj(rho, dims=dims)


def rand_dm_hs(N=None, dims=None, seed=None):
    """
    Returns a Hilbert-Schmidt random density operator.

    The operator has dimensions ``dims`` and rank ``rank`` and
    is obtained using the algorithm of [BCSZ08]_.


    Parameters
    ----------
    N : int, optional
        Dimension of the density operator to be returned. If None, N is
        deduced from dims.
    dims : list of lists of int, optional
        Dimensions of quantum object. Used for specifying tensor
        structure. If None, dims = [[N], [N]].
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    rho : Qobj
        A dim × dim density operator sampled from the Ginibre
        or Hilbert-Schmidt distribution.

    """
    return rand_dm_ginibre(N, rank=None, dims=dims, seed=seed)


def rand_kraus_map(N=None, dims=None, seed=None):
    """
    Creates a random CPTP map on an N-dimensional Hilbert space in Kraus
    form.

    Parameters
    ----------
    N : int, optional
        Length of input/output density matrix.
    dims : list of lists of int, optional
        Dimensions of quantum object. Used for specifying
        tensor structure. Default is dims=[[N],[N]].
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    oper_list : list of qobj
        N^2 x N x N qobj operators.

    """
    if N is None and dims is None:
        raise ValueError('Specify either the number of rows of'
                         ' operator (N) or dimensions of quantum object'
                         ' (dims).')
    elif N is None and dims:
        N = np.prod(dims[0])
    elif N and dims is None:
        dims = [[N], [N]]
    elif N and dims:
        _check_dims(dims, N, N)

    # Random unitary (Stinespring Dilation)
    orthog_cols = rand_unitary(N ** 3, seed=seed).full()[:, :N]
    oper_list = np.reshape(orthog_cols, (N ** 2, N, N))
    return list(map(lambda x: Qobj(inpt=x, dims=dims), oper_list))


def rand_super(N=5, dims=None, seed=None):
    """
    Returns a randomly drawn superoperator acting on operators acting on
    N dimensions.

    Parameters
    ----------
    N : int
        Square root of the dimension of the superoperator to be returned.
    dims : list of lists of int, optional
        Dimensions of quantum object. Used for specifying
        tensor structure. Default is dims=[[[N],[N]], [[N],[N]]].
    seed : int, optional
        Seed for the random number generator.
    """
    from .propagator import propagator
    if dims is not None:
        # TODO: check!
        pass
    else:
        dims = [[[N], [N]], [[N], [N]]]
    H = rand_herm(N, seed=seed)
    S = propagator(H, np.random.rand(), [
        create(N), destroy(N), jmat(float(N - 1) / 2.0, 'z')
    ])
    S.dims = dims
    return S


def rand_super_bcsz(N=2, enforce_tp=True, rank=None, dims=None, seed=None):
    """
    Returns a random superoperator drawn from the Bruzda
    et al. ensemble for CPTP maps [BCSZ08]_. Note that due to
    finite numerical precision, for ranks less than full-rank,
    zero eigenvalues may become slightly negative, such that the
    returned operator is not actually completely positive.


    Parameters
    ----------
    N : int, optional
        Square root of the dimension of the superoperator to be returned.
    enforce_tp : bool
        If True, the trace-preserving condition of [BCSZ08]_ is enforced;
        otherwise only complete positivity is enforced.
    rank : int or None
        Rank of the sampled superoperator. If None, a full-rank
        superoperator is generated.
    dims : list of lists of int, optional
        Dimensions of quantum object. Used for specifying
        tensor structure. Default is dims=[[[N],[N]], [[N],[N]]].
    seed : int, optional
        Seed for the random number generator.

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
        dims = [[[N], [N]], [[N], [N]]]

    if rank is None:
        rank = N**2
    if rank > N**2:
        raise ValueError("Rank cannot exceed superoperator dimension.")

    # We use mainly dense matrices here for speed in low
    # dimensions. In the future, it would likely be better to switch off
    # between sparse and dense matrices as the dimension grows.

    # We start with a Ginibre uniform matrix X of the appropriate rank,
    # and use it to construct a positive semidefinite matrix X X⁺.
    X = randnz((N**2, rank), norm='ginibre', seed=seed)

    # Precompute X X⁺, as we'll need it in two different places.
    XXdag = np.dot(X, X.T.conj())

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
            la.sqrtm(la.inv(Y))
        )

        # Finally, we dot everything together and pack it into a Qobj,
        # marking the dimensions as that of a type=super (that is,
        # with left and right compound indices, each representing
        # left and right indices on the underlying Hilbert space).
        D = Qobj(np.dot(Z, np.dot(XXdag, Z)))
    else:
        D = N * Qobj(XXdag / np.trace(XXdag))

    D.dims = [
        # Left dims
        [[N], [N]],
        # Right dims
        [[N], [N]]
    ]

    # Since [BCSZ08] gives a row-stacking Choi matrix, but QuTiP
    # expects a column-stacking Choi matrix, we must permute the indices.
    D = D.permute([[1], [0]])

    D.dims = dims

    # Mark that we've made a Choi matrix.
    D.superrep = 'choi'

    return sr.to_super(D)


def rand_stochastic(N=None, density=0.75, kind='left', dims=None, seed=None):
    """Generates a random stochastic matrix.

    Parameters
    ----------
    N : int, optional
        Dimension of matrix.
    density : float
        Density between [0,1] of output density matrix.
    kind : str (Default = 'left')
        Generate 'left' or 'right' stochastic matrix.
    dims : list of lists of int, optional
        Dimensions of quantum object. Used for specifying
        tensor structure. Default is dims=[[N],[N]].
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    oper : qobj
        Quantum operator form of stochastic matrix.
    """
    if N is None and dims is None:
        raise ValueError('Specify either the number of rows of matrix'
                         ' (N) or dimensions of quantum object'
                         ' (dims).')
    elif N is None and dims:
        N = np.prod(dims[0])
    elif N and dims is None:
        dims = [[N], [N]]
    elif N and dims:
        _check_dims(dims, N, N)

    if seed is not None:
        np.random.seed(seed=seed)

    num_elems = max([int(np.ceil(N*(N+1)*density)/2), N])
    data = np.random.rand(num_elems)
    # Ensure an element on every row and column
    row_idx = np.hstack([np.random.permutation(N),
                         np.random.choice(N, num_elems-N)])
    col_idx = np.hstack([np.random.permutation(N),
                         np.random.choice(N, num_elems-N)])
    M = sp.coo_matrix((data, (row_idx, col_idx)),
                      dtype=float, shape=(N, N)).tocsr()
    M = 0.5 * (M + M.conj().transpose())
    num_rows = M.indptr.shape[0]-1
    for row in range(num_rows):
        row_start = M.indptr[row]
        row_end = M.indptr[row+1]
        row_sum = np.sum(M.data[row_start:row_end])
        M.data[row_start:row_end] /= row_sum
    if kind == 'left':
        M = M.transpose()
    return Qobj(M, dims=dims, shape=(N, N))


def _check_dims(dims, N1, N2):

    if len(dims) != 2:
        raise Exception("Qobj dimensions must be list of length 2.")
    if (not isinstance(dims[0], list)) or (not isinstance(dims[1], list)):
        raise TypeError(
            "Qobj dimension components must be lists. i.e. dims=[[N],[N]].")
    if np.prod(dims[0]) != N1 or np.prod(dims[1]) != N2:
        raise ValueError("Qobj dimensions must match matrix shape.")
    if len(dims[0]) != len(dims[1]):
        raise TypeError("Qobj dimension components must have same length.")
