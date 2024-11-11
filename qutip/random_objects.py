"""
This module is a collection of random state and operator generators.
"""

__all__ = [
    'rand_herm',
    'rand_unitary',
    'rand_dm',
    'rand_stochastic',
    'rand_ket',
    'rand_kraus_map',
    'rand_super',
    "rand_super_bcsz",
]

import numbers

import numpy as np
from numpy.random import Generator, SeedSequence, default_rng
import scipy.linalg
import scipy.sparse as sp
from typing import Literal, Sequence

from . import (Qobj, create, destroy, jmat, basis,
               to_super, to_choi, to_chi, to_kraus, to_stinespring)
from .core import data as _data
from .core.dimensions import flatten, Dimensions, Space
from . import settings
from .typing import SpaceLike, LayerType


_RAND = default_rng()
_UNITS = np.array([1, 1j])


def _implicit_tensor_dimensions(dimensions, superoper=False):
    """
    Total flattened size and operator dimensions for operator creation routines
    that automatically perform tensor products.

    Parameters
    ----------
    dimensions : (int) or (list of int) or (list of list of int)
        First dimension of an operator which can create an implicit tensor
        product.  If the type is `int`, it is promoted first to `[dimensions]`.
        From there, it should be one of the two-elements `dims` parameter of a
        `qutip.Qobj` representing an `oper` or `super`, with possible tensor
        products.

    Returns
    -------
    size : int
        Dimension of backing matrix required to represent operator.
    dimensions : list
        Dimension list in the form required by ``Qobj`` creation.
    """
    if isinstance(dimensions, Space):
        dimensions = dimensions.as_list()
    if not isinstance(dimensions, list):
        dimensions = [dimensions]
    flat = flatten(dimensions)
    if not all(isinstance(x, numbers.Integral) and x >= 0 for x in flat):
        raise ValueError("All dimensions must be integers >= 0")
    N = np.prod(flat)
    if superoper:
        if isinstance(dimensions[0], numbers.Integral):
            dimensions = [dimensions, dimensions]
        else:
            N = int(N**0.5)
    return N, [dimensions, dimensions]


def _get_generator(seed):
    """
    Obtain a random generator from a seed or generator.

    Parameters
    ----------
    seed: int, SeedSequence, Generator, NoneType
        Seed to create the generator. If it's already a generator return it.
        When ``None`` is suplied, a default generator is provided.
    """
    if seed is None:
        gen = _RAND
    elif isinstance(seed, Generator):
        gen = seed
    else:
        gen = default_rng(seed)
    return gen


def _randnz(shape, generator, norm=np.sqrt(0.5)):
    """
    Returns an array of standard normal complex random variates.
    The Ginibre ensemble corresponds to setting ``norm = 1`` [Mis12]_.

    Parameters
    ----------
    shape : tuple
        Shape of the returned array of random variates.

    generator : Generator
        Random number generator.

    norm : float
        Scale of the returned random variates, or 'ginibre' to draw
        from the Ginibre ensemble.
    """
    # This function is intended for internal use.
    if norm == 'ginibre':
        norm = 1
    return np.sum(generator.normal(size=(shape + (2,))) * _UNITS, axis=-1) * norm


def _rand_jacobi_rotation(A, generator):
    """Random Jacobi rotation of a matrix.

    Parameters
    ----------
    A : qutip.data.Data
        Matrix to rotate as a data layer object.

    generator : numpy.random.generator
        Random number generator.

    Returns
    -------
    qutip.data.Data
        Rotated sparse matrix.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError('Input matrix must be square.')
    n = A.shape[0]
    angle = 2 * generator.random() * np.pi
    a = np.sqrt(0.5) * np.exp(-1j * angle)
    b = np.conj(a)
    i = generator.integers(n)
    j = i
    while i == j:
        j = generator.integers(n)
    data = np.hstack((np.array([a, -b, a, b], dtype=complex),
                      np.ones(n - 2, dtype=complex)))
    diag = np.delete(np.arange(n), [i, j])
    rows = np.hstack(([i, i, j, j], diag))
    cols = np.hstack(([i, j, i, j], diag))
    R = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=complex)
    R = _data.create(R.tocsr())
    return _data.matmul(_data.matmul(R, A), R.adjoint())


def _get_block_sizes(N, density, generator):
    """
    Obtain a list of matrix block sizes in such a way that an NxN matrix
    composed of full matrices of these sizes along the diagonal would be of
    desired density.
    """
    if density <= 0:
        return [1] * N
    elif density >= 1:
        return [N]
    max_block_size = int((N**2 * density)**0.5)
    min_block_size = 1
    if density >= 0.5:
        min_block_size = int(N / 2 * (1 + (2 * density - 1)**0.5))

    if min_block_size >= max_block_size:
        M = min(min_block_size, max_block_size)
    else:
        M = generator.integers(min_block_size, max_block_size) + 1
    M += generator.integers(-1, 2)
    M = min(max(M, 1), N)

    other_blocks = []
    if M < N:
        other_N = N - M
        other_density = (density * N**2 - M**2) / other_N**2
        other_blocks = _get_block_sizes(other_N, other_density, generator)

    return [M] + other_blocks


def _merge_shuffle_blocks(blocks, generator):
    """
    For a list of block, merge them in one matrix along the diagonal and
    shuffle the rows and columns.
    """
    N = sum(block.shape[0] for block in blocks)
    D = sum(block.shape[0]**2 for block in blocks) / N**2
    idx = generator.permutation(N)
    end = 0
    if D < 0.1:
        # TODO: coo_array from sicpy 1.8 would allow to simplify this.
        row = []
        col = []
        data = []
        for block in blocks:
            start = end
            end = end + block.shape[0]
            brow, bcol = np.meshgrid(idx[start:end], idx[start:end])
            row.append(brow.ravel())
            col.append(bcol.ravel())
            data.append(block.ravel())
        data = np.hstack(data)
        row = np.hstack(row)
        col = np.hstack(col)
        matrix = sp.coo_matrix((data, (row, col)), shape=(N, N))
    else:
        matrix = np.zeros((N,N), dtype=complex)
        for block in blocks:
            start = end
            end = end + block.shape[0]
            brow, bcol = np.meshgrid(idx[start:end], idx[start:end])
            matrix[brow, bcol] = block
    return _data.create(matrix, copy=False)


def rand_herm(
    dimensions: SpaceLike,
    density: float = 0.30,
    distribution: Literal["fill", "pos_def", "eigen"] = "fill",
    *,
    eigenvalues: Sequence[float] = (),
    seed: int | SeedSequence | Generator = None,
    dtype: LayerType = None,
):
    """Creates a random sparse Hermitian quantum object.

    Parameters
    ----------
    dimensions : (int) or (list of int) or (list of list of int)
        Dimension of Hilbert space. If provided as a list of ints, then the
        dimension is the product over this list, but the ``dims`` property of
        the new Qobj are set to this list.  This can produce either `oper` or
        `super` depending on the passed `dimensions`.

    density : float, default: 0.30
        Density between [0,1] of output Hermitian operator.

    distribution : str {"fill", "pos_def", "eigen"}, default: "fill"
        Method used to obtain the density matrices.

        - "fill" : Uses :math:`H=0.5*(X+X^{+})` where :math:`X` is a randomly
          generated quantum operator with elements uniformly distributed
          between ``[-1, 1] + [-1j, 1j]``.
        - "eigen" : A density matrix with the given ``eigenvalues``. It uses
          random complex Jacobi rotations to shuffle the operator.
        - "pos_def" : Return a positive semi-definite matrix by diagonal
          dominance.

    eigenvalues : array_like, optional
        Eigenvalues of the output Hermitian matrix. The len must match the
        shape of the matrix.

    seed : int, SeedSequence, Generator, optional
        Seed to create the random number generator or a pre prepared
        generator. When none is suplied, a default generator is used.

    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
        accepted.

    Returns
    -------
    oper : :obj:`.Qobj`
        Hermitian quantum operator.

    Notes
    -----
    If given a list of eigenvalues the object is created using complex Jacobi
    rotations.  While this method is fast for small matrices, it should not be
    repeatedly used for generating matrices larger than ~1000x1000.
    """
    N, dims = _implicit_tensor_dimensions(dimensions)
    generator = _get_generator(seed)
    if distribution not in ["eigen", "fill", "pos_def"]:
        raise ValueError("distribution must be one of {'eigen', 'fill', "
                         "'pos_def'}")

    if distribution == "eigen":
        if N != len(eigenvalues):
            raise ValueError("The number of eigenvalues does not match the "
                             "desired shape.")
        out = _data.diag[_data.CSR](eigenvalues, 0)
        nvals = max([N**2 * density, 1])
        out = _rand_jacobi_rotation(out, generator)
        while _data.csr.nnz(out) < 0.95 * nvals:
            out = _rand_jacobi_rotation(out, generator)
        out = Qobj(out, dims=dims, isherm=True, copy=False)
        dtype = dtype or settings.core["default_dtype"] or _data.CSR

    else:
        pos_def = distribution == "pos_def"
        if density < 0.5:
            M = _rand_herm_sparse(N, density, pos_def, generator)
            dtype = dtype or settings.core["default_dtype"] or _data.CSR
        else:
            M = _rand_herm_dense(N, density, pos_def, generator)
            dtype = dtype or settings.core["default_dtype"] or _data.Dense

        out = Qobj(M, dims=dims, isherm=True, copy=False)

    return out.to(dtype)


def _rand_herm_sparse(N, density, pos_def, generator):
    target = (1 - (1 - density)**0.5)
    num_elems = (N**2 - 0.666 * N) * target + 0.666 * N * density
    num_elems = max([num_elems, 1])
    num_elems = int(num_elems)
    data = (2 * generator.random(num_elems) - 1) + \
           (2 * generator.random(num_elems) - 1) * 1j
    row_idx, col_idx = zip(*[
        divmod(index, N)
        for index in generator.choice(N*N, num_elems, replace=False)
    ])
    M = sp.coo_matrix((data, (row_idx, col_idx)),
                      dtype=complex, shape=(N, N))
    M = 0.5 * (M + M.conj().transpose())
    M.sort_indices()
    rand_mat = _data.create(M)
    if pos_def:
        rand_mat = _data.add(
            rand_mat,
            _data.diag(np.ones(N, dtype=complex)),
            np.sqrt(2) * N + 1
        )
    return rand_mat


def _rand_herm_dense(N, density, pos_def, generator):
    M = (
        (2*generator.random((N, N)) - 1)
        + 1j*(2*generator.random((N, N)) - 1)
    )
    M = 0.5 * (M + M.conj().transpose())
    target = 1 - density**0.5
    num_remove = N * (N - 0.666) * target + 0.666 * N * (1 - density)
    num_remove = max([num_remove, 1])
    num_remove = int(num_remove)
    for index in generator.choice(N*N, num_remove, replace=False):
        row, col = divmod(index, N)
        M[col, row] = 0
        M[row, col] = 0
    if pos_def:
        np.fill_diagonal(M, np.abs(M.diagonal()) + np.sqrt(2) * N )
    return _data.create(M)


def rand_unitary(
    dimensions: SpaceLike,
    density: float = 1,
    distribution: Literal["haar", "exp"] = "haar",
    *,
    seed: int | SeedSequence | Generator = None,
    dtype: LayerType = None,
):
    r"""Creates a random sparse unitary quantum object.

    Parameters
    ----------
    dimensions : (int) or (list of int) or (list of list of int)
        Dimension of Hilbert space. If provided as a list of ints, then the
        dimension is the product over this list, but the ``dims`` property of
        the new Qobj are set to this list.  This can produce either `oper` or
        `super` depending on the passed `dimensions`.

    density : float, default: 1
        Density between [0,1] of output unitary operator.

    distribution : str {"haar", "exp"}, default: "haar"
        Method used to obtain the unitary matrices.

        - haar : Haar random unitary matrix using the algorithm of [Mez07]_.
        - exp : Uses :math:`\exp(-iH)`, where H is a randomly generated
          Hermitian operator.

    seed : int, SeedSequence, Generator, optional
        Seed to create the random number generator or a pre prepared
        generator. When none is suplied, a default generator is used.

    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
        accepted.

    Returns
    -------
    oper : qobj
        Unitary quantum operator.
    """
    dtype = dtype or settings.core["default_dtype"] or _data.Dense
    N, dims = _implicit_tensor_dimensions(dimensions)
    if distribution not in ["haar", "exp"]:
        raise ValueError("distribution must be one of {'haar', 'exp'}")
    generator = _get_generator(seed)
    block_sizes = _get_block_sizes(N, density, generator)

    blocks = []
    for block in block_sizes:
        if distribution == "haar":
            mat = _rand_unitary_haar(block, generator)
        elif distribution == "exp":
            mat = scipy.linalg.expm(
                -1.0j *
                _rand_herm_dense(block, 1, False, generator).as_ndarray()
            )
        blocks.append(mat)

    merged = _merge_shuffle_blocks(blocks, generator)

    mat = Qobj(merged, dims=dims, isunitary=True, copy=False)
    return mat.to(dtype)


def _rand_unitary_haar(N, generator):
    """
    Returns a Haar random unitary matrix of dimension
    ``dim``, using the algorithm of [Mez07]_.

    Parameters
    ----------
    N : int
        Dimension of the unitary to be returned.

    generator : numpy.random.generator
        Random number generator.

    Returns
    -------
    U : qutip.core.data.Dense
        Unitary of shape ``(N, N)`` drawn from the Haar measure.
    """
    # Mez01 STEP 1: Generate an N × N matrix Z of complex standard
    #               normal random variates.
    Z = _randnz((N, N), generator)

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
    return Q * Lambda


def rand_ket(
    dimensions: SpaceLike,
    density: float = 1,
    distribution: Literal["haar", "fill"] = "haar",
    *,
    seed: int | SeedSequence | Generator = None,
    dtype: LayerType = None,
):
    """Creates a random ket vector.

    Parameters
    ----------
    dimensions : (int) or (list of int) or (list of list of int)
        Dimension of Hilbert space. If provided as a list of ints, then the
        dimension is the product over this list, but the ``dims`` property of
        the new Qobj are set to this list.  This can produce either `oper` or
        `super` depending on the passed `dimensions`.

    density : float, default: 1
        Density between [0,1] of output ket state when using the ``fill``
        method.

    distribution : str {"haar", "fill"}, default: "haar"
        Method used to obtain the kets.

        - haar : Haar random pure state obtained by applying a Haar random
          unitary to a fixed pure state.
        - fill : Fill the ket with uniformly distributed random complex number.

    seed : int, SeedSequence, Generator, optional
        Seed to create the random number generator or a pre prepared
        generator. When none is suplied, a default generator is used.

    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
        accepted.

    Returns
    -------
    oper : qobj
        Ket quantum state vector.
    """
    dtype = dtype or settings.core["default_dtype"] or _data.Dense
    generator = _get_generator(seed)
    N, dims = _implicit_tensor_dimensions(dimensions)
    if distribution not in ["haar", "fill"]:
        raise ValueError("distribution must be one of {'haar', 'fill'}")

    if N == 1:
        ket = rand_unitary(1, seed=generator)
    elif distribution == "haar":
        ket = rand_unitary(N, density, "haar", seed=generator) @ basis(N, 0)
    else:
        X = scipy.sparse.rand(N, 1, density, format='csr',
                              random_state=generator)
        while X.nnz == 0:
            # ensure that the ket is not all zeros.
            X = scipy.sparse.rand(N, 1, density+1/N, format='csr',
                                  random_state=generator)
        X.data = X.data - 0.5
        Y = X.copy()
        Y.data = 1.0j * (generator.random(len(X.data)) - 0.5)
        X = _data.csr.CSR(X + Y)
        ket = Qobj(_data.mul(X, 1 / _data.norm.l2(X)),
                   copy=False, isherm=False, isunitary=False)
    ket.dims = [dims[0], [1]]
    return ket.to(dtype)


def rand_dm(
    dimensions: SpaceLike,
    density: float = 0.75,
    distribution: Literal["ginibre", "hs", "pure", "eigen", "uniform"] \
                  = "ginibre",
    *,
    eigenvalues: Sequence[float] = (),
    rank: int = None,
    seed: int | SeedSequence | Generator = None,
    dtype: LayerType = None,
):
    r"""Creates a random density matrix of the desired dimensions.

    Parameters
    ----------
    dimensions : (int) or (list of int) or (list of list of int)
        Dimension of Hilbert space. If provided as a list of ints, then the
        dimension is the product over this list, but the ``dims`` property of
        the new Qobj are set to this list.  This can produce either ``oper`` or
        ``super`` depending on the passed ``dimensions``.

    density : float, default: 0.75
        Density between [0,1] of output density matrix. Used by the "pure",
        "eigen" and "herm".

    distribution : str {"ginibre", "hs", "pure", "eigen", "uniform"}, \
default: "ginibre"
        Method used to obtain the density matrices.

        - "ginibre" : Ginibre random density operator of rank ``rank`` by using
          the algorithm of [BCSZ08]_.
        - "hs" : Hilbert-Schmidt ensemble, equivalent to a full rank ginibre
          operator.
        - "pure" : Density matrix created from a random ket.
        - "eigen" : A density matrix with the given ``eigenvalues``.
        - "herm" : Build from a random hermitian matrix using ``rand_herm``.

    eigenvalues : array_like, optional
        Eigenvalues of the output Hermitian matrix. The len must match the
        shape of the matrix.

    rank : int, optional
        When using the "ginibre" distribution, rank of the density matrix.
        Will default to a full rank operator when not provided.

    seed : int, SeedSequence, Generator, optional
        Seed to create the random number generator or a pre prepared
        generator. When none is suplied, a default generator is used.

    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
        accepted.

    Returns
    -------
    oper : qobj
        Density matrix quantum operator.
    """
    dtype = dtype or settings.core["default_dtype"] or _data.Dense
    generator = _get_generator(seed)
    N, dims = _implicit_tensor_dimensions(dimensions)
    distributions = set(["eigen", "ginibre", "hs", "pure", "herm"])
    if distribution not in distributions:
        raise ValueError(f"distribution must be one of {distributions}")

    if distribution == "eigen":
        if len(eigenvalues) != N:
            raise ValueError("Number of eigenvalues does not match the shape.")
        if np.abs(np.sum(eigenvalues)-1.0) > 1e-15 * N:
            raise ValueError('Eigenvalues of a density matrix '
                             f'must sum to one, not {np.sum(eigenvalues)}')
        H = _data.diag(eigenvalues, 0)
        nvals = N**2 * density
        H = _rand_jacobi_rotation(H, generator)
        while _data.csr.nnz(H) < 0.95*nvals:
            H = _rand_jacobi_rotation(H, generator)
    elif distribution == "ginibre":
        H = _rand_dm_ginibre(N, rank, generator)
    elif distribution == "hs":
        H = _rand_dm_ginibre(N, N, generator)
    elif distribution == "pure":
        dm_density = np.sqrt(density)
        psi = rand_ket(N, density=dm_density,
                       distribution="fill", seed=generator)
        H = psi.proj().data
    elif distribution == "herm":
        block_sizes = _get_block_sizes(N, density, generator)
        blocks = []
        first = True
        for block in block_sizes:
            if block != 1:
                mat = rand_herm(block, 1, seed=generator, dtype="dense").full()
            else:
                mat = np.ones((1,1))
                if not first:
                    mat *= max(generator.random() - 0.5, 0)
                first = False
            blocks.append(mat.T.conj() @ mat)
        H = _merge_shuffle_blocks(blocks, generator)
        H /= H.trace()

    return Qobj(H, dims=dims, isherm=True, copy=False).to(dtype)


def _rand_dm_ginibre(N, rank, generator):
    """
    Returns a Ginibre random density operator of  rank ``rank`` by using the
    algorithm of [BCSZ08]_. If ``rank`` is `None`, a full-rank
    (Hilbert-Schmidt ensemble) random density operator will be
    returned.

    Parameters
    ----------
    N : int
        Dimension of the density operator to be returned.

    rank : int or None
        Rank of the sampled density operator. If None, a full-rank
        density operator is generated.

    Returns
    -------
    rho : Qobj
        An N × N density operator sampled from the Ginibre
        or Hilbert-Schmidt distribution.
    """
    if not rank:
        rank = N
    if rank > N:
        raise ValueError("Rank cannot exceed dimension.")

    X = _randnz((N, rank), generator, norm='ginibre')
    rho = np.dot(X, X.T.conj())
    rho /= np.trace(rho)

    return rho


def rand_kraus_map(
    dimensions: SpaceLike,
    *,
    seed: int | SeedSequence | Generator = None,
    dtype: LayerType = None,
):
    """
    Creates a random CPTP map on an N-dimensional Hilbert space in Kraus
    form.

    Parameters
    ----------
    dimensions : (int) or (list of int) or (list of list of int)
        Dimension of Hilbert space. If provided as a list of ints, then the
        dimension is the product over this list, but the ``dims`` property of
        the new Qobj are set to this list.  This can produce either `oper` or
        `super` depending on the passed `dimensions`.

    seed : int, SeedSequence, Generator, optional
        Seed to create the random number generator or a pre prepared
        generator. When none is suplied, a default generator is used.

    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
        accepted.

    Returns
    -------
    oper_list : list of qobj
        N^2 x N x N qobj operators.

    """
    dtype = dtype or settings.core["default_dtype"] or _data.Dense
    N, dims = _implicit_tensor_dimensions(dimensions)
    dims = Dimensions(dims)
    if dims.issuper:
        raise TypeError("Each kraus operator cannot itself a super operator.")

    # Random unitary (Stinespring Dilation)
    big_unitary = rand_unitary(N ** 3, seed=seed, dtype=dtype).full()
    orthog_cols = np.array(big_unitary[:, :N])
    oper_list = np.reshape(orthog_cols, (N ** 2, N, N))
    return [Qobj(x, dims=dims, copy=False).to(dtype) for x in oper_list]


def rand_super(
    dimensions: SpaceLike,
    *,
    superrep: Literal["super", "choi", "chi"] = "super",
    seed: int | SeedSequence | Generator = None,
    dtype: LayerType = None,
):
    """
    Returns a randomly drawn superoperator acting on operators acting on
    N dimensions.

    Parameters
    ----------
    dimensions : (int) or (list of int) or (list of list of int)
        Dimension of Hilbert space. If provided as a list of ints, then the
        dimension is the product over this list, but the ``dims`` property of
        the new Qobj are set to this list.  This can produce either `oper` or
        `super` depending on the passed `dimensions`.

    superrop : str, default: "super"
        Representation of the super operator

    seed : int, SeedSequence, Generator, optional
        Seed to create the random number generator or a pre prepared
        generator. When none is suplied, a default generator is used.

    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
        accepted.
    """
    dtype = dtype or settings.core["default_dtype"] or _data.Dense
    generator = _get_generator(seed)
    from .solver.propagator import propagator
    N, dims = _implicit_tensor_dimensions(dimensions, superoper=True)
    H = rand_herm(N, seed=generator, dtype=dtype)
    S = propagator(H, generator.random(), [
        create(N), destroy(N), jmat(float(N - 1) / 2.0, 'z')
    ])
    S.dims = dims
    out = {
            "choi" : to_choi,
            "chi" : to_chi,
            "super": to_super,
        }[superrep](S).to(dtype)
    return out


def rand_super_bcsz(
    dimensions: SpaceLike,
    enforce_tp: bool = True,
    rank: int = None,
    *,
    superrep: Literal["super", "choi", "chi"] = "super",
    seed: int | SeedSequence | Generator = None,
    dtype: LayerType = None,
):
    """
    Returns a random superoperator drawn from the Bruzda
    et al ensemble for CPTP maps [BCSZ08]_. Note that due to
    finite numerical precision, for ranks less than full-rank,
    zero eigenvalues may become slightly negative, such that the
    returned operator is not actually completely positive.

    Parameters
    ----------
    dimensions : (int) or (list of int) or (list of list of int)
        Dimension of Hilbert space. If an int is provided, it is understood as
        the Square root of the dimension of the superoperator to be returned,
        with the corresponding dims as ``[[[N],[N]], [[N],[N]]]``. If provided
        as a list of ints, then the dimensions is understood as the space of
        density matrices this superoperator is applied to: ``dimensions=[2,2]``
        ``dims=[[[2,2],[2,2]], [[2,2],[2,2]]]``.

    enforce_tp : bool, default: True
        If True, the trace-preserving condition of [BCSZ08]_ is enforced;
        otherwise only complete positivity is enforced.

    rank : int, optional
        Rank of the sampled superoperator. If None, a full-rank
        superoperator is generated.

    seed : int, SeedSequence, Generator, optional
        Seed to create the random number generator or a pre prepared
        generator. When none is suplied, a default generator is used.

    superrop : str, default: "super"
        representation of the

    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
        accepted.

    Returns
    -------
    rho : Qobj
        A superoperator acting on vectorized dim × dim density operators,
        sampled from the BCSZ distribution.
    """
    dtype = dtype or settings.core["default_dtype"] or _data.CSR
    generator = _get_generator(seed)
    N, dims = _implicit_tensor_dimensions(dimensions, superoper=True)

    if rank is None:
        rank = N**2
    if rank > N**2:
        raise ValueError("Rank cannot exceed superoperator dimension.")

    # We use mainly dense matrices here for speed in low
    # dimensions. In the future, it would likely be better to switch off
    # between sparse and dense matrices as the dimension grows.

    # We start with a Ginibre uniform matrix X of the appropriate rank,
    # and use it to construct a positive semidefinite matrix X X⁺.
    X = _randnz((N**2, rank), generator, norm='ginibre')

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
        D = Qobj(np.dot(Z, np.dot(XXdag, Z)), dims=tmp_dims)
    else:
        D = N * Qobj(XXdag / np.trace(XXdag), dims=tmp_dims)

    # Since [BCSZ08] gives a row-stacking Choi matrix, but QuTiP
    # expects a column-stacking Choi matrix, we must permute the indices.
    D = D.permute([[1], [0]])

    # Mark that we've made a Choi matrix.
    D.superrep = 'choi'
    D.dims = dims
    out = {
            "choi" : to_choi,
            "chi" : to_chi,
            "super": to_super,
        }[superrep](D).to(dtype)
    return out


def rand_stochastic(
    dimensions: SpaceLike,
    density: float = 0.75,
    kind: Literal["left", "right"] = "left",
    *,
    seed: int | SeedSequence | Generator = None,
    dtype: LayerType = None,
):
    """Generates a random stochastic matrix.

    Parameters
    ----------
    dimensions : (int) or (list of int) or (list of list of int)
        Dimension of Hilbert space. If provided as a list of ints, then the
        dimension is the product over this list, but the ``dims`` property of
        the new Qobj are set to this list.  This can produce either `oper` or
        `super` depending on the passed `dimensions`.

    density : float, default: 0.75
        Density between [0,1] of output density matrix.

    kind : str {"left", "right"}, default: "left"
        Generate 'left' or 'right' stochastic matrix.

    seed : int, SeedSequence, Generator, optional
        Seed to create the random number generator or a pre prepared
        generator. When none is suplied, a default generator is used.

    dtype : type or str, optional
        Storage representation. Any data-layer known to ``qutip.data.to`` is
        accepted.

    Returns
    -------
    oper : qobj
        Quantum operator form of stochastic matrix.
    """
    dtype = dtype or settings.core["default_dtype"] or _data.Dense
    generator = _get_generator(seed)
    N, dims = _implicit_tensor_dimensions(dimensions)
    num_elems = max([int(np.ceil(N*(N+1)*density)/2), N])
    data = generator.random(num_elems)
    # Ensure an element on every row and column
    row_idx = np.hstack([generator.permutation(N),
                         generator.choice(N, num_elems-N)])
    col_idx = np.hstack([generator.permutation(N),
                         generator.choice(N, num_elems-N)])
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
    return Qobj(M, dims=dims).to(dtype)
