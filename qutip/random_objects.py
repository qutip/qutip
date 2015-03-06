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
    'rand_herm', 'rand_unitary', 'rand_ket', 'rand_dm',
    'rand_unitary_haar', 'rand_ket_haar', 'rand_dm_ginibre',
    'rand_dm_hs', 'rand_super_bcsz'
]

from scipy import arcsin, sqrt, pi
from scipy.linalg import sqrtm
import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
from qutip.qobj import Qobj
from qutip.operators import create, destroy, jmat
from qutip.states import basis
import qutip.superop_reps as sr


UNITS = np.array([1, 1j])

def randnz(shape, norm=1 / np.sqrt(2)):
    # This function is intended for internal use.
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
    if norm == 'ginibre':
        norm = 1
    return np.sum(np.random.randn(*(shape + (2,))) * UNITS, axis=-1) * norm


def rand_herm(N, density=0.75, dims=None):
    """Creates a random NxN sparse Hermitian quantum object.

    Uses :math:`H=X+X^{+}` where :math:`X` is
    a randomly generated quantum operator with a given `density`.

    Parameters
    ----------
    N : int
        Shape of output quantum operator.
    density : float
        Density between [0,1] of output Hermitian operator.
    dims : list
        Dimensions of quantum object.  Used for specifying
        tensor structure. Default is dims=[[N],[N]].

    Returns
    -------
    oper : qobj
        NxN Hermitian quantum operator.

    """
    if dims:
        _check_dims(dims, N, N)
    # to get appropriate density of output
    # Hermitian operator must convert via:
    herm_density = 2.0 * arcsin(density) / pi

    X = sp.rand(N, N, herm_density, format='csr')
    X.data = X.data - 0.5
    Y = X.copy()
    Y.data = 1.0j * np.random.random(len(X.data)) - (0.5 + 0.5j)
    X = X + Y
    X.sort_indices()
    X = Qobj(X)
    if dims:
        return Qobj((X + X.dag()) / 2.0, dims=dims, shape=[N, N])
    else:
        return Qobj((X + X.dag()) / 2.0)


def rand_unitary(N, density=0.75, dims=None):
    """Creates a random NxN sparse unitary quantum object.

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

    Returns
    -------
    oper : qobj
        NxN Unitary quantum operator.

    """
    if dims:
        _check_dims(dims, N, N)
    U = (-1.0j * rand_herm(N, density)).expm()
    U.data.sort_indices()
    if dims:
        return Qobj(U, dims=dims, shape=[N, N])
    else:
        return Qobj(U)

def rand_unitary_haar(N=2, dims=None):
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
    Z = randnz((N, N))

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

def rand_ket(N, density=1, dims=None):
    """Creates a random Nx1 sparse ket vector.

    Parameters
    ----------
    N : int
        Number of rows for output quantum operator.
    density : float
        Density between [0,1] of output ket state.
    dims : list
        Left-dimensions of quantum object.  Used for specifying
        tensor structure. Default is dims=[[N]].

    Returns
    -------
    oper : qobj
        Nx1 ket state quantum operator.

    """
    if dims:
        _check_ket_dims(dims, N)
    X = sp.rand(N, 1, density, format='csr')
    X.data = X.data - 0.5
    Y = X.copy()
    Y.data = 1.0j * np.random.random(len(X.data)) - (0.5 + 0.5j)
    X = X + Y
    X.sort_indices()
    X = Qobj(X)
    if dims:
        return Qobj(X / X.norm(), dims=[dims, [1]], shape=[N, 1])
    else:
        return Qobj(X / X.norm())


def rand_ket_haar(N=2, dims=None):
    """
    Returns a Haar random pure state of dimension ``dim`` by
    applying a Haar random unitary to a fixed pure state.


    Parameters
    ----------
    N : int
        Dimension of the state vector to be returned.

    dims : list of ints, or None
        Left-dimensions of the resultant quantum object.
        If None, [N] is used.

    Returns
    -------
    psi : Qobj
        A random state vector drawn from the Haar measure.
    """
    if dims:
        _check_ket_dims(dims, N)
    else:
        dims = [N]
    psi = rand_unitary_haar(N) * basis(N, 0)
    psi.dims = [dims, [1]]
    return psi


def rand_dm(N, density=0.75, pure=False, dims=None):
    """Creates a random NxN density matrix.

    Parameters
    ----------
    N : int
        Shape of output density matrix.
    density : float
        Density between [0,1] of output density matrix.
    dims : list
        Dimensions of quantum object.  Used for specifying
        tensor structure. Default is dims=[[N],[N]].


    Returns
    -------
    oper : qobj
        NxN density matrix quantum operator.

    Notes
    -----
    For small density matrices., choosing a low density will result in an error
    as no diagonal elements will be generated such that :math:`Tr(\\rho)=1`.

    """
    if dims:
        _check_dims(dims, N, N)
    if pure:
        dm_density = sqrt(density)
        psi = rand_ket(N, dm_density)
        H = psi * psi.dag()
    else:
        density = density ** 2
        non_zero = 0
        tries = 0
        while non_zero == 0 and tries < 10:
            H = rand_herm(N, density)
            H = H.dag() * H
            non_zero = sum([H.tr()])
            tries += 1
        if tries >= 10:
            raise ValueError(
                "Requested density is too low to generate density matrix.")
    H.data.sort_indices()
    if dims:
        return Qobj(H / H.tr(), dims=dims, shape=[N, N])
    else:
        return Qobj(H / H.tr())

def rand_dm_ginibre(N=2, rank=None, dims=None):
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

    X = randnz((N, rank), norm='ginibre')
    rho = np.dot(X, X.T.conj())
    rho /= np.trace(rho)

    return Qobj(rho, dims=dims)

def rand_dm_hs(N=2, dims=None):
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

    Returns
    -------
    rho : Qobj
        A dim × dim density operator sampled from the Ginibre
        or Hilbert-Schmidt distribution.

    """
    return rand_dm_ginibre(N, rank=None, dims=dims)


def rand_kraus_map(N, dims=None):
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


    Returns
    -------
    oper_list : list of qobj
        N^2 x N x N qobj operators.

    """

    if dims:
        _check_dims(dims, N, N)

    # Random unitary (Stinespring Dilation)
    big_unitary = rand_unitary(N ** 3).data.todense()
    orthog_cols = np.array(big_unitary[:, :N])
    oper_list = np.reshape(orthog_cols, (N ** 2, N, N))
    return list(map(lambda x: Qobj(inpt=x, dims=dims), oper_list))


def rand_super(N=5, dims=None):
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
    """
    if dims is not None:
        # TODO: check!
        pass
    else:
        dims = [[[N],[N]], [[N],[N]]]
    H = rand_herm(N)
    S = propagator(H, np.random.rand(), [
        create(N), destroy(N), jmat(float(N - 1) / 2.0, 'z')
    ])
    S.dims = dims
    return S


def rand_super_bcsz(N=2, enforce_tp=True, rank=None, dims=None):
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
        dims = [[[N],[N]], [[N],[N]]]

    if rank is None:
        rank = N**2
    if rank > N**2:
        raise ValueError("Rank cannot exceed superoperator dimension.")

    # We use mainly dense matrices here for speed in low
    # dimensions. In the future, it would likely be better to switch off
    # between sparse and dense matrices as the dimension grows.

    # We start with a Ginibre uniform matrix X of the appropriate rank,
    # and use it to construct a positive semidefinite matrix X X⁺.
    X = randnz((N**2, rank), norm='ginibre')
    
    # Precompute X X⁺, as we'll need it in two different places.
    XXdag = np.dot(X, X.T.conj())
    
    if enforce_tp:
        # We do the partial trace over the first index by using dense reshape
        # operations, so that we can avoid bouncing to a sparse representation
        # and back.
        Y = np.einsum('ijik->jk', XXdag.reshape((N, N, N, N)))

        # Now we have the matrix 𝟙 ⊗ Y^{-1/2}, which we can find by doing
        # the square root and the inverse separately. As a possible improvement,
        # iterative methods exist to find inverse square root matrices directly,
        # as this is important in statistics.
        Z = np.kron(
            np.eye(N),
            sqrtm(la.inv(Y))
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

def _check_ket_dims(dims, N1):
    if not isinstance(dims, list) or isinstance(dims[0], list):
        raise TypeError("Left and right Qobj dimensions must be lists of ints. E.g.: [2, 3].")
    if np.prod(dims) != N1:
        raise ValueError("Qobj dimensions must match matrix shape.")

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
from qutip.propagator import propagator
