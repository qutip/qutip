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
from scipy import arcsin, sqrt, pi
import numpy as np
import scipy.linalg as la
from scipy.linalg.matfuncs import sqrtm
import scipy.sparse as sp
from qutip.qobj import *
from qutip.operators import create, destroy, jmat
from operator import add


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


def rand_ket(N, density=1, dims=None):
    """Creates a random Nx1 sparse ket vector.

    Parameters
    ----------
    N : int
        Number of rows for output quantum operator.
    density : float
        Density between [0,1] of output ket state.
    dims : list
        Dimensions of quantum object.  Used for specifying
        tensor structure. Default is dims=[[N],[1]].

    Returns
    -------
    oper : qobj
        Nx1 ket state quantum operator.

    """
    if dims:
        _check_dims(dims, N, 1)
    X = sp.rand(N, 1, density, format='csr')
    X.data = X.data - 0.5
    Y = X.copy()
    Y.data = 1.0j * np.random.random(len(X.data)) - (0.5 + 0.5j)
    X = X + Y
    X.sort_indices()
    X = Qobj(X)
    if dims:
        return Qobj(X / X.norm(), dims=dims, shape=[N, 1])
    else:
        return Qobj(X / X.norm())


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


def rand_super(dim=5):
    H = rand_herm(dim)
    return propagator(H, np.random.rand(), [
        create(dim), destroy(dim), jmat(float(dim - 1) / 2.0, 'z')
    ])


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
