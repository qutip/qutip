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
This module contains a collection of functions for calculating metrics
(distance measures) between states and operators.
"""

__all__ = ['fidelity', 'tracedist', 'bures_dist', 'bures_angle',
           'hilbert_dist', 'average_gate_fidelity', 'process_fidelity',
           'unitarity']

import numpy as np
from qutip.sparse import sp_eigs
from qutip.states import ket2dm
from qutip.superop_reps import to_kraus, _super_to_superpauli


def fidelity(A, B):
    """
    Calculates the fidelity (pseudo-metric) between two density matrices.
    See: Nielsen & Chuang, "Quantum Computation and Quantum Information"

    Parameters
    ----------
    A : qobj
        Density matrix or state vector.
    B : qobj
        Density matrix or state vector with same dimensions as A.

    Returns
    -------
    fid : float
        Fidelity pseudo-metric between A and B.

    Examples
    --------
    >>> x = fock_dm(5,3)
    >>> y = coherent_dm(5,1)
    >>> fidelity(x,y)
    0.24104350624628332

    """
    if A.isket or A.isbra:
        A = ket2dm(A)
    if B.isket or B.isbra:
        B = ket2dm(B)

    if A.dims != B.dims:
        raise TypeError('Density matrices do not have same dimensions.')

    A = A.sqrtm()
    return float(np.real((A * (B * A)).sqrtm().tr()))


def process_fidelity(U1, U2, normalize=True):
    """
    Calculate the process fidelity given two process operators.
    """
    if normalize:
        return (U1 * U2).tr() / (U1.tr() * U2.tr())
    else:
        return (U1 * U2).tr()


def average_gate_fidelity(oper):
    """
    Given a Qobj representing the supermatrix form of a map, returns the
    average gate fidelity (pseudo-metric) of that map.

    Parameters
    ----------
    A : Qobj
        Quantum object representing a superoperator.

    Returns
    -------
    fid : float
        Fidelity pseudo-metric between A and the identity superoperator.
    """
    kraus_form = to_kraus(oper)
    d = kraus_form[0].shape[0]

    if kraus_form[0].shape[1] != d:
        return TypeError("Average gate fielity only implemented for square "
                         "superoperators.")

    return (d + np.sum([np.abs(A_k.tr())**2
                        for A_k in kraus_form])) / (d**2 + d)


def tracedist(A, B, sparse=False, tol=0):
    """
    Calculates the trace distance between two density matrices..
    See: Nielsen & Chuang, "Quantum Computation and Quantum Information"

    Parameters
    ----------!=
    A : qobj
        Density matrix or state vector.
    B : qobj
        Density matrix or state vector with same dimensions as A.
    tol : float
        Tolerance used by sparse eigensolver, if used. (0=Machine precision)
    sparse : {False, True}
        Use sparse eigensolver.

    Returns
    -------
    tracedist : float
        Trace distance between A and B.

    Examples
    --------
    >>> x=fock_dm(5,3)
    >>> y=coherent_dm(5,1)
    >>> tracedist(x,y)
    0.9705143161472971

    """
    if A.isket or A.isbra:
        A = ket2dm(A)
    if B.isket or B.isbra:
        B = ket2dm(B)

    if A.dims != B.dims:
        raise TypeError("A and B do not have same dimensions.")

    diff = A - B
    diff = diff.dag() * diff
    vals = sp_eigs(diff.data, diff.isherm, vecs=False, sparse=sparse, tol=tol)
    return float(np.real(0.5 * np.sum(np.sqrt(np.abs(vals)))))


def hilbert_dist(A, B):
    """
    Returns the Hilbert-Schmidt distance between two density matrices A & B.

    Parameters
    ----------
    A : qobj
        Density matrix or state vector.
    B : qobj
        Density matrix or state vector with same dimensions as A.

    Returns
    -------
    dist : float
        Hilbert-Schmidt distance between density matrices.

    Notes
    -----
    See V. Vedral and M. B. Plenio, Phys. Rev. A 57, 1619 (1998).

    """
    if A.isket or A.isbra:
        A = ket2dm(A)
    if B.isket or B.isbra:
        B = ket2dm(B)

    if A.dims != B.dims:
        raise TypeError('A and B do not have same dimensions.')

    return ((A - B)**2).tr()


def bures_dist(A, B):
    """
    Returns the Bures distance between two density matrices A & B.

    The Bures distance ranges from 0, for states with unit fidelity,
    to sqrt(2).

    Parameters
    ----------
    A : qobj
        Density matrix or state vector.
    B : qobj
        Density matrix or state vector with same dimensions as A.

    Returns
    -------
    dist : float
        Bures distance between density matrices.
    """
    if A.isket or A.isbra:
        A = ket2dm(A)
    if B.isket or B.isbra:
        B = ket2dm(B)

    if A.dims != B.dims:
        raise TypeError('A and B do not have same dimensions.')

    dist = np.sqrt(2.0 * (1.0 - fidelity(A, B)))
    return dist


def bures_angle(A, B):
    """
    Returns the Bures Angle between two density matrices A & B.

    The Bures angle ranges from 0, for states with unit fidelity, to pi/2.

    Parameters
    ----------
    A : qobj
        Density matrix or state vector.
    B : qobj
        Density matrix or state vector with same dimensions as A.

    Returns
    -------
    angle : float
        Bures angle between density matrices.
    """
    if A.isket or A.isbra:
        A = ket2dm(A)
    if B.isket or B.isbra:
        B = ket2dm(B)

    if A.dims != B.dims:
        raise TypeError('A and B do not have same dimensions.')

    return np.arccos(fidelity(A, B))

def unitarity(oper):
    """
    Returns the unitarity of a quantum map, defined as the Frobenius norm
    of the unital block of that map's superoperator representation.

    Parameters
    ----------
    oper : Qobj
        Quantum map under consideration.

    Returns
    -------
    u : float
        Unitarity of ``oper``.
    """
    Eu = _super_to_superpauli(oper).full()[1:, 1:]
    #return np.real(np.trace(np.dot(Eu, Eu.conj().T))) / len(Eu)
    return np.linalg.norm(Eu, 'fro')**2 / len(Eu)
