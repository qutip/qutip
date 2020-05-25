# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, The QuTiP Project
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
Module for measuring quantum objects.
"""

import numpy as np

from qutip.qobj import Qobj


def measurement_statistics(op, state):
    """
    Return the measurement eigenvalues, eigenstates (or projectors) and
    measurement probabilities for the given state and measurement operator.

    Parameters
    ----------
    op : Qobj
        The measurement operator.
    state : Qobj
        The ket or density matrix specifying the state to measure.

    Returns
    -------
    eigenvalues: List of floats
        The list of eigenvalues of the measurement operator.
    eigenstates_or_projectors: List of Qobj
        If the state was a ket, return the eigenstates of the measurement
        operator. Otherwise return the projectors onto the eigenstates.
    probabilities: List of floats
        The probability of measuring the state as being in the
        corresponding eigenstate (and the measurement result being
        the corresponding eigenvalue).
    """
    if not isinstance(op, Qobj):
        raise TypeError("op must be a Qobj")
    if not op.isoper:
        raise ValueError("op must be an operator")
    if not isinstance(state, Qobj):
        raise TypeError("state must be a Qobj")
    if state.isket:
        if op.dims[-1] != state.dims[0]:
            raise ValueError(
                "op and state dims should be compatible when state is a ket")
    elif state.isoper:
        if op.dims != state.dims:
            raise ValueError(
                "op and state dims should match"
                " when state is a density matrix")
    else:
        raise ValueError("state must be a ket or a density matrix")
    return _measurement_statistics(op, state)


def _measurement_statistics(op, state):
    eigenvalues, eigenstates = op.eigenstates()
    if state.isket:
        probabilities = [(e.dag() * state).norm() ** 2 for e in eigenstates]
        return eigenvalues, eigenstates, probabilities
    else:
        projectors = [v * v.dag() for v in eigenstates]
        probabilities = [(p * state).tr() for p in projectors]
        return eigenvalues, projectors, probabilities



def measure(op, state):
    """
    Perform a measurement specified by an operator on the given state.

    This function simulates the classic quantum measurement described
    in many introductory texts on quantum mechanics. The measurement
    collapses the state to one of the eigenstates of the given
    operator and the result of the measurement is the corresponding
    eigenvalue.

    Parameters
    ----------
    op : Qobj
        The measurement operator.
    state : Qobj
        The ket or density matrix specifying the state to measure.

    Returns
    -------
    measured_value : float
        The result of the measurement (one of the eigenvalues of op).
    state : Qobj
        The new state (a ket if a ket was given, otherwise a density
        matrix).

    Examples
    --------

    Measure the z-component of the spin of the spin-up basis state:

    >>> measure(sigmaz(), basis(2, 0))
    (1.0, Quantum object: dims = [[2], [1]], shape = (2, 1), type = ket
    Qobj data =
    [[-1.]
     [ 0.]])

    Since the spin-up basis is an eigenstate of sigmaz, this measurement always
    returns 1 as the measurement result (the eigenvalue of the spin-up basis)
    and the original state (up to a global phase).

    Measure the x-component of the spin of the spin-down basis state:

    >>> measure(sigmax(), basis(2, 1))
    (-1.0, Quantum object: dims = [[2], [1]], shape = (2, 1), type = ket
    Qobj data =
    [[-0.70710678]
     [ 0.70710678]])

    This measurement returns 1 fifty percent of the time and -1 the other fifty
    percent of the time. The new state returned is the corresponding eigenstate
    of sigmax.

    One may also perform a measurement on a density matrix. Below we perform
    the same measurement as above, but on the density matrix representing the
    pure spin-down state:

    >>> measure(sigmax(), ket2dm(basis(2, 1)))
    (-1.0, Quantum object: dims = [[2], [2]], shape = (2, 2), type = oper
    Qobj data =
    [[ 0.5 -0.5]
     [-0.5  0.5]])

    The measurement result is the same, but the new state is returned as a
    density matrix.
    """
    eigenvalues, eigenstates_or_projectors, probabilities = (
        measurement_statistics(op, state))
    i = np.random.choice(range(len(eigenvalues)), p=probabilities)
    if state.isket:
        eigenstates = eigenstates_or_projectors
        state = eigenstates[i]
    else:
        projectors = eigenstates_or_projectors
        state = (projectors[i] * state * projectors[i]) / probabilities[i]
    return eigenvalues[i], state
