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
Module for measuring quantum objects.
"""

import numpy as np


def measurement_statistics(op, state):
    """
    Return the measurement eigenvalues, eigenstates and probabilities
    for the given state and measurement operator.

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
    eigenstates: List of Qobj
        The eigenstates of the measurement operator.
    probabilities: List of floats
        The probability of measuring the state as being in the
        corresponding eigenstate (and the measurement result being
        the corresponding eigenvalue).
    """
    # TODO: Add support for density matrix.
    eigenvalues, eigenstates = op.eigenstates()
    projectors = [v * v.dag() for v in eigenstates]
    probabilities = [(p * state).norm() ** 2 for p in projectors]
    return eigenvalues, eigenstates, probabilities


def measure(op, state):
    """
    Perform a measure specified by an operator on the given state.

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
    """
    # TODO: Add support for density matrix.
    eigenvalues, eigenstates, probabilities = measurement_statistics(op, state)
    i = np.random.choice(range(len(eigenvalues)), p=probabilities)
    return eigenvalues[i], eigenstates[i]
