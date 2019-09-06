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

from qutip import basis, tensor, identity


def _qubit_projector(outcome, i, n):
    v = basis(2, outcome)
    id2 = identity(2)
    op = v * v.dag()
    return tensor([id2] * i + [op] + [id2] * (n - i - 1))


def measure_qubit(qobj_in, i):
    """
    Measure a single qubit from a multiple qubit state.

    Parameters
    ----------
    qobj_in : :class:`qutip.Qobj`
        The input multiple qubit state.
    i : int
        The qubit to measure.

    Returns
    -------
    outcome : int
        The outcome of the measurement.
    qobj_out : :class:`qutip.Qobj`
        Quantum object representing the post measurement state.
    """
    n = len(qobj_in.dims[0])

    m0 = _qubit_projector(0, i, n)
    p0 = (m0 * qobj_in).ptrace(i).tr().real

    m1 = _qubit_projector(1, i, n)
    p1 = (m1 * qobj_in).ptrace(i).tr().real

    outcome = int(np.random.choice([0, 1], 1, p=[p0, p1]))

    if outcome == 0:
        qobj_out = m0 * qobj_in / np.sqrt(p0)
    else:
        qobj_out = m1 * qobj_in / np.sqrt(p1)

    return outcome, qobj_out
