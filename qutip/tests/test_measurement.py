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

import numpy as np
from numpy.testing import assert_, assert_almost_equal
from qutip import basis, qubit_states, tensor, sigmax, sigmaz
from qutip.measurement import measure_qubit, measure


def assert_measure_qubits(qobj_in, i, outcomes):
    outcome, qobj_out = measure_qubit(qobj_in, i)
    assert_(outcome in outcomes)
    assert_almost_equal(qobj_out.norm(), 1)
    assert_(qobj_out == outcomes[outcome])


def _test_measure_qubits_n_1():
    q0 = qubit_states(1, [0])
    assert_measure_qubits(q0, 0, {0: q0})

    q1 = qubit_states(1, [1])
    assert_measure_qubits(q1, 0, {1: q1})

    for p in np.linspace(0, 1, 5):
        qb = qubit_states(1, [np.sqrt(p)])
        assert_measure_qubits(qb, 0, {0: q0, 1: q1})


def _test_measure_qubits_n_2():
    q0 = qubit_states(1, [0])
    q1 = qubit_states(1, [1])

    for p in np.linspace(0, 1, 5):
        qb = qubit_states(2, [np.sqrt(p), 0])
        assert_measure_qubits(qb, 0, {
            0: tensor(q0, q0),
            1: tensor(q1, q0),
        })

    for p in np.linspace(0, 1, 5):
        qb = qubit_states(2, [0, np.sqrt(p)])
        assert_measure_qubits(qb, 1, {
            0: tensor(q0, q0),
            1: tensor(q0, q1),
        })

    for p1 in np.linspace(0, 1, 5):
        for p2 in np.linspace(0, 1, 5):
            qb = qubit_states(2, [np.sqrt(p1), np.sqrt(p2)])
            assert_measure_qubits(qb, 0, {
                0: tensor(q0, qubit_states(1, [np.sqrt(p2)])),
                1: tensor(q1, qubit_states(1, [np.sqrt(p2)])),
            })
            assert_measure_qubits(qb, 1, {
                0: tensor(qubit_states(1, [np.sqrt(p1)]), q0),
                1: tensor(qubit_states(1, [np.sqrt(p1)]), q1),
            })


def test_measure_1():
    state = basis(2, 0)
    op = sigmaz()
    e_v, new_state = measure(state, op)
    assert_(e_v == 1)
    assert_(new_state == -1 * state)


def test_measure_2():
    state = (basis(2, 0) + basis(2, 1)).unit()
    op = sigmax()
    e_v, new_state = measure(state, op)
    assert_(e_v == 1)
    assert_(new_state == state)


def test_measure_3():
    state = (basis(2, 0) + basis(2, 1)).unit()
    op = sigmaz()
    e_v, new_state = measure(state, op)
    assert_(e_v in (-1, 1))
    if e_v == 1:
        assert_(new_state == -1 * basis(2, 0))
    else:
        assert_(new_state == -1 * basis(2, 1))
    print(e_v)
