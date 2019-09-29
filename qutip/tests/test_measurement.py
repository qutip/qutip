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
from numpy.testing import assert_, assert_almost_equal, assert_array_equal
from qutip import basis, isequal, sigmax, sigmay, sigmaz
from qutip.measurement import measure, measurement_statistics


class EigenPairs:
    def __init__(self, pairs):
        self.pairs = pairs
        self.eigenvalues = [p[0] for p in pairs]
        self.eigenstates = [p[1] for p in pairs]
        self.projectors = [v * v.dag() for v in self.eigenstates]

    def __getitem__(self, i):
        return self.pairs[i]


SIGMAZ = EigenPairs([
    (-1.0, -basis(2, 1)),
    (1.0, -basis(2, 0)),
])

SIGMAX = EigenPairs([
    (-1.0, (-basis(2, 0) + basis(2, 1)).unit()),
    (1.0, (basis(2, 0) + basis(2, 1)).unit()),
])

SIGMAY = EigenPairs([
    (-1.0, (-basis(2, 0) + 1j * basis(2, 1)).unit()),
    (1.0, (-basis(2, 0) - 1j * basis(2, 1)).unit()),
])


def check_measurement_statistics(
        op, state, pairs, probabilities):
    evs, ess_or_projs, probs = measurement_statistics(op, state)
    assert_array_equal(evs, pairs.eigenvalues)
    if state.isket:
        ess = ess_or_projs
        assert_(len(ess), len(pairs.eigenstates))
        for a, b in zip(ess, pairs.eigenstates):
            assert_(isequal(a, b))
    else:
        projs = ess_or_projs
        assert_(len(projs), len(pairs.projectors))
        for a, b in zip(projs, pairs.projectors):
            assert_(isequal(a, b))
    assert_almost_equal(probs, probabilities)


def test_measurement_statistics_sigmaz():
    check_measurement_statistics(
        sigmaz(), basis(2, 0), SIGMAZ, [0, 1],
    )


def test_measurement_statistics_sigmax():
    check_measurement_statistics(
        sigmax(), basis(2, 0), SIGMAX, [0.5, 0.5],
    )


def test_measurement_statistics_sigmay():
    check_measurement_statistics(
        sigmay(), basis(2, 0), SIGMAY, [0.5, 0.5],
    )


def check_measure(op, state, expected_measurements, seed=0):
    np.random.seed(seed)
    measurements = []
    for _ in expected_measurements:
        value, new_state = measure(op, state)
        measurements.append((value, new_state))
    assert_(measurements == expected_measurements)


def test_measure_sigmaz():
    check_measure(sigmaz(), basis(2, 0), [SIGMAZ[1]] * 5)
    check_measure(sigmaz(), basis(2, 1), [SIGMAZ[0]] * 5)


def test_measure_sigmax():
    check_measure(
        sigmax(), basis(2, 0),
        [SIGMAX[1], SIGMAX[1], SIGMAX[1], SIGMAX[1], SIGMAX[0]],
    )
    check_measure(
        sigmax(), basis(2, 1),
        [SIGMAX[1], SIGMAX[1], SIGMAX[1], SIGMAX[1], SIGMAX[0]],
    )
    check_measure(
        sigmax(), basis(2, 0),
        [SIGMAX[0], SIGMAX[1], SIGMAX[1], SIGMAX[1], SIGMAX[0]],
        seed=42,
    )


def test_measure_sigmay():
    check_measure(
        sigmay(), basis(2, 0),
        [SIGMAY[1], SIGMAY[1], SIGMAY[1], SIGMAY[1], SIGMAY[0]],
    )
    check_measure(
        sigmay(), basis(2, 1),
        [SIGMAY[1], SIGMAY[1], SIGMAY[1], SIGMAY[1], SIGMAY[0]],
    )
    check_measure(
        sigmay(), basis(2, 1),
        [SIGMAY[0], SIGMAY[1], SIGMAY[1], SIGMAY[1], SIGMAY[0]],
        seed=42,
    )
