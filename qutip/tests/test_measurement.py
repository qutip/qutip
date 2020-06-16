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

import numpy as np
from numpy.testing import (
    assert_, assert_almost_equal, assert_array_equal, assert_raises_regex
)
from qutip import basis, isequal, ket2dm, sigmax, sigmay, sigmaz
from qutip.measurement import measure, measurement_statistics


class EigenPairs:
    """ Manage pairs of eigenvalues and eigenstates for an operator. """
    def __init__(self, pairs):
        self.pairs = pairs
        self.eigenvalues = [p[0] for p in pairs]
        self.eigenstates = [p[1] for p in pairs]
        self.projectors = [v * v.dag() for v in self.eigenstates]

    def __getitem__(self, i):
        return self.pairs[i]

    def __contains__(self, other):
        for i, val in enumerate(self.eigenvalues):
            if abs(val - other[0]) < 1e-8:
                break
        else:
            return False
        return _equivalent(other[1], self.eigenstates[i])


def pairs2dm(pairs):
    """ Convert eigenpair entries into eigenvalue and density matrix pairs. """
    return [(v, ket2dm(e)) for v, e in pairs]


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


def _equivalent(left, right, tol=1e-8):
    """ Equal up to a phase """
    return 1 - abs( (left.dag() * right).tr()) < tol


def check_measurement_statistics(
        op, state, pairs, probabilities):
    evs, ess_or_projs, probs = measurement_statistics(op, state)
    assert_almost_equal(evs, pairs.eigenvalues, decimal=1e-14)
    if state.isket:
        ess = ess_or_projs
        assert_(len(ess), len(pairs.eigenstates))
        for a, b in zip(ess, pairs.eigenstates):
            assert_(_equivalent(a, b))
    else:
        projs = ess_or_projs
        assert_(len(projs), len(pairs.projectors))
        for a, b in zip(projs, pairs.projectors):
            assert_(_equivalent(a, b))
    assert_almost_equal(probs, probabilities)


def test_measurement_statistics_sigmaz():
    """ measurement statistics: sigmaz applied to basis states. """
    check_measurement_statistics(
        sigmaz(), basis(2, 0), SIGMAZ, [0, 1],
    )
    check_measurement_statistics(
        sigmaz(), ket2dm(basis(2, 0)), SIGMAZ, [0, 1],
    )


def test_measurement_statistics_sigmax():
    """ measurement statistics: sigmax applied to basis states. """
    check_measurement_statistics(
        sigmax(), basis(2, 0), SIGMAX, [0.5, 0.5],
    )
    check_measurement_statistics(
        sigmax(), ket2dm(basis(2, 0)), SIGMAX, [0.5, 0.5],
    )


def test_measurement_statistics_sigmay():
    """ measurement statistics: sigmay applied to basis states. """
    check_measurement_statistics(
        sigmay(), basis(2, 0), SIGMAY, [0.5, 0.5],
    )
    check_measurement_statistics(
        sigmay(), ket2dm(basis(2, 0)), SIGMAY, [0.5, 0.5],
    )


def test_measurement_statistics_input_errors():
    """ measurement_statistics: check input errors """
    assert_raises_regex(
        TypeError, "op must be a Qobj",
        measurement_statistics, "notqobj", basis(2, 0))
    assert_raises_regex(
        ValueError, "op must be an operator",
        measurement_statistics, basis(2, 1), basis(2, 0))
    assert_raises_regex(
        TypeError, "state must be a Qobj",
        measurement_statistics, sigmaz(), "notqobj")
    assert_raises_regex(
        ValueError, "state must be a ket or a density matrix",
        measurement_statistics, sigmaz(), basis(2, 0).dag())
    assert_raises_regex(
        ValueError,
        "op and state dims should be compatible when state is a ket",
        measurement_statistics, sigmaz(), basis(3, 0))
    assert_raises_regex(
        ValueError,
        "op and state dims should match when state is a density matrix",
        measurement_statistics, sigmaz(), ket2dm(basis(3, 0)))


def check_measure(op, state):
    """ Check that all measure results are in measurement_statistics which
    was previously tested.
    This removes the dependance on the prng.
    """
    evs, ess_or_projs, prob = measurement_statistics(op, state)
    expected_measurements = EigenPairs(list(zip(evs, ess_or_projs)))
    for _ in range(10):
        assert_(measure(op, state) in expected_measurements)


def test_measure_sigmaz():
    """ measure: basis states using sigmaz """
    check_measure(sigmaz(), basis(2, 0) )
    check_measure(sigmaz(), basis(2, 1) )
    check_measure(sigmaz(), ket2dm(basis(2, 0)) )
    check_measure(sigmaz(), ket2dm(basis(2, 1)) )


def test_measure_sigmax():
    """ measure: basis states using sigmax """
    check_measure( sigmax(), basis(2, 0) )
    check_measure( sigmax(), basis(2, 1) )
    check_measure( sigmax(), ket2dm(basis(2, 0)) )


def test_measure_sigmay():
    """ measure: basis states using sigmay """
    check_measure( sigmay(), basis(2, 0) )
    check_measure( sigmay(), basis(2, 1) )
    check_measure( sigmay(), ket2dm(basis(2, 0)) )


def test_measure_input_errors():
    """ measure: check input errors """
    assert_raises_regex(
        TypeError, "op must be a Qobj",
        measure, "notqobj", basis(2, 0))
    assert_raises_regex(
        ValueError, "op must be an operator",
        measure, basis(2, 1), basis(2, 0))
    assert_raises_regex(
        TypeError, "state must be a Qobj",
        measure, sigmaz(), "notqobj")
    assert_raises_regex(
        ValueError, "state must be a ket or a density matrix",
        measure, sigmaz(), basis(2, 0).dag())
    assert_raises_regex(
        ValueError,
        "op and state dims should be compatible when state is a ket",
        measure, sigmaz(), basis(3, 0))
    assert_raises_regex(
        ValueError,
        "op and state dims should match when state is a density matrix",
        measure, sigmaz(), ket2dm(basis(3, 0)))
