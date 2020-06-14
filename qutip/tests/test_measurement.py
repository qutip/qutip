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
import scipy
import pytest
from math import sqrt
from numpy.testing import (
    assert_, assert_almost_equal, assert_array_equal, assert_raises_regex
)
from qutip.qip.circuit import Measurement
from qutip import (Qobj, basis, isequal, ket2dm,
            sigmax, sigmay, sigmaz, identity, tensor, rand_ket)
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


def check_measurement_statistics(
        op, state, pairs, probabilities):
    evs, ess_or_projs, probs = measurement_statistics(op, state)
    assert_array_equal(evs, pairs.eigenvalues)
    if state.isket:
        ess = ess_or_projs
        assert_(len(ess), len(pairs.eigenstates))
        for a, b in zip(ess, pairs.eigenstates):
            assert isequal(a, b)
    else:
        projs = ess_or_projs
        assert_(len(projs), len(pairs.projectors))
        for a, b in zip(projs, pairs.projectors):
            assert isequal(a, b)
    assert_almost_equal(probs, probabilities)


@pytest.mark.parametrize(["op", "state", "pairs", "probabilities"], [
                    pytest.param(sigmaz(), basis(2, 0),
                            SIGMAZ, [0, 1], id="sigmaz_ket"),
                    pytest.param(sigmaz(), ket2dm(basis(2, 0)),
                            SIGMAZ, [0, 1], id="sigmaz_dm"),
                    pytest.param(sigmax(), basis(2, 0),
                            SIGMAX, [0.5, 0.5], id="sigmax_ket"),
                    pytest.param(sigmax(), ket2dm(basis(2, 0)),
                            SIGMAX, [0.5, 0.5], id="sigmax_dm"),
                    pytest.param(sigmay(), basis(2, 0),
                            SIGMAY, [0.5, 0.5], id="sigmay_ket"),
                    pytest.param(sigmay(), ket2dm(basis(2, 0)),
                            SIGMAY, [0.5, 0.5], id="sigmay_dm")])
def test_measurement_statistics(op, state, pairs, probabilities):
    """ measurement statistics: sigmaz applied to basis states. """
    check_measurement_statistics(op, state, pairs, probabilities)


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


def check_measure(op, state, expected_measurements, seed=0):
    np.random.seed(seed)
    measurements = []
    for _ in expected_measurements:
        value, new_state = measure(op, state)
        measurements.append((value, new_state))
    assert measurements == expected_measurements


@pytest.mark.parametrize(["op", "state", "expected_measurements", "seed"], [

                    pytest.param(sigmaz(), basis(2, 0),
                            [SIGMAZ[1]] * 5, 0, id="sigmaz_ket1"),
                    pytest.param(sigmaz(), basis(2, 1),
                            [SIGMAZ[0]] * 5, 0, id="sigmaz_ket2"),
                    pytest.param(sigmaz(), ket2dm(basis(2, 0)),
                            pairs2dm([SIGMAZ[1]] * 5), 0, id="sigmaz_dm1"),
                    pytest.param(sigmaz(), ket2dm(basis(2, 1)),
                            pairs2dm([SIGMAZ[0]] * 5), 0, id="sigmaz_dm2"),

                    pytest.param(sigmax(), basis(2, 0),
                    [SIGMAX[1], SIGMAX[1], SIGMAX[1], SIGMAX[1], SIGMAX[0]], 0,
                    id="sigmax_ket1"),
                    pytest.param(sigmax(), basis(2, 1),
                    [SIGMAX[1], SIGMAX[1], SIGMAX[1], SIGMAX[1], SIGMAX[0]], 0,
                    id="sigmax_ket2"),
                    pytest.param(sigmax(), basis(2, 0),
                    [SIGMAX[0], SIGMAX[1], SIGMAX[1], SIGMAX[1], SIGMAX[0]], 42,
                    id="sigmax_ket3"),
                    pytest.param(sigmax(), ket2dm(basis(2, 0)),
                    pairs2dm([SIGMAX[1], SIGMAX[1], SIGMAX[1], SIGMAX[1],
                            SIGMAX[0]]), 0,
                    id="sigmax_dm"),

                    pytest.param(sigmay(), basis(2, 0),
                    [SIGMAY[1], SIGMAY[1], SIGMAY[1], SIGMAY[1], SIGMAY[0]], 0,
                    id="sigmay_ket1"),
                    pytest.param(sigmay(), basis(2, 1),
                    [SIGMAY[1], SIGMAY[1], SIGMAY[1], SIGMAY[1], SIGMAY[0]], 0,
                    id="sigmax_ket2"),
                    pytest.param(sigmay(), basis(2, 1),
                    [SIGMAY[0], SIGMAY[1], SIGMAY[1], SIGMAY[1], SIGMAY[0]], 42,
                    id="sigmay_ket3"),
                    pytest.param(sigmay(), ket2dm(basis(2, 1)),
                    pairs2dm([SIGMAY[0], SIGMAY[1], SIGMAY[1], SIGMAY[1],
                            SIGMAY[0]]), 42,
                    id="sigmay_dm")])
def test_measure(op, state, expected_measurements, seed):
    """ measure: basis states using different observables """
    check_measure(op, state, expected_measurements, seed)


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


def test_povm():
    """
    Test if povm formulation works correctly by checking probabilities for
    the quantum state discrimination example
    """

    coeff = (sqrt(2)/(1+sqrt(2)))

    E_1 = coeff * ket2dm(basis(2, 1))
    E_2 = coeff * ket2dm((basis(2, 0) - basis(2, 1))/(sqrt(2)))
    E_3 = identity(2) - E_1 - E_2

    M_1 = Qobj(scipy.linalg.sqrtm(E_1))
    M_2 = Qobj(scipy.linalg.sqrtm(E_2))
    M_3 = Qobj(scipy.linalg.sqrtm(E_3))

    ket1 = basis(2, 0)
    ket2 = (basis(2, 0) + basis(2, 1))/(sqrt(2))

    dm1 = ket2dm(ket1)
    dm2 = ket2dm(ket2)

    _, probabilities = measurement_statistics([M_1, M_2, M_3], ket1)
    np.testing.assert_allclose(probabilities, [0, 0.293, 0.707], 0.001)

    _, probabilities = measurement_statistics([M_1, M_2, M_3], ket2)
    np.testing.assert_allclose(probabilities, [0.293, 0, 0.707], 0.001)

    _, probabilities = measurement_statistics([M_1, M_2, M_3], dm1)
    np.testing.assert_allclose(probabilities, [0, 0.293, 0.707], 0.001)

    _, probabilities = measurement_statistics([M_1, M_2, M_3], dm2)
    np.testing.assert_allclose(probabilities, [0.293, 0, 0.707], 0.001)


@pytest.mark.repeat(10)
def test_measurement_comp_basis():
    """
    Test measurements to test probability calculation in
    computational basis measurments on a 3 qubit state
    """

    qubit_kets = [rand_ket(2), rand_ket(2), rand_ket(2)]
    qubit_dms = [ket2dm(qubit_kets[i]) for i in range(3)]

    state = tensor(qubit_kets)
    density_mat = tensor(qubit_dms)

    for i in range(3):
        m_i = Measurement("M" + str(i), i)
        final_states, probabilities_state = m_i.measurement_comp_basis(state)
        final_dms, probabilities_dm = m_i.measurement_comp_basis(density_mat)

        amps = qubit_kets[i].full()
        probabilities_i = [np.abs(amps[0][0])**2, np.abs(amps[1][0])**2]

        np.testing.assert_allclose(probabilities_state, probabilities_dm)
        np.testing.assert_allclose(probabilities_state, probabilities_i)
        for j, final_state in enumerate(final_states):
            np.testing.assert_allclose(final_dms[j], ket2dm(final_state))


@pytest.mark.parametrize("index", [0, 1])
def test_measurement_collapse(index):
    """
    Test if correct state is created after measurement using the example of
    the Bell state
    """

    state_00 = tensor(basis(2, 0), basis(2, 0))
    state_11 = tensor(basis(2, 1), basis(2, 1))

    bell_state = (state_00 + state_11)/sqrt(2)
    M = Measurement("BM", targets=[index])

    states, probabilities = M.measurement_comp_basis(bell_state)
    np.testing.assert_allclose(probabilities, [0.5, 0.5])

    for i, state in enumerate(states):
        if i == 0:
            Mprime = Measurement("00", targets=[1-index])
            states_00, probability_00 = Mprime.measurement_comp_basis(state)
            assert probability_00[0] == 1
            assert states_00[1] is None
        else:
            Mprime = Measurement("11", targets=[1-index])
            states_11, probability_11 = Mprime.measurement_comp_basis(state)
            assert probability_11[1] == 1
            assert states_11[0] is None
