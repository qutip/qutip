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
from qutip.qip.circuit import Measurement
from qutip import (Qobj, basis, isequal, ket2dm,
                    sigmax, sigmay, sigmaz, identity, tensor, rand_ket)
from qutip.measurement import (measure_povm, measurement_statistics_povm,
                                measure_observable,
                                measurement_statistics_observable)


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

state0 = basis(2, 0)
state1 = basis(2, 1)
stateplus = (basis(2, 0) + basis(2, 1)).unit()
stateminus = (basis(2, 0) - basis(2, 1)).unit()
stateR = (basis(2, 0) + 1j * basis(2, 1)).unit()
stateL = (basis(2, 0) - 1j * basis(2, 1)).unit()
PZ = [ket2dm(state0), ket2dm(state1)]
PX = [ket2dm(stateplus), ket2dm(stateminus)]
PY = [ket2dm(stateR), ket2dm(stateL)]
PZ_ket = [state0, state1]

def _equivalent(left, right, tol=1e-8):
    """ Equal up to a phase """
    return 1 - abs( (left.dag() * right).tr()) < tol


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
def test_measurement_statistics_observable(op, state, pairs, probabilities):
    """ measurement_statistics_observable: observables on basis states. """

    evs, ess_or_projs, probs = measurement_statistics_observable(state, op)
    np.testing.assert_almost_equal(evs, pairs.eigenvalues)
    if state.isket:
        ess = ess_or_projs
        assert len(ess) == len(pairs.eigenstates)
        for a, b in zip(ess, pairs.eigenstates):
            assert (_equivalent(a, b))

    else:
        projs = ess_or_projs
        assert len(projs) == len(pairs.projectors)
        for a, b in zip(projs, pairs.projectors):
            assert (_equivalent(a, b))
    np.testing.assert_almost_equal(probs, probabilities)


@pytest.mark.parametrize(["op", "state"], [
                    pytest.param(sigmax(), tensor(basis(2, 0), basis(2, 0)),
                                id="partial_ket_observable"),
                    pytest.param(sigmaz(), tensor(ket2dm(basis(2, 0)),
                                                    ket2dm(basis(2, 0))),
                                id="partial_dm_observable")])
def test_measurement_statistics_observable_ind(op, state):
    """ measurement_statistics_observable: observables on basis
        states with targets. """

    evs1, ess_or_projs1, probs1 = measurement_statistics_observable(
                                                state, tensor(op, identity(2)))
    evs2, ess_or_projs2, probs2 = measurement_statistics_observable(
                                                state, op, targets=[0])
    np.testing.assert_almost_equal(evs1, evs2)
    for a, b in zip(ess_or_projs1, ess_or_projs2):
        assert isequal(a, b)
    np.testing.assert_almost_equal(probs1, probs2)


@pytest.mark.parametrize(["ops", "state", "final_states", "probabilities"], [
                    pytest.param(PZ, basis(2, 0),
                            [state0, None], [1, 0], id="PZ_ket"),
                    pytest.param(PZ, ket2dm(basis(2, 0)),
                            [ket2dm(state0), None], [1, 0], id="PZ_dm"),
                    pytest.param(PZ_ket, basis(2, 0),
                            [state0, None], [1, 0], id="PZket_ket"),
                    pytest.param(PZ_ket, ket2dm(basis(2, 0)),
                            [ket2dm(state0), None], [1, 0], id="PZket_dm"),
                    pytest.param(PX, basis(2, 0),
                            [stateplus, stateminus], [0.5, 0.5], id="PX_ket"),
                    pytest.param(PX, ket2dm(basis(2, 0)),
                            [ket2dm(stateplus), ket2dm(stateminus)],
                            [0.5, 0.5], id="PX_dm"),
                    pytest.param(PY, basis(2, 0),
                            [stateR, stateL], [0.5, 0.5], id="PY_ket"),
                    pytest.param(PY, ket2dm(basis(2, 0)),
                            [ket2dm(stateR), ket2dm(stateL)],
                            [0.5, 0.5], id="PY_dm")])
def test_measurement_statistics_povm(ops, state, final_states, probabilities):
    """ measurement_statistics_povm: projectors applied to basis states. """

    collapsed_states, probs = measurement_statistics_povm(state, ops)
    for i, final_state in enumerate(final_states):
        collapsed_state = collapsed_states[i]
        if final_state:
            assert isequal(collapsed_state, final_state)
        else:
            assert collapsed_state is None
    np.testing.assert_almost_equal(probs, probabilities)


@pytest.mark.parametrize(["ops", "state"], [
                    pytest.param(PX, tensor(basis(2, 0), basis(2, 0)),
                                id="partial_ket"),
                    pytest.param(PX,
                                tensor(ket2dm(basis(2, 0)),
                                        ket2dm(basis(2, 0))),
                                id="partial_dm")])
def test_measurement_statistics_ind(ops, state):
    """ measurement_statistics_povm: projectors on basis states with targets. """

    states1, probs1 = measurement_statistics_povm(
                                    state,
                                    [tensor(op, identity(2)) for op in ops])
    states2, probs2 = measurement_statistics_povm(state, ops, targets=[0])

    for a, b in zip(states1, states2):
        assert isequal(a, b)
    np.testing.assert_almost_equal(probs1, probs2)


def test_measurement_statistics_povm_input_errors():
    """ measurement_statistics_povm: check input errors """

    np.testing.assert_raises_regex(
        ValueError, "op must be all operators or all kets",
        measurement_statistics_povm,
        basis(2, 0), [basis(2, 0), ket2dm(basis(2, 0))])
    np.testing.assert_raises_regex(
        TypeError, "state must be a Qobj",
        measurement_statistics_povm, "notqobj", [sigmaz()])
    np.testing.assert_raises_regex(
        ValueError, "state must be a ket or a density matrix",
        measurement_statistics_povm, basis(2, 0).dag(), [sigmaz()])
    np.testing.assert_raises_regex(
        ValueError,
        "op and state dims should be compatible when state is a ket",
        measurement_statistics_povm, basis(3, 0), [sigmaz()])
    np.testing.assert_raises_regex(
        ValueError,
        "op and state dims should match when state is a density matrix",
        measurement_statistics_povm, ket2dm(basis(3, 0)), [sigmaz()])
    np.testing.assert_raises_regex(
        ValueError,
        "measurement operators must sum to identity",
        measurement_statistics_povm, basis(2, 0), [basis(2, 0)])
    np.testing.assert_raises_regex(
        ValueError,
        "measurement operators must sum to identity",
        measurement_statistics_povm, basis(2, 0), [ket2dm(basis(2, 0))])


def test_measurement_statistics_observable_input_errors():
    """ measurement_statistics_observable: check input errors """

    np.testing.assert_raises_regex(
        TypeError, "op must be a Qobj",
        measurement_statistics_observable, basis(2, 0), "notqobj")
    np.testing.assert_raises_regex(
        ValueError, "op must be all operators or all kets",
        measurement_statistics_observable, basis(2, 0), basis(2, 1))
    np.testing.assert_raises_regex(
        TypeError, "state must be a Qobj",
        measurement_statistics_observable, "notqobj",  sigmaz())
    np.testing.assert_raises_regex(
        ValueError, "state must be a ket or a density matrix",
        measurement_statistics_observable, basis(2, 0).dag(), sigmaz())
    np.testing.assert_raises_regex(
        ValueError,
        "op and state dims should be compatible when state is a ket",
        measurement_statistics_observable, basis(3, 0), sigmaz())
    np.testing.assert_raises_regex(
        ValueError,
        "op and state dims should match when state is a density matrix",
        measurement_statistics_observable, ket2dm(basis(3, 0)), sigmaz())


@pytest.mark.parametrize(["op", "state"], [
                    pytest.param(sigmaz(), basis(2, 0), id="sigmaz_ket1"),
                    pytest.param(sigmaz(), basis(2, 1), id="sigmaz_ket2"),
                    pytest.param(sigmaz(), ket2dm(basis(2, 0)),
                                 id="sigmaz_dm1"),
                    pytest.param(sigmaz(), ket2dm(basis(2, 1)),
                                 id="sigmaz_dm2"),

                    pytest.param(sigmax(), basis(2, 0), id="sigmax_ket1"),
                    pytest.param(sigmax(), basis(2, 1), id="sigmax_ket2"),
                    pytest.param(sigmax(), ket2dm(basis(2, 0)),
                                 id="sigmax_dm"),

                    pytest.param(sigmay(), basis(2, 0), id="sigmay_ket1"),
                    pytest.param(sigmay(), basis(2, 1), id="sigmax_ket2"),
                    pytest.param(sigmay(), ket2dm(basis(2, 1)),
                                 id="sigmay_dm")])
def test_measure_observable(op, state):
    """ measure_observable: basis states using different observables """
    evs, ess_or_projs, prob = measurement_statistics_observable(state, op)

    expected_measurements = EigenPairs(list(zip(evs, ess_or_projs)))
    for _ in range(10):
        assert (measure_observable(state, op) in expected_measurements)


@pytest.mark.parametrize(["ops", "state"], [

                    pytest.param(PZ, basis(2, 0), id="PZ_ket1"),
                    pytest.param(PZ, basis(2, 1), id="PZ_ket2"),
                    pytest.param(PZ, ket2dm(basis(2, 0)), id="PZ_dm1"),
                    pytest.param(PZ, ket2dm(basis(2, 1)), id="PZ_dm2"),

                    pytest.param(PZ_ket, basis(2, 0), id="PZket_ket1"),
                    pytest.param(PZ_ket, basis(2, 1), id="PZket_ket2"),
                    pytest.param(PZ_ket, ket2dm(basis(2, 0)), id="PZket_dm1"),
                    pytest.param(PZ_ket, ket2dm(basis(2, 1)), id="PZket_dm2"),

                    pytest.param(PX, basis(2, 0), id="PX_ket1"),
                    pytest.param(PX, basis(2, 1), id="PX_ket2"),
                    pytest.param(PX, ket2dm(basis(2, 0)), id="PX_dm"),

                    pytest.param(PY, basis(2, 0), id="PY_ket1"),
                    pytest.param(PY, basis(2, 1), id="PY_ket2"),
                    pytest.param(PY, ket2dm(basis(2, 1)), id="PY_dm")])
def test_measure(ops, state):
    """measure_povm: test on basis states using different projectors """

    collapsed_states, _ = measurement_statistics_povm(state, ops)
    for _ in range(10):
        index, final_state = measure_povm(state, ops)
        assert isequal(final_state, collapsed_states[index])


def test_measure_input_errors():
    """ measure_povm: check input errors """
    np.testing.assert_raises_regex(
        ValueError, "op must be all operators or all kets",
        measure_povm, basis(2, 0), [basis(2, 0), ket2dm(basis(2, 0))])
    np.testing.assert_raises_regex(
        TypeError, "state must be a Qobj",
        measure_povm, "notqobj", [sigmaz()])
    np.testing.assert_raises_regex(
        ValueError, "state must be a ket or a density matrix",
        measure_povm, basis(2, 0).dag(), [sigmaz()])
    np.testing.assert_raises_regex(
        ValueError,
        "op and state dims should be compatible when state is a ket",
        measure_povm, basis(3, 0), [sigmaz()])
    np.testing.assert_raises_regex(
        ValueError,
        "op and state dims should match when state is a density matrix",
        measure_povm, ket2dm(basis(3, 0)), [sigmaz()])
    np.testing.assert_raises_regex(
        ValueError,
        "measurement operators must sum to identity",
        measure_povm, basis(2, 0), [basis(2, 0)])
    np.testing.assert_raises_regex(
        ValueError,
        "measurement operators must sum to identity",
        measure_povm, basis(2, 0), [ket2dm(basis(2, 0))])


def test_measure_observable_input_errors():
    """ measure_observable: check input errors """
    np.testing.assert_raises_regex(
        TypeError, "op must be a Qobj",
        measure_observable, basis(2, 0), "notqobj")
    np.testing.assert_raises_regex(
        ValueError, "op must be all operators or all kets",
        measure_observable, basis(2, 0),  basis(2, 1))
    np.testing.assert_raises_regex(
        TypeError, "state must be a Qobj",
        measure_observable, "notqobj", sigmaz())
    np.testing.assert_raises_regex(
        ValueError, "state must be a ket or a density matrix",
        measure_observable, basis(2, 0).dag(), sigmaz())
    np.testing.assert_raises_regex(
        ValueError,
        "op and state dims should be compatible when state is a ket",
        measure_observable, basis(3, 0), sigmaz())
    np.testing.assert_raises_regex(
        ValueError,
        "op and state dims should match when state is a density matrix",
        measure_observable, ket2dm(basis(3, 0)), sigmaz())


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

    M = [M_1, M_2, M_3]

    _, probabilities = measurement_statistics_povm(ket1, M)
    np.testing.assert_allclose(probabilities, [0, 0.293, 0.707], 0.001)

    _, probabilities = measurement_statistics_povm(ket2, M)
    np.testing.assert_allclose(probabilities, [0.293, 0, 0.707], 0.001)

    _, probabilities = measurement_statistics_povm(dm1, M)
    np.testing.assert_allclose(probabilities, [0, 0.293, 0.707], 0.001)

    _, probabilities = measurement_statistics_povm(dm2, M)
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
