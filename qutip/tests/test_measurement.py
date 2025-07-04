import numpy as np
import scipy
import pytest
from math import sqrt
from qutip import (
    Qobj, basis, ket2dm, sigmax, sigmay, sigmaz, identity, num, tensor,
    rand_ket
)
from qutip.measurement import (
    measure_povm, measurement_statistics_povm, measure_observable,
    measurement_statistics_observable,
)


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
    return [(v, e.proj()) for v, e in pairs]


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
    return 1 - abs(left.overlap(right)) < tol


@pytest.mark.parametrize(["op", "state", "pairs", "probabilities"], [
    pytest.param(sigmax(), basis(2, 0), SIGMAX, [0.5, 0.5], id="sigmax_ket"),
    pytest.param(sigmax(), basis(2, 0).proj(), SIGMAX, [0.5, 0.5],
                 id="sigmax_dm"),
    pytest.param(sigmay(), basis(2, 0), SIGMAY, [0.5, 0.5], id="sigmay_ket"),
    pytest.param(sigmay(), basis(2, 0).proj(), SIGMAY, [0.5, 0.5],
                 id="sigmay_dm"),
])
def test_measurement_statistics_observable(op, state, pairs, probabilities):
    """ measurement_statistics_observable: observables on basis states. """

    evs, projs, probs = measurement_statistics_observable(state, op)
    assert len(projs) == len(probabilities)
    np.testing.assert_almost_equal(probs, probabilities)
    for a, b in zip(projs, pairs.projectors):
        assert _equivalent(a, b)


def test_measurement_statistics_observable_degenerate():
    """ measurement_statistics_observable: observables on basis states. """

    state = basis(2, 1) & (basis(2, 0) + basis(2, 1)).unit()
    op = sigmaz() & identity(2)
    expected_projector = num(2) & identity(2)
    evs, projs, probs = measurement_statistics_observable(state, op)
    assert len(probs) == 1
    np.testing.assert_almost_equal(probs, [1.])
    np.testing.assert_almost_equal(evs, [-1.])
    assert _equivalent(projs[0], expected_projector)


@pytest.mark.parametrize(["ops", "state", "final_states", "probabilities"], [
    pytest.param(PZ, basis(2, 0), [state0, None], [1, 0], id="PZ_ket"),
    pytest.param(PZ, basis(2, 0).proj(), [state0.proj(), None], [1, 0],
                 id="PZ_dm"),
    pytest.param(PZ_ket, basis(2, 0), [state0, None], [1, 0], id="PZket_ket"),
    pytest.param(PZ_ket, basis(2, 0).proj(), [state0.proj(), None], [1, 0],
                 id="PZket_dm"),
    pytest.param(PX, basis(2, 0), [stateplus, stateminus], [0.5, 0.5],
                 id="PX_ket"),
    pytest.param(PX, basis(2, 0).proj(), [stateplus.proj(), stateminus.proj()],
                 [0.5, 0.5], id="PX_dm"),
    pytest.param(PY, basis(2, 0), [stateR, stateL], [0.5, 0.5], id="PY_ket"),
    pytest.param(PY, basis(2, 0).proj(), [stateR.proj(), stateL.proj()],
                 [0.5, 0.5], id="PY_dm")])
def test_measurement_statistics_povm(ops, state, final_states, probabilities):
    """ measurement_statistics_povm: projectors applied to basis states. """

    collapsed_states, probs = measurement_statistics_povm(state, ops)
    for i, final_state in enumerate(final_states):
        collapsed_state = collapsed_states[i]
        if final_state:
            assert collapsed_state == final_state
        else:
            assert collapsed_state is None
    np.testing.assert_almost_equal(probs, probabilities)


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
        assert final_state == collapsed_states[index]


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

    M_1 = E_1.sqrtm()
    M_2 = E_2.sqrtm()
    M_3 = E_3.sqrtm()

    ket1 = basis(2, 0)
    ket2 = (basis(2, 0) + basis(2, 1))/(sqrt(2))

    dm1 = ket2dm(ket1)
    dm2 = ket2dm(ket2)

    M = [M_1, M_2, M_3]

    _, probabilities = measurement_statistics_povm(ket1, M)
    np.testing.assert_allclose(probabilities, [0, 0.293, 0.707], atol=0.001)

    _, probabilities = measurement_statistics_povm(ket2, M)
    np.testing.assert_allclose(probabilities, [0.293, 0, 0.707], atol=0.001)

    _, probabilities = measurement_statistics_povm(dm1, M)
    np.testing.assert_allclose(probabilities, [0, 0.293, 0.707], atol=0.001)

    _, probabilities = measurement_statistics_povm(dm2, M)
    np.testing.assert_allclose(probabilities, [0.293, 0, 0.707], atol=0.001)
