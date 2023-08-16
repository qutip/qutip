import pytest
import numpy as np
import qutip
from qutip.core import gates


def _infidelity(a, b):
    """Infidelity between two kets."""
    return 1 - abs(a.overlap(b))


def _remove_global_phase(qobj):
    """
    Return a new Qobj with the gauge fixed for the global phase.  Explicitly,
    we set the first non-zero element to be purely real-positive.
    """
    flat = qobj.tidyup(1e-14).full().flat.copy()
    for phase in flat:
        if phase != 0:
            # Fix the gauge for any global phase.
            flat = flat * np.exp(-1j * np.angle(phase))
            break
    return qutip.Qobj(flat.reshape(qobj.shape), dims=qobj.dims)


def _make_random_three_qubit_gate():
    """Create a random three-qubit gate."""
    operation = qutip.rand_unitary(8, dims=[[2]*3]*2)

    def gate(N=None, controls=None, target=None):
        if N is None:
            return operation
        return gates.gate_expand_3toN(operation, N, controls, target)
    return gate


def _make_controled(op):
    out = qutip.tensor(qutip.fock_dm(2, 0), qutip.qeye_like(op))
    out += qutip.tensor(qutip.fock_dm(2, 1), op)
    return out


class TestExplicitForm:
    def test_swap(self):
        states = [qutip.rand_ket(2) for _ in [None]*2]
        start = qutip.tensor(states)
        swapped = qutip.tensor(states[::-1])
        swap = gates.swap()
        assert _infidelity(swapped, swap*start) < 1e-12
        assert _infidelity(start, swap*swap*start) < 1e-12

    @pytest.mark.parametrize(["gate", "cgate", "args"], [
        pytest.param(qutip.sigmax, gates.cnot, (), id="cnot"),
        pytest.param(qutip.sigmay, gates.cy_gate, (), id="cy_gate"),
        pytest.param(qutip.sigmaz, gates.cz_gate, (), id="cz_gate"),
        pytest.param(gates.s_gate, gates.cs_gate, (), id="cs_gate"),
        pytest.param(gates.t_gate, gates.ct_gate, (), id="ct_gate"),
        pytest.param(gates.phasegate, gates.cphase, (0.1,), id="cphase"),
        pytest.param(qutip.sigmaz, gates.csign, (), id="csign"),
        pytest.param(gates.swap, gates.fredkin, (), id="fredkin"),
        pytest.param(gates.cnot, gates.toffoli, (), id="toffoli"),
    ])
    def test_controled(self, gate, cgate, args):
        expected = _make_controled(gate(*args))
        assert cgate(*args) == expected

    @pytest.mark.parametrize(["gate", "power", "expected"], [
        pytest.param(gates.snot, 2, lambda : qutip.qeye(2), id="snot"),
        pytest.param(gates.s_gate, 4, lambda : qutip.qeye(2), id="s_gate"),
        pytest.param(gates.t_gate, 8, lambda : qutip.qeye(2), id="t_gate"),
        pytest.param(gates.berkeley, 8, lambda : -qutip.qeye([2, 2]),
                     id="berkeley"),
        pytest.param(gates.sqrtnot, 2, qutip.sigmax, id="sqrtnot"),
        pytest.param(gates.sqrtswap, 2, gates.swap, id="cs_gate"),
        pytest.param(gates.sqrtiswap, 2, gates.iswap, id="ct_gate"),
    ])
    def gate_power_relation(self, gate, expected, power):
        assert gate()**power == expected()

    @pytest.mark.parametrize(['angle', 'expected'], [
        pytest.param(np.pi, -1j*qutip.tensor(qutip.sigmax(), qutip.sigmax()),
                     id="pi"),
        pytest.param(2*np.pi, -qutip.qeye([2, 2]), id="2pi"),
    ])
    def test_molmer_sorensen(self, angle, expected):
        np.testing.assert_allclose(gates.molmer_sorensen(angle).full(),
                                   expected.full(), atol=1e-15)

    @pytest.mark.parametrize(["gate", "n_angles"], [
        pytest.param(gates.rx, 1, id="Rx"),
        pytest.param(gates.ry, 1, id="Ry"),
        pytest.param(gates.rz, 1, id="Rz"),
        pytest.param(gates.phasegate, 1, id="phase"),
        pytest.param(gates.qrot, 2, id="Rabi rotation"),
    ])
    def test_zero_rotations_are_identity(self, gate, n_angles):
        np.testing.assert_allclose(np.eye(2), gate(*([0]*n_angles)).full(),
                                   atol=1e-15)

    def test_hadamard(self):
        N = 3
        expected = qutip.tensor([gates.snot()] * N)
        assert gates.hadamard_transform(N) == expected

    def test_globalphase(self):
        assert gates.globalphase(1, 3) * np.exp(-1j) == qutip.qeye([2] * 3)


class TestCliffordGroup:
    """
    Test a sufficient set of conditions to prove that we have a full Clifford
    group for a single qubit.
    """
    clifford = gates.qubit_clifford_group()
    pauli = [qutip.qeye(2), qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()]

    def test_single_qubit_group_dimension_is_24(self):
        assert len(self.clifford) == 24

    def test_all_elements_different(self):
        clifford = [_remove_global_phase(gate) for gate in self.clifford]
        for i, gate in enumerate(clifford):
            for other in clifford[i+1:]:
                # Big tolerance because we actually want to test the inverse.
                assert not np.allclose(gate.full(), other.full(), atol=1e-3)

    @pytest.mark.parametrize("gate", gates.qubit_clifford_group())
    def test_gate_normalises_pauli_group(self, gate):
        """
        Test the fundamental definition of the Clifford group, i.e. that it
        normalises the Pauli group.
        """
        # Assert that each Clifford gate maps the set of Pauli gates back onto
        # itself (though not necessarily in order).  This condition is no
        # stronger than simply considering each (gate, Pauli) pair separately.
        pauli_gates = [_remove_global_phase(x) for x in self.pauli]
        normalised = [_remove_global_phase(gate * pauli * gate.dag())
                      for pauli in self.pauli]
        for gate in normalised:
            for i, pauli in enumerate(pauli_gates):
                if np.allclose(gate.full(), pauli.full(), atol=1e-10):
                    del pauli_gates[i]
                    break
        assert len(pauli_gates) == 0
