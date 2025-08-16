import numpy as np
import qutip as qt
from qutip.quantum_systems.quantum_system import QuantumSystem


def test_quantum_system_initialization():
    qs = QuantumSystem("TestSystem", param1=1.0, param2=2.0)
    assert qs.name == "TestSystem"
    assert qs.parameters["param1"] == 1.0
    assert qs.dimension == 0
    assert isinstance(qs.get_operators(), dict)
    assert isinstance(qs.get_c_ops(), list)
    assert qs.get_latex() == ""


def test_quantum_system_set_and_get_methods():
    qs = QuantumSystem("Test")
    qs.hamiltonian = qt.qeye(2)
    qs.operators = {"op": qt.sigmax()}
    qs.c_ops = [qt.sigmay()]
    qs.latex = "H=σx"

    assert qs.get_hamiltonian() == qt.qeye(2)
    assert "op" in qs.get_operators()
    assert isinstance(qs.get_c_ops()[0], qt.Qobj)
    assert qs.get_latex() == "H=σx"
    assert qs.dimension == 2


def test_quantum_system_eigenvalues_and_ground_state():
    H = qt.sigmaz()
    qs = QuantumSystem("Test")
    qs.hamiltonian = H
    vals = qs.eigenvalues
    assert np.allclose(vals, [-1, 1])
    _, states = qs.eigenstates
    assert len(states) == 2
    assert qs.ground_state.isket
