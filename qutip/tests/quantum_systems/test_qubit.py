import pytest
import numpy as np
from qutip import destroy, sigmax, sigmaz, sigmay, create
from qutip.quantum_systems.qubit import qubit
from qutip.quantum_systems.quantum_system import QuantumSystem


class TestQubit:
    """Balanced test suite for qubit function"""

    def test_default_qubit(self):
        """Test qubit with default parameters"""
        # Random values for parameters
        omega_val = 0.5
        decay_val = 0.1
        dephasing_val = 0.05
        q = qubit(
            omega=omega_val,
            decay_rate=decay_val,
            dephasing_rate=dephasing_val
        )

        assert isinstance(q, QuantumSystem)
        assert q.name == "Qubit"

        # Test existence
        assert "omega" in q.parameters
        assert "decay_rate" in q.parameters
        assert "dephasing_rate" in q.parameters

        # Test correct storage
        assert q.parameters["omega"] == omega_val
        assert q.parameters["decay_rate"] == decay_val
        assert q.parameters["dephasing_rate"] == dephasing_val

        assert q.dimension == [[2], [2]]

    @pytest.mark.parametrize("omega", [0.5, 1.0, 2.0, -1.0, 0.0])
    def test_omega_parameter(self, omega):
        """Test qubit with different omega values"""
        q = qubit(omega=omega)

        assert q.parameters["omega"] == omega

        # Check Hamiltonian scaling
        expected_H = 0.5 * omega * sigmaz()
        assert q.hamiltonian == expected_H

        # Check eigenvalues
        eigs = np.asarray(q.eigenvalues)
        expected = np.array([omega / 2, -omega / 2])
        np.testing.assert_allclose(np.sort(eigs), np.sort(expected))

    @pytest.mark.parametrize("decay_rate", [0.0, 0.1, 0.5])
    def test_decay_parameter(self, decay_rate):
        """Test qubit with different decay rates"""
        q = qubit(decay_rate=decay_rate)

        assert q.parameters["decay_rate"] == decay_rate

        if decay_rate == 0.0:
            assert len(q.c_ops) == 0
        else:
            assert len(q.c_ops) == 1
            expected_decay_op = np.sqrt(decay_rate) * destroy(2)
            assert expected_decay_op in q.c_ops

    @pytest.mark.parametrize("dephasing_rate", [0.0, 0.1, 0.5])
    def test_dephasing_parameter(self, dephasing_rate):
        """Test qubit with different dephasing rates"""
        q = qubit(dephasing_rate=dephasing_rate)

        assert q.parameters["dephasing_rate"] == dephasing_rate

        if dephasing_rate == 0.0:
            assert len(q.c_ops) == 0
        else:
            assert len(q.c_ops) == 1
            expected_dephasing_op = np.sqrt(dephasing_rate) * sigmaz()
            assert expected_dephasing_op in q.c_ops

    @pytest.mark.parametrize("omega, decay_rate, dephasing_rate", [
        (1.0, 0.1, 0.0),    # Decay only
        (1.0, 0.0, 0.1),    # Dephasing only
        (1.0, 0.1, 0.1),    # Both mechanisms
        (2.0, 0.2, 0.3),    # Different rates
    ])
    def test_combined_parameters(self, omega, decay_rate, dephasing_rate):
        """Test qubit with combined parameter variations"""
        q = qubit(
            omega=omega,
            decay_rate=decay_rate,
            dephasing_rate=dephasing_rate
        )

        # Check parameter storage
        assert q.parameters["omega"] == omega
        assert q.parameters["decay_rate"] == decay_rate
        assert q.parameters["dephasing_rate"] == dephasing_rate

        # Check collapse operators count
        expected_c_ops_count = 0
        if decay_rate > 0:
            expected_c_ops_count += 1
        if dephasing_rate > 0:
            expected_c_ops_count += 1
        assert len(q.c_ops) == expected_c_ops_count

    def test_operators_correctness(self):
        """Test that operators are correct"""
        q = qubit()

        assert q.operators["sigma_minus"] == destroy(2)
        assert q.operators["sigma_plus"] == create(2)
        assert q.operators["sigma_z"] == sigmaz()
        assert q.operators["sigma_x"] == sigmax()
        assert q.operators["sigma_y"] == sigmay()

    def test_ground_state(self):
        """Test ground state properties"""
        q = qubit(omega=1.0)
        ground_state = q.ground_state

        # Should be eigenstate with lowest eigenvalue
        eigenvals = q.eigenvalues
        min_eigenval = min(eigenvals)

        H_psi = q.hamiltonian * ground_state
        eigenval_psi = min_eigenval * ground_state
        assert H_psi == eigenval_psi

    def test_hamiltonian_structure(self):
        """Test Hamiltonian structure and properties"""
        q = qubit()

        # Check dimension and Hermiticity
        assert q.hamiltonian.shape == (2, 2)
        assert q.hamiltonian == q.hamiltonian.dag()

    def test_latex_representation(self):
        """Test LaTeX representation"""
        q = qubit()
        expected_latex = r"H = \frac{\omega}{2}\sigma_z"
        assert q.latex == expected_latex

    def test_pretty_print_basic(self, capsys):
        """Test pretty_print output"""
        q = qubit(omega=2.0, decay_rate=0.1)
        q.pretty_print()

        captured = capsys.readouterr()
        assert "Qubit" in captured.out
        assert "Hilbert Space Dimension: [[2], [2]]" in captured.out
        assert "omega" in captured.out
