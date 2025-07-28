import pytest
import numpy as np
import qutip as qt
from qutip.quantum_systems.qubit import qubit
from qutip.quantum_systems.quantum_system import QuantumSystem


class TestQubit:
    """Balanced test suite for qubit function"""

    def test_default_qubit(self):
        """Test qubit with default parameters"""
        q = qubit()

        assert isinstance(q, QuantumSystem)
        assert q.name == "Qubit"
        assert q.parameters["omega"] == 1.0
        assert q.parameters["decay_rate"] == 0.0
        assert q.parameters["dephasing_rate"] == 0.0
        assert q.dimension == 2

    @pytest.mark.parametrize("omega", [0.5, 1.0, 2.0, -1.0, 0.0])
    def test_omega_parameter(self, omega):
        """Test qubit with different omega values"""
        q = qubit(omega=omega)

        assert q.parameters["omega"] == omega

        # Check Hamiltonian scaling
        expected_H = 0.5 * omega * qt.sigmaz()
        assert (q.hamiltonian - expected_H).norm() < 1e-10

        # Check eigenvalues
        eigenvals = q.eigenvalues
        expected_eigenvals = np.array([-omega / 2, omega / 2])
        np.testing.assert_array_almost_equal(
            sorted(eigenvals), sorted(expected_eigenvals))

    @pytest.mark.parametrize("decay_rate", [0.0, 0.1, 0.5])
    def test_decay_parameter(self, decay_rate):
        """Test qubit with different decay rates"""
        q = qubit(decay_rate=decay_rate)

        assert q.parameters["decay_rate"] == decay_rate

        if decay_rate == 0.0:
            assert len(q.c_ops) == 0
        else:
            assert len(q.c_ops) == 1
            expected_decay_op = np.sqrt(decay_rate) * qt.destroy(2)
            decay_found = any((c_op - expected_decay_op).norm()
                              < 1e-10 for c_op in q.c_ops)
            assert decay_found

    @pytest.mark.parametrize("dephasing_rate", [0.0, 0.1, 0.5])
    def test_dephasing_parameter(self, dephasing_rate):
        """Test qubit with different dephasing rates"""
        q = qubit(dephasing_rate=dephasing_rate)

        assert q.parameters["dephasing_rate"] == dephasing_rate

        if dephasing_rate == 0.0:
            assert len(q.c_ops) == 0
        else:
            assert len(q.c_ops) == 1
            expected_dephasing_op = np.sqrt(dephasing_rate) * qt.sigmaz()
            dephasing_found = any(
                (c_op - expected_dephasing_op).norm() < 1e-10 for c_op in q.c_ops)
            assert dephasing_found

    @pytest.mark.parametrize("omega,decay_rate,dephasing_rate", [
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
            dephasing_rate=dephasing_rate)

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

    def test_operators_completeness(self):
        """Test that all expected operators are present"""
        q = qubit()

        expected_operators = [
            "sigma_minus",
            "sigma_plus",
            "sigma_z",
            "sigma_x",
            "sigma_y"]
        for op_name in expected_operators:
            assert op_name in q.operators
        assert len(q.operators) == len(expected_operators)

    @pytest.mark.parametrize("operator_name,expected_operator", [
        ("sigma_minus", qt.destroy(2)),
        ("sigma_plus", qt.create(2)),
        ("sigma_z", qt.sigmaz()),
        ("sigma_x", qt.sigmax()),
        ("sigma_y", qt.sigmay()),
    ])
    def test_operators_correctness(self, operator_name, expected_operator):
        """Test that operators are correct"""
        q = qubit()
        assert q.operators[operator_name] == expected_operator

    @pytest.mark.parametrize("op_name", ["sigma_x", "sigma_y", "sigma_z"])
    def test_pauli_properties(self, op_name):
        """Test Pauli operator properties"""
        q = qubit()
        op = q.operators[op_name]

        # Should be Hermitian
        assert (op - op.dag()).norm() < 1e-10

        # Should square to identity
        assert (op * op - qt.qeye(2)).norm() < 1e-10

    def test_raising_lowering_operators(self):
        """Test raising and lowering operators"""
        q = qubit()

        sigma_plus = q.operators["sigma_plus"]
        sigma_minus = q.operators["sigma_minus"]
        ground = qt.basis(2, 0)
        excited = qt.basis(2, 1)

        # σ_plus = (σ_minus)dagger
        assert (sigma_plus - sigma_minus.dag()).norm() < 1e-10

        # Test basic actions
        assert (sigma_plus * ground - excited).norm() < 1e-10
        assert (sigma_minus * excited - ground).norm() < 1e-10
        assert (sigma_plus * excited).norm() < 1e-10
        assert (sigma_minus * ground).norm() < 1e-10

    @pytest.mark.parametrize("op1,op2", [
        ("sigma_z", "sigma_x"),
        ("sigma_z", "sigma_y"),
        ("sigma_x", "sigma_y"),
    ])
    def test_pauli_anticommutation(self, op1, op2):
        """Test Pauli operator anticommutation relations"""
        q = qubit()

        op_1 = q.operators[op1]
        op_2 = q.operators[op2]

        # Different Pauli operators should anticommute
        anticommutator = op_1 * op_2 + op_2 * op_1
        assert anticommutator.norm() < 1e-10

    def test_ground_state(self):
        """Test ground state properties"""
        q = qubit(omega=1.0)
        ground_state = q.ground_state

        # Ground state should be normalized
        assert abs(ground_state.norm() - 1.0) < 1e-10

        # Should be eigenstate with lowest eigenvalue
        eigenvals = q.eigenvalues
        min_eigenval = min(eigenvals)

        H_psi = q.hamiltonian * ground_state
        eigenval_psi = min_eigenval * ground_state
        assert (H_psi - eigenval_psi).norm() < 1e-10

    def test_hamiltonian_structure(self):
        """Test Hamiltonian structure and properties"""
        q = qubit()

        # Check dimension and Hermiticity
        assert q.hamiltonian.shape == (2, 2)
        H = q.hamiltonian
        assert (H - H.dag()).norm() < 1e-10

    def test_latex_representation(self):
        """Test LaTeX representation"""
        q = qubit()
        expected_latex = r"H = \frac{\omega}{2}\sigma_z"
        assert q.latex == expected_latex

    def test_return_type_and_attributes(self):
        """Test return type and required attributes"""
        q = qubit()

        assert isinstance(q, QuantumSystem)

        required_attributes = [
            'name',
            'operators',
            'hamiltonian',
            'c_ops',
            'latex',
            'parameters']
        for attr in required_attributes:
            assert hasattr(q, attr)

    def test_pretty_print_basic(self, capsys):
        """Test pretty_print output"""
        q = qubit(omega=2.0, decay_rate=0.1)
        q.pretty_print()

        captured = capsys.readouterr()
        assert "Qubit" in captured.out
        assert "Dimension: 2" in captured.out
        assert "omega" in captured.out

    def test_edge_cases(self):
        """Test edge cases"""
        # Very small parameters
        q_small = qubit(omega=1e-10, decay_rate=1e-10)
        assert q_small.dimension == 2

        # Large parameters
        q_large = qubit(omega=1e10, decay_rate=1e5)
        assert q_large.dimension == 2
