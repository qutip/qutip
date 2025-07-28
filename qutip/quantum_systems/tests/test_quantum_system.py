import pytest
import numpy as np
import qutip as qt
from qutip.quantum_systems.quantum_system import QuantumSystem


class TestQuantumSystem:
    """Test suite for QuantumSystem class"""

    def test_basic_initialization(self):
        """Test basic initialization"""
        system = QuantumSystem("Test System")

        assert system.name == "Test System"
        assert system.parameters == {}
        assert system.operators == {}
        assert system.hamiltonian is None
        assert system.c_ops == []
        assert system.latex == ""
        assert system.dimension == 0

    def test_initialization_with_parameters(self):
        """Test initialization with parameters"""
        params = {"omega": 1.0, "gamma": 0.1}
        system = QuantumSystem("Test System", **params)

        assert system.name == "Test System"
        assert system.parameters == params

    @pytest.mark.parametrize("name,params", [
        ("Qubit", {"omega": 1.0}),
        ("Oscillator", {"omega": 1.0, "n_levels": 10}),
        ("Complex", {"freq1": 1.0, "freq2": 1.5, "coupling": 0.1}),
    ])
    def test_various_initializations(self, name, params):
        """Test various initialization scenarios"""
        system = QuantumSystem(name, **params)
        assert system.name == name
        assert system.parameters == params

    def test_get_methods_empty(self):
        """Test getter methods on empty system"""
        system = QuantumSystem("Test")

        assert system.get_operators() == {}
        assert system.get_hamiltonian() is None
        assert system.get_c_ops() == []
        assert system.get_latex() == ""

    def test_get_methods_with_data(self):
        """Test getter methods with data"""
        system = QuantumSystem("Test")

        # Set data
        ops = {"sigma_x": qt.sigmax(), "sigma_z": qt.sigmaz()}
        system.operators = ops
        system.hamiltonian = qt.sigmaz()
        system.c_ops = [qt.destroy(2)]
        system.latex = r"H = \sigma_z"

        # Test getters
        assert system.get_operators() == ops
        assert system.get_hamiltonian() == qt.sigmaz()
        assert system.get_c_ops() == [qt.destroy(2)]
        assert system.get_latex() == r"H = \sigma_z"

    @pytest.mark.parametrize("hamiltonian,expected_dim", [
        (qt.sigmaz(), 2),
        (qt.sigmax(), 2),
        (qt.qeye(3), 3),
        (qt.qeye(5), 5),
        (qt.destroy(4), 4),
        (qt.tensor(qt.sigmaz(), qt.sigmaz()), 4),
    ])
    def test_dimension_property(self, hamiltonian, expected_dim):
        """Test dimension property with various Hamiltonians"""
        system = QuantumSystem("Test")
        system.hamiltonian = hamiltonian
        assert system.dimension == expected_dim

    def test_eigenvalues_qubit(self):
        """Test eigenvalues for qubit systems"""
        system = QuantumSystem("Test")
        system.hamiltonian = 0.5 * qt.sigmaz()

        eigenvals = system.eigenvalues
        expected = np.array([-0.5, 0.5])
        np.testing.assert_array_almost_equal(
            sorted(eigenvals), sorted(expected))

    def test_eigenvalues_harmonic_oscillator(self):
        """Test eigenvalues for harmonic oscillator"""
        system = QuantumSystem("Test")
        n_levels = 5
        system.hamiltonian = qt.num(n_levels) + 0.5 * qt.qeye(n_levels)

        eigenvals = system.eigenvalues
        expected = np.arange(n_levels) + 0.5
        np.testing.assert_array_almost_equal(
            sorted(eigenvals), sorted(expected))

    def test_eigenstates_properties(self):
        """Test eigenstate properties"""
        system = QuantumSystem("Test")
        system.hamiltonian = qt.sigmaz()

        eigenvals, eigenstates = system.eigenstates

        # Check counts and normalization
        assert len(eigenvals) == 2
        assert len(eigenstates) == 2

        for state in eigenstates:
            assert abs(state.norm() - 1.0) < 1e-10

        # Check eigenvalue equation
        for eigenval, eigenstate in zip(eigenvals, eigenstates):
            H_psi = system.hamiltonian * eigenstate
            E_psi = eigenval * eigenstate
            assert (H_psi - E_psi).norm() < 1e-10

    @pytest.mark.parametrize("hamiltonian,expected_ground", [
        (qt.sigmaz(), qt.basis(2, 1)),           # |1⟩ for sigma_z
        (-qt.sigmaz(), qt.basis(2, 0)),          # |0⟩ for -sigma_z
        (qt.num(4), qt.basis(4, 0)),            # |0⟩ for number operator
    ])
    def test_ground_state_identification(self, hamiltonian, expected_ground):
        """Test ground state identification"""
        system = QuantumSystem("Test")
        system.hamiltonian = hamiltonian

        ground_state = system.ground_state

        # Should be normalized
        assert abs(ground_state.norm() - 1.0) < 1e-10

        # Should be eigenstate with lowest eigenvalue
        eigenvals = system.eigenvalues
        min_eigenval = min(eigenvals)

        H_psi = hamiltonian * ground_state
        E_psi = min_eigenval * ground_state
        assert (H_psi - E_psi).norm() < 1e-10

        # Check specific expected ground state
        overlap = abs(ground_state.overlap(expected_ground))
        assert overlap > 0.99

    def test_operators_management(self):
        """Test operator storage and retrieval"""
        system = QuantumSystem("Test")

        # Add operators
        system.operators["sigma_x"] = qt.sigmax()
        system.operators["sigma_z"] = qt.sigmaz()

        assert "sigma_x" in system.operators
        assert "sigma_z" in system.operators
        assert system.operators["sigma_x"] == qt.sigmax()
        assert system.operators["sigma_z"] == qt.sigmaz()

    def test_collapse_operators_management(self):
        """Test collapse operator storage and retrieval"""
        system = QuantumSystem("Test")

        # Add collapse operators
        c_op1 = qt.destroy(2)
        c_op2 = qt.sigmaz()
        system.c_ops.append(c_op1)
        system.c_ops.append(c_op2)

        assert len(system.c_ops) == 2
        assert system.c_ops[0] == c_op1
        assert system.c_ops[1] == c_op2

    def test_latex_management(self):
        """Test LaTeX string management"""
        system = QuantumSystem("Test")

        latex_str = r"H = \omega \sigma_z"
        system.latex = latex_str

        assert system.get_latex() == latex_str
        assert system.latex == latex_str

    def test_repr_string(self):
        """Test string representation"""
        system = QuantumSystem("Test System")
        system.hamiltonian = qt.sigmaz()

        repr_str = repr(system)
        assert "QuantumSystem(name='Test System'" in repr_str
        assert "dim=2" in repr_str

    def test_pretty_print_empty(self, capsys):
        """Test pretty_print with empty system"""
        system = QuantumSystem("Empty System")
        system.pretty_print()

        captured = capsys.readouterr()
        assert "Quantum System: Empty System" in captured.out
        assert "Hilbert Space Dimension: 0" in captured.out
        assert "Parameters: {}" in captured.out
        assert "Number of Operators: 0" in captured.out
        assert "Number of Collapse Operators: 0" in captured.out

    def test_pretty_print_populated(self, capsys):
        """Test pretty_print with populated system"""
        system = QuantumSystem("Populated System", omega=1.0, gamma=0.1)
        system.hamiltonian = qt.sigmaz()
        system.operators = {"sz": qt.sigmaz(), "sx": qt.sigmax()}
        system.c_ops = [qt.destroy(2)]
        system.latex = r"H = \sigma_z"

        system.pretty_print()

        captured = capsys.readouterr()
        assert "Quantum System: Populated System" in captured.out
        assert "Hilbert Space Dimension: 2" in captured.out
        assert "omega" in captured.out
        assert "gamma" in captured.out
        assert "Number of Operators: 2" in captured.out
        assert "Number of Collapse Operators: 1" in captured.out
        assert r"H = \sigma_z" in captured.out

    def test_mathematical_properties(self):
        """Test mathematical properties for Hermitian Hamiltonians"""
        hamiltonians = [
            qt.sigmaz(),
            qt.sigmax() + qt.sigmay(),
            qt.num(4),
            qt.tensor(qt.sigmaz(), qt.sigmaz()),
        ]

        for H in hamiltonians:
            system = QuantumSystem("Test")
            system.hamiltonian = H

            # Should be Hermitian
            assert (H - H.dag()).norm() < 1e-10

            # Eigenvalues should be real
            eigenvals = system.eigenvalues
            assert all(np.isreal(eigenvals))

            # Ground state should have lowest energy
            ground_state = system.ground_state
            ground_energy = qt.expect(H, ground_state)
            assert abs(ground_energy - min(eigenvals)) < 1e-10

    def test_parameter_modification(self):
        """Test parameter modification"""
        system = QuantumSystem("Test", omega=1.0)

        # Modify existing parameter
        system.parameters["omega"] = 2.0
        assert system.parameters["omega"] == 2.0

        # Add new parameter
        system.parameters["gamma"] = 0.1
        assert system.parameters["gamma"] == 0.1

    def test_edge_cases(self):
        """Test edge cases"""
        # Large dimension system
        system_large = QuantumSystem("Large")
        system_large.hamiltonian = qt.qeye(50)
        assert system_large.dimension == 50

        # Zero Hamiltonian
        system_zero = QuantumSystem("Zero")
        system_zero.hamiltonian = qt.qzero(3)
        eigenvals = system_zero.eigenvalues
        assert all(abs(e) < 1e-10 for e in eigenvals)

        # Identity Hamiltonian
        system_identity = QuantumSystem("Identity")
        system_identity.hamiltonian = qt.qeye(3)
        eigenvals = system_identity.eigenvalues
        expected = [1.0, 1.0, 1.0]
        np.testing.assert_array_almost_equal(
            sorted(eigenvals), sorted(expected))

    def test_tensor_product_systems(self):
        """Test tensor product Hamiltonians"""
        system = QuantumSystem("Two Qubits")

        # Two-qubit Hamiltonian
        H = qt.tensor(qt.sigmaz(), qt.qeye(2)) + \
            qt.tensor(qt.qeye(2), qt.sigmaz())
        system.hamiltonian = H

        assert system.dimension == 4
        eigenvals = system.eigenvalues
        assert len(eigenvals) == 4

        # Should be Hermitian
        assert (H - H.dag()).norm() < 1e-10
