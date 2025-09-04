import pytest
import numpy as np
from qutip import tensor, qeye, destroy, sigmax, sigmaz, basis, num
from qutip.quantum_systems import QuantumSystem


class TestQuantumSystem:
    """Test suite for QuantumSystem class"""

    def test_basic_initialization(self):
        """Test basic initialization"""
        hamiltonian = None
        system = QuantumSystem(hamiltonian, "Test System")

        assert system.name == "Test System"
        assert system.parameters == {}
        assert system.operators == {}
        assert system.hamiltonian == hamiltonian
        assert system.c_ops == []
        assert system.latex == ""
        assert system.dimension == 0

    @pytest.mark.parametrize("name,params", [
        ("Qubit", {"omega": 1.0}),
        ("Oscillator", {"omega": 1.0, "n_levels": 10}),
        ("Complex", {"freq1": 1.0, "freq2": 1.5, "coupling": 0.1}),
    ])
    def test_various_initializations(self, name, params):
        """Test various initialization scenarios"""
        system = QuantumSystem(qeye(2), name, **params)
        assert system.name == name
        assert system.parameters == params

    @pytest.mark.parametrize("hamiltonian,expected_dim", [
        (sigmaz(), [[2], [2]]),
        (sigmax(), [[2], [2]]),
        (qeye(3), [[3], [3]]),
        (qeye(5), [[5], [5]]),
        (destroy(4), [[4], [4]]),
        (tensor(sigmaz(), sigmaz()), [[2, 2], [2, 2]]),
    ])
    def test_dimension_property(self, hamiltonian, expected_dim):
        """Test dimension property with various Hamiltonians"""
        system = QuantumSystem(hamiltonian, "Test")
        assert system.dimension == expected_dim

    def test_eigenvalues_qubit(self):
        """Test eigenvalues for qubit systems"""
        system = QuantumSystem(0.5 * sigmaz(), "Test")

        eigenvals = system.eigenvalues
        expected = np.array([-0.5, 0.5])
        np.testing.assert_array_almost_equal(eigenvals, expected)

    def test_eigenvalues_harmonic_oscillator(self):
        """Test eigenvalues for harmonic oscillator"""
        n_levels = 5
        hamiltonian = num(n_levels) + 0.5 * qeye(n_levels)
        system = QuantumSystem(hamiltonian, "Test")

        eigenvals = system.eigenvalues
        expected = np.arange(n_levels) + 0.5
        np.testing.assert_array_almost_equal(eigenvals, expected)

    def test_eigenstates_properties(self):
        """Test eigenstate properties"""
        system = QuantumSystem(sigmaz(), "Test")
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
            assert H_psi == E_psi

    @pytest.mark.parametrize("hamiltonian,expected_ground", [
        (sigmaz(), basis(2, 1)),           # |1⟩ for sigma_z
        (-sigmaz(), basis(2, 0)),          # |0⟩ for -sigma_z
        (num(4), basis(4, 0)),            # |0⟩ for number operator
    ])
    def test_ground_state_identification(self, hamiltonian, expected_ground):
        """Test ground state identification"""
        system = QuantumSystem(hamiltonian, "Test")

        ground_state = system.ground_state

        # Should be normalized
        assert abs(ground_state.norm() - 1.0) < 1e-10

        # Should be eigenstate with lowest eigenvalue
        eigenvals = system.eigenvalues
        min_eigenval = min(eigenvals)

        H_psi = hamiltonian * ground_state
        E_psi = min_eigenval * ground_state
        assert H_psi == E_psi

        # Check specific expected ground state
        overlap = abs(ground_state.overlap(expected_ground))
        assert overlap > 0.99

    def test_repr_string(self):
        """Test string representation"""
        system = QuantumSystem(sigmaz(), "Test System")

        repr_str = repr(system)
        assert "QuantumSystem(name='Test System'" in repr_str
        assert "dim=[[2], [2]]" in repr_str

    @pytest.mark.parametrize("system_type", ["empty", "populated"])
    def test_pretty_print(self, capsys, system_type):
        """Test pretty_print with different system states"""

        if system_type == "empty":
            system = QuantumSystem(qeye(2), "Empty System")  # For empty case
            system.pretty_print()

            captured = capsys.readouterr()
            assert "Quantum System: Empty System" in captured.out
            assert "Hilbert Space Dimension: [[2], [2]]" in captured.out
            assert "Parameters: {}" in captured.out
            assert "Number of Operators: 0" in captured.out
            assert "Number of Collapse Operators: 0" in captured.out

        else:  # populated
            system = QuantumSystem(
                sigmaz(),
                "Populated System",
                omega=1.0,
                gamma=0.1)  # For populated case
            system.operators = {"sz": sigmaz(), "sx": sigmax()}
            system.c_ops = [destroy(2)]
            system.latex = r"H = \sigma_z"

            system.pretty_print()

            captured = capsys.readouterr()
            assert "Quantum System: Populated System" in captured.out
            assert "Hilbert Space Dimension: [[2], [2]]" in captured.out
            assert "omega" in captured.out
            assert "gamma" in captured.out
            assert "Number of Operators: 2" in captured.out
            assert "Number of Collapse Operators: 1" in captured.out
            assert r"H = \sigma_z" in captured.out

    def test_tensor_product_systems(self):
        """Test tensor product Hamiltonians"""
        # Two-qubit Hamiltonian
        H = tensor(sigmaz(), qeye(2)) + \
            tensor(qeye(2), sigmaz())

        system = QuantumSystem(H, "Two Qubits")

        assert system.dimension == [[2, 2], [2, 2]]
        eigenvals = system.eigenvalues
        assert len(eigenvals) == 4

        # Should be Hermitian
        assert H == H.dag()

    def test_repr_latex_behavior(self):
        """Test different behaviors of _repr_latex_ depending on the latex attribute."""
        hamiltonian = 0.1 * (0.5 * sigmaz())

        # Case 1: plain LaTeX string (should get wrapped in $...$)
        sys_plain = QuantumSystem(
            hamiltonian,
            name="Qubit",
            latex=r"H = \frac{\omega}{2}\sigma_z")
        assert sys_plain._repr_latex_() == r"$H = \frac{\omega}{2}\sigma_z$"

        # Case 2: already wrapped in $...$ (should remain unchanged)
        sys_wrapped = QuantumSystem(
            hamiltonian,
            name="Qubit",
            latex=r"$H = \frac{\omega}{2}\sigma_z$")
        assert sys_wrapped._repr_latex_() == r"$H = \frac{\omega}{2}\sigma_z$"

        # Case 3: no latex provided (should fall back to system name)
        sys_fallback = QuantumSystem(hamiltonian, name="Qubit")
        assert sys_fallback._repr_latex_() == r"$\text{Qubit}$"
