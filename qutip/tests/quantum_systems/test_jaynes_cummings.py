import pytest
import numpy as np
from qutip import tensor, qeye, destroy, sigmaz, basis, expect, coefficient
from qutip.quantum_systems.jaynes_cummings import jaynes_cummings
from qutip.quantum_systems.quantum_system import QuantumSystem


class TestJaynesCummings:
    """Balanced test suite for jaynes_cummings function"""

    def test_default_jaynes_cummings(self):
        """Test Jaynes-Cummings with default parameters"""
        jc = jaynes_cummings()

        assert isinstance(jc, QuantumSystem)
        assert jc.name == "Jaynes-Cummings"
        assert jc.parameters["omega_c"] == 1.0
        assert jc.parameters["omega_a"] == 1.0
        assert jc.parameters["g"] == 0.1
        assert jc.parameters["n_cavity"] == 10
        assert jc.parameters["rotating_wave"]

    @pytest.mark.parametrize("n_cavity", [3, 5, 10, 20])
    def test_dimension_scaling(self, n_cavity):
        """Test dimension scaling with cavity size"""
        jc = jaynes_cummings(n_cavity=n_cavity)
        expected_dim = [[n_cavity, 2], [n_cavity, 2]]
        assert jc.dimension == expected_dim

    def test_operators_completeness(self):
        """Test that all expected operators are present"""
        jc = jaynes_cummings(n_cavity=5)

        expected_operators = [
            "a", "a_dag", "n_c",  # Cavity operators
            "sigma_minus", "sigma_plus", "sigma_z", "sigma_x", "sigma_y"  # Atomic operators
        ]

        for op_name in expected_operators:
            assert op_name in jc.operators

    def test_cavity_operators(self):
        """Test cavity operator properties"""
        n_cavity = 5
        jc = jaynes_cummings(n_cavity=n_cavity)

        a = jc.operators["a"]
        a_dag = jc.operators["a_dag"]
        n_c = jc.operators["n_c"]

        # Basic relationships
        assert a_dag == a.dag()
        assert n_c == a_dag * a

    def test_atomic_operators(self):
        """Test atomic operator properties"""
        jc = jaynes_cummings(n_cavity=3)

        sigma_plus = jc.operators["sigma_plus"]
        sigma_minus = jc.operators["sigma_minus"]

        assert sigma_plus == sigma_minus.dag()

    @pytest.mark.parametrize("omega_c, omega_a, g", [
        (1.0, 1.0, 0.1),    # Resonant
        (1.0, 1.2, 0.1),    # Detuned
        (2.0, 1.0, 0.2),    # Different frequencies
        (1.0, 1.0, 0.0),    # No coupling
    ])
    def test_hamiltonian_parameters(self, omega_c, omega_a, g):
        """Test Hamiltonian with different parameters"""
        jc = jaynes_cummings(omega_c=omega_c, omega_a=omega_a, g=g, n_cavity=3)

        # Check parameter storage
        assert jc.parameters["omega_c"] == omega_c
        assert jc.parameters["omega_a"] == omega_a
        assert jc.parameters["g"] == g

        # Hamiltonian should be Hermitian
        H = jc.hamiltonian
        assert H == H.dag()

    def test_rotating_wave_approximation(self):
        """Test rotating wave approximation vs full interaction"""
        params = {"omega_c": 1.0, "omega_a": 1.0, "g": 0.1, "n_cavity": 3}

        jc_rwa = jaynes_cummings(rotating_wave=True, **params)
        jc_full = jaynes_cummings(rotating_wave=False, **params)

        # Hamiltonians should be different
        H_rwa = jc_rwa.hamiltonian
        H_full = jc_full.hamiltonian
        assert (H_rwa - H_full).norm() > 1e-10

    def test_hamiltonian_structure_rwa(self):
        """Test Hamiltonian structure with RWA"""
        omega_c, omega_a, g = 1.5, 1.2, 0.1
        jc = jaynes_cummings(omega_c=omega_c, omega_a=omega_a, g=g,
                             n_cavity=3, rotating_wave=True)

        # Build expected Hamiltonian
        a = jc.operators["a"]
        a_dag = jc.operators["a_dag"]
        sigma_plus = jc.operators["sigma_plus"]
        sigma_minus = jc.operators["sigma_minus"]
        sigma_z = jc.operators["sigma_z"]

        H_expected = (omega_c * a_dag * a +
                      omega_a * sigma_plus * sigma_minus +
                      g * (a_dag * sigma_minus + a * sigma_plus))

        assert jc.hamiltonian == H_expected

    def test_hamiltonian_structure_no_rwa(self):
        """Test Hamiltonian structure without RWA (Rabi model)"""
        omega_c, omega_a, g = 1.5, 1.2, 0.1
        jc = jaynes_cummings(omega_c=omega_c, omega_a=omega_a, g=g,
                             n_cavity=3, rotating_wave=False)

        # Build expected Rabi Hamiltonian
        a = jc.operators["a"]
        a_dag = jc.operators["a_dag"]
        sigma_plus = jc.operators["sigma_plus"]
        sigma_minus = jc.operators["sigma_minus"]
        sigma_z = jc.operators["sigma_z"]

        H_expected = (omega_c * a_dag * a +
                      omega_a * sigma_plus * sigma_minus +
                      g * (a_dag + a) * (sigma_plus + sigma_minus))

        assert jc.hamiltonian == H_expected

    @pytest.mark.parametrize("decay_type, rate, operator_name", [
        ("cavity_decay", 0.0, "a"),
        ("cavity_decay", 0.1, "a"),
        ("cavity_decay", 0.5, "a"),
        ("atomic_decay", 0.0, "sigma_minus"),
        ("atomic_decay", 0.1, "sigma_minus"),
        ("atomic_decay", 0.5, "sigma_minus"),
        ("atomic_dephasing", 0.0, "sigma_z"),
        ("atomic_dephasing", 0.1, "sigma_z"),
        ("atomic_dephasing", 0.5, "sigma_z"),
    ])
    def test_dissipation_mechanisms(self, decay_type, rate, operator_name):
        """Test individual dissipation mechanisms"""
        kwargs = {decay_type: rate, "n_cavity": 3}
        jc = jaynes_cummings(**kwargs)

        if rate == 0.0:
            assert len(jc.c_ops) == 0
        else:
            assert len(jc.c_ops) == 1
            expected_op = np.sqrt(rate) * jc.operators[operator_name]
            assert expected_op in jc.c_ops

    def test_thermal_photons(self):
        """Test thermal photon effects"""
        cavity_decay = 0.1
        thermal_photons = 0.1
        jc = jaynes_cummings(
            cavity_decay=cavity_decay,
            thermal_photons=thermal_photons,
            n_cavity=3)

        # Should have both relaxation and excitation operators
        cavity_relax_rate = cavity_decay * (1 + thermal_photons)
        expected_relax_op = cavity_relax_rate**0.5 * jc.operators["a"]
        assert expected_relax_op in jc.c_ops

        cavity_excite_rate = cavity_decay * thermal_photons
        expected_excite_op = cavity_excite_rate**0.5 * jc.operators["a_dag"]
        assert expected_excite_op in jc.c_ops

    def test_all_dissipation_mechanisms(self):
        """Test system with all dissipation mechanisms"""
        jc = jaynes_cummings(
            cavity_decay=0.1, atomic_decay=0.05,
            atomic_dephasing=0.02, thermal_photons=0.1,
            n_cavity=3
        )

        # Should have 4 collapse operators
        assert len(jc.c_ops) == 4

    @pytest.mark.parametrize("rotating_wave", [True, False])
    def test_latex_representation(self, rotating_wave):
        """Test LaTeX representation"""
        jc = jaynes_cummings(rotating_wave=rotating_wave)

        if rotating_wave:
            expected_latex = (
                r"H = \omega_c a^\dagger a + \frac{\omega_a}{2}\sigma_z + "
                r"g(a^\dagger\sigma_- + a\sigma_+)")
        else:
            expected_latex = (
                r"H = \omega_c a^\dagger a + \frac{\omega_a}{2}\sigma_z + "
                r"g(a^\dagger + a)(\sigma_+ + \sigma_-)")

        assert jc.latex == expected_latex

    def test_ground_state(self):
        """Test ground state properties"""
        jc = jaynes_cummings(omega_c=1.0, omega_a=1.0, g=0.1, n_cavity=3)
        ground_state = jc.ground_state

        # Ground state should be normalized
        assert ground_state.norm() == pytest.approx(1., abs=1e-10)

        # Should be eigenstate with lowest energy
        eigenvals = jc.eigenvalues
        min_eigenval = min(eigenvals)

        H_psi = jc.hamiltonian * ground_state
        eigenval_psi = min_eigenval * ground_state
        assert H_psi == eigenval_psi

    def test_photon_number_operator(self):
        """Test photon number operator"""
        n_cavity = 4
        # Uncoupled for simplicity
        jc = jaynes_cummings(g=0.0, n_cavity=n_cavity)

        n_c = jc.operators["n_c"]

        # Test expectation values on basis states
        for n in range(min(3, n_cavity)):
            for atom_state in [0, 1]:
                state = basis([n_cavity, 2], [n, atom_state])
                expectation = expect(n_c, state)
                assert abs(expectation - n) < 1e-10

    def test_operator_tensor_structure(self):
        """Test tensor product structure of operators"""
        n_cavity = 3
        jc = jaynes_cummings(n_cavity=n_cavity)

        # Cavity operators act on first subsystem
        a = jc.operators["a"]
        expected_a = tensor(destroy(n_cavity), qeye(2))
        assert a == expected_a

        # Atomic operators act on second subsystem
        sigma_z = jc.operators["sigma_z"]
        expected_sigma_z = tensor(qeye(n_cavity), sigmaz())
        assert sigma_z == expected_sigma_z

    def test_commutation_relations(self):
        """Test important commutation relations"""
        jc = jaynes_cummings(n_cavity=3)

        a = jc.operators["a"]
        a_dag = jc.operators["a_dag"]
        n_c = jc.operators["n_c"]

        # [n_c, a] = -a
        commutator_n_a = n_c * a - a * n_c
        assert commutator_n_a == -a

        # [n_c, a_dag] = a_dag
        commutator_n_adag = n_c * a_dag - a_dag * n_c
        assert commutator_n_adag == a_dag

    def test_return_type_and_attributes(self):
        """Test return type and required attributes"""
        jc = jaynes_cummings()

        assert isinstance(jc, QuantumSystem)

    def test_pretty_print_basic(self, capsys):
        """Test pretty_print output"""
        jc = jaynes_cummings(omega_c=1.5, omega_a=1.2, g=0.1, n_cavity=5)
        jc.pretty_print()

        captured = capsys.readouterr()
        assert "Jaynes-Cummings" in captured.out
        assert "Hilbert Space Dimension: [[5, 2], [5, 2]]" in captured.out
        assert "omega_c" in captured.out

    def test_coefficient_parameters(self):
        """Test Jaynes-Cummings with Coefficient parameters"""
        # Create coefficient parameters
        # Simple linear time-dependent omega_c
        omega_c_coeff = coefficient("omega_c0 * 2 * t", args={'omega_c0': 1.0})

        # sligtly more complex time dependant g
        g_coeff = coefficient("g0 * sin(omega_d * t)",
                args={'g0': 0.1,'omega_d': 2.0})

        jc = jaynes_cummings(omega_c=omega_c_coeff,
            g=g_coeff,n_cavity=3)

        # Test that coefficients are stored
        assert jc.parameters["omega_c"] == omega_c_coeff
        assert jc.parameters["g"] == g_coeff

        # Test that system is created successfully
        assert isinstance(jc, QuantumSystem)
