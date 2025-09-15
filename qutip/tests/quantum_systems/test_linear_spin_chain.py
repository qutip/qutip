import pytest
import numpy as np
from qutip import qeye, sigmax, sigmay, sigmaz, tensor, coefficient, QobjEvo 
from qutip.quantum_systems.linear_spin_chain import linear_spin_chain
from qutip.quantum_systems.quantum_system import QuantumSystem


class TestLinearSpinChain:
    """Test suite for linear_spin_chain function"""

    def test_default_heisenberg_chain(self):
        """Test default Heisenberg chain creation"""
        lsc = linear_spin_chain()
        
        assert isinstance(lsc, QuantumSystem)
        assert lsc.name == "Linear Spin Chain (HEISENBERG)"
        assert lsc.parameters["model_type"] == "heisenberg"
        assert lsc.parameters["N"] == 4
        assert lsc.parameters["J"] == 1.0
        assert lsc.parameters["Jz"] == 1.0  # Should equal J for Heisenberg
        assert lsc.parameters["boundary_conditions"] == "open"

    @pytest.mark.parametrize("model_type,expected_jz", [
        ("heisenberg", 1.0),  # Jz = J
        ("xy", 0.0),          # Jz = 0
        ("xxz", 1.0),         # Jz = J
        ("ising", 1.0),       # Jz = J
    ])
    def test_model_types(self, model_type, expected_jz):
        """Test different spin chain models"""
        J_val = 1.0
        lsc = linear_spin_chain(model_type=model_type, J=J_val, N=3)
        
        assert lsc.parameters["model_type"] == model_type
        assert lsc.parameters["Jz"] == expected_jz
        assert lsc.name == f"Linear Spin Chain ({model_type.upper()})"

    @pytest.mark.parametrize("N", [2, 3, 4, 5])
    def test_chain_length_scaling(self, N):
        """Test dimension scaling with chain length"""
        lsc = linear_spin_chain(N=N)
        
        # Dimension should be 2^N for N spins
        expected_dims = [[2] * N, [2] * N]  # N spin-1/2 particles
        assert lsc.dimension == expected_dims
        assert lsc.parameters["N"] == N

    @pytest.mark.parametrize("boundary_conditions", ["open", "periodic"])
    def test_boundary_conditions(self, boundary_conditions):
        """Test different boundary conditions"""
        lsc = linear_spin_chain(N=3, boundary_conditions=boundary_conditions)
        
        assert lsc.parameters["boundary_conditions"] == boundary_conditions
        
        # Both should create valid systems
        assert lsc.hamiltonian is not None
        assert lsc.hamiltonian.shape[0] == 8  # 2^3 for 3 spins

    def test_heisenberg_hamiltonian_structure(self):
        """Test Heisenberg Hamiltonian construction"""
        N = 3
        J = 1.5
        lsc = linear_spin_chain(model_type="heisenberg", N=N, J=J, 
                               boundary_conditions="open")
        
        # Get operators to build expected Hamiltonian
        ops = lsc.operators
        
        # Expected Heisenberg Hamiltonian (open boundary)
        H_expected = 0
        for k in range(N - 1):  # N-1 interactions for open boundary
            H_expected += J * (ops[f"S_{k}_x"] * ops[f"S_{k+1}_x"] +
                             ops[f"S_{k}_y"] * ops[f"S_{k+1}_y"] +
                             ops[f"S_{k}_z"] * ops[f"S_{k+1}_z"])
        
        assert lsc.hamiltonian == H_expected

    def test_xy_model_hamiltonian(self):
        """Test XY model Hamiltonian (no Z interactions)"""
        N = 3
        J = 1.0
        lsc = linear_spin_chain(model_type="xy", N=N, J=J)
        
        ops = lsc.operators
        
        # XY model: only X and Y interactions
        H_expected = 0
        for k in range(N - 1):
            H_expected += J * (ops[f"S_{k}_x"] * ops[f"S_{k+1}_x"] +
                             ops[f"S_{k}_y"] * ops[f"S_{k+1}_y"])
        
        assert lsc.hamiltonian == H_expected

    def test_ising_model_hamiltonian(self):
        """Test Ising model Hamiltonian (only Z interactions)"""
        N = 3
        J = 2.0
        lsc = linear_spin_chain(model_type="ising", N=N, J=J)
        
        ops = lsc.operators
        
        # Ising model: only Z interactions
        H_expected = 0
        for k in range(N - 1):
            H_expected += J * ops[f"S_{k}_z"] * ops[f"S_{k+1}_z"]
        
        assert lsc.hamiltonian == H_expected

    @pytest.mark.parametrize("gamma_dephasing", [0.0, 0.1])
    def test_dephasing_dissipation(self, gamma_dephasing):
        """Test pure dephasing collapse operators"""
        N = 2  # Keep small for simpler testing
        lsc = linear_spin_chain(N=N, gamma_dephasing=gamma_dephasing)
        
        if gamma_dephasing == 0.0:
            assert len(lsc.c_ops) == 0
        else:
            assert len(lsc.c_ops) == N
            
            for k in range(N):
                expected_op = np.sqrt(gamma_dephasing) * lsc.operators[f"S_{k}_z"]
                assert expected_op in lsc.c_ops

    def test_thermal_dissipation(self):
        """Test thermal bath coupling"""
        lsc = linear_spin_chain(N=2, gamma_thermal=0.1, temperature=1.0,
                               transition_frequency=2.0)
        
        # Thermal coupling adds both up and down transitions (2 per site)
        assert len(lsc.c_ops) == 4  # 2 sites * 2 operators each
        assert lsc.parameters["temperature"] == 1.0

    def test_correlation_operators(self):
        """Test nearest-neighbor correlation operators"""
        lsc = linear_spin_chain(N=3, boundary_conditions="open")
        ops = lsc.operators
        
        # For open boundaries with N=3, should have 2 correlation terms
        expected_xx = (ops["S_0_x"] * ops["S_1_x"] + 
                      ops["S_1_x"] * ops["S_2_x"])
        assert ops["correlation_xx_nn"] == expected_xx

    def test_periodic_vs_open_boundaries(self):
        """Test difference between periodic and open boundaries"""
        N = 3
        lsc_open = linear_spin_chain(N=N, boundary_conditions="open")
        lsc_periodic = linear_spin_chain(N=N, boundary_conditions="periodic")
        
        # Both should have same dimension
        assert lsc_open.dimension == lsc_periodic.dimension
        
        # But different Hamiltonians due to boundary terms
        assert (lsc_open.hamiltonian - lsc_periodic.hamiltonian).norm() > 1e-10

    @pytest.mark.parametrize("latex_model,expected_text", [
        ("heisenberg", r"\vec{S}_i \cdot \vec{S}_j"),
        ("xy", r"S_i^x S_j^x + S_i^y S_j^y"),
        ("ising", r"S_i^z S_j^z"),
    ])
    def test_latex_representation(self, latex_model, expected_text):
        """Test LaTeX representation for different models"""
        lsc = linear_spin_chain(model_type=latex_model)
        assert expected_text in lsc.latex
        
        # Should include boundary condition info
        assert "OBC" in lsc.latex  # Open boundary condition

    def test_input_validation(self):
        """Test input validation and error handling"""
        # Test invalid chain length
        with pytest.raises(ValueError, match="Chain length N must be at least 2"):
            linear_spin_chain(N=1)
        
        # Test invalid model type
        with pytest.raises(ValueError, match="model_type must be one of"):
            linear_spin_chain(model_type="invalid")
        
        # Test invalid boundary conditions
        with pytest.raises(ValueError, match="boundary_conditions must be one of"):
            linear_spin_chain(boundary_conditions="invalid")

    def test_custom_jz_parameter(self):
        """Test custom Jz parameter override"""
        J = 1.0
        Jz = 0.5
        lsc = linear_spin_chain(model_type="heisenberg", J=J, Jz=Jz)
        
        # Jz should be custom value, not equal to J
        assert lsc.parameters["Jz"] == Jz
        assert lsc.parameters["Jz"] != lsc.parameters["J"]

    def test_coefficient_parameters(self):
        """Test Linear Spin Chain with Coefficient parameters"""
        # Create coefficient parameters
        # Simple linear time-dependent J coupling
        J_coeff = coefficient("J0 + slope * t", args={'J0': 1.0, 'slope': 0.1})
        
        # Time-dependent magnetic field
        Bz_coeff = coefficient("B0 * sin(omega * t)", args={'B0': 0.5, 'omega': 0.3})
        
        # Time-dependent dephasing
        gamma_deph_coeff = coefficient("gamma0 + rate * t", args={'gamma0': 0.1, 'rate': 0.02})
        
        # Time-dependent depolarizing channel
        gamma_depol_coeff = coefficient("gamma_d * exp(-decay * t)", args={'gamma_d': 0.05, 'decay': 0.1})

        spin_chain = linear_spin_chain(
            model_type="heisenberg",
            N=4,
            J=J_coeff,
            B_z=Bz_coeff, 
            gamma_dephasing=gamma_deph_coeff,
            gamma_depolarizing=gamma_depol_coeff,
            boundary_conditions="open"
        )

        # Test that coefficients are stored
        assert spin_chain.parameters["J"] == J_coeff
        assert spin_chain.parameters["B_z"] == Bz_coeff
        assert spin_chain.parameters["gamma_dephasing"] == gamma_deph_coeff
        assert spin_chain.parameters["gamma_depolarizing"] == gamma_depol_coeff
        
        # Test that the Hamiltonian is the right format (QobjEvo for time-dependent)
        assert isinstance(spin_chain.hamiltonian, QobjEvo)
        
        # Test that system is created successfully
        assert isinstance(spin_chain, QuantumSystem)
