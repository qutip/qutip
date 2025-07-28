import numpy as np
import qutip as qt
from .quantum_system import QuantumSystem  # Import the QuantumSystem class


def jaynes_cummings(
        omega_c: float = 1.0,
        omega_a: float = 1.0,
        g: float = 0.1,
        n_cavity: int = 10,
        rotating_wave: bool = True,
        cavity_decay: float = 0.0,
        atomic_decay: float = 0.0,
        atomic_dephasing: float = 0.0,
        thermal_photons: float = 0.0) -> QuantumSystem:
    """
    Create Jaynes-Cummings system

    The Jaynes-Cummings model describes a two-level atom interacting with a
    single cavity mode. The Hamiltonian is:

    H = omega_c * a_dag * a + (omega_a/2) * sigma_z + g * (a_dag * sigma_minus + a * sigma_plus)  [with RWA]
    H = omega_c * a_dag * a + (omega_a / 2) * sigma_z + g * (a_dag + a) * (sigma_plus + sigma_minus) [without RWA]

    Parameters:
    -----------
    omega_c : float, default=1.0
        Cavity frequency
    omega_a : float, default=1.0
        Atomic transition frequency
    g : float, default=0.1
        Atom-cavity coupling strength
    n_cavity : int, default=10
        Cavity Fock space truncation (number of photon states)
    rotating_wave : bool, default=True
        Whether to apply rotating wave approximation
    cavity_decay : float, default=0.0
        Cavity decay rate, kappa (photon loss rate)
    atomic_decay : float, default=0.0
        Atomic decay rate, gamma (spontaneous emission rate)
    atomic_dephasing : float, default=0.0
        Atomic pure dephasing rate, gamma_phi
    thermal_photons : float, default=0.0
        Mean thermal photon number, n_th (for thermal bath)

    Returns:
    --------
    QuantumSystem
        Configured Jaynes-Cummings system instance

    """

    # Create QuantumSystem instance with parameters
    system = QuantumSystem(
        "Jaynes-Cummings",
        omega_c=omega_c,
        omega_a=omega_a,
        g=g,
        n_cavity=n_cavity,
        rotating_wave=rotating_wave,
        cavity_decay=cavity_decay,
        atomic_decay=atomic_decay,
        atomic_dephasing=atomic_dephasing,
        thermal_photons=thermal_photons
    )

    # Build operators
    operators = {}

    # Cavity operators (tensor with 2-level atomic system)
    operators['a'] = qt.tensor(qt.destroy(n_cavity), qt.qeye(2))
    operators['a_dag'] = operators['a'].dag()
    operators['n_c'] = operators['a_dag'] * operators['a']  # Photon number

    # Atomic operators (tensor with cavity system)
    operators['sigma_minus'] = qt.tensor(qt.qeye(n_cavity), qt.destroy(2))
    operators['sigma_plus'] = operators['sigma_minus'].dag()
    operators['sigma_z'] = qt.tensor(qt.qeye(n_cavity), qt.sigmaz())
    operators['sigma_x'] = qt.tensor(qt.qeye(n_cavity), qt.sigmax())
    operators['sigma_y'] = qt.tensor(qt.qeye(n_cavity), qt.sigmay())

    # Build Hamiltonian
    # Free cavity evolution: ω_c a†a
    H_cavity = omega_c * operators['n_c']

    # Free atomic evolution: (ω_a/2)σ_z
    H_atom = 0.5 * omega_a * operators['sigma_z']

    # Interaction term
    if rotating_wave:
        # Rotating wave approximation: g(a†σ_- + aσ_+)
        H_interaction = g * (operators['a_dag'] * operators['sigma_minus'] +
                             operators['a'] * operators['sigma_plus'])
    else:
        # Full interaction (Rabi model): g(a† + a)(σ_+ + σ_-)
        H_interaction = g * (operators['a_dag'] + operators['a']) * \
            (operators['sigma_plus'] + operators['sigma_minus'])

    # Total Hamiltonian
    hamiltonian = H_cavity + H_atom + H_interaction

    # Build collapse operators for dissipation
    c_ops = []

    # Cavity relaxation: kappa(1 + n_th) * a
    cavity_relax_rate = cavity_decay * (1 + thermal_photons)
    if cavity_relax_rate > 0.0:
        c_ops.append(np.sqrt(cavity_relax_rate) * operators['a'])

    # Cavity excitation (thermal): kappa * n_th * a_dag
    cavity_excite_rate = cavity_decay * thermal_photons
    if cavity_excite_rate > 0.0:
        c_ops.append(np.sqrt(cavity_excite_rate) * operators['a_dag'])

    # Atomic spontaneous emission: gamma * sigma_minus
    if atomic_decay > 0.0:
        c_ops.append(np.sqrt(atomic_decay) * operators['sigma_minus'])

    # Atomic pure dephasing: gamma_phi * sigma_z
    if atomic_dephasing > 0.0:
        c_ops.append(np.sqrt(atomic_dephasing) * operators['sigma_z'])

    # LaTeX representation
    if rotating_wave:
        latex = (r"H = \omega_c a^\dagger a + \frac{\omega_a}{2}\sigma_z + "
                 r"g(a^\dagger\sigma_- + a\sigma_+)")
    else:
        latex = (r"H = \omega_c a^\dagger a + \frac{\omega_a}{2}\sigma_z + "
                 r"g(a^\dagger + a)(\sigma_+ + \sigma_-)")

    # Set all attributes in the system
    system.operators = operators
    system.hamiltonian = hamiltonian
    system.c_ops = c_ops
    system.latex = latex

    return system


# Example usage and demonstrations
if __name__ == "__main__":
    print("Jaynes-Cummings Model Examples")
    print("=" * 40)

    # Example 1: Basic resonant system
    print("\n1. Basic Resonant Jaynes-Cummings System:")
    jc_basic = jaynes_cummings(omega_c=1.0, omega_a=1.0, g=0.1, n_cavity=5)
    jc_basic.pretty_print()

    # Example 2: Detuned system with dissipation
    print("\n2. Detuned System with Dissipation:")
    jc_dissipative = jaynes_cummings(
        omega_c=1.0,
        omega_a=1.1,  # 10% detuning
        g=0.05,
        n_cavity=8,
        cavity_decay=0.01,    # kappa = 0.01
        atomic_decay=0.005,   # gamma = 0.005
        thermal_photons=0.1   # n_th = 0.1
    )
    jc_dissipative.pretty_print()

    # Example 3: Rabi model (no rotating wave approximation)
    print("\n3. Rabi Model (No RWA):")
    rabi = jaynes_cummings(
        omega_c=1.0,
        omega_a=1.0,
        g=0.2,
        rotating_wave=False
    )
    print(f"LaTeX: {rabi.latex}")

    # Example 4: Accessing operators and Hamiltonian
    print("\n4. Accessing System Components:")
    jc = jaynes_cummings(omega_c=1.0, omega_a=1.0, g=0.1, n_cavity=3)

    print(f"Available operators: {list(jc.operators.keys())}")
    print(f"Hamiltonian dimension: {jc.hamiltonian.shape}")
    print(f"Number of collapse operators: {len(jc.c_ops)}")

    # Show both access methods
    print(f"Direct access - operators: {type(jc.operators)}")
    print(f"Method access - operators: {type(jc.get_operators())}")

    # Example 6: Energy eigenvalues
    print("\n6. Energy Spectrum:")
    jc_small = jaynes_cummings(omega_c=1.0, omega_a=1.0, g=0.1, n_cavity=3)
    eigenvals = jc_small.eigenvalues()
    print(f"First few eigenvalues: {eigenvals[:6]}")

    print(f"\nGround state energy: {eigenvals[0]:.3f}")
    print(f"First excited energy: {eigenvals[1]:.3f}")
