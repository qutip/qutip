import numpy as np
from qutip import tensor, qeye, destroy, sigmax, sigmay, sigmaz, coefficient
from typing import Union
from .quantum_system import QuantumSystem


def jaynes_cummings(
    omega_c: Union[float, coefficient] = 1.0,
    omega_a: Union[float, coefficient] = 1.0,
    g: Union[float, coefficient] = 0.1,
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
    omega_c : float or coefficient, default=1.0
        Cavity frequency, can be constant or time-dependent
    omega_a : float or coefficient, default=1.0
        Atomic transition frequency, can be constant or time-dependent
    g : float or coefficient, default=0.1
        Atom-cavity coupling strength, can be constant or time-dependent
    n_cavity : int, default=10
        Cavity Fock space truncation (number of photon states)
    rotating_wave : bool, default=True
        Whether to apply rotating wave approximation
    cavity_decay : float, default=0.0
        Cavity decay rate, kappa (photon loss rate)
    atomic_decay : float or coefficient, default=0.0
        Atomic decay rate, gamma (spontaneous emission rate)
    atomic_dephasing : float or coefficient, default=0.0
        Atomic pure dephasing rate, gamma_phi
    thermal_photons : float, default=0.0
        Mean thermal photon number, n_th (for thermal bath)

    Returns:
    --------
    QuantumSystem
        Configured Jaynes-Cummings system instance

    """
    # Build operators
    operators = {}

    # Cavity operators (tensor with 2-level atomic system)
    operators['a'] = tensor(destroy(n_cavity), qeye(2))
    operators['a_dag'] = operators['a'].dag()
    operators['n_c'] = operators['a_dag'] * operators['a']  # Photon number

    # Atomic operators (tensor with cavity system)
    operators['sigma_minus'] = tensor(qeye(n_cavity), destroy(2))
    operators['sigma_plus'] = operators['sigma_minus'].dag()
    operators['sigma_z'] = tensor(qeye(n_cavity), sigmaz())
    operators['sigma_x'] = tensor(qeye(n_cavity), sigmax())
    operators['sigma_y'] = tensor(qeye(n_cavity), sigmay())

    # Build Hamiltonian
    # Free cavity evolution: ω_c a†a
    H_cavity = omega_c * operators['n_c']

    # Free atomic evolution: (ω_a/2)σ_z
    H_atom = omega_a * operators['sigma_plus'] * operators['sigma_minus']

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

    return QuantumSystem(
        hamiltonian=hamiltonian,
        name="Jaynes-Cummings",
        operators=operators,
        c_ops=c_ops,
        latex=latex,
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
