import numpy as np
from qutip import tensor, qeye, destroy, sigmax, sigmay, sigmaz, coefficient
from qutip.core.cy.coefficient import Coefficient
from typing import Union
from .quantum_system import QuantumSystem

def _create_sqrt_coefficient(rate):
    """Helper function to create sqrt coefficient from decay rate"""
    if isinstance(rate, Coefficient):
        # Extract coefficient information and create sqrt version
        def sqrt_func(t, args):
            return np.sqrt(rate(t, args))

        return coefficient(sqrt_func, args={})
    else:
        return np.sqrt(rate)

def jaynes_cummings(
    omega_c: Union[float, Coefficient] = 1.0,
    omega_a: Union[float, Coefficient] = 1.0,
    g: Union[float, Coefficient] = 0.1,
    n_cavity: int = 10,
    rotating_wave: bool = True,
    cavity_decay: Union[float, Coefficient] = 0.0,
    atomic_decay: Union[float, Coefficient] = 0.0,
    atomic_dephasing: Union[float, Coefficient] = 0.0,
    thermal_photons: float = 0.0) -> QuantumSystem:
    """
    Create Jaynes-Cummings system

    The Jaynes-Cummings model is one of the most fundamental models in quantum optics,
    describing the interaction between a two-level atom and a single mode of the
    electromagnetic field (cavity mode). It was introduced by Edwin Jaynes and
    Fred Cummings in 1963 and remains central to cavity quantum electrodynamics (cavity QED).

    The Hamiltonian is:

    **With rotating wave approximation (RWA):**

    .. math::

        H = \\omega_c a^\\dagger a + \\frac{\\omega_a}{2}\\sigma_z + g(a^\\dagger\\sigma_- + a\\sigma_+)

    **Without rotating wave approximation:**

    .. math::

        H = \\omega_c a^\\dagger a + \\frac{\\omega_a}{2}\\sigma_z + g(a^\\dagger + a)(\\sigma_+ + \\sigma_-)

    Parameters
    ----------
    omega_c : float or Coefficient, default=1.0
        Cavity frequency, can be constant or time-dependent
    omega_a : float or Coefficient, default=1.0
        Atomic transition frequency, can be constant or time-dependent
    g : float or Coefficient, default=0.1
        Atom-cavity coupling strength, can be constant or time-dependent
    n_cavity : int, default=10
        Cavity Fock space truncation (number of photon states)
    rotating_wave : bool, default=True
        Whether to apply rotating wave approximation
    cavity_decay : float or Coefficient, default=0.0
        Cavity decay rate, kappa (photon loss rate), can be constant or time-dependent
    atomic_decay : float or Coefficient, default=0.0
        Atomic decay rate, gamma (spontaneous emission rate), can be constant or time-dependent
    atomic_dephasing : float or Coefficient, default=0.0
        Atomic pure dephasing rate, gamma_phi, can be constant or time-dependent
    thermal_photons : float, default=0.0
        Mean thermal photon number, n_th (for thermal bath)

    Returns
    -------
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
    operators['sigma_z'] = operators['sigma_plus'] * operators['sigma_minus'] - operators['sigma_minus'] * operators['sigma_plus']

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

    # Cavity decay with thermal effects
    if isinstance(cavity_decay, Coefficient):
        # cavity_decay is coefficient, thermal_photons is float
        def cavity_relax_func(t, args):
            kappa = cavity_decay(t, args)
            return np.sqrt(kappa * (1 + thermal_photons))

        sqrt_cavity_relax = coefficient(cavity_relax_func, args={})
        c_ops.append(sqrt_cavity_relax * operators['a'])
    elif cavity_decay > 0.0:
        cavity_relax_rate = cavity_decay * (1 + thermal_photons)
        c_ops.append(np.sqrt(cavity_relax_rate) * operators['a'])

    # Cavity excitation (thermal): sqrt(kappa * n_th) * a_dag
    if thermal_photons > 0.0:
        if isinstance(cavity_decay, Coefficient):
            # cavity_decay is coefficient, thermal_photons is float
            def cavity_excite_func(t, args):
                kappa = cavity_decay(t, args)
                return np.sqrt(kappa * thermal_photons)

            sqrt_cavity_excite = coefficient(cavity_excite_func, args={})
            c_ops.append(sqrt_cavity_excite * operators['a_dag'])
        elif cavity_decay > 0.0:
            cavity_excite_rate = cavity_decay * thermal_photons
            c_ops.append(np.sqrt(cavity_excite_rate) * operators['a_dag'])

    # Atomic spontaneous emission: sqrt(gamma) * sigma_minus
    # Check if it's a Coefficient
    if isinstance(atomic_decay, Coefficient) or atomic_decay > 0.0:
        sqrt_atomic_decay = _create_sqrt_coefficient(atomic_decay)
        c_ops.append(sqrt_atomic_decay * operators['sigma_minus'])

    # Atomic pure dephasing:  sqrt(gamma_phi)* sigma_z
    if isinstance(atomic_dephasing, Coefficient) or atomic_dephasing > 0.0:
        sqrt_atomic_dephasing = _create_sqrt_coefficient(atomic_dephasing)
        c_ops.append(sqrt_atomic_dephasing * operators['sigma_z'])

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
