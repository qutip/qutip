import numpy as np
import qutip as qt


def build_jc_operators(n_cavity: int):
    """
    Build the fundamental operators for the Jaynes-Cummings system

    Parameters:
    -----------
    n_cavity : int
        Number of cavity Fock states (cavity truncation)

    Returns:
    --------
    ops : dict
        Dictionary containing all operators
    """
    ops = {}

    # Cavity operators
    ops['a'] = qt.tensor(qt.destroy(n_cavity), qt.qeye(2))
    ops['a_dag'] = ops['a'].dag()
    ops['n_c'] = ops['a_dag'] * ops['a']

    # Atomic operators
    ops['sigma_plus'] = qt.tensor(qt.qeye(n_cavity), qt.sigmap())
    ops['sigma_minus'] = qt.tensor(qt.qeye(n_cavity), qt.sigmam())
    ops['sigma_z'] = qt.tensor(qt.qeye(n_cavity), qt.sigmaz())
    ops['sigma_x'] = qt.tensor(qt.qeye(n_cavity), qt.sigmax())
    ops['sigma_y'] = qt.tensor(qt.qeye(n_cavity), qt.sigmay())

    return ops


def build_jc_hamiltonian(
        n_cavity: int = 10,
        omega_c: float = 1.0,
        omega_a: float = 1.0,
        g: float = 0.1,
        rotating_wave: bool = True,
        operators: dict = None):
    """
    Build the Jaynes-Cummings Hamiltonian

    H = omega_c * a_dag * a + (omega_a/2) * sigma_z + g * (a_dag * sigma_minus + a * sigma_plus)

    Parameters:
    -----------
    n_cavity : int
        Number of cavity Fock states (cavity truncation)
    omega_c : float
        Cavity frequency
    omega_a : float
        Atomic transition frequency
    g : float
        Atom-cavity coupling strength
    rotating_wave : bool
        Whether to apply rotating wave approximation
    operators : dict
        Pre-computed operators dictionary

    Returns:
    --------
    H : qutip.Qobj
        Hamiltonian as QuTip Qobj
    """

    if operators is None:
        operators = build_jc_operators(n_cavity)

    # Free cavity energy
    H_cavity = omega_c * operators['n_c']

    # Free atomic energy
    H_atom = 0.5 * omega_a * operators['sigma_z']

    # Interaction term
    if rotating_wave:
        H_int = g * (operators['a_dag'] * operators['sigma_minus'] +
                     operators['a'] * operators['sigma_plus'])
    else:
        H_int = g * (operators['a_dag'] + operators['a']) * \
            (operators['sigma_plus'] + operators['sigma_minus'])

    return H_cavity + H_atom + H_int


def build_jc_collapse_operators(
        operators: dict,
        cavity_decay: float = 0.0,
        atomic_decay: float = 0.0,
        atomic_dephasing: float = 0.0,
        thermal_noise: tuple = None):
    """
    Build collapse operators with rates incorporated for Jaynes-Cummings dissipation

    Parameters:
    -----------
    operators : dict
        Dictionary of operators from build_jc_operators
    cavity_decay : float
        Cavity photon decay rate (kappa)
    atomic_decay : float
        Atomic spontaneous emission rate (gamma)
    atomic_dephasing : float
        Atomic pure dephasing rate (gamma_phi)
    thermal_noise : tuple
        (n_th, kappa) where n_th is mean thermal photon number
        and kappa is the cavity decay rate

    Returns:
    --------
    c_ops : list
        List of collapse operators with rates incorporated (sqrt(rate) * operator)
    """
    c_ops = []

    # Cavity photon decay
    if cavity_decay > 0:
        c_ops.append(np.sqrt(cavity_decay) * operators['a'])

    # Atomic spontaneous emission
    if atomic_decay > 0:
        c_ops.append(np.sqrt(atomic_decay) * operators['sigma_minus'])

    # Atomic pure dephasing
    if atomic_dephasing > 0:
        c_ops.append(np.sqrt(atomic_dephasing) * operators['sigma_z'])

    # Thermal noise
    if thermal_noise:
        n_th, kappa = thermal_noise
        if n_th > 0:
            c_ops.append(np.sqrt(kappa * n_th) * operators['a_dag'])

        c_ops.append(np.sqrt(kappa * (n_th + 1)) * operators['a'])

    return c_ops


def jc_latex_representation(rotating_wave: bool = True):
    """Return LaTeX representation of the Hamiltonian"""
    if rotating_wave:
        return (r"H = \omega_c a^\dagger a + \frac{\omega_a}{2}\sigma_z + "
                r"g(a^\dagger\sigma_- + a\sigma_+)")
    else:
        return (r"H = \omega_c a^\dagger a + \frac{\omega_a}{2}\sigma_z + "
                r"g(a^\dagger + a)(\sigma_+ + \sigma_-)")
