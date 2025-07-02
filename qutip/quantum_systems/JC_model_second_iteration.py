import numpy as np
import qutip as qt


def jaynes_cummings_model(n_cavity: int = 10,
                          omega_c: float = 1.0,
                          omega_a: float = 1.0,
                          g: float = 0.1,
                          rotating_wave: bool = True,
                          cavity_decay: float = 0.0,
                          atomic_decay: float = 0.0,
                          atomic_dephasing: float = 0.0,
                          thermal_noise: tuple = None):
    """
    Build complete Jaynes-Cummings model with operators, Hamiltonian, collapse operators, and LaTeX

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
    operators : dict
        Dictionary containing all JC operators
    hamiltonian : qutip.Qobj
        Hamiltonian as QuTip Qobj
    c_ops : list
        List of collapse operators with rates incorporated
    latex : str
        LaTeX representation of the Hamiltonian
    """

    # Build operators
    operators = {}

    # Cavity operators
    operators['a'] = qt.tensor(qt.destroy(n_cavity), qt.qeye(2))
    operators['a_dag'] = operators['a'].dag()
    operators['n_c'] = operators['a_dag'] * operators['a']

    # Atomic operators
    operators['sigma_minus'] = qt.tensor(qt.qeye(n_cavity), qt.destroy(2))
    operators['sigma_plus'] = operators['sigma_minus'].dag()
    operators['sigma_z'] = qt.tensor(qt.qeye(n_cavity), qt.sigmaz())
    operators['sigma_x'] = qt.tensor(qt.qeye(n_cavity), qt.sigmax())
    operators['sigma_y'] = qt.tensor(qt.qeye(n_cavity), qt.sigmay())

    # Build Hamiltonian
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

    hamiltonian = H_cavity + H_atom + H_int

    # Build collapse operators
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

    # LaTeX representation
    if rotating_wave:
        latex = (r"H = \omega_c a^\dagger a + \frac{\omega_a}{2}\sigma_z + "
                 r"g(a^\dagger\sigma_- + a\sigma_+)")
    else:
        latex = (r"H = \omega_c a^\dagger a + \frac{\omega_a}{2}\sigma_z + "
                 r"g(a^\dagger + a)(\sigma_+ + \sigma_-)")

    return operators, hamiltonian, c_ops, latex
