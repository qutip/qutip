import numpy as np
from typing import Union
from qutip import destroy, create, sigmax, sigmay, sigmaz, coefficient
from .quantum_system import QuantumSystem


def qubit(omega: Union[float, coefficient] = 1.0, decay_rate: float = 0.0,
    dephasing_rate: float = 0.0) -> QuantumSystem:
    """
    Create two-level system (qubit)

    H = (omega_a/2) * sigma_z

    Parameters:
    -----------
    omega : float or coefficient, default=1.0
        Transition frequency, can be constant or time-dependent
    decay_rate : float, default=0.0
        Relaxation rate (1/T1)
    dephasing_rate : float, default=0.0
        Dephasing rate (1/T2)

    Returns:
    --------
    QuantumSystem instance configured as qubit
    """
    # Build operators
    operators = {}
    operators['sigma_minus'] = destroy(2)
    operators['sigma_plus'] = create(2)
    operators['sigma_z'] = sigmaz()
    operators['sigma_x'] = sigmax()
    operators['sigma_y'] = sigmay()

    # Build Hamiltonian
    hamiltonian = omega * (0.5 * operators['sigma_z'])

    # Build collapse operators
    c_ops = []
    if decay_rate > 0.0:
        c_ops.append(np.sqrt(decay_rate) * operators['sigma_minus'])
    if dephasing_rate > 0.0:
        c_ops.append(np.sqrt(dephasing_rate) * operators['sigma_z'])

    # LaTeX representation
    latex = r"H = \frac{\omega}{2}\sigma_z"

    # Create system with all components
    return QuantumSystem(
        hamiltonian=hamiltonian,
        name="Qubit",
        operators=operators,
        c_ops=c_ops,
        latex=latex,
        omega=omega,
        decay_rate=decay_rate,
        dephasing_rate=dephasing_rate
    )
