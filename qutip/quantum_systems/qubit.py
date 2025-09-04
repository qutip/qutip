import numpy as np
from typing import Union
from qutip import destroy, create, sigmax, sigmay, sigmaz, coefficient
from qutip.core.cy.coefficient import Coefficient
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

def qubit(omega: Union[float, Coefficient] = 1.0, decay_rate: Union[float, Coefficient] = 0.0,
    dephasing_rate: Union[float, Coefficient] = 0.0) -> QuantumSystem:
    """
    Create two-level system (qubit)

    H = (omega_a/2) * sigma_z

    Parameters:
    -----------
    omega : float or Coefficient, default=1.0
        Transition frequency, can be constant or time-dependent
    decay_rate : float or Coefficient, default=0.0
        Relaxation rate (1/T1), can be constant or time-dependent
    dephasing_rate : float or Coefficient, default=0.0
        Dephasing rate (1/T2), can be constant or time-dependent

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
    
    # Handle decay_rate: coefficient object OR positive numeric value
    if isinstance(decay_rate, Coefficient) or decay_rate > 0.0:
        sqrt_decay_rate = _create_sqrt_coefficient(decay_rate)
        c_ops.append(sqrt_decay_rate * operators['sigma_minus'])
    
    # Handle dephasing_rate: coefficient object OR positive numeric value
    if isinstance(dephasing_rate, Coefficient) or dephasing_rate > 0.0:
        sqrt_dephasing_rate = _create_sqrt_coefficient(dephasing_rate)
        c_ops.append(sqrt_dephasing_rate * operators['sigma_z'])

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
