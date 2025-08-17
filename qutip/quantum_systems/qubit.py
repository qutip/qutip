import numpy as np
import qutip as qt
from .quantum_system import QuantumSystem  # Import the QuantumSystem class


def qubit(omega: float = 1.0, decay_rate: float = 0.0, 
          dephasing_rate: float = 0.0) -> QuantumSystem:
    """
    Create two-level system (qubit)

    H = (omega_a/2) * sigma_z
    
    Parameters:
    -----------
    omega : float
        Transition frequency
    decay_rate : float
        Relaxation rate (1/T1)
    dephasing_rate : float
        Dephasing rate (1/T2)
        
    Returns:
    --------
    QuantumSystem instance configured as qubit
    """
    # Build operators 
    operators = {}
    operators['sigma_minus'] = qt.destroy(2)
    operators['sigma_plus'] = qt.create(2)
    operators['sigma_z'] = qt.sigmaz()
    operators['sigma_x'] = qt.sigmax()
    operators['sigma_y'] = qt.sigmay()
    
    # Build Hamiltonian 
    hamiltonian = 0.5 * omega * operators['sigma_z']
    
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
        name="Qubit",
        hamiltonian=hamiltonian,
        operators=operators,
        c_ops=c_ops,
        latex=latex,
        omega=omega,
        decay_rate=decay_rate,
        dephasing_rate=dephasing_rate
    )
