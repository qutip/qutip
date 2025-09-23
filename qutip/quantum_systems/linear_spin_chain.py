import numpy as np
from qutip import tensor, qeye, sigmax, sigmay, sigmaz, coefficient, Qobj, sigmap, sigmam
from qutip.core.cy.coefficient import Coefficient
from typing import Union
from .quantum_system import QuantumSystem
from .jaynes_cummings import _create_sqrt_coefficient

def linear_spin_chain(
    model_type: str = "heisenberg",
    N: int = 4,
    J: Union[float, Coefficient] = 1.0,
    Jz: Union[float, Coefficient] = None,
    boundary_conditions: str = "open",
    B_x: Union[float, Coefficient] = 0.0,
    B_y: Union[float, Coefficient] = 0.0,
    B_z: Union[float, Coefficient] = 0.0,
    gamma_dephasing: Union[float, Coefficient] = 0.0,
    gamma_depolarizing: Union[float, Coefficient] = 0.0,
    gamma_thermal: Union[float, Coefficient] = 0.0,
    temperature: float = 0.0,
    transition_frequency: float = 1.0,
) -> QuantumSystem:
    """
    Create linear spin chain system

    Creates a 1D chain of spin-1/2 particles with nearest-neighbor interactions.
    Supports various spin models and open system dynamics.
    
    **Model Hamiltonians:**

    Heisenberg model:

    .. math::
        H = J \\sum_{\\langle i,j \\rangle} \\vec{S}_i \\cdot \\vec{S}_j + B_x \\sum_i S_i^x + B_y \\sum_i S_i^y + B_z \\sum_i S_i^z

    XXZ model:

    .. math::
        H = J \\sum_{\\langle i,j \\rangle} (S_i^x S_j^x + S_i^y S_j^y) + J_z \\sum_{\\langle i,j \\rangle} S_i^z S_j^z + \\vec{B} \\cdot \\sum_i \\vec{S}_i

    XY model:

    .. math::
        H = J \\sum_{\\langle i,j \\rangle} (S_i^x S_j^x + S_i^y S_j^y) + \\vec{B} \\cdot \\sum_i \\vec{S}_i

    Ising model (with transverse field when :math:`B_x, B_y \\neq 0`):

    .. math::
        H = J \\sum_{\\langle i,j \\rangle} S_i^z S_j^z + \\vec{B} \\cdot \\sum_i \\vec{S}_i

    **Dissipation Channels:**

    Pure dephasing:

    .. math::
        \\mathcal{L}[\\rho] = \\gamma_{\\text{deph}} \\sum_k \\left( S_k^z \\rho S_k^z - \\frac{1}{2}\\{(S_k^z)^2, \\rho\\} \\right)

    Depolarizing channel:

    .. math::
        \\mathcal{L}[\\rho] = \\frac{\\gamma_{\\text{depol}}}{3} \\sum_{k,\\alpha} \\left( S_k^\\alpha \\rho S_k^\\alpha - \\frac{1}{2}\\{(S_k^\\alpha)^2, \\rho\\} \\right)

    where :math:`\\alpha \\in \\{x, y, z\\}`.

    Thermal bath (:math:`T = 0`):

    .. math::
        \\mathcal{L}[\\rho] = \\gamma_{\\text{thermal}} \\sum_k \\left( S_k^- \\rho S_k^+ - \\frac{1}{2}\\{S_k^+ S_k^-, \\rho\\} \\right)

    Thermal bath (:math:`T > 0`):

    .. math::
        \\mathcal{L}[\\rho] = \\gamma_{\\text{thermal}} \\sum_k \\left[ p_{\\text{down}} \\left( S_k^- \\rho S_k^+ - \\frac{1}{2}\\{S_k^+ S_k^-, \\rho\\} \\right) + p_{\\text{up}} \\left( S_k^+ \\rho S_k^- - \\frac{1}{2}\\{S_k^- S_k^+, \\rho\\} \\right) \\right]

    with thermal factors:

    .. math::
        p_{\\text{down}} = \\frac{1}{1 + e^{-\\beta\\omega}}, \\quad p_{\\text{up}} = \\frac{e^{-\\beta\\omega}}{1 + e^{-\\beta\\omega}}

    **Note:** These thermal factors assume **Fermi-Dirac statistics** (coupling to a fermionic bath, 
    e.g., other spins). For bosonic environments (e.g., phonons), the rates would follow 
    Bose-Einstein statistics with different thermal factors.

    Parameters
    ----------
    model_type : str, default="heisenberg"
        Type of spin chain model: "heisenberg", "xxz", "xy", "ising"
    N : int, default=4
        Number of spins in the chain
    J : float or coefficient, default=1.0
        Coupling strength for XY interactions :math:`(S_x S_x + S_y S_y)` terms
    Jz : float or coefficient, default=None
        Z-coupling strength :math:`(S_z S_z)` terms. If None, equals J for Heisenberg, 0 for XY
    boundary_conditions : str, default="open"
        Boundary conditions: "open" or "periodic"
    B_x, B_y, B_z : float or coefficient, default=0.0
        External magnetic field components :math:`\\vec{B} = (B_x, B_y, B_z)`
    gamma_dephasing : float or coefficient, default=0.0
        Pure dephasing rate :math:`\\gamma_{\\text{deph}} ` (T2* process)
    gamma_depolarizing : float or coefficient, default=0.0
        Depolarizing channel rate :math:`\\gamma_{\\text{depol}}`
    gamma_thermal : float or coefficient, default=0.0
        Thermal bath coupling rate :math:`\\gamma_{\\text{thermal}}`. At :math:`T=0`, this gives pure 
        spontaneous emission (T1 process). At finite temperature, includes
        both emission and absorption processes.
    temperature : float, default=0.0
        Bath temperature :math:`T` (in units of transition_frequency)
    transition_frequency : float, default=1.0
        Energy scale :math:`\\omega` for thermal factors

    Returns
    -------
    QuantumSystem
        Configured linear spin chain system instance
    """

    # Validate inputs
    if N < 2:
        raise ValueError("Chain length N must be at least 2")

    model_types = ["heisenberg", "xxz", "xy", "ising"]
    if model_type.lower() not in model_types:
        raise ValueError(f"model_type must be one of {model_types}")

    boundary_types = ["open", "periodic"]
    if boundary_conditions.lower() not in boundary_types:
        raise ValueError(f"boundary_conditions must be one of {boundary_types}")

    model_type = model_type.lower()
    boundary_conditions = boundary_conditions.lower()

    # Set default Jz values
    if Jz is None:
        if model_type == "heisenberg":
            Jz = J
        elif model_type == "xy":
            Jz = 0.0
        else:  # xxz, ising
            Jz = J

    # Build operators
    operators = {}

    # Individual site operators
    for k in range(N):
        # Create single-site operators in full chain Hilbert space
        operators[f"S_{k}_x"] = _create_site_operator(N, k, sigmax() / 2)
        operators[f"S_{k}_y"] = _create_site_operator(N, k, sigmay() / 2)
        operators[f"S_{k}_z"] = _create_site_operator(N, k, sigmaz() / 2)
        operators[f"S_{k}_plus"] = _create_site_operator(N, k, sigmap())
        operators[f"S_{k}_minus"] = _create_site_operator(N, k, sigmam())

    # Total spin operators
    operators["S_x_total"] = sum(operators[f"S_{k}_x"] for k in range(N))
    operators["S_y_total"] = sum(operators[f"S_{k}_y"] for k in range(N))
    operators["S_z_total"] = sum(operators[f"S_{k}_z"] for k in range(N))
    operators["S_plus_total"] = sum(operators[f"S_{k}_plus"] for k in range(N))
    operators["S_minus_total"] = sum(operators[f"S_{k}_minus"] for k in range(N))

    # Magnetization (alias for S_z_total)
    operators["magnetization"] = operators["S_z_total"]

    # Nearest-neighbor correlation operators
    operators["correlation_xx_nn"] = sum(
        operators[f"S_{k}_x"] * operators[f"S_{(k+1)%N}_x"]
        for k in range(N if boundary_conditions == "periodic" else N - 1)
    )
    operators["correlation_zz_nn"] = sum(
        operators[f"S_{k}_z"] * operators[f"S_{(k+1)%N}_z"]
        for k in range(N if boundary_conditions == "periodic" else N - 1)
    )

    # Build Hamiltonian
    hamiltonian = _build_hamiltonian(
        operators, model_type, N, J, Jz, boundary_conditions, B_x, B_y, B_z
    )

    # Build collapse operators for dissipation
    c_ops = _build_collapse_operators(
        operators,
        N,
        gamma_dephasing,
        gamma_depolarizing,
        gamma_thermal,
        temperature,
        transition_frequency,
    )

    # Generate LaTeX representation
    latex = _generate_latex(model_type, J, Jz, B_x, B_y, B_z, boundary_conditions, gamma_dephasing, gamma_depolarizing, gamma_thermal, temperature)

    # Return system with all components
    return QuantumSystem(
        hamiltonian=hamiltonian,
        name=f"Linear Spin Chain ({model_type.upper()})",
        operators=operators,
        c_ops=c_ops,
        latex=latex,
        model_type=model_type,
        N=N,
        J=J,
        Jz=Jz,
        boundary_conditions=boundary_conditions,
        B_x=B_x,
        B_y=B_y,
        B_z=B_z,
        gamma_dephasing=gamma_dephasing,
        gamma_depolarizing=gamma_depolarizing,
        gamma_thermal=gamma_thermal,
        temperature=temperature,
        transition_frequency=transition_frequency,
    )


def _create_site_operator(N: int, site: int, single_op: Qobj) -> Qobj:
    """Create operator acting on specific site in N-spin chain"""
    op_list = [qeye(2)] * N
    op_list[site] = single_op
    return tensor(op_list)


def _build_hamiltonian(
    operators: dict,
    model_type: str,
    N: int,
    J: float,
    Jz: float,
    boundary_conditions: str,
    B_x: float,
    B_y: float,
    B_z: float,
) -> Qobj:
    """Build Hamiltonian for specified model type"""

    # Determine number of interaction terms
    num_interactions = N if boundary_conditions == "periodic" else N - 1

    # Interaction Hamiltonian
    H_interaction = 0

    for k in range(num_interactions):
        k_next = (k + 1) % N

        if model_type in ["heisenberg", "xxz", "xy"]:
            # XY interactions (present in all except Ising)
            H_interaction += J * (
                operators[f"S_{k}_x"] * operators[f"S_{k_next}_x"]
                + operators[f"S_{k}_y"] * operators[f"S_{k_next}_y"]
            )

        if model_type in ["heisenberg", "xxz", "ising"]:
            # Z interactions (present in all except XY)
            H_interaction += Jz * operators[f"S_{k}_z"] * operators[f"S_{k_next}_z"]

    # External magnetic field
    H_field = (
        B_x * operators["S_x_total"]
        + B_y * operators["S_y_total"]
        + B_z * operators["S_z_total"]
    )

    return H_interaction + H_field


def _build_collapse_operators(
    operators: dict,
    N: int,
    gamma_dephasing: Union[float, Coefficient],
    gamma_depolarizing: Union[float, Coefficient],
    gamma_thermal: Union[float, Coefficient],
    temperature: float,
    transition_frequency: float,
) -> list:
    """Build collapse operators for open system dynamics"""
    c_ops = []

    # Local dissipation on each site
    for k in range(N):

        # Pure dephasing
        if isinstance(gamma_dephasing, Coefficient) or gamma_dephasing > 0.0:
            sqrt_gamma_dephasing = _create_sqrt_coefficient(gamma_dephasing)
            c_ops.append(sqrt_gamma_dephasing * operators[f"S_{k}_z"])

        # Depolarizing channel
        if isinstance(gamma_depolarizing, Coefficient) or gamma_depolarizing > 0.0:
            if isinstance(gamma_depolarizing, Coefficient):
                # Create coefficient for gamma/3
                def depol_func(t, args):
                    gamma = gamma_depolarizing(t, args) 
                    return np.sqrt(gamma / 3.0)
                
                sqrt_depol_rate = coefficient(depol_func, args={})
            else:
                sqrt_depol_rate = np.sqrt(gamma_depolarizing / 3.0)
            
            c_ops.append(sqrt_depol_rate * operators[f"S_{k}_x"])
            c_ops.append(sqrt_depol_rate * operators[f"S_{k}_y"])
            c_ops.append(sqrt_depol_rate * operators[f"S_{k}_z"])

        # Thermal bath coupling
        if isinstance(gamma_thermal, Coefficient) or gamma_thermal > 0.0:
            if temperature > 0.0:
                # Finite temperature case
                beta = 1.0 / temperature
                exp_factor = np.exp(-beta * transition_frequency)
                p_down = 1.0 / (1.0 + exp_factor)
                p_up = exp_factor / (1.0 + exp_factor)
                
                if isinstance(gamma_thermal, Coefficient):
                    # Create coefficients for thermal rates
                    def thermal_down_func(t, args):
                        gamma = gamma_thermal(t, args)
                        return np.sqrt(gamma * p_down)
                    
                    def thermal_up_func(t, args):
                        gamma = gamma_thermal(t, args)
                        return np.sqrt(gamma * p_up)
                    
                    sqrt_thermal_down = coefficient(thermal_down_func, args={})
                    sqrt_thermal_up = coefficient(thermal_up_func, args={})
                else:
                    sqrt_thermal_down = np.sqrt(gamma_thermal * p_down)
                    sqrt_thermal_up = np.sqrt(gamma_thermal * p_up)
                
                c_ops.append(sqrt_thermal_down * operators[f"S_{k}_minus"])
                c_ops.append(sqrt_thermal_up * operators[f"S_{k}_plus"])
            else:
                # T=0 case: pure spontaneous emission (only downward transitions)
                sqrt_gamma_thermal = _create_sqrt_coefficient(gamma_thermal)
                c_ops.append(sqrt_gamma_thermal * operators[f"S_{k}_minus"])

    return c_ops


def _generate_latex(
    model_type: str,
    J: float,
    Jz: float,
    B_x: float,
    B_y: float,
    B_z: float,
    boundary_conditions: str,
    gamma_dephasing: Union[float, Coefficient],
    gamma_depolarizing: Union[float, Coefficient],
    gamma_thermal: Union[float, Coefficient],
    temperature: float,
) -> str:
    """Generate LaTeX representation of the Hamiltonian"""

    # Interaction terms
    if model_type == "heisenberg":
        interaction = r"J \sum_{\langle i,j \rangle} \vec{S}_i \cdot \vec{S}_j"
    elif model_type == "xxz":
        interaction = r"J \sum_{\langle i,j \rangle} (S_i^x S_j^x + S_i^y S_j^y) + J_z \sum_{\langle i,j \rangle} S_i^z S_j^z"
    elif model_type == "xy":
        interaction = r"J \sum_{\langle i,j \rangle} (S_i^x S_j^x + S_i^y S_j^y)"
    elif model_type == "ising":
        interaction = r"J \sum_{\langle i,j \rangle} S_i^z S_j^z"

    # Magnetic field terms
    field_terms = []
    if B_x != 0.0:
        field_terms.append(f"B_x S^x_{{total}}")
    if B_y != 0.0:
        field_terms.append(f"B_y S^y_{{total}}")
    if B_z != 0.0:
        field_terms.append(f"B_z S^z_{{total}}")

    field_str = " + ".join(field_terms) if field_terms else ""

    # Combine terms
    latex = f"H = {interaction}"
    if field_str:
        latex += f" + {field_str}"

    # Add boundary condition note
    bc_note = "PBC" if boundary_conditions == "periodic" else "OBC"
    latex += f" \\quad ({bc_note})"

    # Add Lindblad operators 
    noise_terms = []
    
    # Check for dephasing (handle both Coefficient and float)
    if isinstance(gamma_dephasing, Coefficient) or gamma_dephasing > 0.0:
        noise_terms.append(r"\sqrt{\gamma_{deph}} \sigma_z^{(k)}")
    
    # Check for depolarizing (handle both Coefficient and float)  
    if isinstance(gamma_depolarizing, Coefficient) or gamma_depolarizing > 0.0:
        noise_terms.append(r"\sqrt{\frac{\gamma_{depol}}{3}} \sigma_{\alpha}^{(k)}")
    
    # Check for thermal (handle both Coefficient and float)
    if isinstance(gamma_thermal, Coefficient) or gamma_thermal > 0.0:
        if temperature > 0.0:
            noise_terms.append(r"\sqrt{\gamma_{th} p_{\downarrow}} \sigma_-^{(k)}, \sqrt{\gamma_{th} p_{\uparrow}} \sigma_+^{(k)}")
        else:
            noise_terms.append(r"\sqrt{\gamma_{th}} \sigma_-^{(k)}")

    if noise_terms:
        latex += r" \\ \text{Lindblad ops: } " + ", ".join(noise_terms)

    return latex
