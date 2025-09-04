import numpy as np
import qutip as qt
from .quantum_system import QuantumSystem


def linear_spin_chain(
    model_type: str = "heisenberg",
    N: int = 4,
    J: float = 1.0,
    Jz: float = None,
    boundary_conditions: str = "open",
    B_x: float = 0.0,
    B_y: float = 0.0,
    B_z: float = 0.0,
    gamma_dephasing: float = 0.0,
    gamma_depolarizing: float = 0.0,
    gamma_thermal: float = 0.0,
    temperature: float = 0.0,
    transition_frequency: float = 1.0,
) -> QuantumSystem:
    """
    Create linear spin chain system

    Creates a 1D chain of spin-1/2 particles with nearest-neighbor interactions.
    Supports various spin models and open system dynamics.

    Model Hamiltonians:
    - Heisenberg: H = J * sum_i [S_i^x * S_{i+1}^x + S_i^y * S_{i+1}^y + S_i^z * S_{i+1}^z] + fields
    - XXZ: H = J * sum_i [S_i^x * S_{i+1}^x + S_i^y * S_{i+1}^y] + Jz * sum_i [S_i^z * S_{i+1}^z] + fields
    - XY: H = J * sum_i [S_i^x * S_{i+1}^x + S_i^y * S_{i+1}^y] + fields
    - Ising: H = J * sum_i [S_i^z * S_{i+1}^z] + fields

    Parameters:
    -----------
    model_type : str, default="heisenberg"
        Type of spin chain model: "heisenberg", "xxz", "xy", "ising"
    N : int, default=4
        Number of spins in the chain
    J : float, default=1.0
        Coupling strength for XY interactions (Sx*Sx + Sy*Sy terms)
    Jz : float, default=None
        Z-coupling strength (Sz*Sz terms). If None, equals J for Heisenberg, 0 for XY
    boundary_conditions : str, default="open"
        Boundary conditions: "open" or "periodic"
    B_x, B_y, B_z : float, default=0.0
        External magnetic field components
    gamma_dephasing : float, default=0.0
        Pure dephasing rate (T2* process)
    gamma_depolarizing : float, default=0.0
        Depolarizing channel rate
    gamma_thermal : float, default=0.0
        Thermal bath coupling rate. At temperature=0, this gives pure 
        spontaneous emission (T1 process). At finite temperature, includes
        both emission and absorption processes.
    temperature : float, default=0.0
        Bath temperature (in units of transition_frequency)
    transition_frequency : float, default=1.0
        Energy scale for thermal factors

    Returns:
    --------
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
        operators[f"S_{k}_x"] = _create_site_operator(N, k, qt.sigmax() / 2)
        operators[f"S_{k}_y"] = _create_site_operator(N, k, qt.sigmay() / 2)
        operators[f"S_{k}_z"] = _create_site_operator(N, k, qt.sigmaz() / 2)
        operators[f"S_{k}_plus"] = _create_site_operator(N, k, qt.sigmap())
        operators[f"S_{k}_minus"] = _create_site_operator(N, k, qt.sigmam())

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
    latex = _generate_latex(model_type, J, Jz, B_x, B_y, B_z, boundary_conditions)

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


def _create_site_operator(N: int, site: int, single_op: qt.Qobj) -> qt.Qobj:
    """Create operator acting on specific site in N-spin chain"""
    op_list = [qt.qeye(2)] * N
    op_list[site] = single_op
    return qt.tensor(op_list)


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
) -> qt.Qobj:
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
    gamma_dephasing: float,
    gamma_depolarizing: float,
    gamma_thermal: float,
    temperature: float,
    transition_frequency: float,
) -> list:
    """Build collapse operators for open system dynamics"""
    c_ops = []

    # Local dissipation on each site
    for k in range(N):

        # Pure dephasing
        if gamma_dephasing > 0.0:
            c_ops.append(np.sqrt(gamma_dephasing) * operators[f"S_{k}_z"])

        # Depolarizing channel
        if gamma_depolarizing > 0.0:
            rate = gamma_depolarizing / 3.0
            c_ops.append(np.sqrt(rate) * operators[f"S_{k}_x"])
            c_ops.append(np.sqrt(rate) * operators[f"S_{k}_y"])
            c_ops.append(np.sqrt(rate) * operators[f"S_{k}_z"])

        # Thermal bath coupling
        if gamma_thermal > 0.0:
            if temperature > 0.0:
                # Finite temperature case
                beta = 1.0 / temperature
                exp_factor = np.exp(-beta * transition_frequency)
                p_down = 1.0 / (1.0 + exp_factor)
                p_up = exp_factor / (1.0 + exp_factor)
                
                c_ops.append(np.sqrt(gamma_thermal * p_down) * operators[f"S_{k}_minus"])
                c_ops.append(np.sqrt(gamma_thermal * p_up) * operators[f"S_{k}_plus"])
            else:
                # T=0 case: pure spontaneous emission (only downward transitions)
                c_ops.append(np.sqrt(gamma_thermal) * operators[f"S_{k}_minus"])

    return c_ops


def _generate_latex(
    model_type: str,
    J: float,
    Jz: float,
    B_x: float,
    B_y: float,
    B_z: float,
    boundary_conditions: str,
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

    return latex
