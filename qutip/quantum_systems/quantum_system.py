import numpy as np
from qutip import Qobj


class QuantumSystem:
    """
    General class for quantum systems

    All quantum systems are instances of this class, configured by factory functions.
    """

    def __init__(self, hamiltonian: Qobj, name: str = "Quantum System",
                 operators: dict = None, c_ops: list = None,
                 latex: str = "", **kwargs):
        """
        Initialize quantum system

        Parameters:
        -----------
        name : str
            Name/type of the quantum system
        hamiltonian : Qobj, optional
            System Hamiltonian
        operators : dict, optional
            Dictionary of system operators
        c_ops : list, optional
            List of collapse operators
        latex : str, optional
            LaTeX representation of the system
        **kwargs : dict
            Additional system-specific parameters
        """
        self.name = name
        self.parameters = kwargs

        # Set system components with defaults
        self.hamiltonian = hamiltonian
        self.operators = operators if operators is not None else {}
        self.c_ops = c_ops if c_ops is not None else []
        self.latex = latex

    @property
    def dimension(self) -> list:
        """Get Hilbert space dimension"""
        return self.hamiltonian.dims if self.hamiltonian else 0

    @property
    def eigenvalues(self) -> np.ndarray:
        """Get eigenvalues of the Hamiltonian"""
        return self.hamiltonian.eigenenergies()

    @property
    def eigenstates(self) -> tuple:
        """Get eigenvalues and eigenstates"""
        return self.hamiltonian.eigenstates()

    @property
    def ground_state(self) -> Qobj:
        """Get ground state"""
        _, states = self.eigenstates
        return states[0]

    def pretty_print(self):
        """Pretty print system information"""
        # Check if we're in a Jupyter environment
        try:
            from IPython.display import display, Markdown, Latex
            in_jupyter = True
        except ImportError:
            in_jupyter = False

        print(f"Quantum System: {self.name}")
        print(f"Hilbert Space Dimension: {self.dimension}")
        print(f"Parameters: {self.parameters}")
        print(f"Number of Operators: {len(self.operators)}")
        print(f"Number of Collapse Operators: {len(self.c_ops)}")
        print(f"LaTeX: {self.latex}")
        # Display LaTeX if available and in Jupyter
        if self.latex and in_jupyter:
            print("LaTeX Representation:")
            display(Latex(f"$${self.latex}$$"))
        else:
            print(f"LaTeX: {self.latex}")

    def __repr__(self):
        return f"QuantumSystem(name='{self.name}', dim={self.dimension})"

    def _repr_latex_(self):
        """
        Jupyter LaTeX representation.
        Uses self.latex if provided, otherwise shows the system name.
        """
        if getattr(self, "latex", None):
            s = self.latex.strip()
            if not (s.startswith("$") or s.startswith(
                    r"\[") or s.startswith(r"\begin{")):
                s = f"${s}$"
            return s
        return rf"$\text{{{self.name}}}$"
