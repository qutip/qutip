import numpy as np
import qutip as qt



class QuantumSystem:
    """
    General class for quantum systems

    All quantum systems are instances of this class, configured by factory functions.
    """

    def __init__(self, name: str, **kwargs):
        """
        Initialize quantum system

        Parameters:
        -----------
        name : str
            Name/type of the quantum system
        **kwargs : dict
            System-specific parameters
        """
        self.name = name
        self.parameters = kwargs

        self.operators = {}
        self.hamiltonian = None
        self.c_ops = []
        self.latex = ""

    def get_operators(self) -> dict:
        """Get operators dictionary"""
        return self.operators

    def get_hamiltonian(self) -> qt.Qobj:
        """Get Hamiltonian"""
        return self.hamiltonian

    def get_c_ops(self) -> list[qt.Qobj]:
        """Get collapse operators"""
        return self.c_ops

    def get_latex(self) -> str:
        """Get LaTeX representation"""
        return self.latex

    @property
    def dimension(self) -> int:
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
    def ground_state(self) -> qt.Qobj:
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
        #print(f"LaTeX: {self.latex}")
        # Display LaTeX if available and in Jupyter
        if self.latex and in_jupyter:
            print("LaTeX Representation:")
            display(Latex(f"$${self.latex}$$"))
        else:
            print(f"LaTeX: {self.latex}")

    def __repr__(self):
        return f"QuantumSystem(name='{self.name}', dim={self.dimension})"

