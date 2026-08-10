class HEOMBackend:
    """
    Base class for HEOM backends.
    
    This class provides the interface for different backends to handle the
    construction of the RHS, steady state solving, and state preparation.
    """

    def __init__(self, solver):
        self.solver = solver

    def get_local_labels(self):
        """Return the ADO labels to process on this node."""
        raise NotImplementedError

    def add_op(self, row_he, col_he, op):
        """Add a block operator."""
        raise NotImplementedError

    def finalize(self):
        """Assemble the final RHS."""
        raise NotImplementedError

    def configure_solver(self, rhs, options):
        """Configure the base Solver using the built RHS."""
        raise NotImplementedError

    def steady_state(self, **kwargs):
        """Compute the steady state."""
        raise NotImplementedError

    def prepare_state(self, state):
        """Prepare the state for the integrator."""
        raise NotImplementedError

    def restore_state(self, state, *, copy=True):
        """Restore the state from the integrator output."""
        raise NotImplementedError
