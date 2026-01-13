#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

"""
Matrix-form Lindblad master equation integrand.

Computes dρ/dt = -i[H,ρ] + Σᵢ(cᵢρcᵢ† - ½{cᵢ†cᵢ,ρ}) using matrix-matrix
multiplications instead of superoperator-vector multiplication.
"""

from functools import partial

from qutip.core.data cimport Data, Dense
from qutip.core.data.add import iadd
from qutip.core.data.mul import imul
from qutip.core.data import dense
from qutip.core.cy.qobjevo cimport QobjEvo

__all__ = ['LindbladMatrixForm']


cdef class LindbladMatrixForm:
    """
    Computes the Lindblad master equation RHS in matrix form.

    Instead of building an n²×n² superoperator and vectorizing the density
    matrix, this class keeps ρ as an n×n matrix and computes the RHS using
    matrix-matrix products:

        dρ/dt = -i[H,ρ] + Σᵢ(cᵢρcᵢ† - ½{cᵢ†cᵢ,ρ})

    This class mimics the QobjEvo interface (specifically the matmul_data
    method) so it can be used as a drop-in replacement in solvers.

    Parameters
    ----------
    H : QobjEvo
        Hamiltonian (n×n operator)
    c_ops : list of QobjEvo
        Collapse operators (each n×n operator)

    Attributes
    ----------
    c_ops : list of QobjEvo
        The collapse operators
    H_nh : QobjEvo
        Non-Hermitian Hamiltonian H - (i/2)Σᵢ(cᵢ†cᵢ)
    num_collapse : int
        Number of collapse operators
    shape : tuple
        Shape of operators (n, n)
    isconstant : bool
        Whether all operators are time-independent
    """

    def __init__(self, H, c_ops, *, _H_nh=None):
        """
        Initialize matrix-form Lindblad system.

        Parameters
        ----------
        H : QobjEvo
            Hamiltonian
        c_ops : list of QobjEvo
            Collapse operators
        _H_nh : QobjEvo, optional
            Pre-computed non-Hermitian Hamiltonian. If provided, skips the
            H_nh computation. This is used internally for pickling.
        """
        self.c_ops = list(c_ops) if c_ops else []
        self.num_collapse = len(self.c_ops)

        # Use pre-computed H_nh if provided (e.g., from unpickling)
        if _H_nh is not None:
            self.H_nh = _H_nh
        # Construct non-Hermitian Hamiltonian: H_nh = H - (i/2)Σᵢ(cᵢ†cᵢ)
        # This allows us to write: dρ/dt = -i(H_nh ρ - ρ H_nh†) + Σᵢ(cᵢρcᵢ†)
        elif len(c_ops) > 0:
            H_nh = H.copy()
            for c_op in self.c_ops:
                H_nh = H_nh - (0.5j) * (c_op.dag() * c_op)
            H_nh.compress()
            self.H_nh = H_nh
        else:
            self.H_nh = H

        self._dims = H._dims
        self.shape = H.shape
        self.type = 'oper'
        self.issuper = False

        # Check if all operators are time-independent
        self.isconstant = (H.isconstant and
                          all(c.isconstant for c in self.c_ops))
        
        # Pre-allocate temporary buffer for intermediate calculations
        # This avoids allocation in the hot path
        # Note: We'll set the memory ordering dynamically on first use
        n = self.shape[0]
        self._temp_buffer = None
        self._temp_buffer_size = n

    cpdef Data matmul_data(LindbladMatrixForm self, object t, Data rho, Data out=None):
        """
        Compute dρ/dt using matrix form of Lindblad equation.

        Uses non-Hermitian Hamiltonian formulation:
            dρ/dt = -i(H_nh ρ - ρ H_nh†) + Σᵢ(cᵢρcᵢ†)
        where H_nh = H - (i/2)Σᵢ(cᵢ†cᵢ)

        This reduces operations from 6 per collapse op to 2 + 1 per collapse op.

        Parameters
        ----------
        t : float
            Time
        rho : Dense
            Density matrix (n×n dense matrix, not vectorized)
        out : Dense, optional
            Output buffer. If None, allocates new Dense object.

        Returns
        -------
        drho_dt : Dense
            Time derivative dρ/dt as n×n dense matrix
        """
        cdef Dense rho_dense, out_dense, temp_dense
        cdef int i
        cdef int n = rho.shape[0]

        # Check that rho is Dense - LindbladMatrixForm requires dense matrices
        if type(rho) is not Dense:
            raise TypeError(
                f"LindbladMatrixForm.matmul_data() requires Dense input, "
                f"got {type(rho).__name__}. Convert to dense first or use "
                f"MESolverMatrixForm which handles conversion automatically."
            )

        rho_dense = <Dense>rho

        # Initialize output to zero
        if out is None:
            out_dense = dense.zeros(n, n, rho_dense.fortran)
        else:
            out_dense = <Dense>out
            out_dense = <Dense>imul(out_dense, 0)

        # Use pre-allocated temporary buffer for intermediate results
        # Allocate on first use to match input memory ordering
        if self._temp_buffer is None:
            self._temp_buffer = dense.zeros(self._temp_buffer_size, self._temp_buffer_size, 
                                          fortran=rho_dense.fortran)
        temp_dense = <Dense>self._temp_buffer

        # Compute non-Hermitian commutator: -i(H_nh ρ - ρ H_nh†)
        # Use scale parameters for efficient accumulation
        
        # H_nh @ ρ with scale -1j (accumulate directly into out)
        self.H_nh.matmul_data(t, rho_dense, out_dense, -1j)
        
        # ρ @ H_nh† with scale +1j (accumulate directly into out)
        self.H_nh.adjoint_rmatmul_data(t, rho_dense, out_dense, 1j)

        # Lindblad jump terms: Σᵢ(cᵢρcᵢ†)
        # Use adjoint_rmatmul_data for efficient on-the-fly adjoint operations
        for i in range(self.num_collapse):
            # Step 1: temp = c @ ρ
            temp_dense = <Dense>imul(temp_dense, 0)  # Zero out temp
            self.c_ops[i].matmul_data(t, rho_dense, temp_dense)
            # Step 2: out += temp @ c†
            self.c_ops[i].adjoint_rmatmul_data(t, temp_dense, out_dense)

        return out_dense

    def __call__(LindbladMatrixForm self, double t, dict _args=None, **kwargs):
        """Not implemented - this class doesn't produce operators at time t."""
        raise NotImplementedError(
            "LindbladMatrixForm cannot be called to produce an operator. "
            "Use matmul_data() to compute the RHS."
        )

    def arguments(LindbladMatrixForm self, dict _args=None, **kwargs):
        """Update arguments in H_nh and c_ops.

        Parameters
        ----------
        _args : dict, optional
            Dictionary of arguments to update. Can also pass as keyword args.
        **kwargs :
            Keyword arguments to update.

        Notes
        -----
        If both the positional ``_args`` and keywords are passed new values
        from both will be used. If a key is present with both, the ``_args``
        dict value will take priority.
        """
        if _args is not None:
            kwargs.update(_args)
        self.H_nh.arguments(**kwargs)
        for c_op in self.c_ops:
            c_op.arguments(**kwargs)

    def __reduce__(self):
        """
        Support pickling by excluding temporary buffers.
        
        The _temp_buffer is a workspace that will be lazily recreated on first
        use, so we don't need to pickle it. We pass H_nh via _H_nh to avoid
        recomputing it on unpickle.
        """
        return (
            partial(LindbladMatrixForm, _H_nh=self.H_nh),
            (self.H_nh, self.c_ops)
        )

    def _register_feedback(self, solvers_feeds, solver):
        """
        Forward feedback registration to underlying QobjEvo operators.

        Parameters
        ----------
        solvers_feeds : dict
            Solver-specific feedback sources.
        solver : str
            Name of the solver for error messages.
        """
        self.H_nh._register_feedback(solvers_feeds, solver)
        for c_op in self.c_ops:
            c_op._register_feedback(solvers_feeds, solver)
