#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

"""
Matrix-form Lindblad master equation integrand.

Computes the Lindblad master equation using matrix-matrix multiplications
instead of superoperator-vector multiplication.

.. math::

    \\frac{d\\rho}{dt} = -i[H,\\rho] + \\sum_i \\left(c_i \\rho c_i^\\dagger 
    - \\frac{1}{2}\\{c_i^\\dagger c_i, \\rho\\}\\right)
"""

from functools import partial
from libc.string cimport memcpy

from qutip.core.data cimport Data, Dense
from qutip.core.data.adjoint cimport iconj_dense
from qutip.core.data.add cimport iadd_dense
from qutip.core.data.mul cimport imul_dense
from qutip.core.data import dense
from qutip.core.cy.qobjevo cimport QobjEvo

__all__ = ['LindbladMatrixForm']


cdef class LindbladMatrixForm:
    """
    Computes the Lindblad master equation RHS in matrix form.

    Instead of building an n^2 x n^2 superoperator and vectorizing the density
    matrix, this class keeps rho as an n x n matrix and computes the RHS using
    matrix-matrix products:

    .. math::

        \\frac{d\\rho}{dt} = -i[H,\\rho] + \\sum_i \\left(c_i \\rho c_i^\\dagger 
        - \\frac{1}{2}\\{c_i^\\dagger c_i, \\rho\\}\\right)

    This class mimics the QobjEvo interface (specifically the matmul_data
    method) so it can be used as a drop-in replacement in solvers.

    Parameters
    ----------
    H : QobjEvo
        Hamiltonian (n x n operator)
    c_ops : list of QobjEvo
        Collapse operators (each n x n operator)

    Attributes
    ----------
    c_ops : list of QobjEvo
        The collapse operators
    H_nh : QobjEvo
        Non-Hermitian Hamiltonian: ``H - (i/2) * sum(dag(c_i) * c_i)``
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
        # Construct non-Hermitian Hamiltonian: H_nh = H - (i/2) * sum(dag(c_i) * c_i)
        # This allows us to write: drho/dt = -i(H_nh * rho - rho * dag(H_nh)) + sum(c_i * rho * dag(c_i))
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
        self.issuper = True

        # Check if all operators are time-independent
        self.isconstant = (H.isconstant and
                          all(c.isconstant for c in self.c_ops))
        
        # Pre-allocate temporary buffer for intermediate calculations
        n = self.shape[0]
        self._temp_buffer = None
        self._buffer_size = n

    cpdef Data matmul_data(LindbladMatrixForm self, object t, Data rho, Data out=None):
        """
        Compute drho/dt using matrix form of Lindblad equation.

        Exploits Hermiticity of rho to compute A + A.dag() where:

        .. math::

            A = -i H_{nh} \\rho + \\frac{1}{2} \\sum_i c_i \\rho c_i^\\dagger

        and H_nh = H - (i/2) sum(c_i^dag c_i).

        Parameters
        ----------
        t : float
            Time
        rho : Dense
            Density matrix (n x n dense matrix, not vectorized)
        out : Dense, optional
            Output buffer. If None, allocates new Dense object.

        Returns
        -------
        drho_dt : Dense
            Time derivative drho/dt as n x n dense matrix
        """
        cdef Dense rho_dense, out_dense, temp_dense
        cdef int i
        cdef int n = rho.shape[0]

        if type(rho) is not Dense:
            raise TypeError(
                f"LindbladMatrixForm.matmul_data() requires Dense input, "
                f"got {type(rho).__name__}."
            )

        rho_dense = <Dense>rho

        # Initialize output to zero
        if out is None:
            out_dense = dense.zeros(n, n, rho_dense.fortran)
        else:
            out_dense = <Dense>out
            imul_dense(out_dense, 0)

        # Allocate temp buffer on first use
        if self._temp_buffer is None:
            self._temp_buffer = dense.zeros(self._buffer_size, 
                                            self._buffer_size, 
                                            fortran=rho_dense.fortran)
        temp_dense = <Dense>self._temp_buffer

        # Compute A = -i H_nh @ rho + 0.5 sum(L @ rho @ Ld)
        # Then drho/dt = A + A.dag()

        # -i H_nh @ rho
        self.H_nh.matmul_data(t, rho_dense, out_dense, -1j)

        # +0.5 sum(L @ rho @ Ld)
        cdef QobjEvo c_op
        for i in range(self.num_collapse):
            c_op = <QobjEvo>self.c_ops[i]
            imul_dense(temp_dense, 0)
            c_op.matmul_data(t, rho_dense, temp_dense, 0.5)
            c_op.adjoint_rmatmul_data(t, temp_dense, out_dense)

        # Add Hermitian conjugate: out += out.dag()
        # Copy out to temp, conjugate temp in-place, then add with transpose
        # (transpose is achieved by flipping fortran flag for iadd_dense)
        cdef size_t nbytes = n * n * sizeof(double complex)
        memcpy(temp_dense.data, out_dense.data, nbytes)
        iconj_dense(temp_dense)
        temp_dense.fortran = not temp_dense.fortran  # transpose via flag
        iadd_dense(out_dense, temp_dense, 1)         # out += temp.T = out.dag()
        temp_dense.fortran = not temp_dense.fortran  # restore flag

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
