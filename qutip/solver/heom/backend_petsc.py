"""
PETSc backend for HEOM RHS matrix assembly.

This module provides a PETSc-based builder for the HEOM right-hand side
matrix. It streams sparse operator blocks directly into a distributed
PETSc matrix, avoiding the memory overhead of collecting Python objects
before assembly.
"""

import numpy as np
from petsc4py import PETSc


from .backend_base import HEOMBackend
from qutip import Qobj
from qutip.core import data as _data
from .bofin_solvers import HierarchyADOsState

__all__ = ["PETScHEOMBackend"]


class _PETScRHS:
    """Wrapper around a PETSc.Mat that provides the interface expected
    by IntegratorPETSc and HEOMSolver.

    This bridges the gap between PETSc's distributed matrix and the
    QuTiP solver infrastructure.

    Parameters
    ----------
    mat : PETSc.Mat
        The assembled PETSc matrix representing the HEOM RHS.
    sys_size : int
        The size of the system density matrix (i.e. ``N**2`` for an
        ``N``-level system).
    """

    def __init__(self, mat, sys_size):
        self.mat = mat
        self.sys_size = sys_size
        self.isconstant = True

    def arguments(self, args):
        """
        Update the arguments of the RHS.

        Raises
        ------
        NotImplementedError
            If any time-dependent arguments are supplied (time-dependence 
            is completely unsupported by the PETSc backend).
        """
        if args:
            raise NotImplementedError(
                "Time-dependent arguments are completely unsupported "
                "with the PETSc backend."
            )

    def __getattr__(self, name):
        return getattr(self.mat, name)


class PETScHEOMBackend(HEOMBackend):
    """A class for collecting elements of the right-hand side matrix
    of the HEOM and streaming them directly into a distributed PETSc
    matrix to avoid Python object memory overhead.

    This class follows the same backend interface as ``CSRHEOMBackend``
    but assembles blocks directly into a distributed PETSc ``Mat``.
    """

    def __init__(self, solver):
        super().__init__(solver)
        self._block_size = solver._sup_shape
        self._n_blocks = solver._n_ados
        self._f_idx = solver.ados.idx

        comm = PETSc.COMM_WORLD
        size = comm.getSize()
        rank = comm.getRank()

        global_size = self._block_size * self._n_blocks
        n_local_blocks = self._n_blocks // size
        remainder = self._n_blocks % size

        if rank < remainder:
            local_blocks = n_local_blocks + 1
        else:
            local_blocks = n_local_blocks

        local_size = local_blocks * self._block_size

        self.mat = PETSc.Mat().create(comm)
        self.mat.setSizes(
            ((local_size, global_size), (local_size, global_size))
        )
        self.mat.setType(PETSc.Mat.Type.MPIAIJ)

        # Estimate non-zeros per row for PETSc matrix preallocation.
        # - Diagonal block (L_sys): sys_shape non-zeros per row.
        # - Off-diagonal blocks: up to 2 * num_exponents connections per ADO,
        #   each having up to sys_shape non-zeros.
        d_nnz = self.solver._sys_shape + 2 * self.solver._n_exponents * self.solver._sys_shape
        o_nnz = d_nnz  # Distributed off-diagonal can be up to the same amount
        self.mat.setPreallocationNNZ((d_nnz, o_nnz))
        self.mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)


    # -- MPI label distribution ------------------------------------------

    def get_local_labels(self):
        """Return the subset of ADO *labels* owned by the current MPI rank.

        The labels are distributed across ranks using the same
        block-cyclic scheme that the PETSc matrix rows use, so each
        rank only builds the matrix rows it owns.

        Returns
        -------
        local_labels : array_like
            The slice of *labels* assigned to this MPI rank.
        """
        labels = self.solver.ados.labels
        comm = PETSc.COMM_WORLD
        size = comm.getSize()
        rank = comm.getRank()

        n_blocks = len(labels)
        n_local_blocks = n_blocks // size
        remainder = n_blocks % size

        if rank < remainder:
            start_block = rank * (n_local_blocks + 1)
            end_block = start_block + n_local_blocks + 1
        else:
            start_block = rank * n_local_blocks + remainder
            end_block = start_block + n_local_blocks

        return labels[start_block:end_block]

    # -- Operator insertion ----------------------------------------------

    def add_op(self, row_he, col_he, op):
        """Add a block operator into the PETSc matrix.

        The operator *op* is placed at the block position
        ``(row_he, col_he)`` in the HEOM matrix.

        Parameters
        ----------
        row_he : hashable
            The ADO label for the row block.
        col_he : hashable
            The ADO label for the column block.
        op : :class:`qutip.data.Data`
            The sparse operator to insert.
        """
        row_blk = self._f_idx(row_he)
        col_blk = self._f_idx(col_he)

        sp_csr = op.as_scipy().tocsr()
        for i in range(sp_csr.shape[0]):
            start, end = sp_csr.indptr[i], sp_csr.indptr[i + 1]
            if start < end:
                global_row = row_blk * self._block_size + i
                global_cols = (
                    col_blk * self._block_size + sp_csr.indices[start:end]
                )
                vals = sp_csr.data[start:end]
                self.mat.setValues(
                    [global_row], global_cols, vals,
                    addv=PETSc.InsertMode.ADD_VALUES,
                )

    # -- Assembly --------------------------------------------------------

    def gather(self, L_sys=None):
        """Assemble the PETSc matrix, optionally adding the system
        Liouvillian on the diagonal blocks.

        Parameters
        ----------
        L_sys : :class:`qutip.coefficient.Coefficient`, optional
            The system Liouvillian.  If provided and time-independent,
            its value is added to every diagonal block of the HEOM
            matrix.

        Returns
        -------
        mat : PETSc.Mat
            The assembled distributed PETSc matrix.
        """
        if L_sys is not None and L_sys.isconstant:
            from qutip.core import data as _data

            L_sys_csr = _data.to(_data.CSR, L_sys(0).data).as_scipy()

            comm = PETSc.COMM_WORLD
            size = comm.getSize()
            rank = comm.getRank()

            n_local_blocks = self._n_blocks // size
            remainder = self._n_blocks % size
            if rank < remainder:
                start_block = rank * (n_local_blocks + 1)
                end_block = start_block + n_local_blocks + 1
            else:
                start_block = rank * n_local_blocks + remainder
                end_block = start_block + n_local_blocks

            for r_blk in range(start_block, end_block):
                for i in range(L_sys_csr.shape[0]):
                    start, end = L_sys_csr.indptr[i], L_sys_csr.indptr[i + 1]
                    if start < end:
                        global_row = r_blk * self._block_size + i
                        global_cols = (
                            r_blk * self._block_size
                            + L_sys_csr.indices[start:end]
                        )
                        vals = L_sys_csr.data[start:end]
                        self.mat.setValues(
                            [global_row], global_cols, vals,
                            addv=PETSc.InsertMode.ADD_VALUES,
                        )

        self.mat.assemblyBegin()
        self.mat.assemblyEnd()
        return self.mat

    # -- Finalize --------------------------------------------------------

    def finalize(self):
        """Assemble the final PETSc matrix and wrap it in a
        :class:`_PETScRHS` object.
        """
        L_sys = self.solver.L_sys
        if not L_sys.isconstant:
            raise NotImplementedError(
                "PETSc backend currently supports only "
                "time-independent Liouvillians."
            )

        rhs_mat = self.gather(L_sys=L_sys)
        return _PETScRHS(rhs_mat, self.solver._sup_shape)

    def configure_solver(self, rhs, options):
        import time
        self.solver.rhs = rhs
        self.solver.options = options
        
        # We manually initialize stats because we must bypass `Solver.__init__` 
        # (which enforces QobjEvo) but we still use the official `_get_integrator` factory.
        _integrator_start = time.time()
        self.solver._integrator = self.solver._get_integrator()
        self.solver._init_integrator_time = time.time() - _integrator_start
        
        self.solver._state_metadata = {}
        self.solver.stats = self.solver._initialize_stats()

    def steady_state(
        self,
        ksp_type="gmres", pc_type="none", ksp_rtol=1e-8, ksp_atol=1e-8,
        **kwargs
    ):
        """
        Compute the steady state of the system using PETSc KSP solvers.

        Parameters
        ----------
        ksp_type : str, default='gmres'
            The PETSc KSP (Krylov Subspace) method type (e.g., 'bcgs', 'gmres').
        pc_type : str, default='none'
            The PETSc Preconditioner (PC) method type. Block Jacobi ('bjacobi') 
            is generally mathematically incompatible with distributed master 
            equations because ranks > 0 do not receive the trace constraint, 
            leaving their local sub-matrices singular and causing PCSETUP_FAILED.
        ksp_rtol : float, default=1e-8
            The relative tolerance for the KSP solver.
        ksp_atol : float, default=1e-8
            The absolute tolerance for the KSP solver.

        Returns
        -------
        steady_state : Qobj
            The steady state density matrix of the system.
        steady_ados : :class:`HierarchyADOsState`
            The steady state of the full ADO hierarchy.
        """
        from petsc4py import PETSc
        n = self.solver._sys_shape
        mat = self.solver.rhs.duplicate(copy=True)
        mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

        rstart, rend = mat.getOwnershipRange()

        mat.zeroRows([0], diag=0.0)
        mat.assemblyBegin()
        mat.assemblyEnd()

        # Force allocation of diagonal entries to prevent PCSetUp_ILU from failing
        # with "Matrix is missing diagonal entries"
        for i in range(rstart, rend):
            mat.setValue(i, i, 0.0, addv=PETSc.InsertMode.ADD_VALUES)
        mat.assemblyBegin()
        mat.assemblyEnd()

        if rstart <= 0 < rend:
            cols = [num * (n + 1) for num in range(n)]
            vals = [1.0] * n
            mat.setValues(
                [0], cols, vals, addv=PETSc.InsertMode.INSERT_VALUES,
            )

        mat.assemblyBegin()
        mat.assemblyEnd()

        b = mat.createVecRight()
        b.set(0.0)
        if rstart <= 0 < rend:
            b.setValue(0, 1.0)
        b.assemblyBegin()
        b.assemblyEnd()

        x = mat.createVecRight()

        ksp = PETSc.KSP().create(comm=mat.getComm())
        ksp.setOperators(mat)
        ksp.setType(ksp_type)
        pc = ksp.getPC()
        pc.setType(pc_type)
        ksp.setTolerances(rtol=ksp_rtol, atol=ksp_atol)
        ksp.setFromOptions()

        ksp.solve(b, x)
        
        reason = ksp.getConvergedReason()
        if reason < 0:
            raise RuntimeError(f"PETSc KSP solve failed to converge. Reason code: {reason}")

        scatter, vec_seq = PETSc.Scatter.toAll(x)
        scatter.scatter(
            x, vec_seq,
            PETSc.InsertMode.INSERT_VALUES,
            PETSc.ScatterMode.FORWARD,
        )

        solution = vec_seq.getArray().copy()
        data = _data.Dense(solution[:n ** 2].reshape((n, n), order='F'))
        data = _data.mul(_data.add(data, data.adjoint()), 0.5)
        steady_state = Qobj(data, dims=self.solver._sys_dims, copy=False)

        solution = solution.reshape((self.solver._n_ados, n, n))
        steady_ados = HierarchyADOsState(steady_state, self.solver.ados, solution)

        return steady_state, steady_ados

    def prepare_state(self, state):
        n = self.solver._sys_shape
        rho_dims = self.solver._sys_dims
        hierarchy_shape = (self.solver._n_ados, n, n)

        rho0 = state
        ado_init = not isinstance(rho0, Qobj)

        if ado_init:
            if isinstance(rho0, HierarchyADOsState):
                rho0_he = rho0._ado_state
            elif hasattr(rho0, "shape"):
                rho0_he = rho0
            else:
                raise TypeError(
                    f"Initial ADOs passed have type {type(rho0)}"
                    " but a HierarchyADOsState or a numpy array-like instance"
                    " was expected"
                )
            if rho0_he.shape != hierarchy_shape:
                raise ValueError(
                    f"Initial ADOs passed have shape {rho0_he.shape}"
                    f" but the solver hierarchy shape is {hierarchy_shape}"
                )
            rho0_he = rho0_he.reshape(n ** 2 * self.solver._n_ados)
            rho0_he = _data.create(rho0_he)
        else:
            if rho0._dims != rho_dims:
                raise ValueError(
                    f"Initial state rho has dims {rho0.dims}"
                    f" but the system dims are {rho_dims}"
                )
            import numpy as np
            rho0_he = np.zeros(n ** 2 * self.solver._n_ados, dtype=complex)
            rho0_he[: n ** 2] = rho0.full().ravel('F')
            rho0_he = _data.create(rho0_he)

        return rho0_he

    def restore_state(self, state, *, copy=True):
        n = self.solver._sys_shape
        rho_shape = (n, n)
        rho_dims = self.solver._sys_dims
        hierarchy_shape = (self.solver._n_ados, n, n)

        rho = Qobj(
            state.to_array()[:n ** 2].reshape(rho_shape, order='F'),
            dims=rho_dims,
        )
        # When using the PETSc backend with store_ados=False, the integrator 
        # avoids gathering the full hierarchy state across MPI nodes for performance, 
        # and instead only returns the system density matrix (size n**2). 
        # In this case, we populate the _ado_state with None.
        if state.shape[0] == n ** 2:
            ado_state = HierarchyADOsState(
                rho, self.solver.ados, None
            )
        else:
            ado_state = HierarchyADOsState(
                rho, self.solver.ados, state.to_array().reshape(hierarchy_shape)
            )
        return ado_state

# -- Backend Registration ----------------------------------------------
# Register PETSc backend with HEOMSolver if petsc4py is available.
try:
    from petsc4py import PETSc  # noqa: F401
    from .bofin_solvers import HEOMSolver
    HEOMSolver.add_backend(PETScHEOMBackend, "petsc")
except ImportError:
    pass
