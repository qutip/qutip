"""
PETSc backend for HEOM RHS matrix assembly and solving.
"""

from time import time

import numpy as np
from petsc4py import PETSc

from qutip import Qobj
from qutip.core import data as _data
from qutip.core import QobjEvo
from .bofin_solvers import HEOMSolver, HierarchyADOsState
from qutip.solver.integrator.integrator import Integrator

__all__ = ["PETScHEOMSolver", "IntegratorPETSc"]


class _PETScRHS:
    """Wrapper around a PETSc.Mat that provides the interface expected
    by IntegratorPETSc and HEOMSolver.
    """

    def __init__(self, mat, sys_size):
        self.mat = mat
        self.sys_size = sys_size
        self.isconstant = True
        self.shape = (mat.getSize()[0], mat.getSize()[1])

    def arguments(self, args):
        if args:
            raise NotImplementedError(
                "Time-dependent arguments are completely unsupported "
                "with the PETSc backend."
            )

    def __getattr__(self, name):
        return getattr(self.mat, name)

    def __call__(self, t):
        return self


class PETScHEOMSolver(HEOMSolver):
    """
    PETSc-based solver for the Hierarchical Equations of Motion (HEOM).

    This solver distributes the HEOM Liouvillian across MPI ranks using
    PETSc, enabling parallel time evolution and steady-state solving for
    large open quantum system problems.

    .. note::
        This solver only supports **time-independent** Hamiltonians. Passing
        a time-dependent ``H`` will raise a ``NotImplementedError`` during
        initialization.

    .. note::
        To run with MPI parallelism, scripts must be launched with an MPI
        runner (e.g. ``mpirun -np 4 python script.py``).

    Parameters
    ----------
    H, bath, max_depth, odd_parity, options :
        Same as :class:`.HEOMSolver`.

    options : dict, optional
        Solver options. In addition to the options accepted by
        :class:`.HEOMSolver`, the PETSc backend accepts the following keys
        (all passed through to :class:`IntegratorPETSc`):

        - ``method`` : str, default ``"bdf"``
            ODE time-stepping method. Maps to PETSc's ``ts_type``. Common
            values: ``"bdf"``, ``"cn"`` (Crank-Nicolson), ``"beuler"``
            (backward Euler), ``"rk"`` (Runge-Kutta). See the full list in
            the `PETSc TS documentation
            <https://petsc.org/release/manual/ts/>`_.
        - ``ts_adapt`` : str, default ``"basic"``
            Adaptive step-size controller type (e.g. ``"basic"``, ``"none"``).
        - ``dt`` : float, default ``1e-4``
            Initial time step.
        - ``max_steps`` : int, default ``100000``
            Maximum number of time steps.
        - ``atol`` : float, default ``1e-8``
            Absolute tolerance for the ODE integrator.
        - ``rtol`` : float, default ``1e-6``
            Relative tolerance for the ODE integrator.
        - ``ksp_type`` : str, default ``"bcgs"``
            Krylov subspace method used by PETSc's linear solver inside the
            implicit time-stepper. Common values: ``"bcgs"``, ``"gmres"``.
        - ``pc_type`` : str, default ``"bjacobi"``
            Preconditioner type for the KSP solver. Common values:
            ``"bjacobi"``, ``"ilu"``, ``"none"``.
        - ``ksp_atol`` : float, default ``1e-8``
            Absolute tolerance for the KSP solver.
        - ``ksp_rtol`` : float, default ``1e-6``
            Relative tolerance for the KSP solver.
        - ``store_ados`` : bool, default ``False``
            Whether to gather and store the full ADO state vector at each
            output time. Requires an all-gather across MPI ranks and increases
            memory use.
    """

    name = "petsc heomsolve"
    # Extend solver_options with all PETSc-specific integrator keys so that
    # the base Solver.options setter (which rejects unknown keys) accepts them.
    # These mirror IntegratorPETSc.integrator_options and must be kept in sync.
    solver_options = {
        **HEOMSolver.solver_options,
        "ts_type": "bdf",
        "ts_adapt": "basic",
        "dt": 1e-4,
        "max_steps": 100000,
        "atol": 1e-8,
        "rtol": 1e-6,
        "ksp_type": "bcgs",
        "pc_type": "bjacobi",
        "ksp_atol": 1e-8,
        "ksp_rtol": 1e-6,
    }

    def __init__(
        self, H, bath, max_depth, *, odd_parity=False, options=None, backend=None
    ):
        # The system Liouvillian (L_sys) should not be too large because this
        # backend constructs the complete L of the system in memory. The
        # distributed "partial constructions" only apply to the ADOs (Auxiliary
        # Density Operators). This is generally fine because the system L is
        # usually relatively small.
        super().__init__(
            H, bath, max_depth, odd_parity=odd_parity, options=options, backend=backend
        )

        # Now we construct the actual distributed PETSc RHS
        self.rhs = self._petsc_calc_rhs()

        # Re-initialize the integrator with the new RHS
        self._integrator = self._get_integrator()

    def _get_integrator(self):
        """Return an IntegratorPETSc, always, regardless of options['method'].

        The standard QuTiP ``"method"`` option is re-mapped to ``"ts_type"``
        so that users can write ``options={"method": "bdf"}`` consistently
        with other QuTiP solvers.
        """
        _time_start = time()

        options = self._options.copy()

        # Map the standard qutip "method" key to PETSc's "ts_type" key,
        # unless the user already specified "ts_type" explicitly.
        user_method = options.get("method", "bdf")
        if user_method not in ("adams", "petsc"):
            # A concrete time-stepping method was requested via "method";
            # forward it to ts_type.
            options.setdefault("ts_type", user_method)

        integrator = IntegratorPETSc(self, options)
        # _init_integrator_time is read by Solver._initialize_stats(); the
        # base _get_integrator() normally sets it, but we override that method.
        self._init_integrator_time = time() - _time_start
        return integrator

    def _calculate_rhs(self):
        """Return a zero QobjEvo with the correct dimensions to satisfy
        Solver.__init__."""
        dim = self._sup_shape * self._n_ados
        dummy_mat = _data.csr.zeros(dim, dim)
        dummy_qobj = Qobj(dummy_mat, dims=[[dim], [dim]])
        return QobjEvo(dummy_qobj)

    def _petsc_calc_rhs(self):
        """Make the full distributed PETSc RHS required by the solver."""
        if not self.L_sys.isconstant:
            raise NotImplementedError(
                "PETSc backend currently supports only "
                "time-independent Liouvillians."
            )

        comm = PETSc.COMM_WORLD
        size = comm.getSize()
        rank = comm.getRank()

        block_size = self._sup_shape
        n_blocks = self._n_ados

        global_size = block_size * n_blocks
        n_local_blocks = n_blocks // size
        remainder = n_blocks % size

        if rank < remainder:
            local_blocks = n_local_blocks + 1
            start_block = rank * (n_local_blocks + 1)
            end_block = start_block + local_blocks
        else:
            local_blocks = n_local_blocks
            start_block = rank * n_local_blocks + remainder
            end_block = start_block + local_blocks

        local_size = local_blocks * block_size

        mat = PETSc.Mat().create(comm)
        mat.setSizes(((local_size, global_size), (local_size, global_size)))
        mat.setType(PETSc.Mat.Type.MPIAIJ)

        d_nnz = self._sys_shape + 2 * self._n_exponents * self._sys_shape
        o_nnz = d_nnz
        mat.setPreallocationNNZ((d_nnz, o_nnz))
        mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

        def add_op(row_he, col_he, op):
            row_blk = self.ados.idx(row_he)
            col_blk = self.ados.idx(col_he)
            sp_csr = op.as_scipy().tocsr()
            for i in range(sp_csr.shape[0]):
                start, end = sp_csr.indptr[i], sp_csr.indptr[i + 1]
                if start < end:
                    global_row = row_blk * block_size + i
                    global_cols = col_blk * block_size + sp_csr.indices[start:end]
                    vals = sp_csr.data[start:end]
                    mat.setValues(
                        [global_row], global_cols, vals,
                        addv=PETSc.InsertMode.ADD_VALUES,
                    )

        local_labels = self.ados.labels[start_block:end_block]

        for he_n in local_labels:
            op = self._grad_n(he_n)
            add_op(he_n, he_n, op)
            for k in range(len(self.ados.dims)):
                next_he = self.ados.next(he_n, k)
                if next_he is not None:
                    op = self._grad_next(he_n, k)
                    add_op(he_n, next_he, op)
                prev_he = self.ados.prev(he_n, k)
                if prev_he is not None:
                    op = self._grad_prev(he_n, k)
                    add_op(he_n, prev_he, op)

        if self.L_sys.isconstant:
            L_sys_csr = _data.to(_data.CSR, self.L_sys(0).data).as_scipy()
            for r_blk in range(start_block, end_block):
                for i in range(L_sys_csr.shape[0]):
                    start, end = L_sys_csr.indptr[i], L_sys_csr.indptr[i + 1]
                    if start < end:
                        global_row = r_blk * block_size + i
                        global_cols = r_blk * block_size + L_sys_csr.indices[start:end]
                        vals = L_sys_csr.data[start:end]
                        mat.setValues(
                            [global_row], global_cols, vals,
                            addv=PETSc.InsertMode.ADD_VALUES,
                        )

        mat.assemblyBegin()
        mat.assemblyEnd()

        return _PETScRHS(mat, self._sup_shape)

    def steady_state(
        self,
        ksp_type="gmres",
        pc_type="none",
        ksp_rtol=1e-8,
        ksp_atol=1e-8,
        **kwargs
    ):
        """
        Compute the steady state using PETSc's KSP linear solver.

        Solves :math:`\\mathcal{L} \\vec{\\rho}_{\\mathrm{heom}} = 0`
        in parallel, modifying the first row of the Liouvillian matrix to
        impose the trace normalisation constraint.

        Parameters
        ----------
        ksp_type : str, default ``"gmres"``
            Krylov Subspace method used by PETSc KSP. Common choices:
            ``"gmres"``, ``"bcgs"``, ``"cgs"``, ``"minres"``.
            See the `PETSc KSP documentation
            <https://petsc.org/release/manual/ksp/>`_ for the full list.
        pc_type : str, default ``"none"``
            Preconditioner type. Common choices: ``"none"``, ``"ilu"``,
            ``"bjacobi"``, ``"jacobi"``.
            See the `PETSc PC documentation
            <https://petsc.org/release/manual/ksp/#preconditioners>`_.
        ksp_rtol : float, default ``1e-8``
            Relative convergence tolerance for the KSP solver.
        ksp_atol : float, default ``1e-8``
            Absolute convergence tolerance for the KSP solver.

        Returns
        -------
        steady_state : :class:`.Qobj`
            The steady-state density matrix.
        steady_ados : :class:`~.HierarchyADOsState`
            The full steady-state ADO state.

        Raises
        ------
        RuntimeError
            If the KSP solver fails to converge.
        """
        n = self._sys_shape
        mat = self.rhs.duplicate(copy=True)
        mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

        rstart, rend = mat.getOwnershipRange()

        mat.zeroRows([0], diag=0.0)
        mat.assemblyBegin()
        mat.assemblyEnd()

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
            raise RuntimeError(
                f"PETSc KSP solve failed to converge. Reason code: {reason}"
            )

        scatter, vec_seq = PETSc.Scatter.toAll(x)
        scatter.scatter(
            x, vec_seq,
            PETSc.InsertMode.INSERT_VALUES,
            PETSc.ScatterMode.FORWARD,
        )

        solution = vec_seq.getArray().copy()
        data = _data.Dense(solution[:n ** 2].reshape((n, n), order="F"))
        data = _data.mul(_data.add(data, data.adjoint()), 0.5)
        steady_state = Qobj(data, dims=self._sys_dims, copy=False)

        solution = solution.reshape((self._n_ados, n, n))
        steady_ados = HierarchyADOsState(steady_state, self.ados, solution)

        return steady_state, steady_ados

    def _restore_state(self, state, *, copy=True):
        if state.to_array().shape[0] == self._sys_shape ** 2:
            n = self._sys_shape
            rho = Qobj(
                state.to_array().reshape((n, n), order="F"),
                dims=self._sys_dims,
            )
            return HierarchyADOsState(rho, self.ados, None)
        return super()._restore_state(state, copy=copy)


class IntegratorPETSc(Integrator):
    """
    ODE Integrator that uses petsc4py TS (Time Stepping) solver.

    This integrator distributes the state vector across MPI ranks via
    PETSc's parallel vector and matrix types. It only supports
    constant-coefficient (time-independent) systems.

    Options
    -------
    ts_type : str, default ``"bdf"``
        PETSc time-stepping method. Common values:

        - ``"bdf"`` — Backward Differentiation Formula (good for stiff ODEs)
        - ``"cn"`` — Crank-Nicolson
        - ``"beuler"`` — Backward Euler
        - ``"rk"`` — explicit Runge-Kutta (not recommended for stiff HEOM)

        See the `PETSc TS documentation
        <https://petsc.org/release/manual/ts/>`_ for the complete list.
    ts_adapt : str, default ``"basic"``
        Adaptive step-size controller. Use ``"none"`` to disable adaptivity.
    dt : float, default ``1e-4``
        Initial time step.
    max_steps : int, default ``100000``
        Maximum number of time-stepping iterations.
    atol : float, default ``1e-8``
        Absolute tolerance for the ODE integrator.
    rtol : float, default ``1e-6``
        Relative tolerance for the ODE integrator.
    ksp_type : str, default ``"bcgs"``
        Krylov subspace method for PETSc's internal linear solver.
        Common values: ``"bcgs"``, ``"gmres"``, ``"cgs"``.
    pc_type : str, default ``"bjacobi"``
        Preconditioner for the KSP solver.
        Common values: ``"bjacobi"``, ``"ilu"``, ``"none"``.
    ksp_atol : float, default ``1e-8``
        Absolute tolerance for the KSP solver.
    ksp_rtol : float, default ``1e-6``
        Relative tolerance for the KSP solver.
    store_ados : bool, default ``False``
        If ``True``, the full ADO state is gathered across all MPI ranks at
        each output time. Increases memory usage and communication cost.
    """

    integrator_options = {
        "ts_type": "bdf",
        "ts_adapt": "basic",
        "dt": 1e-4,
        "max_steps": 100000,
        "atol": 1e-8,
        "rtol": 1e-6,
        "ksp_type": "bcgs",
        "pc_type": "bjacobi",
        "ksp_atol": 1e-8,
        "ksp_rtol": 1e-6,
        "store_ados": False,
    }

    name = "petsc"
    method = "petsc"
    rhs_format = "solver"

    def __init__(self, system, options):
        if hasattr(system, "rhs") and hasattr(system.rhs, "mat"):
            self.rhs_obj = system.rhs
        elif hasattr(system, "mat"):
            self.rhs_obj = system
        elif isinstance(system, QobjEvo) or (
            hasattr(system, "rhs") and isinstance(system.rhs, QobjEvo)
        ):
            self.rhs_obj = None
        else:
            raise TypeError("Unsupported system type passed to IntegratorPETSc.")

        self._is_set = False
        self._options = self.integrator_options.copy()
        self.options = options

        if self.rhs_obj is not None:
            self._prepare()

    @property
    def options(self):
        """Supported options by PETSc TS."""
        return self._options

    @options.setter
    def options(self, new_options):
        Integrator.options.fset(self, new_options)

    def _prepare(self):
        self.mat = self.rhs_obj.mat

        self.ts = PETSc.TS().create()
        self.ts.setProblemType(PETSc.TS.ProblemType.LINEAR)

        self.ts.setRHSFunction(PETSc.TS.computeRHSFunctionLinear)
        self.ts.setRHSJacobian(
            PETSc.TS.computeRHSJacobianConstant, self.mat, self.mat
        )

        self.ts.setType(self.options["ts_type"])
        self.ts.setTimeStep(self.options["dt"])
        self.ts.setMaxSteps(self.options["max_steps"])
        self.ts.setTolerances(
            atol=self.options["atol"],
            rtol=self.options["rtol"],
        )

        snes = self.ts.getSNES()
        ksp = snes.getKSP()
        ksp.setType(self.options["ksp_type"])
        pc = ksp.getPC()
        pc.setType(self.options["pc_type"])
        ksp.setTolerances(
            atol=self.options["ksp_atol"],
            rtol=self.options["ksp_rtol"],
        )

        adapt_type = self.options["ts_adapt"]
        if adapt_type:
            PETSc.Options().setValue("-ts_adapt_type", adapt_type)

        self.ts.setFromOptions()

        rstart, rend = self.mat.getOwnershipRange()
        self.vec = self.mat.createVecRight()
        self.vec.setFromOptions()

        self.ts.setSolution(self.vec)
        self.ts.setUp()
        self.name = f"petsc_ts_{self.options['ts_type']}"

        sys_size = self.rhs_obj.sys_size
        self.store_ados = self.options["store_ados"]

        self._gather_full = self.store_ados or not sys_size

        if not self._gather_full:
            comm = self.mat.getComm()
            idx_gather = np.arange(sys_size, dtype=np.int32)
            is_global = PETSc.IS().createGeneral(idx_gather, comm=comm)

            self.vec_seq = PETSc.Vec().createSeq(sys_size)
            is_local = PETSc.IS().createStride(
                sys_size, first=0, step=1, comm=PETSc.COMM_SELF
            )

            self.scatter = PETSc.Scatter().create(
                self.vec, is_global, self.vec_seq, is_local
            )
        else:
            self.scatter, self.vec_seq = PETSc.Scatter.toAll(self.vec)

    def set_state(self, t, state0):
        state_np = state0.to_array().flatten()
        rstart, rend = self.mat.getOwnershipRange()
        self.vec.set(0.0)

        if len(state_np) == self.mat.getSize()[1]:
            local_state = state_np[rstart:rend]
            self.vec.setValues(range(rstart, rend), local_state)
        elif len(state_np) == self.rhs_obj.sys_size:
            start_idx = max(0, rstart)
            end_idx = min(self.rhs_obj.sys_size, rend)
            if start_idx < end_idx:
                self.vec.setValues(
                    range(start_idx, end_idx), state_np[start_idx:end_idx]
                )
        else:
            raise ValueError(f"Unexpected state0 size: {len(state_np)}")

        self.vec.assemblyBegin()
        self.vec.assemblyEnd()

        self.ts.setTime(t)
        self.ts.setSolution(self.vec)
        self._is_set = True

    def get_state(self, copy=True):
        self.scatter.scatter(
            self.vec, self.vec_seq,
            PETSc.InsertMode.INSERT_VALUES,
            PETSc.ScatterMode.FORWARD,
        )
        gathered_np = self.vec_seq.getArray().copy()

        if self._gather_full:
            shape = (self.mat.getSize()[1], 1)
        else:
            shape = (self.rhs_obj.sys_size, 1)

        state_data = _data.Dense(gathered_np.reshape(shape))
        current_t = self.ts.getTime()
        return current_t, state_data

    def integrate(self, t, copy=True):
        if not self._is_set:
            raise RuntimeError(
                "The initial state must be set using set_state before integrating."
            )

        self.ts.setMaxTime(t)
        self.ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)
        self.ts.solve(self.vec)

        return self.get_state(copy=copy)

    def mcstep(self, t, copy=True):
        raise NotImplementedError(
            "Monte Carlo steps are not supported for PETSc integrator."
        )


HEOMSolver.add_backend("petsc", PETScHEOMSolver)
HEOMSolver.add_integrator(IntegratorPETSc, "petsc")
