
"""
PETSc backend for HEOM RHS matrix assembly and solving.
"""

import numpy as np
from petsc4py import PETSc

from qutip import Qobj
from qutip.core import data as _data
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
    """
    
    name = "petsc_heomsolver"
    solver_options = HEOMSolver.solver_options.copy()
    solver_options["method"] = "petsc"

    def __init__(
        self, H, bath, max_depth, *, odd_parity=False, options=None, backend=None
    ):
        import time
        from qutip.core import QobjEvo, CoreOptions
        from qutip.core.superoperator import liouvillian, spre, spost
        from .bofin_solvers import HierarchyADOs

        _time_start = time.time()
        self.odd_parity = bool(odd_parity)
        if not isinstance(H, (Qobj, QobjEvo)):
            raise TypeError("The Hamiltonian (H) must be a Qobj or QobjEvo")

        H = QobjEvo(H)
        self.L_sys = (
            liouvillian(H) if H.type == "oper"
            else H
        )

        self._sys_shape = int(np.sqrt(self.L_sys.shape[0]))
        self._sup_shape = self.L_sys.shape[0]

        self.ados = HierarchyADOs(
            self._combine_bath_exponents(bath), max_depth,
        )
        self._n_ados = len(self.ados.labels)
        self._n_exponents = len(self.ados.exponents)

        self._init_ados_time = time.time() - _time_start
        _time_start = time.time()

        with CoreOptions(default_dtype="csr"):
            self._sId = _data.identity(self._sup_shape, dtype="csr")

            Qs = [exp.Q.to("csr") for exp in self.ados.exponents]
            self._spreQ = [spre(op).data for op in Qs]
            self._spostQ = [spost(op).data for op in Qs]
            self._s_pre_minus_post_Q = [
                _data.sub(self._spreQ[k], self._spostQ[k])
                for k in range(self._n_exponents)
            ]
            self._s_pre_plus_post_Q = [
                _data.add(self._spreQ[k], self._spostQ[k])
                for k in range(self._n_exponents)
            ]
            self._spreQdag = [spre(op.dag()).data for op in Qs]
            self._spostQdag = [spost(op.dag()).data for op in Qs]
            self._s_pre_minus_post_Qdag = [
                _data.sub(self._spreQdag[k], self._spostQdag[k])
                for k in range(self._n_exponents)
            ]
            self._s_pre_plus_post_Qdag = [
                _data.add(self._spreQdag[k], self._spostQdag[k])
                for k in range(self._n_exponents)
            ]

            self._init_superop_cache_time = time.time() - _time_start
            _time_start = time.time()

            self.rhs = self._calculate_rhs()

        self._init_rhs_time = time.time() - _time_start

        self.options = options
        self._state_metadata = {}
        _integrator_start = time.time()
        self._integrator = self._get_integrator()
        self._init_integrator_time = time.time() - _integrator_start
        self.stats = self._initialize_stats()

    def _calculate_rhs(self):
        """ Make the full RHS required by the solver. """
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
        ksp_type="gmres", pc_type="none", ksp_rtol=1e-8, ksp_atol=1e-8,
        **kwargs
    ):
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
            raise RuntimeError(f"PETSc KSP solve failed to converge. Reason code: {reason}")

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

    def _prepare_state(self, state):
        n = self._sys_shape
        rho_dims = self._sys_dims
        hierarchy_shape = (self._n_ados, n, n)

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
            rho0_he = rho0_he.reshape(n ** 2 * self._n_ados)
            rho0_he = _data.create(rho0_he)
        else:
            if rho0._dims != rho_dims:
                raise ValueError(
                    f"Initial state rho has dims {rho0.dims}"
                    f" but the system dims are {rho_dims}"
                )
            rho0_he = np.zeros(n ** 2 * self._n_ados, dtype=complex)
            rho0_he[: n ** 2] = rho0.full().ravel("F")
            rho0_he = _data.create(rho0_he)

        return rho0_he

    def _restore_state(self, state, *, copy=True):
        n = self._sys_shape
        rho_shape = (n, n)
        rho_dims = self._sys_dims
        hierarchy_shape = (self._n_ados, n, n)

        state_arr = state.to_array()
        rho = Qobj(
            state_arr[:n ** 2].reshape(rho_shape, order="F"),
            dims=rho_dims,
        )
        if state_arr.shape[0] == n ** 2:
            ado_state = HierarchyADOsState(
                rho, self.ados, None
            )
        else:
            ado_state = HierarchyADOsState(
                rho, self.ados, state_arr.reshape(hierarchy_shape)
            )
        return ado_state


class IntegratorPETSc(Integrator):
    """
    ODE Integrator that uses petsc4py TS (Time Stepping) solver.
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
    }

    support_time_dependant = False
    supports_blackbox = False
    name = "petsc"
    method = "petsc"
    rhs_format = "solver"

    def __init__(self, system, options):
        if hasattr(system, "rhs") and hasattr(system.rhs, "mat"):
            self.rhs_obj = system.rhs
        elif hasattr(system, "mat"):
            self.rhs_obj = system
        else:
            raise TypeError("Unsupported system type passed to IntegratorPETSc.")

        self._is_set = False
        self._back = (np.inf, None)
        self._options = self.integrator_options.copy()
        self.options = options
        self._prepare()

    @property
    def options(self):
        """Supported options by PETSc TS."""
        return self._options

    @options.setter
    def options(self, new_options):
        Integrator.options.fset(self, new_options)

    def _prepare(self):
        self.PETSc = PETSc
        self.mat = self.rhs_obj.mat

        self.ts = PETSc.TS().create()
        self.ts.setProblemType(PETSc.TS.ProblemType.LINEAR)

        self.ts.setRHSFunction(PETSc.TS.computeRHSFunctionLinear)
        self.ts.setRHSJacobian(PETSc.TS.computeRHSJacobianConstant, self.mat, self.mat)

        self.ts.setType(self.options.get("ts_type", "bdf"))
        self.ts.setTimeStep(self.options.get("dt", 1e-4))
        self.ts.setMaxSteps(self.options.get("max_steps", 100000))
        self.ts.setTolerances(
            atol=self.options.get("atol", 1e-8),
            rtol=self.options.get("rtol", 1e-6)
        )

        snes = self.ts.getSNES()
        ksp = snes.getKSP()
        ksp.setType(self.options.get("ksp_type", "bcgs"))
        pc = ksp.getPC()
        pc.setType(self.options.get("pc_type", "bjacobi"))
        ksp.setTolerances(
            atol=self.options.get("ksp_atol", 1e-8),
            rtol=self.options.get("ksp_rtol", 1e-6)
        )

        adapt_type = self.options.get("ts_adapt", "basic")
        if adapt_type:
            self.PETSc.Options().setValue("-ts_adapt_type", adapt_type)

        self.ts.setFromOptions()

        rstart, rend = self.mat.getOwnershipRange()
        self.vec = self.mat.createVecRight()
        self.vec.setFromOptions()

        self.ts.setSolution(self.vec)
        self.ts.setUp()
        self.name = f"petsc_ts_{self.options.get('ts_type', 'bdf')}"

        sys_size = self.rhs_obj.sys_size
        self.store_ados = self.options.get("store_ados", False)

        self._gather_full = self.store_ados or not sys_size

        if not self._gather_full:
            comm = self.mat.getComm()
            idx_gather = np.arange(sys_size, dtype=np.int32)
            is_global = self.PETSc.IS().createGeneral(idx_gather, comm=comm)

            self.vec_seq = self.PETSc.Vec().createSeq(sys_size)
            is_local = self.PETSc.IS().createStride(sys_size, first=0, step=1, comm=self.PETSc.COMM_SELF)

            self.scatter = self.PETSc.Scatter().create(self.vec, is_global, self.vec_seq, is_local)
        else:
            self.scatter, self.vec_seq = self.PETSc.Scatter.toAll(self.vec)

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
                self.vec.setValues(range(start_idx, end_idx), state_np[start_idx:end_idx])
        else:
            raise ValueError(f"Unexpected state0 size: {len(state_np)}")

        self.vec.assemblyBegin()
        self.vec.assemblyEnd()

        self.ts.setTime(t)
        self.ts.setSolution(self.vec)
        self._is_set = True

    def get_state(self, copy=True):
        self.scatter.scatter(self.vec, self.vec_seq, self.PETSc.InsertMode.INSERT_VALUES, self.PETSc.ScatterMode.FORWARD)
        gathered_np = self.vec_seq.getArray()
        if copy:
            gathered_np = gathered_np.copy()

        if self._gather_full:
            shape = (self.mat.getSize()[1], 1)
        else:
            shape = (self.rhs_obj.sys_size, 1)

        state_data = _data.Dense(gathered_np.reshape(shape))
        current_t = self.ts.getTime()
        return current_t, state_data

    def integrate(self, t, copy=True):
        if not self._is_set:
            raise RuntimeError("The initial state must be set using set_state before integrating.")

        self.ts.setMaxTime(t)
        self.ts.setExactFinalTime(self.PETSc.TS.ExactFinalTime.MATCHSTEP)
        self.ts.solve(self.vec)

        return self.get_state(copy=copy)

    def mcstep(self, t, copy=True):
        raise NotImplementedError("Monte Carlo steps are not supported for PETSc integrator.")


HEOMSolver.add_backend("petsc", PETScHEOMSolver)
HEOMSolver.add_integrator(IntegratorPETSc, "petsc")
