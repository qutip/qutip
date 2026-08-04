import numpy as np
from qutip.core import data as _data
from qutip.solver.integrator.integrator import Integrator

__all__ = ["IntegratorPETSc"]

class IntegratorPETSc(Integrator):
    """
    ODE Integrator that uses petsc4py TS (Time Stepping) solver.
    This integrator handles PETSc.Mat directly instead of QobjEvo.
    """
    integrator_options = {
        "ts_type": "bdf",    # Backward Differentiation Formula (best for stiff HEOM)
        "ts_adapt": "basic", # Automatic step size adaptivity
        "dt": 1e-4,          # Initial time step
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

    @property
    def options(self):
        """
        Supported options by PETSc TS integrator:

        ts_type : str, default: "bdf"
            PETSc TS solver type. Common choices: "bdf" (stiff),
            "rk" (explicit Runge-Kutta), "cn" (Crank-Nicolson).

        ts_adapt : str, default: "basic"
            Adaptive time-stepping scheme. Use "basic" for automatic
            step size control, or "none" to disable adaptivity.

        dt : float, default: 1e-4
            Initial time step size.

        max_steps : int, default: 100000
            Maximum number of internal time steps.

        atol : float, default: 1e-8
            Absolute tolerance for the TS solver.

        rtol : float, default: 1e-6
            Relative tolerance for the TS solver.

        ksp_type : str, default: "bcgs"
            Krylov subspace method for the linear solver.
            Common choices: "bcgs", "gmres", "preonly".

        pc_type : str, default: "bjacobi"
            Preconditioner type for the linear solver.
            Common choices: "bjacobi", "ilu", "none".

        ksp_atol : float, default: 1e-8
            Absolute tolerance for the KSP linear solver.

        ksp_rtol : float, default: 1e-6
            Relative tolerance for the KSP linear solver.
    """
        return self._options

    @options.setter
    def options(self, new_options):
        Integrator.options.fset(self, new_options)

    def _prepare(self):
        from petsc4py import PETSc

        self.PETSc = PETSc
        # self.system is the PETScRhsWrapper from bofin_solvers.py
        self.mat = self.system.mat

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

        # Configure the internal linear solver (KSP) for implicit methods
        snes = self.ts.getSNES()
        ksp = snes.getKSP()
        ksp.setType(self.options.get("ksp_type", "bcgs"))
        pc = ksp.getPC()
        pc.setType(self.options.get("pc_type", "bjacobi"))
        ksp.setTolerances(
            atol=self.options.get("ksp_atol", 1e-8),
            rtol=self.options.get("ksp_rtol", 1e-6)
        )

        # Explicitly configure adaptivity via PETSc options dictionary
        adapt_type = self.options.get("ts_adapt", "basic")
        if adapt_type:
            self.PETSc.Options().setValue("-ts_adapt_type", adapt_type)

        # Allow command-line options to override TS settings and apply our set values
        self.ts.setFromOptions()

        # We need a PETSc Vec for the state
        rstart, rend = self.mat.getOwnershipRange()
        self.vec = self.mat.createVecRight()
        self.vec.setFromOptions()

        self.ts.setSolution(self.vec)
        self.ts.setUp()
        self.name = f"petsc_ts_{self.options.get('ts_type', 'bdf')}"

        # Determine whether to gather the full hierarchy or just the density matrix
        sys_size = self.system.sys_size
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
        # state0 is a qutip.Data object (usually Dense), we need to extract its values
        # state0 represents the full hierarchy state OR just the system state
        state_np = state0.to_array().flatten()

        rstart, rend = self.mat.getOwnershipRange()
        self.vec.set(0.0) # Zero out the global vector first

        if len(state_np) == self.mat.getSize()[1]:
            local_state = state_np[rstart:rend]
            self.vec.setValues(range(rstart, rend), local_state)
        elif len(state_np) == self.system.sys_size:
            start_idx = max(0, rstart)
            end_idx = min(self.system.sys_size, rend)
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
        # Gather the distributed vector back to a QuTiP Data object
        self.scatter.scatter(self.vec, self.vec_seq, self.PETSc.InsertMode.INSERT_VALUES, self.PETSc.ScatterMode.FORWARD)

        gathered_np = self.vec_seq.getArray()
        if copy:
            gathered_np = gathered_np.copy()

        # Convert back to qutip.Data Dense
        if self._gather_full:
            shape = (self.mat.getSize()[1], 1)
        else:
            shape = (self.system.sys_size, 1)

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
