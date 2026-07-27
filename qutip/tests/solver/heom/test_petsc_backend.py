"""
Tests for the PETSc backend of HEOM solver.
"""

import numpy as np
import pytest

from qutip import basis, sigmaz, sigmay, sigmax, fidelity, Qobj, expect
from qutip.solver.heom.bofin_solvers import HEOMSolver
from qutip.solver.heom.bofin_baths import DrudeLorentzBath

petsc4py_installed = False
try:
    import petsc4py
    from petsc4py import PETSc
    petsc4py_installed = True
except ImportError:
    pass

@pytest.mark.skipif(not petsc4py_installed, reason="petsc4py is not installed")
class TestPETScBackend:
    def test_pure_dephasing_model(self, atol=1e-3):
        # Parameters
        lam = 0.025
        gamma = 0.05
        T = 1 / 0.95
        Nk = 2

        H_sys = 1e-5 * Qobj(np.ones((2, 2))) # Simple small Hamiltonian
        Q = sigmaz()

        bath = DrudeLorentzBath(Q, lam=lam, gamma=gamma, T=T, Nk=Nk)
        
        # Test standard CSR backend
        options_csr = {"backend": "csr", "nsteps": 15000, "store_states": True}
        solver_csr = HEOMSolver(H_sys, bath, max_depth=5, options=options_csr)
        
        # Test PETSc backend
        options_petsc = {"backend": "petsc", "ts_type": "bdf", "atol": 1e-8, "rtol": 1e-6}
        solver_petsc = HEOMSolver(H_sys, bath, max_depth=5, options=options_petsc)
        
        tlist = np.linspace(0, 10, 21)
        rho0 = 0.5 * Qobj(np.ones((2, 2)))

        result_csr = solver_csr.run(rho0, tlist)
        result_petsc = solver_petsc.run(rho0, tlist)

        for state_csr, state_petsc in zip(result_csr.states, result_petsc.states):
            np.testing.assert_allclose(state_csr.full(), state_petsc.full(), atol=atol)

    def test_steady_state_fidelity(self, atol=1e-3):
        H_sys = 0.25 * sigmaz() + 0.5 * sigmay()
        bath = DrudeLorentzBath(sigmaz(), lam=0.025, gamma=0.05, T=1/0.95, Nk=2)
        
        options_petsc = {"backend": "petsc", "ts_type": "bdf", "atol": 1e-8, "rtol": 1e-6}
        solver_petsc = HEOMSolver(H_sys, bath, 5, options=options_petsc)
        
        options_csr = {"backend": "csr"}
        solver_csr = HEOMSolver(H_sys, bath, 5, options=options_csr)
        
        tlist = np.linspace(0, 50, 11)
        rho0 = basis(2, 0) * basis(2, 0).dag()

        result_petsc = solver_petsc.run(rho0, tlist)
        result_csr = solver_csr.run(rho0, tlist)
        
        # Check fidelity at the end
        fid = fidelity(result_petsc.states[-1], result_csr.states[-1])
        np.testing.assert_allclose(fid, 1.0, atol=atol)

    def test_e_ops(self, atol=1e-2):
        H_sys = sigmax()
        bath = DrudeLorentzBath(sigmaz(), lam=0.05, gamma=0.1, T=1.0, Nk=2)
        
        options_petsc = {"backend": "petsc", "ts_type": "bdf"}
        solver_petsc = HEOMSolver(H_sys, bath, 4, options=options_petsc)
        
        options_csr = {"backend": "csr"}
        solver_csr = HEOMSolver(H_sys, bath, 4, options=options_csr)
        
        tlist = np.linspace(0, 5, 11)
        rho0 = basis(2, 1) * basis(2, 1).dag()
        e_ops = [sigmaz(), sigmay()]

        result_petsc = solver_petsc.run(rho0, tlist, e_ops=e_ops)
        result_csr = solver_csr.run(rho0, tlist, e_ops=e_ops)
        
        for e_csr, e_petsc in zip(result_csr.expect, result_petsc.expect):
            np.testing.assert_allclose(e_csr, e_petsc, atol=atol)

    # Steady-state solver tests (KSP path)

    def test_steady_state_vs_csr(self, atol=1e-4):
        """Compare PETSc steady_state() against CSR steady_state()."""
        H_sys = 0.25 * sigmaz() + 0.5 * sigmax()
        bath = DrudeLorentzBath(
            sigmaz(), lam=0.025, gamma=0.05, T=1 / 0.95, Nk=2,
        )

        solver_petsc = HEOMSolver(
            H_sys, bath, 5, options={"backend": "petsc"},
        )
        solver_csr = HEOMSolver(
            H_sys, bath, 5, options={"backend": "csr"},
        )

        rho_ss_petsc, ados_petsc = solver_petsc.steady_state()
        rho_ss_csr, ados_csr = solver_csr.steady_state(use_mkl=False)

        np.testing.assert_allclose(
            rho_ss_petsc.full(), rho_ss_csr.full(), atol=atol,
            err_msg="PETSc and CSR steady states differ element-wise",
        )

    def test_steady_state_trace_and_hermiticity(self, atol=1e-6):
        """Verify the PETSc steady state has Tr=1 and is Hermitian."""
        H_sys = 0.5 * sigmaz() + 0.25 * sigmay()
        bath = DrudeLorentzBath(
            sigmaz(), lam=0.05, gamma=0.1, T=1.0, Nk=2,
        )

        solver = HEOMSolver(
            H_sys, bath, 4, options={"backend": "petsc"},
        )
        rho_ss, _ = solver.steady_state()

        # Trace should be 1
        np.testing.assert_allclose(
            rho_ss.tr(), 1.0, atol=atol,
            err_msg="Steady state trace is not 1",
        )

        # Should be Hermitian: rho == rho^dag
        np.testing.assert_allclose(
            rho_ss.full(), rho_ss.dag().full(), atol=atol,
            err_msg="Steady state is not Hermitian",
        )

        # Eigenvalues should be non-negative (valid density matrix)
        eigvals = np.linalg.eigvalsh(rho_ss.full())
        assert np.all(eigvals > -atol), (
            f"Steady state has negative eigenvalue: {eigvals.min()}"
        )

    def test_steady_state_is_fixed_point(self, atol=1e-3):
        """Evolving from the steady state should leave it unchanged."""
        H_sys = 0.25 * sigmaz() + 0.5 * sigmax()
        bath = DrudeLorentzBath(
            sigmaz(), lam=0.025, gamma=0.05, T=1 / 0.95, Nk=2,
        )

        solver = HEOMSolver(
            H_sys, bath, 5,
            options={"backend": "petsc", "ts_type": "bdf",
                     "atol": 1e-8, "rtol": 1e-6},
        )
        rho_ss, steady_ados = solver.steady_state()

        # Now evolve from the full steady state hierarchy for some time
        tlist = np.linspace(0, 20, 11)
        result = solver.run(steady_ados, tlist)

        # The state at each time should remain close to rho_ss
        for t, state in zip(result.times, result.states):
            np.testing.assert_allclose(
                state.full(), rho_ss.full(), atol=atol,
                err_msg=f"State drifted from steady state at t={t}",
            )

    def test_steady_state_ksp_options(self, atol=1e-4):
        """Test that custom KSP solver options work correctly."""
        H_sys = sigmax()
        bath = DrudeLorentzBath(
            sigmaz(), lam=0.05, gamma=0.1, T=1.0, Nk=2,
        )

        solver = HEOMSolver(
            H_sys, bath, 4, options={"backend": "petsc"},
        )

        # Solve with default KSP options (preonly + lu = direct solver)
        rho_ss_default, _ = solver.steady_state()

        # Solve with GMRES preconditioned by LU (converges in 1 iteration)
        rho_ss_gmres_lu, _ = solver.steady_state(
            ksp_type="gmres", pc_type="lu", tol=1e-10, atol=1e-10,
        )

        # Both should give the same steady state
        np.testing.assert_allclose(
            rho_ss_default.full(), rho_ss_gmres_lu.full(), atol=atol,
            err_msg="Different KSP options produced different steady states",
        )

        # Also verify against CSR backend
        solver_csr = HEOMSolver(
            H_sys, bath, 4, options={"backend": "csr"},
        )
        rho_ss_csr, _ = solver_csr.steady_state(use_mkl=False)
        np.testing.assert_allclose(
            rho_ss_default.full(), rho_ss_csr.full(), atol=atol,
            err_msg="PETSc steady state differs from CSR reference",
        )
