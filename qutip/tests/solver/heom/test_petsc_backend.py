"""
Tests for the PETSc backend of HEOM solver.
"""

import numpy as np
import pytest

from qutip import basis, sigmaz, sigmay, sigmax, fidelity, Qobj, expect
from qutip.solver.heom.bofin_solvers import HEOMSolver
from qutip.solver.heom.bofin_baths import DrudeLorentzBath

from .conftest import requires_petsc4py


@requires_petsc4py
class TestPETScBackend:
    def test_pure_dephasing_model(self, atol=1e-3):
        lam = 0.025
        gamma = 0.05
        T = 1 / 0.95
        Nk = 2
        H_sys = 1e-5 * Qobj(np.ones((2, 2)))
        Q = sigmaz()
        bath = DrudeLorentzBath(Q, lam=lam, gamma=gamma, T=T, Nk=Nk)
        options_csr = {"nsteps": 15000, "store_states": True}
        solver_csr = HEOMSolver(H_sys, bath, max_depth=5, options=options_csr)
        options_petsc = {"method": "petsc", "ts_type": "bdf", "atol": 1e-8, "rtol": 1e-6}
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
        options_petsc = {"method": "petsc", "ts_type": "bdf", "atol": 1e-8, "rtol": 1e-6}
        solver_petsc = HEOMSolver(H_sys, bath, 5, options=options_petsc)
        solver_csr = HEOMSolver(H_sys, bath, 5)
        tlist = np.linspace(0, 50, 11)
        rho0 = basis(2, 0) * basis(2, 0).dag()
        result_petsc = solver_petsc.run(rho0, tlist)
        result_csr = solver_csr.run(rho0, tlist)
        fid = fidelity(result_petsc.states[-1], result_csr.states[-1])
        np.testing.assert_allclose(fid, 1.0, atol=atol)

    def test_e_ops(self, atol=1e-2):
        H_sys = sigmax()
        bath = DrudeLorentzBath(sigmaz(), lam=0.05, gamma=0.1, T=1.0, Nk=2)
        options_petsc = {"method": "petsc", "ts_type": "bdf"}
        solver_petsc = HEOMSolver(H_sys, bath, 4, options=options_petsc)
        solver_csr = HEOMSolver(H_sys, bath, 4)
        tlist = np.linspace(0, 5, 11)
        rho0 = basis(2, 1) * basis(2, 1).dag()
        e_ops = [sigmaz(), sigmay()]
        result_petsc = solver_petsc.run(rho0, tlist, e_ops=e_ops)
        result_csr = solver_csr.run(rho0, tlist, e_ops=e_ops)
        for e_csr, e_petsc in zip(result_csr.expect, result_petsc.expect):
            np.testing.assert_allclose(e_csr, e_petsc, atol=atol)

    def test_steady_state_vs_csr(self, atol=1e-4):
        H_sys = 0.25 * sigmaz() + 0.5 * sigmax()
        bath = DrudeLorentzBath(sigmaz(), lam=0.025, gamma=0.05, T=1 / 0.95, Nk=2)
        solver_petsc = HEOMSolver(H_sys, bath, 5, options={"method": "petsc"})
        solver_csr = HEOMSolver(H_sys, bath, 5)
        rho_ss_petsc, ados_petsc = solver_petsc.steady_state()
        rho_ss_csr, ados_csr = solver_csr.steady_state(use_mkl=False)
        np.testing.assert_allclose(rho_ss_petsc.full(), rho_ss_csr.full(), atol=atol, err_msg="...")

    def test_steady_state_trace_and_hermiticity(self, atol=1e-6):
        H_sys = 0.5 * sigmaz() + 0.25 * sigmay()
        bath = DrudeLorentzBath(sigmaz(), lam=0.05, gamma=0.1, T=1.0, Nk=2)
        solver = HEOMSolver(H_sys, bath, 4, options={"method": "petsc"})
        rho_ss, _ = solver.steady_state()
        np.testing.assert_allclose(rho_ss.tr(), 1.0, atol=atol, err_msg="...")
        np.testing.assert_allclose(rho_ss.full(), rho_ss.dag().full(), atol=atol, err_msg="...")
        eigvals = np.linalg.eigvalsh(rho_ss.full())
        assert np.all(eigvals > -atol), f"..."

    def test_steady_state_is_fixed_point(self, atol=1e-3):
        H_sys = 0.25 * sigmaz() + 0.5 * sigmax()
        bath = DrudeLorentzBath(sigmaz(), lam=0.025, gamma=0.05, T=1 / 0.95, Nk=2)
        solver = HEOMSolver(H_sys, bath, 5, options={"method": "petsc", "ts_type": "bdf", "atol": 1e-8, "rtol": 1e-6})
        rho_ss, steady_ados = solver.steady_state()
        tlist = np.linspace(0, 20, 11)
        result = solver.run(steady_ados, tlist)
        for t, state in zip(result.times, result.states):
            np.testing.assert_allclose(state.full(), rho_ss.full(), atol=atol, err_msg=f"...")

    def test_steady_state_ksp_options(self, atol=1e-4):
        H_sys = sigmax()
        bath = DrudeLorentzBath(sigmaz(), lam=0.05, gamma=0.1, T=1.0, Nk=2)
        solver = HEOMSolver(H_sys, bath, 4, options={"method": "petsc"})
        rho_ss_default, _ = solver.steady_state()
        rho_ss_gmres_lu, _ = solver.steady_state(ksp_type="gmres", pc_type="lu", ksp_rtol=1e-10, ksp_atol=1e-10)
        np.testing.assert_allclose(rho_ss_default.full(), rho_ss_gmres_lu.full(), atol=atol, err_msg="...")
        solver_csr = HEOMSolver(H_sys, bath, 4)
        rho_ss_csr, _ = solver_csr.steady_state(use_mkl=False)
        np.testing.assert_allclose(rho_ss_default.full(), rho_ss_csr.full(), atol=atol, err_msg="...")
