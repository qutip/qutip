"""
Tests for LindbladMatrixForm class.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

import qutip
from qutip.core.cy.lindblad_matrix_form import LindbladMatrixForm
from qutip.core.cy.qobjevo import QobjEvo
from qutip.core.data import dense


class TestLindbladMatrixFormInit:
    """Test LindbladMatrixForm initialization."""

    @pytest.mark.parametrize("N", [5, 10, 20])
    def test_constant_operators(self, N):
        """Test initialization with time-independent operators."""
        H = qutip.num(N)
        c_ops = [np.sqrt(0.1) * qutip.destroy(N)]

        rhs = LindbladMatrixForm(QobjEvo(H), [QobjEvo(c) for c in c_ops])

        assert rhs.shape == (N, N)
        assert rhs.num_collapse == 1
        assert rhs.isconstant is True

    def test_time_dependent_operators(self):
        """Test initialization with time-dependent operators."""
        N = 10
        H0 = qutip.num(N)
        H1 = 0.1 * qutip.create(N)
        H_td = [H0, [H1, lambda t, args: np.cos(t)]]
        c_ops = [np.sqrt(0.1) * qutip.destroy(N)]

        rhs = LindbladMatrixForm(QobjEvo(H_td), [QobjEvo(c) for c in c_ops])

        assert rhs.shape == (N, N)
        assert rhs.isconstant is False

    def test_multiple_collapse_ops(self):
        """Test with multiple collapse operators."""
        N = 10
        H = qutip.num(N)
        c_ops = [
            np.sqrt(0.1) * qutip.destroy(N),
            np.sqrt(0.05) * qutip.create(N),
        ]

        rhs = LindbladMatrixForm(QobjEvo(H), [QobjEvo(c) for c in c_ops])

        assert rhs.num_collapse == 2

    def test_non_hermitian_hamiltonian(self):
        """Test that H_nh is constructed correctly."""
        N = 10
        H = qutip.num(N)
        c = np.sqrt(0.1) * qutip.destroy(N)
        c_ops = [c]

        rhs = LindbladMatrixForm(QobjEvo(H), [QobjEvo(c) for c in c_ops])

        # H_nh should be H - (i/2) * c†c
        H_nh_expected = H - 0.5j * (c.dag() * c)
        H_nh_actual = rhs.H_nh(0.0)  # Get H_nh at t=0

        diff = (H_nh_actual - H_nh_expected).norm()
        assert diff < 1e-12


class TestLindbladMatrixFormProperties:
    """
    Property tests for LindbladMatrixForm correctness with different
    operator types.
    """

    @pytest.mark.parametrize("case", [
        pytest.param({
            'N': 10,
            'H': qutip.num(10),
            'c_ops': [np.sqrt(0.1) * qutip.destroy(10)],
        }, id='simple_decay'),
        pytest.param({
            'N': 8,
            'H': [qutip.num(8),
                  [qutip.create(8) + qutip.destroy(8), 'cos(t)']],
            'c_ops': [np.sqrt(0.05) * qutip.destroy(8)],
        }, id='driven_system'),
        pytest.param({
            'N': 6,
            'H': qutip.rand_herm(6, seed=42),
            'c_ops': [
                np.sqrt(0.1) * qutip.destroy(6),
                np.sqrt(0.05) * qutip.create(6),
                np.sqrt(0.02) * qutip.qeye(6),
            ],
        }, id='multiple_collapse'),
        pytest.param({
            'N': 8,
            'H': qutip.num(8),
            'c_ops': [[np.sqrt(0.1) * qutip.destroy(8), 'exp(-0.1*t)']],
        }, id='time_dependent_collapse'),
    ])
    def test_lindblad_matrix_form_vs_superoperator(self, case):
        """
        Compare LindbladMatrixForm against time-dependent Liouvillian
        superoperator.
        """
        N = case['N']
        H = case['H']
        c_ops = case['c_ops']

        # Initial state
        psi0 = qutip.fock(N, N // 2)
        rho0 = qutip.ket2dm(psi0)
        rho_dense = dense.Dense(rho0.data.to_array(), copy=False)
        rho_vec = qutip.operator_to_vector(rho0)

        # LindbladMatrixForm
        rhs_matrix = LindbladMatrixForm(
            QobjEvo(H), [QobjEvo(c) for c in c_ops]
        )

        # Time-dependent Liouvillian superoperator (built once)
        L = qutip.liouvillian(QobjEvo(H), [QobjEvo(c) for c in c_ops])

        for t in [0.0, 0.5, 1.0, 2.0]:
            drho_matrix = rhs_matrix.matmul_data(t, rho_dense)

            drho_super_vec = L(t) * rho_vec
            drho_super = qutip.vector_to_operator(drho_super_vec)

            diff = np.max(np.abs(drho_matrix.to_array() - drho_super.full()))
            assert diff < 1e-10, f"Failed at t={t}: diff = {diff}"
