from qutip.sparse import eigh
from qutip import krylovsolve, sesolve
from qutip.solver import Options, Result
from qutip import tensor, Qobj, basis, expect, ket
from qutip import sigmax, sigmay, sigmaz, qeye, jmat
from qutip.qip.operations import x_gate, y_gate, z_gate
from qutip import rand_herm, rand_ket, rand_unitary_haar, num, destroy, create
import os
import math
import types
import pytest
from scipy.linalg import expm
import numpy as np

np.random.seed(0)


def h_sho(dim):
    U = rand_unitary_haar(dim)
    H = U * (num(dim) + 0.5) * U.dag()
    H = H / np.linalg.norm(H.full(), ord=2)
    return H


def h_ising_transverse(
    N: int, hx: float, hz: float, Jx: float, Jy: float, Jz: float
):

    hx = hx * np.ones(N)
    hz = hz * np.ones(N)
    Jx, Jy, Jz = Jx * np.ones(N), Jy * np.ones(N), Jz * np.ones(N)

    sx_list = [x_gate(N, i) for i in range(N)]
    sy_list = [y_gate(N, i) for i in range(N)]
    sz_list = [z_gate(N, i) for i in range(N)]

    # construct the hamiltonian
    H = 0

    # energy splitting terms
    for n in range(N):
        H += hz[n] * sz_list[n]
        H += hx[n] * sx_list[n]

        if n < N-1:
            # interaction terms
            H += -Jx[n] * sx_list[n] * sx_list[n + 1]
            H += -Jy[n] * sy_list[n] * sy_list[n + 1]
            H += -Jz[n] * sz_list[n] * sz_list[n + 1]

    return H


def err_psi(psi_a, psi_b):
    err = 1 - np.abs(psi_a.overlap(psi_b)) ** 2
    return err


def assert_err_states_less_than_tol(res_1, res_2, tol):
    """Asserts error of states of two results less than tol."""

    err_states = [err_psi(psi_1, psi_2) for (psi_1, psi_2)
                  in zip(res_1.states, res_2.states)]
    for err in err_states:
        assert err <= tol,\
            f"err in states {err} is > than tolerance {tol}."


def assert_err_expect_less_than_tol(exp_1, exp_2, tol):
    """Asserts error of expect values of two results less than tol."""
    err_expect = [np.abs(ex_1 - ex_2) for (ex_1, ex_2)
                  in zip(exp_1, exp_2)]

    for err in err_expect:
        assert err <= tol, \
            f"err in expect values {err} is > than tol {tol}."


def exactsolve(H, psi0, tlist):
    """Calculates exact solution by direct diagonalization."""

    dims = psi0.dims
    H = H.full()
    psi0 = psi0.full()

    eigenvalues, eigenvectors = eigh(H)

    psi_base_diag = np.matmul(eigenvectors.conj().T, psi0)
    U = np.exp(np.outer(-1j * eigenvalues, tlist))
    psi_list = np.matmul(
        eigenvectors, np.multiply(U, psi_base_diag.reshape([-1, 1]))
    )
    exact_results = Result()
    exact_results.states = [Qobj(state, dims=dims) for state in psi_list.T]

    return exact_results


class TestKrylovSolve:
    """
    A test class for the QuTiP Krylov Approximation method Solver.
    """

    def check_sparse_vs_dense(self, output_sparse, output_dense, tol=1e-5):
        "krylovsolve: comparing sparse vs non sparse results"

        assert_err_states_less_than_tol(res_1=outout_sparse,
                                        res_2=output_dense
                                        tol=tol)

    @pytest.mark.parametrize("density,dim", [(0.1, 512), (0.9, 800)],
                             ids=["sparse H check", "dense H check"])
    def test_01_check_sparse_vs_non_sparse_with_density_H(
        self, density, dim, krylov_dim=20
    ):
        "krylovsolve: comparing sparse vs non sparse."
        psi0 = rand_ket(dim)
        H_sparse = rand_herm(dim, density=0.1)
        tlist = np.linspace(0, 10, 200)

        output_sparse = krylovsolve(
            H_sparse, psi0, tlist, krylov_dim, sparse=True
        )
        output_dense = krylovsolve(
            H_sparse, psi0, tlist, krylov_dim, sparse=False
        )

        self.check_sparse_vs_dense(output_sparse, output_dense)

    def simple_check_states_e_ops(
        self,
        H,
        psi0,
        tlist,
        tol=1e-5,
        tol2=1e-4,
        krylov_dim=25,
        square_hamiltonian=True,
    ):
        """
        Compare integrated evolution with sesolve and exactsolve result.
        """

        options = Options(store_states=True)

        e_ops = [
            jmat((H.shape[0] - 1) / 2.0, "x"),
            jmat((H.shape[0] - 1) / 2.0, "y"),
            jmat((H.shape[0] - 1) / 2.0, "z"),
        ]
        if not square_hamiltonian:
            _e_ops = []
            for op in e_ops:
                op2 = op.copy()
                op2.dims = H.dims
                _e_ops.append(op2)
            e_ops = _e_ops

        output = krylovsolve(
            H, psi0, tlist, krylov_dim, e_ops=e_ops, options=options
        )
        output_ss = sesolve(H, psi0, tlist, e_ops=e_ops, options=options)
        output_exact = exactsolve(H, psi0, tlist)

        assert_err_states_less_than_tol(res_1=output,
                                        res_2=output_ss, tol=tol)

        assert_err_states_less_than_tol(res_1=output,
                                        res_2=output_exact, tol=tol)

        # for the operators, test against exactsolve for accuracy
        for i in range(len(e_ops)):
            output_exact.expect = [
                expect(e_ops[i], state) for state in output_exact.states
            ]
            assert_err_expect_less_than_tol(exp_1=output.expect[i],
                                            exp_2=output_exact.expect,
                                            tol=tol)

    def test_02_simple_check_states_e_ops_H_random(self):
        "krylovsolve: states with const H random"
        dim = 512
        psi0 = rand_ket(dim)
        H = rand_herm(dim)
        tlist = np.linspace(0, 10, 200)

        self.simple_check_states_e_ops(H, psi0, tlist)

    def test_03_simple_check_states_e_ops_H_ising_transverse(self):
        "krylovsolve: states with const H Ising Transverse Field"
        N = 8
        dim = 2**N
        H = h_ising_transverse(N, hx=0.1, hz=0.5, Jx=1, Jy=0, Jz=1)
        _dims = H.dims
        _dims2 = [1] * N
        psi0 = rand_ket(dim, dims=[_dims[0], _dims2])
        tlist = np.linspace(0, 20, 200)

        self.simple_check_states_e_ops(
            H, psi0, tlist, square_hamiltonian=False)

    def test_04_simple_check_states_e_ops_H_sho(self):
        "krylovsolve: states with const H SHO"
        dim = 100
        psi0 = rand_ket(dim)
        H = h_sho(dim)
        tlist = np.linspace(0, 20, 200)

        self.simple_check_states_e_ops(H, psi0, tlist)

    def check_e_ops_none(
        self, H, psi0, tlist, dim, krylov_dim=30, tol=1e-5, tol2=1e-4
    ):
        "krylovsolve: testing inputs when e_ops=None"

        krylov_outputs = krylovsolve(H, psi0, tlist, krylov_dim, e_ops=None)

        try:
            sesolve_outputs = sesolve(H, psi0, tlist, e_ops=None)
        except IndexError:  # if tlist=[], sesolve breaks but krylov doesn't
            pass

        if len(tlist) > 1:
            err_outputs = [
                err_psi(psi_k, psi_ss)
                for (psi_k, psi_ss) in zip(
                    krylov_outputs.states, sesolve_outputs.states
                )
            ]
            for err in err_outputs:
                assert (
                    err < tol
                ), f"difference between krylov and sesolve states evolution is\
                    greater than tol={tol} with err={err}"

        elif len(tlist) == 1:
            assert krylov_outputs.states == sesolve_outputs.states
        else:
            assert krylov_outputs.states == []

    @pytest.mark.parametrize(
        "dim,tlist", [(128, np.linspace(0, 5, 200)), (400, [2]), (560, [])],
        ids=["normal tlist", "single element tlist", "empty tlist"]
    )
    def test_05_check_e_ops_none(self, dim, tlist):
        "krylovsolve: check e_ops inputs with random H and different tlists."
        psi0 = rand_ket(dim)
        H = rand_herm(dim, density=0.5)
        self.check_e_ops_none(H, psi0, tlist, dim)

    def check_e_ops_callable(
        self,
        H,
        psi0,
        tlist,
        dim,
        krylov_dim=35,
        tol=1e-5,
        square_hamiltonian=True,
    ):
        "krylovsolve: testing inputs when e_ops=callable"

        def e_ops(t, psi): return expect(num(dim), psi)

        if not square_hamiltonian:
            H.dims = [[H.shape[0]], [H.shape[0]]]
            psi0.dims = [[H.shape[0]], [1]]

        krylov_outputs = krylovsolve(H, psi0, tlist, krylov_dim, e_ops=e_ops)
        exact_output = exactsolve(H, psi0, tlist)
        exact_output.expect = [
            e_ops(t, state) for (t, state) in zip(tlist, exact_output.states)
        ]

        try:
            sesolve_outputs = sesolve(H, psi0, tlist, e_ops=e_ops)
        except IndexError:  # if tlist=[], sesolve breaks but krylov doesn't
            pass

        if len(tlist) > 1:

            assert len(krylov_outputs.expect) == len(
                sesolve_outputs.expect
            ), "shape of outputs between krylov and sesolve differs"

            assert_err_expect_less_than_tol(exp_1=krylov_outputs.expect,
                                            exp_2=exact_output.expect,
                                            tol=tol)
        elif len(tlist) == 1:
            assert (
                np.abs(krylov_outputs.expect[0] - sesolve_outputs.expect[0])
                <= 1e-7
            ), "krylov and sesolve outputs differ for len(tlist)=1"
        else:
            assert krylov_outputs.states == []

    @pytest.mark.parametrize(
        "dim,tlist", [(128, np.linspace(0, 5, 200)), (400, [2]), (560, [])],
        ids=["normal tlist", "single element tlist", "empty tlist"]
    )
    def test_06_check_e_ops_callable(self, dim, tlist):
        "krylovsolve: check e_ops inputs with random H and different tlists."
        psi0 = rand_ket(dim)
        H = rand_herm(dim, density=0.5)

        self.check_e_ops_callable(H, psi0, tlist, dim)

    def check_e_ops_list_single_callable(
        self,
        H,
        psi0,
        tlist,
        dim,
        krylov_dim=35,
        tol=1e-5,
        square_hamiltonian=True,
    ):
        "krylovsolve: testing inputs when e_ops=[callable]"

        e_ops = [lambda t, psi: expect(num(dim), psi)]
        if not square_hamiltonian:
            H.dims = [[H.shape[0]], [H.shape[0]]]
            psi0.dims = [[H.shape[0]], [1]]

        krylov_outputs = krylovsolve(H, psi0, tlist, krylov_dim, e_ops=e_ops)
        exact_output = exactsolve(H, psi0, tlist)
        exact_output.expect = [
            e_ops[0](t, state)
            for (t, state) in zip(tlist, exact_output.states)
        ]

        try:
            sesolve_outputs = sesolve(H, psi0, tlist, e_ops=e_ops)
        except IndexError:  # if tlist=[], sesolve breaks but krylov doesn't
            pass

        if len(tlist) > 1:

            assert len(krylov_outputs.expect) == len(
                sesolve_outputs.expect
            ), "shape of outputs between krylov and sesolve differs"

            assert_err_expect_less_than_tol(exp_1=krylov_outputs.expect[0],
                                            exp_2=exact_output.expect,
                                            tol=tol)

        elif len(tlist) == 1:
            assert (
                np.abs(krylov_outputs.expect[0] - sesolve_outputs.expect[0])
                <= tol
            ), "expect outputs from krylovsolve and sesolve are not equal"
        else:
            assert krylov_outputs.states == []

    @pytest.mark.parametrize(
        "dim,tlist", [(128, np.linspace(0, 5, 200)), (400, [2]), (560, [])],
        ids=["normal tlist", "single element tlist", "empty tlist"]
    )
    def test_07_check_e_ops_list_single_callable(self, dim, tlist):
        "krylovsolve: check e_ops inputs with random H and different tlists."
        psi0 = rand_ket(dim)
        H = rand_herm(dim, density=0.5)

        self.check_e_ops_list_single_callable(H, psi0, tlist, dim)

    def check_e_ops_mixed_list(
        self, H, psi0, tlist, dim, krylov_dim=35, tol=1e-5
    ):
        "krylovsolve: testing inputs when e_ops=[callable]"

        e_ops = [
            lambda t, psi: expect(num(dim), psi),
            jmat((H.shape[0] - 1) / 2.0, "x"),
        ]

        krylov_outputs = krylovsolve(H, psi0, tlist, krylov_dim, e_ops=e_ops)
        exact_output = exactsolve(H, psi0, tlist)

        try:
            sesolve_outputs = sesolve(H, psi0, tlist, e_ops=e_ops)
        except IndexError:  # if tlist=[], sesolve breaks but krylov doesn't
            pass

        if len(tlist) > 1:

            assert len(krylov_outputs.expect) == len(
                sesolve_outputs.expect
            ), "shape of outputs between krylov and sesolve differs"

            for idx, k_expect in enumerate(krylov_outputs.expect):
                if idx == 0:
                    exact_output.expect = [
                        e_ops[idx](t, state)
                        for (t, state) in zip(tlist, exact_output.states)
                    ]
                else:
                    exact_output.expect = [
                        expect(e_ops[idx], state)
                        for state in exact_output.states
                    ]

                assert_err_expect_less_than_tol(exp_1=k_expect,
                                                exp_2=exact_output.expect,
                                                tol=tol)
        elif len(tlist) == 1:
            assert len(krylov_outputs.expect) == len(
                sesolve_outputs.expect
            ), "shape of outputs between krylov and sesolve differs"
            for k_expect, ss_expect in zip(
                krylov_outputs.expect, sesolve_outputs.expect
            ):
                assert (
                    np.abs(k_expect - ss_expect) <= tol
                ), "expect outputs from krylovsolve and sesolve are not equal"
        else:
            assert krylov_outputs.states == []

    @pytest.mark.parametrize(
        "dim,tlist", [(128, np.linspace(0, 5, 200)), (400, [2]), (560, [])],
        ids=["normal tlist", "single element tlist", "empty tlist"]
    )
    def test_08_check_e_ops_mixed_list(self, dim, tlist):
        "krylovsolve: check e_ops inputs with random H and different tlists."
        psi0 = rand_ket(dim)
        H = rand_herm(dim, density=0.5)

        self.check_e_ops_mixed_list(H, psi0, tlist, dim)

    @pytest.mark.parametrize(
        "psi0,hz,Jx,Jz",
        [
            # eigenstate
            (ket([1, 0, 0, 0]), 0.5, 0, 1),
            # state in the magnetization subspace of the XXZ model
            (ket([0, 0, 1, 0]), 1.0, 1, 0),
        ],
        ids=["happy_breakdown_eigenstate", "happy_breakdown_symmetry"],
    )
    def test_09_happy_breakdown_simple(self, psi0, hz, Jx, Jz):

        krylov_dim = 12
        N = 4
        dim = 2**N
        H = h_ising_transverse(N, hx=0, hz=hz, Jx=Jx, Jy=0, Jz=Jz)
        _dims = H.dims
        _dims2 = [1] * N

        tlist = np.linspace(0, 20, 200)
        self.simple_check_states_e_ops(
            H, psi0, tlist, krylov_dim=krylov_dim, square_hamiltonian=False
        )

    @pytest.mark.parametrize(
        "psi0,hz,Jx,Jz",
        [
            # eigenstate
            (ket([1, 0, 0, 0]), 0.5, 0, 1),
            # state in the magnetization subspace of the XXZ model
            (ket([0, 0, 1, 0]), 1.0, 1, 0),
        ],
        ids=["happy_breakdown_eigenstate", "happy_breakdown_symmetry"],
    )
    def test_10_happy_breakdown_e_ops_none(self, psi0, hz, Jx, Jz):

        krylov_dim = 12
        N = 4
        dim = 2**N
        H = h_ising_transverse(N, hx=0, hz=hz, Jx=Jx, Jy=0, Jz=Jz)
        _dims = H.dims
        _dims2 = [1] * N

        tlist = np.linspace(0, 20, 200)
        self.check_e_ops_none(
            H, psi0, tlist, dim, krylov_dim=krylov_dim
        )

    @pytest.mark.parametrize(
        "psi0,hz,Jx,Jz",
        [
            # eigenstate
            (ket([1, 0, 0, 0]), 0.5, 0, 1),
            # state in the magnetization subspace of the XXZ model
            (ket([0, 0, 1, 0]), 1.0, 1, 0),
        ],
        ids=["happy_breakdown_eigenstate", "happy_breakdown_symmetry"],
    )
    def test_11_happy_breakdown_e_ops_callable(self, psi0, hz, Jx, Jz):

        krylov_dim = 12
        N = 4
        dim = 2**N
        H = h_ising_transverse(N, hx=0, hz=hz, Jx=Jx, Jy=0, Jz=Jz)
        _dims = H.dims
        _dims2 = [1] * N

        tlist = np.linspace(0, 20, 200)
        self.check_e_ops_callable(
            H,
            psi0,
            tlist,
            dim,
            krylov_dim=krylov_dim,
            square_hamiltonian=False,
        )

    @pytest.mark.parametrize(
        "psi0,hz,Jx,Jz",
        [
            # eigenstate
            (ket([1, 0, 0, 0]), 0.5, 0, 1),
            # state in the magnetization subspace of the XXZ model
            (ket([0, 0, 1, 0]), 1.0, 1, 0),
        ],
        ids=["happy_breakdown_eigenstate", "happy_breakdown_symmetry"],
    )
    def test_12_happy_breakdown_e_ops_list_single_callable(self,
                                                           psi0,
                                                           hz,
                                                           Jx,
                                                           Jz):

        krylov_dim = 12
        N = 4
        dim = 2**N
        H = h_ising_transverse(N, hx=0, hz=hz, Jx=Jx, Jy=0, Jz=Jz)
        _dims = H.dims
        _dims2 = [1] * N

        tlist = np.linspace(0, 20, 200)
        self.check_e_ops_list_single_callable(
            H,
            psi0,
            tlist,
            dim,
            krylov_dim=krylov_dim,
            square_hamiltonian=False,
        )
