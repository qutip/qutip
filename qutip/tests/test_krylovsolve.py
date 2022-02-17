from scipy.linalg import expm
import numpy as np

np.random.seed(0)
import pytest
import math
import os

from qutip import rand_herm, rand_ket, rand_unitary_haar, num, destroy, create
from qutip import sigmax, sigmay, sigmaz, qeye, jmat
from qutip import tensor, Qobj, basis, expect
from qutip.solver import Options, Result
from qutip import krylovsolve, sesolve
from qutip.sparse import eigh


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

    si = qeye(2)
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()

    sx_list = []
    sy_list = []
    sz_list = []

    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)

        op_list[n] = sx
        sx_list.append(tensor(op_list))

        op_list[n] = sy
        sy_list.append(tensor(op_list))

        op_list[n] = sz
        sz_list.append(tensor(op_list))

    # construct the hamiltonian
    H = 0

    # energy splitting terms
    for n in range(N):
        H += hz[n] * sz_list[n]

    for n in range(N):
        H += hx[n] * sx_list[n]

    # interaction terms
    for n in range(N - 1):
        H += -Jx[n] * sx_list[n] * sx_list[n + 1]
        H += -Jy[n] * sy_list[n] * sy_list[n + 1]
        H += -Jz[n] * sz_list[n] * sz_list[n + 1]

    _dims = H.dims
    H = Qobj(H, dims=_dims)
    return H


def err_psi(psi_a, psi_b):
    err = 1 - np.abs(psi_a.overlap(psi_b)) ** 2
    return err


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

    def check_evolution_states(
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

        # check errors between krylov and sesolve for states
        errors_states = [
            err_psi(psi_k, psi_ss)
            for (psi_k, psi_ss) in zip(output.states, output_ss.states)
        ]

        for err in errors_states:
            assert (
                err <= tol
            ), f"error between sesolve states and krylov states its {err}."

        # check errors between krylov and exactsolve for states
        errors_states = [
            err_psi(psi_k, psi_ex)
            for (psi_k, psi_ex) in zip(output.states, output_exact.states)
        ]

        for err in errors_states:
            assert (
                err <= tol
            ), f"error between sesolve states and krylov states its {err}."

        # for the operators, I'm forced to test against exactsolve and not
        # sesolve, because sesolve cumulates too much error (see test at
        # the end of the file)

        for i in range(len(e_ops)):
            output_exact.expect = [
                expect(e_ops[i], state) for state in output_exact.states
            ]

            errors_operators = [
                np.abs(op_k - op_ex)
                for (op_k, op_ex) in zip(output.expect[i], output_exact.expect)
            ]

            for err in errors_operators:
                assert (
                    err <= tol
                ), f"error between exactsolve and krylov operators is {err}."

    def test_01_states_with_constant_H_random(self):
        "krylovsolve: states with const H random"
        dim = 512
        psi0 = rand_ket(dim)
        H = rand_herm(dim)
        tlist = np.linspace(0, 10, 200)

        self.check_evolution_states(H, psi0, tlist)

    def test_02_states_with_constant_H_ising_transverse(self):
        "krylovsolve: states with const H Ising Transverse Field"
        N = 8
        dim = 2**N
        H = h_ising_transverse(N, hx=0.1, hz=0.5, Jx=1, Jy=0, Jz=1)
        _dims = H.dims
        _dims2 = [d - 1 for d in _dims[0]]
        psi0 = rand_ket(dim, dims=[_dims[0], _dims2])
        tlist = np.linspace(0, 20, 200)

        self.check_evolution_states(H, psi0, tlist, square_hamiltonian=False)

    def test_03_states_with_constant_H_sho(self):
        "krylovsolve: states with const H SHO"
        dim = 100
        psi0 = rand_ket(dim)
        H = h_sho(dim)
        tlist = np.linspace(0, 20, 200)

        self.check_evolution_states(H, psi0, tlist)

    def check_sparse_vs_dense(self, output_sparse, output_dense, tol=1e-5):
        "krylovsolve: comparing sparse vs non sparse"

        err_outputs = [
            err_psi(psi_sparse, psi_dense)
            for (psi_sparse, psi_dense) in zip(
                output_sparse.states, output_dense.states
            )
        ]
        for err in err_outputs:
            assert (
                err < tol
            ), f"difference between sparse and dense methods with err={err}"

    def test_04_check_sparse_vs_non_sparse_with_sparse_H(
        self, dim=512, krylov_dim=20
    ):
        "krylovsolve: comparing sparse vs non sparse with non dense H"
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

    def test_05_check_sparse_vs_non_sparse_with_dense_H(
        self, dim=512, krylov_dim=20
    ):
        "krylovsolve: comparing sparse vs non sparse with dense H"

        psi0 = rand_ket(dim)
        H_dense = rand_herm(dim, density=0.9)
        tlist = np.linspace(0, 10, 200)

        output_sparse = krylovsolve(
            H_dense, psi0, tlist, krylov_dim, sparse=True
        )
        output_dense = krylovsolve(
            H_dense, psi0, tlist, krylov_dim, sparse=False
        )

        self.check_sparse_vs_dense(output_sparse, output_dense)

    def check_e_ops_input_types_None(
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

    def test_06_check_e_ops_input_types_None(self):
        "krylovsolve: testing inputs when e_ops=None and tlist is common."
        dim = 128
        psi0 = rand_ket(dim)
        H = rand_herm(dim, density=0.5)
        tlist = np.linspace(0, 10, 200)

        self.check_e_ops_input_types_None(H, psi0, tlist, dim)

    def test_07_check_e_ops_input_types_None_single_element_tlist(self):
        "krylovsolve: test inputs when e_ops=None and len(tlist)=1."
        dim = 128
        psi0 = rand_ket(dim)
        H = rand_herm(dim, density=0.5)
        tlist = [2]

        self.check_e_ops_input_types_None(H, psi0, tlist, dim)

    def test_08_check_e_ops_input_types_None_empty_tlist(self):
        "krylovsolve: testing inputs when e_ops=None and tlist is empty."
        dim = 128
        psi0 = rand_ket(dim)
        H = rand_herm(dim, density=0.5)
        tlist = []

        self.check_e_ops_input_types_None(H, psi0, tlist, dim)

    def check_e_ops_input_types_callable(
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

        e_ops = lambda t, psi: expect(num(dim), psi)

        # this will break if we have spins, but we want to test certain 
        # things as well thus we reformat it:
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
            err_outputs = [
                np.abs(e_k - e_ex)
                for (e_k, e_ex) in zip(
                    krylov_outputs.expect, exact_output.expect
                )
            ]
            assert len(krylov_outputs.expect) == len(
                sesolve_outputs.expect
            ), "shape of outputs between krylov and sesolve differs"
            for err in err_outputs:
                assert (
                    err <= tol
                ), f"difference between krylov and exact e_ops evolution is \
                    greater than tol={tol} with err={err}"
                    
        elif len(tlist) == 1:
            assert krylov_outputs.expect == sesolve_outputs.expect
        else:
            assert krylov_outputs.states == []

    def test_09_check_e_ops_input_types_callable(self):
        "krylovsolve: check e_ops inputs with const H Ising Transverse Field"
        dim = 128
        psi0 = rand_ket(dim)
        H = rand_herm(dim, density=0.5)
        tlist = np.linspace(0, 5, 200)

        self.check_e_ops_input_types_callable(H, psi0, tlist, dim)

    def test_10_check_e_ops_input_types_callable_single_element_tlist(self):
        "krylovsolve: check e_ops inputs with const H Ising Transverse Field"
        dim = 128
        psi0 = rand_ket(dim)
        H = rand_herm(dim, density=0.5)
        tlist = [2]

        self.check_e_ops_input_types_callable(H, psi0, tlist, dim)

    def test_11_check_e_ops_input_types_callable_empty_tlist(self):
        "krylovsolve: check e_ops inputs with const H Ising Transverse Field"
        dim = 128
        psi0 = rand_ket(dim)
        H = rand_herm(dim, density=0.5)
        tlist = []

        self.check_e_ops_input_types_callable(H, psi0, tlist, dim)

    def check_e_ops_input_types_callable_single_list(
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

            err_outputs = [
                np.abs(e_k - e_ex)
                for (e_k, e_ex) in zip(
                    krylov_outputs.expect[0], exact_output.expect
                )
            ]
            for err in err_outputs:
                assert (
                    err <= tol
                ), f"difference between krylov and exact e_ops evolution is \
                    greater than tol={tol} with err={err}"
        elif len(tlist) == 1:
            assert (
                np.abs(krylov_outputs.expect[0] - sesolve_outputs.expect[0])
                <= tol
            ), "expect outputs from krylovsolve and sesolve are not equal"
        else:
            assert krylov_outputs.states == []

    def test_12_check_e_ops_input_types_callable_single_list(self):
        "krylovsolve: check e_ops inputs with const H Ising Transverse Field"
        dim = 128
        psi0 = rand_ket(dim)
        H = rand_herm(dim, density=0.5)
        tlist = np.linspace(0, 5, 200)

        self.check_e_ops_input_types_callable_single_list(H, psi0, tlist, dim)

    def test_13_ck_e_ops_input_types_callable_single_list_single_element_tlist(
        self,
    ):
        "krylovsolve: check e_ops inputs with const H Ising Transverse Field"
        dim = 128
        psi0 = rand_ket(dim)
        H = rand_herm(dim, density=0.5)
        tlist = [2]

        self.check_e_ops_input_types_callable_single_list(H, psi0, tlist, dim)

    def test_14_check_e_ops_input_types_callable_single_element_empty_tlist(
        self,
    ):
        "krylovsolve: check e_ops inputs with const H Ising Transverse Field"
        dim = 128
        psi0 = rand_ket(dim)
        H = rand_herm(dim, density=0.5)
        tlist = []

        self.check_e_ops_input_types_callable_single_list(H, psi0, tlist, dim)

    def check_e_ops_input_types_callable_mixed_list(
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

                err_outputs = [
                    np.abs(e_k - e_ex)
                    for (e_k, e_ex) in zip(k_expect, exact_output.expect)
                ]
                for err in err_outputs:
                    assert (
                        err <= tol
                    ), f"difference between krylov and exact e_ops evolution \
                        is greater than tol={tol} with err={err}"
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

    def test_15_check_e_ops_input_types_callable_mixed_list(self):
        "krylovsolve: check e_ops inputs with const H Ising Transverse Field"
        dim = 128
        psi0 = rand_ket(dim)
        H = rand_herm(dim, density=0.5)
        tlist = np.linspace(0, 5, 200)

        self.check_e_ops_input_types_callable_mixed_list(H, psi0, tlist, dim)

    def test_16_chk_e_ops_input_types_callable_mixed_list_single_element_tlist(
        self,
    ):
        "krylovsolve: check e_ops inputs with const H Ising Transverse Field"
        dim = 128
        psi0 = rand_ket(dim)
        H = rand_herm(dim, density=0.5)
        tlist = [2]

        self.check_e_ops_input_types_callable_mixed_list(H, psi0, tlist, dim)

    def test_17_check_e_ops_input_types_callable_mixed_element_empty_tlist(
        self,
    ):
        "krylovsolve: check e_ops inputs with const H Ising Transverse Field"
        dim = 128
        psi0 = rand_ket(dim)
        H = rand_herm(dim, density=0.5)
        tlist = []

        self.check_e_ops_input_types_callable_mixed_list(H, psi0, tlist, dim)

    def test_18_check_happy_breakdown_eigenstate(self):
        N = 8
        dim = 2**N
        H = h_ising_transverse(N, hx=0.0, hz=0.5, Jx=0, Jy=0, Jz=1)
        _dims = H.dims
        _dims2 = [d - 1 for d in _dims[0]]

        # create a ket with all spins down except the first one,
        # which is an eigenstate of the Hamiltonian
        psi0 = None
        for i in range(N):
            if not psi0:
                psi0 = basis(2, 1)
            else:
                psi0 = tensor(psi0, basis(2, 0))

        # psi0 = rand_ket(dim, dims=[_dims[0], _dims2])
        tlist = np.linspace(0, 20, 200)
        self.check_evolution_states(H, psi0, tlist, square_hamiltonian=False)
        self.check_e_ops_input_types_None(H, psi0, tlist, dim)
        self.check_e_ops_input_types_callable(
            H, psi0, tlist, dim, square_hamiltonian=False
        )
        self.check_e_ops_input_types_callable_single_list(
            H, psi0, tlist, dim, square_hamiltonian=False
        )

    def test_19_check_happy_breakdown_no_eigenstate(self):

        krylov_dim = 12
        N = 4
        dim = 2**N
        H = h_ising_transverse(N, hx=0.0, hz=1.0, Jx=1, Jy=0, Jz=0)
        _dims = H.dims
        _dims2 = [d - 1 for d in _dims[0]]

        # create a ket (0,0,1,0), which lies on the symmetry subspace
        # of magnetization. This guarantees a happy breakdown to occur, and it
        # is not an eigenstate.
        psi0 = basis(2, 0)
        for i in range(1, N):
            if i == 2:
                psi0 = tensor(psi0, basis(2, 1))
            else:
                psi0 = tensor(psi0, basis(2, 0))

        tlist = np.linspace(0, 20, 200)
        self.check_evolution_states(
            H, psi0, tlist, krylov_dim=krylov_dim, square_hamiltonian=False
        )
        self.check_e_ops_input_types_None(
            H, psi0, tlist, dim, krylov_dim=krylov_dim
        )
        self.check_e_ops_input_types_callable(
            H,
            psi0,
            tlist,
            dim,
            krylov_dim=krylov_dim,
            square_hamiltonian=False,
        )
        self.check_e_ops_input_types_callable_single_list(
            H,
            psi0,
            tlist,
            dim,
            krylov_dim=krylov_dim,
            square_hamiltonian=False,
        )

    def test_20_check_err_magnitude_krylov_vs_sesolve(self, tol=1e-7):

        dim = 512
        H = rand_herm(dim)
        psi0 = rand_ket(dim)
        krylov_dim = 20
        tlist = np.linspace(0, 20, 100)

        output_k = krylovsolve(H, psi0, tlist, krylov_dim)
        output_ss = sesolve(H, psi0, tlist)
        output_ex = exactsolve(H, psi0, tlist)

        err_k_ex = [
            err_psi(psi_k, psi_ex)
            for (psi_k, psi_ex) in zip(output_k.states, output_ex.states)
        ]
        err_ss_ex = [
            err_psi(psi_ss, psi_ex)
            for (psi_ss, psi_ex) in zip(output_ss.states, output_ex.states)
        ]

        # check that krylov has better accuracy in this case
        for e_k_ex, e_ss_ex in zip(err_k_ex, err_ss_ex):
            assert (
                e_k_ex <= e_ss_ex
            ), "krylov failed to have better accuracy than sesolve"

