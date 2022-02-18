from scipy.linalg import expm
import numpy as np

np.random.seed(0)
import pytest
import types
import math
import os

from qutip import rand_herm, rand_ket, rand_unitary_haar, num, destroy, create
from qutip.qip.operations import x_gate, y_gate, z_gate
from qutip import sigmax, sigmay, sigmaz, qeye, jmat
from qutip import tensor, Qobj, basis, expect, ket
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

    def general_check(self, H, psi0, krylov_dim, tol):
        """generally checks krylovsolve for different possibilities of e_ops
        and t_lists"""
        dim = H.shape[0]

        options = Options(store_states=True)    

        # posibilities
        e_ops_possibilities = [None, 
                               
                               lambda t, psi: expect(num(dim), psi),
                            
                               [lambda t, psi: expect(num(dim), psi)],
                            
                               [lambda t, psi: expect(num(dim), psi), 
                                jmat((dim-1)/2., 'x')],
                            
                               [jmat((dim-1)/2., 'x'), 
                                jmat((dim-1)/2., 'y'), 
                                jmat((dim-1)/2., 'z')],
                            
                               jmat((dim-1)/2., 'x'),
                            
                               [jmat((dim-1)/2., 'x')]
                              ]

        t_list_possibilities = [np.linspace(0, 2, 100),
                                np.linspace(0, 1, 2),
                                [0]]

        k_res, se_res, ex_res = [], [], []


        for idx_e, e_ops in enumerate(e_ops_possibilities):
            for idx_t, tlist in enumerate(t_list_possibilities):
                print(f"idx_e={idx_e}, idx_t={idx_t}")
                k = krylovsolve(H, psi0, tlist=tlist, krylov_dim=krylov_dim, 
                                progress_bar=None, sparse=False, 
                                options=options, e_ops=e_ops)
                k_res.append(k)
                ex_res = exactsolve(H, psi0, tlist)
                try:
                    s = sesolve(H, psi0, tlist=tlist, progress_bar=None, 
                                options=options, e_ops=e_ops)
                    se_res.append(s)
                except UnboundLocalError:
                    print("here sesolve failed")
                    s = k_res[-1]
                    se_res.append(s)
                    pass

                assert len(k.states) == len(s.states), \
                    "states output has different length"
                assert len(k.expect) == len(s.expect), \
                    "expect output has different length"

                if (not isinstance(e_ops, list)) and (e_ops is not None):

                    if isinstance(e_ops, types.FunctionType):
                        # expect should be a list of len=len(tlist) with the 
                        # expectation values, thus:
                        ex_expect = [e_ops(t, state) for (t, state) \
                            in zip(tlist, ex_res.states)]
                        err_expect = [np.abs(k_ex - ex_ex) for (k_ex, ex_ex) \
                            in zip(k.expect, ex_expect)]

                    elif isinstance(e_ops, Qobj):
                        ex_expect = [expect(e_ops, state) for state \
                            in ex_res.states]
                        err_expect = [np.abs(k_ex - ex_ex) for (k_ex, ex_ex) \
                            in zip(k.expect[0], ex_expect)]

                        assert k.expect[0].shape == s.expect[0].shape, \
                            "different shape for krylov and sesolve outputs"

                    for err in err_expect:
                        assert err <= tol, \
                            f"err in expec values {err} is > than tol {tol}."

                elif e_ops is None:
                    err_states = [err_psi(psi_k, psi_ex) for (psi_k, psi_ex) \
                        in zip(k.states, ex_res.states)]
                    for err in err_states:
                        assert err <= tol,\
                        f"err in states {err} is > than tolerance {tol}."
                else:
                    for idx, op in enumerate(e_ops):
            
                        if isinstance(op, types.FunctionType):
                            # expect should be a list of len=len(tlist) with 
                            # the expectation values, thus:
                            ex_expect = [op(t, state) for (t, state) \
                                in zip(tlist, ex_res.states)]
                            err_expect = [np.abs(k_ex - ex_ex) for \
                                (k_ex, ex_ex) in zip(k.expect[idx], ex_expect)]

                        elif isinstance(op, Qobj):
                            ex_expect = [expect(op, state) for state \
                                in ex_res.states]
                            err_expect = [np.abs(k_ex - ex_ex) for \
                                (k_ex, ex_ex) in zip(k.expect[idx], ex_expect)]
                    
                        assert k.expect[idx].shape == s.expect[idx].shape, \
                        "different shape for krylov and sesolve outputs"
            
                        for err in err_expect:
                            assert err <= tol, \
                                f"err in expect values {err} is > than {tol}."


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

        # for the operators, test against exactsolve for accuracy
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
        self.general_check(H, psi0, krylov_dim=20, tol=1e-5)

    def test_02_states_with_constant_H_ising_transverse(self):
        "krylovsolve: states with const H Ising Transverse Field"
        N = 8
        dim = 2**N
        H = h_ising_transverse(N, hx=0.1, hz=0.5, Jx=1, Jy=0, Jz=1)
        _dims = H.dims
        _dims2 = [1] * N
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
        self.general_check(H, psi0, krylov_dim=20, tol=1e-5)

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

    @pytest.mark.parametrize("density,dim", [(0.1, 512), (0.9, 800)],
                             ids=["sparse H check", "dense H check"])
    def test_check_sparse_vs_non_sparse_with_density_H(
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
            assert (
                np.abs(krylov_outputs.expect[0] - sesolve_outputs.expect[0])
                <= 1e-7
            ), "krylov and sesolve outputs differ for len(tlist)=1"
        else:
            assert krylov_outputs.states == []

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

    @pytest.mark.parametrize(
        "dim,tlist", [(128, np.linspace(0, 5, 200)), (400, [2]), (560, [])]
        ,ids=["normal tlist", "single element tlist", "empty tlist"]
    )
    def test_04_check_e_ops_input_types_and_tlist_sizes(self, dim, tlist):
        "krylovsolve: check e_ops inputs with random H and different tlists."
        psi0 = rand_ket(dim)
        H = rand_herm(dim, density=0.5)

        self.check_e_ops_input_types_None(H, psi0, tlist, dim)
        self.check_e_ops_input_types_callable(H, psi0, tlist, dim)
        self.check_e_ops_input_types_callable_single_list(H, psi0, tlist, dim)
        self.check_e_ops_input_types_callable_mixed_list(H, psi0, tlist, dim)

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
    def test_05_check_happy_breakdown(self, psi0, hz, Jx, Jz):

        krylov_dim = 12
        N = 4
        dim = 2**N
        H = h_ising_transverse(N, hx=0, hz=hz, Jx=Jx, Jy=0, Jz=Jz)
        _dims = H.dims
        _dims2 = [1] * N

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
