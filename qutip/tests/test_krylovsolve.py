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


@pytest.fixture(params=[
    pytest.param(100, id="small dim"),
    pytest.param(500, id="intermediate dim"),
    pytest.param(1000, id="large dim"),
])
def dimensions(request):
    return request.param


@pytest.fixture(params=[
    pytest.param(np.linspace(0, 5, 200), id="normal tlist"),
    pytest.param([2], id="single element tlist"),
    pytest.param([], id="empty tlist"),
])
def tlists(request):
    return request.param


@pytest.fixture(params=[
    pytest.param((ket([1, 0, 0, 0]), 0.5, 0, 1), id="eigenstate"),
    pytest.param((ket([0, 0, 1, 0]), 1.0, 1, 0),
                 id="magnetization subspace state XXZ model"),
])
def happy_breakdown_parameters(request):
    return request.param


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


def create_test_e_ops(e_ops_type, dim):
    """Creates e_ops used for testing give H and dim."""
    if e_ops_type == "[c]":
        e_ops = [lambda t, psi: expect(num(dim), psi)]
    if e_ops_type == "[q]":
        e_ops = [jmat((dim - 1)/2, "x")]
    if e_ops_type == "[c, c]":
        e_ops = [lambda t, psi: expect(num(dim), psi),
                 lambda t, psi: expect(num(dim)/2, psi)]
    if e_ops_type == "[c, q]":
        e_ops = [lambda t, psi: expect(num(dim), psi),
                 jmat((dim - 1) / 2.0, "x")]
    if e_ops_type == "[q, q]":
        e_ops = [jmat((dim - 1) / 2.0, "x"),
                 jmat((dim - 1) / 2.0, "y"),
                 jmat((dim - 1) / 2.0, "z")]
    return e_ops


def err_psi(psi_a, psi_b):
    """Error between to kets."""
    err = 1 - np.abs(psi_a.overlap(psi_b)) ** 2
    return err


def expect_value(e_ops, res_1, tlist):
    """Calculates expectation values of results object."""
    if isinstance(e_ops, types.FunctionType):
        expect_values = [
            e_ops(t, state) for (t, state) in zip(tlist, res_1.states)
        ]
    else:
        expect_values = [expect(e_ops, state) for state in res_1.states]
    return expect_values


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
        "simple check of errors between two outputs states"

        assert_err_states_less_than_tol(res_1=output_sparse,
                                        res_2=output_dense,
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

            output_exact.expect = expect_value(e_ops[i], output_exact, tlist)

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
        "Check input possibilities when e_ops=None"

        krylov_outputs = krylovsolve(H, psi0, tlist, krylov_dim, e_ops=None)

        try:
            sesolve_outputs = sesolve(H, psi0, tlist, e_ops=None)
        except IndexError:  # if tlist=[], sesolve breaks but krylov doesn't
            pass

        if len(tlist) > 1:
            assert_err_states_less_than_tol(
                krylov_outputs, sesolve_outputs, tol)
        elif len(tlist) == 1:
            assert krylov_outputs.states == sesolve_outputs.states
        else:
            assert krylov_outputs.states == []

    def test_05_check_e_ops_none(self, dimensions, tlists):
        "krylovsolve: check e_ops=None inputs with random H different tlists."
        psi0 = rand_ket(dimensions)
        H = rand_herm(dimensions, density=0.5)
        self.check_e_ops_none(H, psi0, tlists, dimensions)

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
        "Check input possibilities when e_ops=callable"

        def e_ops(t, psi): return expect(num(dim), psi)

        if not square_hamiltonian:
            H.dims = [[H.shape[0]], [H.shape[0]]]
            psi0.dims = [[H.shape[0]], [1]]

        krylov_outputs = krylovsolve(H, psi0, tlist, krylov_dim, e_ops=e_ops)
        exact_output = exactsolve(H, psi0, tlist)
        exact_output.expect = expect_value(e_ops=e_ops,
                                           res_1=exact_output,
                                           tlist=tlist)

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

    def test_06_check_e_ops_callable(self, dimensions, tlists):
        "krylovsolve: check e_ops=call inputs with random H different tlists."
        psi0 = rand_ket(dimensions)
        H = rand_herm(dimensions, density=0.5)
        self.check_e_ops_callable(H, psi0, tlists, dimensions)

    def check_e_ops_list_single_operator(
        self,
        e_ops,
        H,
        psi0,
        tlist,
        dim,
        krylov_dim=35,
        tol=1e-5,
        square_hamiltonian=True,
    ):
        "Check input possibilities when e_ops=[callable | qobj]"

        if not square_hamiltonian:
            H.dims = [[H.shape[0]], [H.shape[0]]]
            psi0.dims = [[H.shape[0]], [1]]

        krylov_outputs = krylovsolve(H, psi0, tlist, krylov_dim, e_ops=e_ops)
        exact_output = exactsolve(H, psi0, tlist)
        exact_output.expect = expect_value(e_ops[0], exact_output, tlist)

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

    @pytest.mark.parametrize("e_ops_type", [("[c]"), ("[q]")], ids=["[c]", "[q]"])
    def test_07_check_e_ops_list_single_callable(self, e_ops_type, dimensions, tlists):
        "krylovsolve: check e_ops=[call | qobj] random H different tlists."
        psi0 = rand_ket(dimensions)
        H = rand_herm(dimensions, density=0.5)
        e_ops = create_test_e_ops(e_ops_type, dimensions)
        self.check_e_ops_list_single_operator(
            e_ops, H, psi0, tlists, dimensions)

    def check_e_ops_mixed_list(
        self, e_ops, H, psi0, tlist, dim, krylov_dim=35, tol=1e-5,
    ):
        "Check input possibilities when e_ops=[call | qobj] and len(e_ops) > 1"

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
                exact_output.expect = expect_value(
                    e_ops[idx], exact_output, tlist)
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

    @pytest.mark.parametrize("e_ops_type",
                             [("[c, c]"), ("[c, q]"), ("[q, q]")],
                             ids=["[c, c]", "[c, q]", "[q, q]"])
    def test_08_check_e_ops_mixed_list(self, e_ops_type, dimensions, tlists):
        "krylovsolve: check e_ops=[call | qobj] with len(e_ops)>1 for"
        "random H different tlists."
        psi0 = rand_ket(dimensions)
        H = rand_herm(dimensions, density=0.5)
        e_ops = create_test_e_ops(e_ops_type, dimensions)
        self.check_e_ops_mixed_list(e_ops, H, psi0, tlists, dimensions)

    def test_9_happy_breakdown_simple(self, happy_breakdown_parameters):
        "krylovsolve: check simple at happy breakdowns"
        psi0, hz, Jx, Jz = happy_breakdown_parameters
        krylov_dim = 12
        N = 4
        dim = 2**N
        H = h_ising_transverse(N, hx=0, hz=hz, Jx=Jx, Jy=0, Jz=Jz)
        tlist = np.linspace(0, 20, 200)
        self.simple_check_states_e_ops(
            H, psi0, tlist, krylov_dim=krylov_dim, square_hamiltonian=False
        )

    def test_10_happy_breakdown_e_ops_none(self, happy_breakdown_parameters):
        "krylovsolve: check e_ops=None at happy breakdowns"
        psi0, hz, Jx, Jz = happy_breakdown_parameters
        krylov_dim = 12
        N = 4
        dim = 2**N
        H = h_ising_transverse(N, hx=0, hz=hz, Jx=Jx, Jy=0, Jz=Jz)
        tlist = np.linspace(0, 20, 200)
        self.check_e_ops_none(
            H, psi0, tlist, dim, krylov_dim=krylov_dim
        )

    def test_11_happy_breakdown_e_ops_callable(self, happy_breakdown_parameters):
        "krylovsolve: check e_ops=callable at happy breakdowns"
        psi0, hz, Jx, Jz = happy_breakdown_parameters
        krylov_dim = 12
        N = 4
        dim = 2**N
        H = h_ising_transverse(N, hx=0, hz=hz, Jx=Jx, Jy=0, Jz=Jz)
        def e_ops(t, psi): return expect(num(dim), psi)
        tlist = np.linspace(0, 20, 200)

        self.check_e_ops_callable(
            H,
            psi0,
            tlist,
            dim,
            krylov_dim=krylov_dim,
            square_hamiltonian=False,
        )

    def test_12_happy_breakdown_e_ops_list_single_callable(self, happy_breakdown_parameters):
        "krylovsolve: check e_ops=[callable] at happy breakdowns"
        psi0, hz, Jx, Jz = happy_breakdown_parameters
        krylov_dim = 12
        N = 4
        dim = 2**N
        H = h_ising_transverse(N, hx=0, hz=hz, Jx=Jx, Jy=0, Jz=Jz)
        e_ops = [lambda t, psi: expect(num(dim), psi)]

        tlist = np.linspace(0, 20, 200)
        self.check_e_ops_list_single_operator(
            e_ops,
            H,
            psi0,
            tlist,
            dim,
            krylov_dim=krylov_dim,
            square_hamiltonian=False,
        )


def test_13_krylovsolve_bad_krylov_dim(dim=15, krylov_dim=20):
    """Check errors from bad krylov dimension inputs."""
    H = rand_herm(dim)
    psi0 = basis(dim, 0)
    tlist = np.linspace(0, 1, 100)
    with pytest.raises(ValueError) as exc:
        krylovsolve(H, psi0, tlist, krylov_dim)
