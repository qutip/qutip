import numpy as np

np.random.seed(0)
import pytest
import math
import os

from qutip import rand_herm, rand_ket, rand_unitary_haar, num, destroy, create
from qutip import sigmax, sigmay, sigmaz, qeye, jmat
from qutip import tensor, Qobj, basis, expect
from qutip import krylovsolve, sesolve
from qutip.solver import Options
from scipy.linalg import expm
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


def magnitude(value):
    if value == 0:
        return 0
    return int(math.floor(math.log10(abs(value))))


class TestKrylovSolve:
    """
    A test class for the QuTiP Krylov Approximation method Solver.
    """

    def check_evolution_states(
        self,
        H,
        psi0,
        tlist,
        td_args={},
        tol=1e-5,
        tol2=1e-4,
        krylov_dim=25,
        square_hamiltonian=True,
    ):
        """
        Compare integrated evolution with analytical result
        If U0 is not None then operator evo is checked
        Otherwise state evo
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
        output_ss = sesolve(
            H, psi0, tlist, e_ops=e_ops, options=Options(atol=1e-8)
        )

        errors_states = [
            1 - np.abs(psi_ss.overlap(psi_k)) ** 2
            for (psi_k, psi_ss) in zip(output.states, output_ss.states)
        ]

        for err in errors_states:
            assert (
                err <= tol
            ), f"error between sesolve states and krylov states its {err}."

        for i in range(len(e_ops)):

            errors_operators = [
                np.abs(op_k - op_ss)
                for (op_k, op_ss) in zip(output.expect[i], output_ss.expect[i])
            ]

            # magnitud of the error is checked here, because the optimization 
            # was done for the states not the operators.
            for err in errors_operators:
                mag_err = magnitude(err)
                if mag_err == 0:
                    pass
                else:
                    assert magnitude(err) <= magnitude(
                        tol2
                    ), f"error between sesolve and krylov operators is {err}."

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
