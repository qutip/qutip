# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

import itertools
import numpy as np
from numpy.testing import assert_, assert_allclose, run_module_suite
from qutip.states import basis, ket2dm
from qutip.operators import identity, qeye, sigmax, sigmay, sigmaz
from qutip.qip import (rx, ry, rz, phasegate, qrot, cnot, swap, iswap,
                       sqrtswap, molmer_sorensen,
                       toffoli, fredkin, gate_expand_3toN, 
                       qubit_clifford_group, expand_oper)
from qutip.random_objects import rand_ket, rand_herm, rand_unitary
from qutip.tensor import tensor
from qutip.qobj import Qobj


class TestGates:
    """
    A test class for the QuTiP functions for generating quantum gates
    """

    def testSwapGate(self):
        """
        gates: swap gate
        """
        a, b = np.random.rand(), np.random.rand()
        psi1 = (a * basis(2, 0) + b * basis(2, 1)).unit()

        c, d = np.random.rand(), np.random.rand()
        psi2 = (c * basis(2, 0) + d * basis(2, 1)).unit()

        psi_in = tensor(psi1, psi2)
        psi_out = tensor(psi2, psi1)

        psi_res = swap() * psi_in
        assert_((psi_out - psi_res).norm() < 1e-12)

        psi_res = swap() * swap() * psi_in
        assert_((psi_in - psi_res).norm() < 1e-12)

    def test_clifford_group_len(self):
        assert_(len(list(qubit_clifford_group())) == 24)

    def _prop_identity(self, U, tol=1e-6):
        """
        Returns True if and only if U is proportional to the
        identity.
        """
        U0 = complex(U[0, 0])  # scipy 1.3 return 0 dims array.
        if U0 != 0:
            norm_U = U / U0
            return (qeye(U.dims[0]) - norm_U).norm() <= tol
        else:
            return False

    def case_is_clifford(self, U):
        paulis = (identity(2), sigmax(), sigmay(), sigmaz())

        for P in paulis:
            U_P = U * P * U.dag()

            out = (np.any(
                np.array([self._prop_identity(U_P * Q) for Q in paulis])
            ))
        return out

    def test_are_cliffords(self):
        for U in qubit_clifford_group():
            assert_(self.case_is_clifford(U))

    def testExpandGate1toN(self):
        """
        gates: expand 1 to N
        """
        N = 7

        for g in [rx, ry, rz, phasegate]:

            theta = np.random.rand() * 2 * 3.1415
            a, b = np.random.rand(), np.random.rand()
            psi1 = (a * basis(2, 0) + b * basis(2, 1)).unit()
            psi2 = g(theta) * psi1

            psi_rand_list = [rand_ket(2) for k in range(N)]

            for m in range(N):

                psi_in = tensor([psi1 if k == m else psi_rand_list[k]
                                 for k in range(N)])
                psi_out = tensor([psi2 if k == m else psi_rand_list[k]
                                  for k in range(N)])

                G = g(theta, N, m)
                psi_res = G * psi_in

                assert_((psi_out - psi_res).norm() < 1e-12)

    def testExpandGate2toNSwap(self):
        """
        gates: expand 2 to N (using swap)
        """

        a, b = np.random.rand(), np.random.rand()
        k1 = (a * basis(2, 0) + b * basis(2, 1)).unit()

        c, d = np.random.rand(), np.random.rand()
        k2 = (c * basis(2, 0) + d * basis(2, 1)).unit()

        N = 6
        kets = [rand_ket(2) for k in range(N)]

        for m in range(N):
            for n in set(range(N)) - {m}:

                psi_in = tensor([k1 if k == m else k2 if k == n else kets[k]
                                 for k in range(N)])
                psi_out = tensor([k2 if k == m else k1 if k == n else kets[k]
                                  for k in range(N)])

                targets = [m, n]
                G = swap(N, targets)

                psi_out = G * psi_in

                assert_((psi_out - G * psi_in).norm() < 1e-12)

    def testExpandGate2toN(self):
        """
        gates: expand 2 to N (using cnot, iswap, sqrtswap)
        """

        a, b = np.random.rand(), np.random.rand()
        k1 = (a * basis(2, 0) + b * basis(2, 1)).unit()

        c, d = np.random.rand(), np.random.rand()
        k2 = (c * basis(2, 0) + d * basis(2, 1)).unit()

        psi_ref_in = tensor(k1, k2)

        N = 6
        psi_rand_list = [rand_ket(2) for k in range(N)]

        for g in [cnot, iswap, sqrtswap]:

            psi_ref_out = g() * psi_ref_in
            rho_ref_out = ket2dm(psi_ref_out)

            for m in range(N):
                for n in set(range(N)) - {m}:

                    psi_list = [psi_rand_list[k] for k in range(N)]
                    psi_list[m] = k1
                    psi_list[n] = k2
                    psi_in = tensor(psi_list)

                    if g == cnot:
                        G = g(N, m, n)
                    else:
                        G = g(N, [m, n])

                    psi_out = G * psi_in

                    o1 = psi_out.overlap(psi_in)
                    o2 = psi_ref_out.overlap(psi_ref_in)
                    assert_(abs(o1 - o2) < 1e-12)

                    p = [0, 1] if m < n else [1, 0]
                    rho_out = psi_out.ptrace([m, n]).permute(p)

                    assert_((rho_ref_out - rho_out).norm() < 1e-12)

    def testExpandGate3toN_permutation(self):
        """
        gates: expand 3 to 3 with permuTation (using toffoli)
        """
        for _p in itertools.permutations([0, 1, 2]):
            controls, target = [_p[0], _p[1]], _p[2]

            controls = [1, 2]
            target = 0

            p = [1, 2, 3]
            p[controls[0]] = 0
            p[controls[1]] = 1
            p[target] = 2

            U = toffoli(N=3, controls=controls, target=target)

            ops = [basis(2, 0).dag(),  basis(2, 0).dag(), identity(2)]
            P = tensor(ops[p[0]], ops[p[1]], ops[p[2]])
            assert_(P * U * P.dag() == identity(2))

            ops = [basis(2, 1).dag(),  basis(2, 0).dag(), identity(2)]
            P = tensor(ops[p[0]], ops[p[1]], ops[p[2]])
            assert_(P * U * P.dag() == identity(2))

            ops = [basis(2, 0).dag(),  basis(2, 1).dag(), identity(2)]
            P = tensor(ops[p[0]], ops[p[1]], ops[p[2]])
            assert_(P * U * P.dag() == identity(2))

            ops = [basis(2, 1).dag(),  basis(2, 1).dag(), identity(2)]
            P = tensor(ops[p[0]], ops[p[1]], ops[p[2]])
            assert_(P * U * P.dag() == sigmax())

    def testExpandGate3toN(self):
        """
        gates: expand 3 to N (using toffoli, fredkin, and random 3 qubit gate)
        """

        a, b = np.random.rand(), np.random.rand()
        psi1 = (a * basis(2, 0) + b * basis(2, 1)).unit()

        c, d = np.random.rand(), np.random.rand()
        psi2 = (c * basis(2, 0) + d * basis(2, 1)).unit()

        e, f = np.random.rand(), np.random.rand()
        psi3 = (e * basis(2, 0) + f * basis(2, 1)).unit()

        N = 4
        psi_rand_list = [rand_ket(2) for k in range(N)]

        _rand_gate_U = tensor([rand_herm(2, density=1) for k in range(3)])

        def _rand_3qubit_gate(N=None, controls=None, k=None):
            if N is None:
                return _rand_gate_U
            else:
                return gate_expand_3toN(_rand_gate_U, N, controls, k)

        for g in [fredkin, toffoli, _rand_3qubit_gate]:

            psi_ref_in = tensor(psi1, psi2, psi3)
            psi_ref_out = g() * psi_ref_in

            for m in range(N):
                for n in set(range(N)) - {m}:
                    for k in set(range(N)) - {m, n}:

                        psi_list = [psi_rand_list[p] for p in range(N)]
                        psi_list[m] = psi1
                        psi_list[n] = psi2
                        psi_list[k] = psi3
                        psi_in = tensor(psi_list)

                        if g == fredkin:
                            targets = [n, k]
                            G = g(N, control=m, targets=targets)
                        else:
                            controls = [m, n]
                            G = g(N, controls, k)

                        psi_out = G * psi_in

                        o1 = psi_out.overlap(psi_in)
                        o2 = psi_ref_out.overlap(psi_ref_in)
                        assert_(abs(o1 - o2) < 1e-12)

    def test_expand_oper(self):
        """
        gate : expand qubits operator to a N qubits system.
        """
        # random single qubit gate test, integer as target
        r = rand_unitary(2)
        assert(expand_oper(r, 3, 0) == tensor([r, identity(2), identity(2)]))
        assert(expand_oper(r, 3, 1) == tensor([identity(2), r, identity(2)]))
        assert(expand_oper(r, 3, 2) == tensor([identity(2), identity(2), r]))

        # random 2qubits gate test, list as target
        r2 = rand_unitary(4)
        r2.dims = [[2, 2], [2, 2]]
        assert(expand_oper(r2, 3, [2, 1]) == tensor(
            [identity(2), r2.permute([1, 0])]))
        assert(expand_oper(r2, 3, [0, 1]) == tensor(
            [r2, identity(2)]))
        assert(expand_oper(r2, 3, [0, 2]) == tensor(
            [r2, identity(2)]).permute([0, 2, 1]))

        # cnot expantion, qubit 2 control qubit 0
        assert(expand_oper(cnot(), 3, [2, 0]) == Qobj([
            [1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1.],
            [0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0.]],
            dims=[[2, 2, 2], [2, 2, 2]]))

    def test_molmer_sorensen(self):
        """
        gate: test for the molmer_sorensen gate
        """
        assert_allclose(
            molmer_sorensen(np.pi, targets=[0, 1]),
            Qobj(-1j*np.array(
                 [[0, 0, 0, 1],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0],
                  [1, 0, 0, 0]]), dims=[[2, 2], [2, 2]]),
            atol=1e-15)
        assert_allclose(
            molmer_sorensen(2*np.pi, targets=[0, 1]),
            Qobj(-np.array(
                 [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]), dims=[[2, 2], [2, 2]]),
            atol=1e-15)
        assert_allclose(
            molmer_sorensen(np.pi/2, N=4, targets=[1, 2]),
            tensor([identity(2), molmer_sorensen(np.pi/2), identity(2)])
        )

    def test_qrot(self):
        """
        gate: test for the qubit rotation gate
        """
        assert_allclose(qrot(0, 0), identity(2))
        assert_allclose(
            qrot(np.pi, np.pi/2),
            Qobj([[0, -1], [1, 0]]),
            atol=1e-15)
        assert_allclose(
            qrot(np.pi/4., np.pi/3., N=2, target=1),
            tensor([identity(2), qrot(np.pi/4., np.pi/3.)])
        )

if __name__ == "__main__":
    run_module_suite()
