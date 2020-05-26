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

import pytest
import numpy as np
import qutip


class TestVonNeumannEntropy:
    @pytest.mark.parametrize("p", np.linspace(0, 1, 17))
    def test_binary(self, p):
        dm = qutip.qdiags([p, 1 - p], 0)
        expected = 0 if p in [0, 1] else p*np.log2(p) + (1-p)*np.log2(1-p)
        assert abs(-qutip.entropy_vn(dm, 2) - expected) < 1e-12

    @pytest.mark.repeat(10)
    def test_pure_state(self):
        assert abs(qutip.entropy_vn(qutip.rand_ket(10))) < 1e-12


class TestLinearEntropy:
    @pytest.mark.repeat(10)
    def test_less_than_von_neumann(self):
        dm = qutip.rand_dm(10)
        assert qutip.entropy_linear(dm) <= qutip.entropy_vn(dm)

    @pytest.mark.repeat(10)
    def test_pure_state(self):
        assert abs(qutip.entropy_linear(qutip.rand_ket(10))) < 1e-12


class TestConcurrence:
    @pytest.mark.parametrize("dm", [
        pytest.param(qutip.bell_state(x).proj(), id='bell'+x)
        for x in ['00', '01', '10', '11']
    ])
    def test_maximally_entangled(self, dm):
        assert abs(qutip.concurrence(dm) - 1) < 1e-12

    @pytest.mark.repeat(10)
    def test_nonzero(self):
        dm = qutip.rand_dm(4, dims=[[2, 2], [2, 2]])
        assert qutip.concurrence(dm) >= 0


@pytest.mark.repeat(10)
class TestMutualInformation:
    def test_pure_state_additive(self):
        # Verify mutual information = S(A) + S(B) for pure states.
        dm = qutip.rand_dm(25, dims=[[5, 5], [5, 5]], pure=True)
        expect = (qutip.entropy_vn(dm.ptrace(0))
                  + qutip.entropy_vn(dm.ptrace(1)))
        assert abs(qutip.entropy_mutual(dm, [0], [1]) - expect) < 1e-13

    def test_component_selection(self):
        dm = qutip.rand_dm(8, dims=[[2, 2, 2], [2, 2, 2]], pure=True)
        expect = (qutip.entropy_vn(dm.ptrace([0, 2]))
                  + qutip.entropy_vn(dm.ptrace(1)))
        assert abs(qutip.entropy_mutual(dm, [0, 2], [1]) - expect) < 1e-13


@pytest.mark.repeat(20)
class TestConditionalEntropy:
    def test_inequality_3_qubits(self):
        # S(A | B,C) <= S(A|B)
        full = qutip.rand_dm(8, dims=[[2]*3]*2, pure=True)
        ab = full.ptrace([0, 1])
        assert (qutip.entropy_conditional(full, [1, 2])
                <= qutip.entropy_conditional(ab, 1))

    def test_triangle_inequality_4_qubits(self):
        # S(A,B | C,D) <= S(A|C) + S(B|D)
        full = qutip.rand_dm(16, dims=[[2]*4]*2, pure=True)
        ac, bd = full.ptrace([0, 2]), full.ptrace([1, 3])
        assert (qutip.entropy_conditional(full, [2, 3])
                <= (qutip.entropy_conditional(ac, 1)
                    + qutip.entropy_conditional(bd, 1)))


_alpha = 2*np.pi * np.random.rand()


@pytest.mark.parametrize(["gate", "expected"], [
    pytest.param(qutip.qip.operations.gates.cnot(), 2/9, id="CNOT"),
    pytest.param(qutip.qip.operations.gates.iswap(), 2/9, id="ISWAP"),
    pytest.param(qutip.qip.operations.gates.berkeley(), 2/9, id="Berkeley"),
    pytest.param(qutip.qip.operations.gates.swap(), 0, id="SWAP"),
    pytest.param(qutip.qip.operations.gates.sqrtswap(), 1/6, id="sqrt(SWAP)"),
    pytest.param(qutip.qip.operations.gates.swapalpha(_alpha),
                 np.sin(np.pi*_alpha)**2 / 6, id="SWAP(alpha)"),
])
def test_entangling_power(gate, expected):
    assert abs(qutip.entangling_power(gate) - expected) < 1e-12
