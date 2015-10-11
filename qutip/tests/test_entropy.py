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

from __future__ import division

import numpy as np
from numpy.testing import assert_, assert_equal, run_module_suite

from qutip import (basis, ket2dm, cnot, entropy_vn, entropy_linear, rand_ket,
                   rand_dm, tensor, concurrence, entropy_mutual, ptrace,
                   entropy_conditional, entangling_power, iswap, swap,
                   berkeley, sqrtswap, swapalpha)


def test_EntropyVN():
    "Entropy: von-Neumann entropy"
    # verify that entropy_vn gives correct binary entropy
    a = np.linspace(0, 1, 20)
    for k in range(len(a)):
        # a*|0><0|
        x = a[k] * ket2dm(basis(2, 0))
        # (1-a)*|1><1|
        y = (1 - a[k]) * ket2dm(basis(2, 1))
        rho = x + y
        # Von-Neumann entropy (base 2) of rho
        out = entropy_vn(rho, 2)
        if k == 0 or k == 19:
            assert_equal(out, -0.0)
        else:
            assert_(abs(-out - a[k] * np.log2(a[k])
                        - (1. - a[k]) * np.log2((1. - a[k]))) < 1e-12)

    # test_ entropy_vn = 0 for pure state
    psi = rand_ket(10)
    assert_equal(abs(entropy_vn(psi)) <= 1e-13, True)


def test_EntropyLinear():
    "Entropy: Linear entropy"
    # test_ entropy_vn = 0 for pure state
    psi = rand_ket(10)
    assert_equal(abs(entropy_linear(psi)) <= 1e-13, True)

    # test_ linear entropy always less than or equal to VN entropy
    rhos = [rand_dm(6) for k in range(10)]
    for k in rhos:
        assert_equal(entropy_linear(k) <= entropy_vn(k), True)


def test_EntropyConcurrence():
    "Entropy: Concurrence"
    # check concurrence = 1 for maximal entangled (Bell) state
    bell = ket2dm(
        (tensor(basis(2), basis(2)) + tensor(basis(2, 1), basis(2, 1))).unit())
    assert_equal(abs(concurrence(bell) - 1.0) < 1e-15, True)

    # check all concurrence values >=0
    rhos = [rand_dm(4, dims=[[2, 2], [2, 2]]) for k in range(10)]
    for k in rhos:
        assert_equal(concurrence(k) >= 0, True)


def test_EntropyMutual():
    "Entropy: Mutual information"
    # verify mutual information = S(A)+S(B) for pure state
    rhos = [rand_dm(25, dims=[[5, 5], [5, 5]], pure=True) for k in range(10)]

    for r in rhos:
        assert_equal(abs(entropy_mutual(r, [0], [1]) - (
            entropy_vn(ptrace(r, 0)) + entropy_vn(ptrace(r, 1)))) < 1e-13,
            True)

    # check component selection
    rhos = [rand_dm(8, dims=[[2, 2, 2], [2, 2, 2]], pure=True)
            for k in range(10)]

    for r in rhos:
        assert_equal(abs(entropy_mutual(r, [0, 2], [1]) - (entropy_vn(
            ptrace(r, [0, 2])) + entropy_vn(ptrace(r, 1)))) < 1e-13, True)


def test_EntropyConditional():
    "Entropy: Conditional entropy"
    # test_ S(A,B|C,D)<=S(A|C)+S(B|D)
    rhos = [rand_dm(16, dims=[[2, 2, 2, 2], [2, 2, 2, 2]], pure=True)
            for k in range(20)]

    for ABCD in rhos:
        AC = ptrace(ABCD, [0, 2])
        BD = ptrace(ABCD, [1, 3])
        assert_equal(entropy_conditional(ABCD, [2, 3]) <= (
            entropy_conditional(AC, 1) + entropy_conditional(BD, 1)), True)

    # test_ S(A|B,C)<=S(A|B)
    rhos = [rand_dm(8, dims=[[2, 2, 2], [2, 2, 2]], pure=True)
            for k in range(20)]

    for ABC in rhos:
        AB = ptrace(ABC, [0, 1])
        assert_equal(entropy_conditional(
            ABC, [1, 2]) <= entropy_conditional(AB, 1), True)


def test_EntanglingPower():
    "Entropy: Entangling power"
    assert_(abs(entangling_power(cnot()) - 2/9) < 1e-12)
    assert_(abs(entangling_power(iswap()) - 2/9) < 1e-12)
    assert_(abs(entangling_power(berkeley()) - 2/9) < 1e-12)
    assert_(abs(entangling_power(sqrtswap()) - 1/6) < 1e-12)
    alpha = 2 * np.pi * np.random.rand()
    assert_(abs(entangling_power(swapalpha(alpha))
                - 1/6 * np.sin(np.pi * alpha) ** 2) < 1e-12)
    assert_(abs(entangling_power(swap()) - 0) < 1e-12)


if __name__ == "__main__":
    run_module_suite()
