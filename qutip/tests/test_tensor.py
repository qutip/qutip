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

from numpy.testing import assert_equal, assert_, run_module_suite

from qutip.operators import identity
from qutip.superop_reps import to_super
from qutip.tensor import (
    tensor_contract, flatten, enumerate_flat, deep_remove, unflatten
)


def test_flatten():
    l = [[[0], 1], 2]
    assert_equal(flatten(l), [0, 1, 2])


def test_enumerate_flat():
    l = [[[10], [20, 30]], 40]
    labels = enumerate_flat(l)
    assert_equal(labels, [[[0], [1, 2]], 3])


def test_deep_remove():
    l = [[[0], 1], 2]
    l = deep_remove(l, 1)
    assert_equal(l, [[[0]], 2])

    # Harder case...
    l = [[[[0, 1, 2]], [3, 4], [5], [6, 7]]]
    l = deep_remove(l, 0, 5)
    assert l == [[[[1, 2]], [3, 4], [], [6, 7]]]


def test_unflatten():
    l = [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]]
    labels = enumerate_flat(l)
    assert unflatten(flatten(l), labels) == l


def test_tensor_contract():
    qobj = identity([2, 3, 4])
    ans = 3 * identity([2, 4])

    assert_(ans == tensor_contract(qobj, (1, 4)))

    # Now try for superoperators.
    # For now, we just ensure the dims are correct.
    sqobj = to_super(qobj)
    correct_dims = [[[2, 4], [2, 4]], [[2, 4], [2, 4]]]
    assert_equal(correct_dims, tensor_contract(sqobj, (1, 4), (7, 10)).dims)

if __name__ == "__main__":
    run_module_suite()