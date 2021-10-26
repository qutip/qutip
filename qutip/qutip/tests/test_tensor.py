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

import numpy as np

from numpy.testing import assert_equal, assert_, run_module_suite

from qutip.qobj import Qobj
from qutip.operators import identity, sigmax, sigmay
from qutip.superop_reps import to_super, to_choi
from qutip.random_objects import rand_super_bcsz
from qutip.tensor import (
    tensor_contract, tensor_swap
)

import warnings

def test_tensor_contract_ident():
    qobj = identity([2, 3, 4])
    ans = 3 * identity([2, 4])

    assert_(ans == tensor_contract(qobj, (1, 4)))

    # Now try for superoperators.
    # For now, we just ensure the dims are correct.
    sqobj = to_super(qobj)
    correct_dims = [[[2, 4], [2, 4]], [[2, 4], [2, 4]]]
    assert_equal(correct_dims, tensor_contract(sqobj, (1, 4), (7, 10)).dims)

def case_tensor_contract_other(left, right, pairs, expected_dims, expected_data=None):
    dat = np.arange(np.product(left) * np.product(right)).reshape((np.product(left), np.product(right)))
    
    qobj = Qobj(dat, dims=[left, right])
    cqobj = tensor_contract(qobj, *pairs)

    assert_equal(cqobj.dims, expected_dims)
    if expected_data is not None:
        assert_equal(cqobj.data.toarray(), expected_data)
    else:
        warnings.warn("tensor_contract test case without checking returned data.")

def test_tensor_contract_other():
    case_tensor_contract_other([2, 3], [3, 4], [(1, 2)], [[2], [4]],
        np.einsum('abbc', np.arange(2 * 3 * 3 * 4).reshape((2, 3, 3, 4))))

    case_tensor_contract_other([2, 3], [4, 3], [(1, 3)], [[2], [4]],
        np.einsum('abcb', np.arange(2 * 3 * 3 * 4).reshape((2, 3, 4, 3))))

    case_tensor_contract_other([2, 3], [4, 3], [(1, 3)], [[2], [4]],
        np.einsum('abcb', np.arange(2 * 3 * 3 * 4).reshape((2, 3, 4, 3))))

    # Make non-rectangular outputs in a column-/row-symmetric fashion.
    big_dat = np.arange(2 * 3 * 2 * 3 * 2 * 3 * 2 * 3).reshape((2, 3) * 4)

    case_tensor_contract_other([[2, 3], [2, 3]], [[2, 3], [2, 3]],
        [(0, 2)], [[[3], [3]], [[2, 3], [2, 3]]],
        np.einsum('ibidwxyz', big_dat).reshape((3 * 3, 3 * 2 * 3 * 2)))

    case_tensor_contract_other([[2, 3], [2, 3]], [[2, 3], [2, 3]],
        [(0, 2), (5, 7)], [[[3], [3]], [[2], [2]]],
        # We separate einsum into two calls due to a bug internal to
        # einsum.
        np.einsum('ibidwy', np.einsum('abcdwjyj', big_dat)).reshape((3 * 3, 2 * 2)))

    # Now we try collapsing in a way that's sensitive to column- and row-
    # stacking conventions.
    big_dat = np.arange(2 * 2 * 3 * 3 * 2 * 3 * 2 * 3).reshape((3, 3, 2, 2, 2, 3, 2, 3))
    # Note that the order of [2, 2] and [3, 3] is swapped on the left!
    big_dims = [[[2, 2], [3, 3]], [[2, 3], [2, 3]]]

    # Let's try a weird tensor contraction; this will likely never come up in practice,
    # but it should serve as a good canary for more reasonable contractions.
    case_tensor_contract_other(big_dims[0], big_dims[1],
        [(0, 4)], [[[2], [3, 3]], [[3], [2, 3]]],
        # We separate einsum into two calls due to a bug internal to
        # einsum.
        np.einsum('abidwxiz', big_dat).reshape((2 * 3 * 3, 3 * 2 * 3)))


def case_tensor_swap(qobj, pairs, expected_dims, expected_data=None):
    sqobj = tensor_swap(qobj, *pairs)

    assert_equal(sqobj.dims, expected_dims)
    if expected_data is not None:
        assert_equal(sqobj.data.toarray(), expected_data.data.toarray())
    else:
        warnings.warn("tensor_contract test case without checking returned data.")


def test_tensor_swap_other():
    dims = (2, 3, 4, 5, 7)

    for dim in dims:
        S = to_super(rand_super_bcsz(dim))

        # Swapping the inner indices on a superoperator should give a Choi matrix.
        J = to_choi(S)
        case_tensor_swap(S, [(1, 2)], [[[dim], [dim]], [[dim], [dim]]], J)


if __name__ == "__main__":
    run_module_suite()