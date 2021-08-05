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

from numpy.testing import assert_equal, assert_

from qutip import (
    Qobj, identity, sigmax, to_super, to_choi, rand_super_bcsz, basis,
    tensor_contract, tensor_swap, num, QobjEvo, destroy, tensor
)


def test_tensor_contract_ident():
    qobj = identity([2, 3, 4])
    ans = 3 * identity([2, 4])

    assert_(ans == tensor_contract(qobj, (1, 4)))

    # Now try for superoperators.
    # For now, we just ensure the dims are correct.
    sqobj = to_super(qobj)
    correct_dims = [[[2, 4], [2, 4]], [[2, 4], [2, 4]]]
    assert_equal(correct_dims, tensor_contract(sqobj, (1, 4), (7, 10)).dims)


def case_tensor_contract_other(left, right, pairs,
                               expected_dims, expected_data):
    dat = np.arange(np.prod(left) * np.prod(right))
    dat = dat.reshape((np.prod(left), np.prod(right)))

    qobj = Qobj(dat, dims=[left, right])
    cqobj = tensor_contract(qobj, *pairs)
    assert_equal(cqobj.dims, expected_dims)
    assert_equal(cqobj.full(), expected_data)


def test_tensor_contract_other():
    case_tensor_contract_other(
        [2, 3], [3, 4], [(1, 2)],
        [[2], [4]],
        np.einsum('abbc', np.arange(2 * 3 * 3 * 4).reshape((2, 3, 3, 4)))
    )

    case_tensor_contract_other(
        [2, 3], [4, 3], [(1, 3)],
        [[2], [4]],
        np.einsum('abcb', np.arange(2 * 3 * 3 * 4).reshape((2, 3, 4, 3)))
    )

    case_tensor_contract_other(
        [2, 3], [4, 3], [(1, 3)],
        [[2], [4]],
        np.einsum('abcb', np.arange(2 * 3 * 3 * 4).reshape((2, 3, 4, 3)))
    )

    # Make non-rectangular outputs in a column-/row-symmetric fashion.
    big_dat = np.arange(2 * 3 * 2 * 3 * 2 * 3 * 2 * 3).reshape((2, 3) * 4)

    case_tensor_contract_other(
        [[2, 3], [2, 3]], [[2, 3], [2, 3]], [(0, 2)],
        [[[3], [3]], [[2, 3], [2, 3]]],
        np.einsum('ibidwxyz', big_dat).reshape((3 * 3, 3 * 2 * 3 * 2)))

    case_tensor_contract_other(
        [[2, 3], [2, 3]], [[2, 3], [2, 3]],
        [(0, 2), (5, 7)], [[[3], [3]], [[2], [2]]],
        # We separate einsum into two calls due to a bug internal to
        # einsum.
        np.einsum('ibidwy', np.einsum('abcdwjyj', big_dat)).reshape(3*3, 2*2))

    # Now we try collapsing in a way that's sensitive to column- and row-
    # stacking conventions.
    big_dat = np.arange(2 * 2 * 3 * 3 * 2 * 3 * 2 * 3)
    big_dat = big_dat.reshape((3, 3, 2, 2, 2, 3, 2, 3))
    # Note that the order of [2, 2] and [3, 3] is swapped on the left!
    big_dims = [[[2, 2], [3, 3]], [[2, 3], [2, 3]]]

    # Let's try a weird tensor contraction; this will likely never come up in
    # practice, but it should serve as a good canary for more reasonable
    # contractions.
    case_tensor_contract_other(
        big_dims[0], big_dims[1], [(0, 4)],
        [[[2], [3, 3]], [[3], [2, 3]]],
        np.einsum('abidwxiz', big_dat).reshape((2 * 3 * 3, 3 * 2 * 3)))


def case_tensor_swap(qobj, pairs, expected_dims, expected_data=None):
    sqobj = tensor_swap(qobj, *pairs)

    assert_equal(sqobj.dims, expected_dims)
    assert_equal(sqobj.full(), expected_data.full())


def test_tensor_swap_other():
    dims = (2, 3, 4, 5, 7)
    for dim in dims:
        S = to_super(rand_super_bcsz(dim))
        # Swapping the inner indices on a superoperator should give a Choi
        # matrix.
        J = to_choi(S)
        case_tensor_swap(S, [(1, 2)], [[[dim], [dim]], [[dim], [dim]]], J)


def test_tensor_qobjevo():
    N = 5
    t = 1.5
    left = QobjEvo([num(N),[destroy(N),"t"]])
    right = QobjEvo([identity(2),[sigmax(),"t"]])
    assert tensor(left, right)(t) == tensor(left(t), right(t))
    assert tensor(left, sigmax())(t) == tensor(left(t), sigmax())
    assert tensor(num(N), right)(t) == tensor(num(N), right(t))


def test_tensor_qobjevo_non_square():
    N = 5
    t = 1.5
    left = QobjEvo([basis(N, 0), [basis(N, 1), "t"]])
    right = QobjEvo([basis(2, 0).dag(), [basis(2, 1).dag(), "t"]])
    assert tensor(left, right)(t) == tensor(left(t), right(t))


def test_tensor_qobjevo_multiple():
    N = 5
    t = 1.5
    left = QobjEvo([basis(N, 0), [basis(N, 1), "t"]])
    center = QobjEvo([basis(2, 0).dag(), [basis(2, 1).dag(), "t"]])
    right = QobjEvo([sigmax()])
    as_QobjEvo = tensor(left, center, right)(t)
    as_Qobj = tensor(left(t), center(t), right(t))
    assert as_QobjEvo == as_Qobj


def test_tensor_and():
    N = 5
    t = 1.5
    sx = sigmax()
    evo = QobjEvo([num(N),[destroy(N),"t"]])
    assert tensor([sx, sx, sx]) == sx & sx & sx
    assert tensor(evo, sx)(t) == (evo & sx)(t)
    assert tensor(sx, evo)(t) == (sx & evo)(t)
    assert tensor(evo, evo)(t) == (evo & evo)(t)
