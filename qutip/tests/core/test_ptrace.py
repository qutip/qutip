# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, The QuTiP Project
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
import pytest

import qutip


def expected(qobj, sel):
    if qobj.isbra or qobj.isket:
        qobj = qobj.proj()
    sel = sorted(sel)
    dims = [[x for i, x in enumerate(qobj.dims[0]) if i in sel]]*2
    new_shape = (np.prod(dims[0]),) * 2
    out = qobj.full()
    before, after = 1, qobj.shape[0]
    for i, dim in enumerate(qobj.dims[0]):
        after //= dim
        if i in sel:
            before = before * dim
            continue
        tmp_dims = (before, dim, after) * 2
        out = np.einsum('aibcid->abcd', out.reshape(tmp_dims))
    return qutip.Qobj(out.reshape(new_shape), dims=dims)


def test_ptrace_rand_sparse():
    'ptrace : randomized tests, sparse'
    for _ in range(5):
        A = qutip.tensor(
            qutip.rand_ket(5), qutip.rand_ket(2), qutip.rand_ket(3),
        )
        for sel in ([2, 1], [0, 2], [0, 1]):
            assert A.ptrace(sel) == expected(A, sel)

        A = qutip.tensor(
            qutip.rand_dm(2), qutip.thermal_dm(10, 1), qutip.rand_unitary(3),
        )
        for sel in ([1, 2], [0, 2], [0, 1]):
            assert A.ptrace(sel) == expected(A, sel)

        A = qutip.tensor(
            qutip.rand_ket(2), qutip.rand_ket(2), qutip.rand_ket(2),
            qutip.rand_ket(2), qutip.rand_ket(2), qutip.rand_ket(2),
        )
        for sel in ([3, 2], [0, 2], [0, 1]):
            assert A.ptrace(sel) == expected(A, sel)

        A = qutip.rand_dm(64, 0.5, dims=[[4, 4, 4], [4, 4, 4]])
        for sel in ([0], [1], [0, 2]):
            assert A.ptrace(sel) == expected(A, sel)


@pytest.mark.skip(reason="dense data layer not yet implemented")
def test_ptrace_rand_dense():
    'ptrace : randomized tests, dense'
    for k in range(5):
        A = tensor(rand_ket(5), rand_ket(2), rand_ket(3))
        B = A.ptrace([1, 2], False)
        bdat, bd, _ = _pt(A, [1, 2])
        C = Qobj(bdat, dims=bd)
        assert B == C

        B = A.ptrace([0, 2], False)
        bdat, bd, _ = _pt(A, [0, 2])
        C = Qobj(bdat, dims=bd)
        assert B == C

        B = A.ptrace([0, 1], False)
        bdat, bd, _ = _pt(A, [0, 1])
        C = Qobj(bdat, dims=bd)
        assert B == C

    for k in range(5):
        A = tensor(rand_dm(2), thermal_dm(10, 1), rand_unitary(3))
        B = A.ptrace([1, 2], False)
        bdat, bd, _ = _pt(A, [1, 2])
        C = Qobj(bdat, dims=bd)
        assert B == C

        B = A.ptrace([0, 2], False)
        bdat, bd, _ = _pt(A, [0, 2])
        C = Qobj(bdat, dims=bd)
        assert B == C

        B = A.ptrace([0, 1], False)
        bdat, bd, _ = _pt(A, [0, 1])
        C = Qobj(bdat, dims=bd)
        assert B == C

    for k in range(5):
        A = tensor(rand_ket(2), rand_ket(2), rand_ket(2),
                   rand_ket(2), rand_ket(2), rand_ket(2))
        B = A.ptrace([3, 2], False)
        bdat, bd, _ = _pt(A, [3, 2])
        C = Qobj(bdat, dims=bd)
        assert B == C

        B = A.ptrace([0, 2], False)
        bdat, bd, _ = _pt(A, [0, 2])
        C = Qobj(bdat, dims=bd)
        assert B == C

        B = A.ptrace([0, 1], False)
        bdat, bd, _ = _pt(A, [0, 1])
        C = Qobj(bdat, dims=bd)
        assert B == C

    for k in range(5):
        A = rand_dm(64, 0.5, dims=[[4, 4, 4], [4, 4, 4]])
        B = A.ptrace([0], False)
        bdat, bd, _ = _pt(A, [0])
        C = Qobj(bdat, dims=bd)
        assert B == C

        B = A.ptrace([1], False)
        bdat, bd, _ = _pt(A, [1])
        C = Qobj(bdat, dims=bd)
        assert B == C

        B = A.ptrace([0, 2], False)
        bdat, bd, _ = _pt(A, [0, 2])
        C = Qobj(bdat, dims=bd)
        assert B == C
