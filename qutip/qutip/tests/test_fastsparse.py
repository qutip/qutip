# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, The QuTiP Project.
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
import scipy.sparse
import qutip
from qutip.fastsparse import fast_csr_matrix


class TestOperationEffectsOnType:
    @pytest.mark.parametrize("operation", [
        pytest.param(lambda x: x, id="identity"),
        pytest.param(lambda x: x + x, id="addition"),
        pytest.param(lambda x: x - x, id="subtraction"),
        pytest.param(lambda x: x * x, id="multiplication by op"),
        pytest.param(lambda x: 2*x, id="multiplication by scalar"),
        pytest.param(lambda x: x/3, id="division by scalar"),
        pytest.param(lambda x: -x, id="negation"),
        pytest.param(lambda x: x.copy(), id="copy"),
        pytest.param(lambda x: x.T, id="transpose [.T]"),
        pytest.param(lambda x: x.trans(), id="transpose [.trans()]"),
        pytest.param(lambda x: x.transpose(), id="transpose [.transpose()]"),
        pytest.param(lambda x: x.H, id="adjoint [.H]"),
        pytest.param(lambda x: x.getH(), id="adjoint [.getH()]"),
        pytest.param(lambda x: x.adjoint(), id="adjoint [.adjoint()]"),
    ])
    def test_operations_preserve_type(self, operation):
        op = qutip.rand_herm(5).data
        assert isinstance(operation(op), fast_csr_matrix)

    @pytest.mark.parametrize("operation", [
        pytest.param(lambda x, y: y, id="identity of other"),
        pytest.param(lambda x, y: x + y, id="addition"),
        pytest.param(lambda x, y: y + x, id="r-addition"),
        pytest.param(lambda x, y: x - y, id="subtraction"),
        pytest.param(lambda x, y: y - x, id="r-subtraction"),
        pytest.param(lambda x, y: x * y, id="multiplication"),
        pytest.param(lambda x, y: y * x, id="r-multiplication"),
    ])
    def test_mixed_operations_yield_type(self, operation):
        op = qutip.rand_herm(5).data
        other = scipy.sparse.csr_matrix((op.data, op.indices, op.indptr),
                                        copy=True, shape=op.shape)
        assert not isinstance(operation(op, other), fast_csr_matrix)
