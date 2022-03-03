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

import scipy
import numpy as np
import pytest
from qutip import num, rand_herm, expect, rand_unitary


def is_eigen_set(oper, vals, vecs):
    for val, vec in zip(vals, vecs):
        assert abs(vec.norm() - 1) < 1e-13
        assert abs(expect(oper, vec) - val) < 1e-13


@pytest.mark.parametrize(["sparse", 'dtype'], [
    pytest.param(True, 'csr', id="sparse"),
    pytest.param(False, 'csr', id="sparse2dense"),
    pytest.param(False, 'dense', id="dense"),
])
def test_eigen_known_oper(sparse, dtype):
    N = num(10, dtype=dtype)
    spvals, spvecs = N.eigenstates(sparse=sparse)
    expected = np.arange(10)
    is_eigen_set(N, spvals, spvecs)
    np.testing.assert_allclose(spvals, expected, atol=1e-13)


@pytest.mark.parametrize(["sparse", 'dtype'], [
    pytest.param(True, 'csr', id="sparse"),
    pytest.param(False, 'csr', id="sparse2dense"),
    pytest.param(False, 'dense', id="dense"),
])
@pytest.mark.parametrize(["rand"], [
    pytest.param(rand_herm, id="hermitian"),
    pytest.param(rand_unitary, id="non-hermitian"),
])
@pytest.mark.parametrize("order", ['low', 'high'])
def test_eigen_rand_oper(rand, sparse, dtype, order):
    H = rand(10, dtype=dtype)
    spvals, spvecs = H.eigenstates(sparse=sparse, sort=order)
    if order == 'low':
        assert np.all(np.diff(spvals).real >= 0)
    else:
        assert np.all(np.diff(spvals).real <= 0)
    is_eigen_set(H, spvals, spvecs)


@pytest.mark.parametrize(["sparse", 'dtype'], [
    pytest.param(True, 'csr', id="sparse"),
    pytest.param(False, 'csr', id="sparse2dense"),
    pytest.param(False, 'dense', id="dense"),
])
@pytest.mark.parametrize("rand", [
    pytest.param(rand_herm, id="hermitian"),
    pytest.param(rand_unitary, id="non-hermitian"),
])
@pytest.mark.parametrize("order", ['low', 'high'])
@pytest.mark.parametrize("N", [1, 5, 8, 9])
def test_FewState(rand, sparse, dtype, order, N):
    H = rand(10, dtype=dtype)
    all_spvals = H.eigenenergies(sparse=sparse, sort=order)
    spvals, spvecs = H.eigenstates(sparse=sparse, sort=order, eigvals=N)
    assert np.allclose(all_spvals[:N], spvals)
    is_eigen_set(H, spvals, spvecs)
    if order == 'low':
        assert np.all(np.diff(spvals).real >= 0)
    else:
        assert np.all(np.diff(spvals).real <= 0)


@pytest.mark.parametrize(["sparse", 'dtype'], [
    pytest.param(True, 'csr', id="sparse"),
    pytest.param(False, 'csr', id="sparse2dense"),
    pytest.param(False, 'dense', id="dense"),
])
@pytest.mark.parametrize(["rand"], [
    pytest.param(rand_herm, id="hermitian"),
    pytest.param(rand_unitary, id="non-hermitian"),
])
@pytest.mark.parametrize("order", ['low', 'high'])
@pytest.mark.parametrize("N", [1, 5, 8, 9])
def test_ValsOnly(rand, sparse, dtype, order, N):
    H = rand(10, dtype=dtype)
    all_spvals = H.eigenenergies(sparse=sparse, sort=order)
    spvals = H.eigenenergies(sparse=sparse, sort=order, eigvals=N)
    assert np.allclose(all_spvals[:N], spvals)
    if order == 'low':
        assert np.all(np.diff(spvals).real >= 0)
    else:
        assert np.all(np.diff(spvals).real <= 0)
