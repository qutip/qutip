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
from numpy.testing import assert_equal, run_module_suite
import unittest

from qutip import num, rand_herm, expect, rand_unitary
from qutip import _version2int


@unittest.skipIf(_version2int(scipy.__version__) < _version2int('0.10'),
                 'Known to fail on SciPy ' + scipy.__version__)
def test_SparseHermValsVecs():
    """
    Sparse eigs Hermitian
    """

    # check using number operator
    N = num(10)
    spvals, spvecs = N.eigenstates(sparse=True)
    for k in range(10):
        # check that eigvals are in proper order
        assert_equal(abs(spvals[k] - k) <= 1e-13, True)
        # check that eigenvectors are right and in right order
        assert_equal(abs(expect(N, spvecs[k]) - spvals[k]) < 5e-14, True)

    # check ouput of only a few eigenvals/vecs
    spvals, spvecs = N.eigenstates(sparse=True, eigvals=7)
    assert_equal(len(spvals), 7)
    assert_equal(spvals[0] <= spvals[-1], True)
    for k in range(7):
        assert_equal(abs(spvals[k] - k) < 1e-12, True)

    spvals, spvecs = N.eigenstates(sparse=True, sort='high', eigvals=5)
    assert_equal(len(spvals), 5)
    assert_equal(spvals[0] >= spvals[-1], True)
    vals = np.arange(9, 4, -1)
    for k in range(5):
        # check that eigvals are ordered from high to low
        assert_equal(abs(spvals[k] - vals[k]) < 5e-14, True)
        assert_equal(abs(expect(N, spvecs[k]) - vals[k]) < 1e-14, True)
    # check using random Hermitian
    H = rand_herm(10)
    spvals, spvecs = H.eigenstates(sparse=True)
    # check that sorting is lowest eigval first
    assert_equal(spvals[0] <= spvals[-1], True)
    # check that spvals equal expect vals
    for k in range(10):
        assert_equal(abs(expect(H, spvecs[k]) - spvals[k]) < 5e-14, True)
        # check that ouput is real for Hermitian operator
        assert_equal(np.isreal(spvals[k]), True)


def test_SparseValsVecs():
    """
    Sparse eigs non-Hermitian
    """
    U = rand_unitary(10)
    spvals, spvecs = U.eigenstates(sparse=True)
    assert_equal(np.real(spvals[0]) <= np.real(spvals[-1]), True)
    for k in range(10):
        # check that eigenvectors are right and in right order
        assert_equal(abs(expect(U, spvecs[k]) - spvals[k]) < 5e-14, True)
        assert_equal(np.iscomplex(spvals[k]), True)

    # check sorting
    spvals, spvecs = U.eigenstates(sparse=True, sort='high')
    assert_equal(np.real(spvals[0]) >= np.real(spvals[-1]), True)

    # check for N-1 eigenvals
    U = rand_unitary(10)
    spvals, spvecs = U.eigenstates(sparse=True, eigvals=9)
    assert_equal(len(spvals), 9)


@unittest.skipIf(_version2int(scipy.__version__) < _version2int('0.10'),
                 'Known to fail on SciPy ' + scipy.__version__)
def test_SparseValsOnly():
    """
    Sparse eigvals only Hermitian.
    """
    H = rand_herm(10)
    spvals = H.eigenenergies(sparse=True)
    assert_equal(len(spvals), 10)
    # check that sorting is lowest eigval first
    assert_equal(spvals[0] <= spvals[-1], True)
    # check that spvals equal expect vals
    for k in range(10):
        # check that ouput is real for Hermitian operator
        assert_equal(np.isreal(spvals[k]), True)
    spvals = H.eigenenergies(sparse=True, sort='high')
    # check that sorting is lowest eigval first
    assert_equal(spvals[0] >= spvals[-1], True)
    spvals = H.eigenenergies(sparse=True, sort='high', eigvals=4)
    assert_equal(len(spvals), 4)

    U = rand_unitary(10)
    spvals = U.eigenenergies(sparse=True)
    assert_equal(len(spvals), 10)
    # check that sorting is lowest eigval first
    assert_equal(spvals[0] <= spvals[-1], True)
    # check that spvals equal expect vals
    for k in range(10):
        # check that ouput is real for Hermitian operator
        assert_equal(np.iscomplex(spvals[k]), True)
    spvals = U.eigenenergies(sparse=True, sort='high')
    # check that sorting is lowest eigval first
    assert_equal(spvals[0] >= spvals[-1], True)
    spvals = U.eigenenergies(sparse=True, sort='high', eigvals=4)
    assert_equal(len(spvals), 4)


def test_DenseHermValsVecs():
    """
    Dense eigs Hermitian.
    """
    # check using number operator
    N = num(10)
    spvals, spvecs = N.eigenstates(sparse=False)
    for k in range(10):
        # check that eigvals are in proper order
        assert_equal(abs(spvals[k] - k) < 1e-14, True)
        # check that eigenvectors are right and in right order
        assert_equal(abs(expect(N, spvecs[k]) - spvals[k]) < 5e-14, True)

    # check ouput of only a few eigenvals/vecs
    spvals, spvecs = N.eigenstates(sparse=False, eigvals=7)
    assert_equal(len(spvals), 7)
    assert_equal(spvals[0] <= spvals[-1], True)
    for k in range(7):
        assert_equal(abs(spvals[k] - k) < 1e-14, True)

    spvals, spvecs = N.eigenstates(sparse=False, sort='high', eigvals=5)
    assert_equal(len(spvals), 5)
    assert_equal(spvals[0] >= spvals[-1], True)
    vals = np.arange(9, 4, -1)
    for k in range(5):
        # check that eigvals are ordered from high to low
        assert_equal(abs(spvals[k] - vals[k]) < 5e-14, True)
        assert_equal(abs(expect(N, spvecs[k]) - vals[k]) < 5e-14, True)
    # check using random Hermitian
    H = rand_herm(10)
    spvals, spvecs = H.eigenstates(sparse=False)
    # check that sorting is lowest eigval first
    assert_equal(spvals[0] <= spvals[-1], True)
    # check that spvals equal expect vals
    for k in range(10):
        assert_equal(abs(expect(H, spvecs[k]) - spvals[k]) < 5e-14, True)
        # check that ouput is real for Hermitian operator
        assert_equal(np.isreal(spvals[k]), True)


def test_DenseValsVecs():
    """
    Dense eigs non-Hermitian
    """
    U = rand_unitary(10)
    spvals, spvecs = U.eigenstates(sparse=False)
    assert_equal(np.real(spvals[0]) <= np.real(spvals[-1]), True)
    for k in range(10):
        # check that eigenvectors are right and in right order
        assert_equal(abs(expect(U, spvecs[k]) - spvals[k]) < 1e-14, True)
        assert_equal(np.iscomplex(spvals[k]), True)

    # check sorting
    spvals, spvecs = U.eigenstates(sparse=False, sort='high')
    assert_equal(np.real(spvals[0]) >= np.real(spvals[-1]), True)

    # check for N-1 eigenvals
    U = rand_unitary(10)
    spvals, spvecs = U.eigenstates(sparse=False, eigvals=9)
    assert_equal(len(spvals), 9)


def test_DenseValsOnly():
    """
    Dense eigvals only Hermitian
    """
    H = rand_herm(10)
    spvals = H.eigenenergies(sparse=False)
    assert_equal(len(spvals), 10)
    # check that sorting is lowest eigval first
    assert_equal(spvals[0] <= spvals[-1], True)
    # check that spvals equal expect vals
    for k in range(10):
        # check that ouput is real for Hermitian operator
        assert_equal(np.isreal(spvals[k]), True)
    spvals = H.eigenenergies(sparse=False, sort='high')
    # check that sorting is lowest eigval first
    assert_equal(spvals[0] >= spvals[-1], True)
    spvals = H.eigenenergies(sparse=False, sort='high', eigvals=4)
    assert_equal(len(spvals), 4)

    U = rand_unitary(10)
    spvals = U.eigenenergies(sparse=False)
    assert_equal(len(spvals), 10)
    # check that sorting is lowest eigval first
    assert_equal(spvals[0] <= spvals[-1], True)
    # check that spvals equal expect vals
    for k in range(10):
        # check that ouput is real for Hermitian operator
        assert_equal(np.iscomplex(spvals[k]), True)
    spvals = U.eigenenergies(sparse=False, sort='high')
    # check that sorting is lowest eigval first
    assert_equal(spvals[0] >= spvals[-1], True)
    spvals = U.eigenenergies(sparse=False, sort='high', eigvals=4)
    assert_equal(len(spvals), 4)

if __name__ == "__main__":
    run_module_suite()
