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
#    "AS IS" AND ANY np.exp(RESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
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
from numpy.testing import run_module_suite, assert_equal
from qutip import (sigmax, sigmay, sigmaz, sigmam, mesolve, mcsolve, essolve,
                   basis)


def _qubit_integrate(tlist, psi0, epsilon, delta, g1, g2, solver):

    H = epsilon / 2.0 * sigmaz() + delta / 2.0 * sigmax()

    c_op_list = []

    rate = g1
    if rate > 0.0:
        c_op_list.append(np.sqrt(rate) * sigmam())

    rate = g2
    if rate > 0.0:
        c_op_list.append(np.sqrt(rate) * sigmaz())

    e_ops = [sigmax(), sigmay(), sigmaz()]

    if solver == "me":
        output = mesolve(H, psi0, tlist, c_op_list, e_ops)
    elif solver == "es":
        output = essolve(H, psi0, tlist, c_op_list, e_ops)
    elif solver == "mc":
        output = mcsolve(H, psi0, tlist, c_op_list, e_ops, ntraj=750)
    else:
        raise ValueError("unknown solver")

    return output.expect[0], output.expect[1], output.expect[2]


def test_MESolverCase1():
    """
    Test mesolve qubit, with dissipation
    """

    epsilon = 0.0 * 2 * np.pi   # cavity frequency
    delta = 1.0 * 2 * np.pi   # atom frequency
    g2 = 0.1
    g1 = 0.0
    psi0 = basis(2, 0)        # initial state
    tlist = np.linspace(0, 5, 200)

    sx, sy, sz = _qubit_integrate(tlist, psi0, epsilon, delta, g1, g2, "me")

    sx_analytic = np.zeros(np.shape(tlist))
    sy_analytic = -np.sin(2 * np.pi * tlist) * np.exp(-tlist * g2)
    sz_analytic = np.cos(2 * np.pi * tlist) * np.exp(-tlist * g2)

    assert_equal(max(abs(sx - sx_analytic)) < 0.05, True)
    assert_equal(max(abs(sy - sy_analytic)) < 0.05, True)
    assert_equal(max(abs(sz - sz_analytic)) < 0.05, True)


def test_MESolverCase2():
    """
    Test mesolve qubit, no dissipation
    """

    epsilon = 0.0 * 2 * np.pi   # cavity frequency
    delta = 1.0 * 2 * np.pi   # atom frequency
    g2 = 0.0
    g1 = 0.0
    psi0 = basis(2, 0)        # initial state
    tlist = np.linspace(0, 5, 200)

    sx, sy, sz = _qubit_integrate(tlist, psi0, epsilon, delta, g1, g2, "me")

    sx_analytic = np.zeros(np.shape(tlist))
    sy_analytic = -np.sin(2 * np.pi * tlist) * np.exp(-tlist * g2)
    sz_analytic = np.cos(2 * np.pi * tlist) * np.exp(-tlist * g2)

    assert_equal(max(abs(sx - sx_analytic)) < 0.05, True)
    assert_equal(max(abs(sy - sy_analytic)) < 0.05, True)
    assert_equal(max(abs(sz - sz_analytic)) < 0.05, True)


def test_ESSolverCase1():
    """
    Test essolve qubit, with dissipation
    """
    epsilon = 0.0 * 2 * np.pi      # cavity frequency
    delta = 1.0 * 2 * np.pi        # atom frequency
    g2 = 0.1
    g1 = 0.0
    psi0 = basis(2, 0)          # initial state
    tlist = np.linspace(0, 5, 200)

    sx, sy, sz = _qubit_integrate(tlist, psi0, epsilon, delta, g1, g2, "es")

    sx_analytic = np.zeros(np.shape(tlist))
    sy_analytic = -np.sin(2 * np.pi * tlist) * np.exp(-tlist * g2)
    sz_analytic = np.cos(2 * np.pi * tlist) * np.exp(-tlist * g2)

    assert_equal(max(abs(sx - sx_analytic)) < 0.05, True)
    assert_equal(max(abs(sy - sy_analytic)) < 0.05, True)
    assert_equal(max(abs(sz - sz_analytic)) < 0.05, True)


def test_MCSolverCase1():
    """
    Test mcsolve qubit, with dissipation
    """

    epsilon = 0.0 * 2 * np.pi      # cavity frequency
    delta = 1.0 * 2 * np.pi        # atom frequency
    g2 = 0.1
    g1 = 0.0
    psi0 = basis(2, 0)          # initial state
    tlist = np.linspace(0, 5, 200)

    sx, sy, sz = _qubit_integrate(tlist, psi0, epsilon, delta, g1, g2, "mc")

    sx_analytic = np.zeros(np.shape(tlist))
    sy_analytic = -np.sin(2 * np.pi * tlist) * np.exp(-tlist * g2)
    sz_analytic = np.cos(2 * np.pi * tlist) * np.exp(-tlist * g2)

    assert_equal(max(abs(sx - sx_analytic)) < 0.25, True)
    assert_equal(max(abs(sy - sy_analytic)) < 0.25, True)
    assert_equal(max(abs(sz - sz_analytic)) < 0.25, True)


def test_MCSolverCase2():
    """
    Test mcsolve qubit, no dissipation
    """

    epsilon = 0.0 * 2 * np.pi      # cavity frequency
    delta = 1.0 * 2 * np.pi        # atom frequency
    g2 = 0.0
    g1 = 0.0
    psi0 = basis(2, 0)          # initial state
    tlist = np.linspace(0, 5, 200)

    sx, sy, sz = _qubit_integrate(tlist, psi0, epsilon, delta, g1, g2, "mc")

    sx_analytic = np.zeros(np.shape(tlist))
    sy_analytic = -np.sin(2 * np.pi * tlist) * np.exp(-tlist * g2)
    sz_analytic = np.cos(2 * np.pi * tlist) * np.exp(-tlist * g2)

    assert_equal(max(abs(sx - sx_analytic)) < 0.25, True)
    assert_equal(max(abs(sy - sy_analytic)) < 0.25, True)
    assert_equal(max(abs(sz - sz_analytic)) < 0.25, True)


if __name__ == "__main__":
    run_module_suite()
