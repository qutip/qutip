import numpy as np
from numpy.testing import run_module_suite, assert_equal
import pytest

from qutip import (
    sigmax, sigmay, sigmaz, sigmam, mesolve, mcsolve, essolve, basis,
)


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

    with pytest.deprecated_call():
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
