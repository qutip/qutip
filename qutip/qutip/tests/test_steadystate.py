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
from numpy.testing import assert_, assert_equal, run_module_suite

from qutip import (sigmaz, destroy, steadystate, expect, coherent_dm,
                    build_preconditioner)


def test_qubit_direct():
    "Steady state: Thermal qubit - direct solver"
    # thermal steadystate of a qubit: compare numerics with analytical formula
    sz = sigmaz()
    sm = destroy(2)

    H = 0.5 * 2 * np.pi * sz
    gamma1 = 0.05

    wth_vec = np.linspace(0.1, 3, 20)
    p_ss = np.zeros(np.shape(wth_vec))

    for idx, wth in enumerate(wth_vec):

        n_th = 1.0 / (np.exp(1.0 / wth) - 1)  # bath temperature
        c_op_list = []
        rate = gamma1 * (1 + n_th)
        c_op_list.append(np.sqrt(rate) * sm)
        rate = gamma1 * n_th
        c_op_list.append(np.sqrt(rate) * sm.dag())
        rho_ss = steadystate(H, c_op_list, method='direct')
        p_ss[idx] = expect(sm.dag() * sm, rho_ss)

    p_ss_analytic = np.exp(-1.0 / wth_vec) / (1 + np.exp(-1.0 / wth_vec))
    delta = sum(abs(p_ss_analytic - p_ss))
    assert_equal(delta < 1e-5, True)


def test_qubit_eigen():
    "Steady state: Thermal qubit - eigen solver"
    # thermal steadystate of a qubit: compare numerics with analytical formula
    sz = sigmaz()
    sm = destroy(2)

    H = 0.5 * 2 * np.pi * sz
    gamma1 = 0.05

    wth_vec = np.linspace(0.1, 3, 20)
    p_ss = np.zeros(np.shape(wth_vec))

    for idx, wth in enumerate(wth_vec):

        n_th = 1.0 / (np.exp(1.0 / wth) - 1)  # bath temperature
        c_op_list = []
        rate = gamma1 * (1 + n_th)
        c_op_list.append(np.sqrt(rate) * sm)
        rate = gamma1 * n_th
        c_op_list.append(np.sqrt(rate) * sm.dag())
        rho_ss = steadystate(H, c_op_list, method='eigen')
        p_ss[idx] = expect(sm.dag() * sm, rho_ss)

    p_ss_analytic = np.exp(-1.0 / wth_vec) / (1 + np.exp(-1.0 / wth_vec))
    delta = sum(abs(p_ss_analytic - p_ss))
    assert_equal(delta < 1e-5, True)


def test_qubit_power():
    "Steady state: Thermal qubit - power solver"
    # thermal steadystate of a qubit: compare numerics with analytical formula
    sz = sigmaz()
    sm = destroy(2)

    H = 0.5 * 2 * np.pi * sz
    gamma1 = 0.05

    wth_vec = np.linspace(0.1, 3, 20)
    p_ss = np.zeros(np.shape(wth_vec))

    for idx, wth in enumerate(wth_vec):

        n_th = 1.0 / (np.exp(1.0 / wth) - 1)  # bath temperature
        c_op_list = []
        rate = gamma1 * (1 + n_th)
        c_op_list.append(np.sqrt(rate) * sm)
        rate = gamma1 * n_th
        c_op_list.append(np.sqrt(rate) * sm.dag())
        rho_ss = steadystate(H, c_op_list, method='power', mtol=1e-5)
        p_ss[idx] = expect(sm.dag() * sm, rho_ss)

    p_ss_analytic = np.exp(-1.0 / wth_vec) / (1 + np.exp(-1.0 / wth_vec))
    delta = sum(abs(p_ss_analytic - p_ss))
    assert_equal(delta < 1e-5, True)


def test_qubit_power_gmres():
    "Steady state: Thermal qubit - power-gmres solver"
    # thermal steadystate of a qubit: compare numerics with analytical formula
    sz = sigmaz()
    sm = destroy(2)

    H = 0.5 * 2 * np.pi * sz
    gamma1 = 0.05

    wth_vec = np.linspace(0.1, 3, 20)
    p_ss = np.zeros(np.shape(wth_vec))

    for idx, wth in enumerate(wth_vec):
        n_th = 1.0 / (np.exp(1.0 / wth) - 1)  # bath temperature
        c_op_list = []
        rate = gamma1 * (1 + n_th)
        c_op_list.append(np.sqrt(rate) * sm)
        rate = gamma1 * n_th
        c_op_list.append(np.sqrt(rate) * sm.dag())
        rho_ss = steadystate(H, c_op_list, method='power-gmres', mtol=1e-1)
        p_ss[idx] = expect(sm.dag() * sm, rho_ss)

    p_ss_analytic = np.exp(-1.0 / wth_vec) / (1 + np.exp(-1.0 / wth_vec))
    delta = sum(abs(p_ss_analytic - p_ss))
    assert_equal(delta < 1e-5, True)


def test_qubit_power_bicgstab():
    "Steady state: Thermal qubit - power-bicgstab solver"
    # thermal steadystate of a qubit: compare numerics with analytical formula
    sz = sigmaz()
    sm = destroy(2)

    H = 0.5 * 2 * np.pi * sz
    gamma1 = 0.05

    wth_vec = np.linspace(0.1, 3, 20)
    p_ss = np.zeros(np.shape(wth_vec))

    for idx, wth in enumerate(wth_vec):

        n_th = 1.0 / (np.exp(1.0 / wth) - 1)  # bath temperature
        c_op_list = []
        rate = gamma1 * (1 + n_th)
        c_op_list.append(np.sqrt(rate) * sm)
        rate = gamma1 * n_th
        c_op_list.append(np.sqrt(rate) * sm.dag())
        rho_ss = steadystate(H, c_op_list, method='power-bicgstab', use_precond=1)
        p_ss[idx] = expect(sm.dag() * sm, rho_ss)

    p_ss_analytic = np.exp(-1.0 / wth_vec) / (1 + np.exp(-1.0 / wth_vec))
    delta = sum(abs(p_ss_analytic - p_ss))
    assert_equal(delta < 1e-5, True)



def test_qubit_gmres():
    "Steady state: Thermal qubit - iterative-gmres solver"
    # thermal steadystate of a qubit: compare numerics with analytical formula
    sz = sigmaz()
    sm = destroy(2)

    H = 0.5 * 2 * np.pi * sz
    gamma1 = 0.05

    wth_vec = np.linspace(0.1, 3, 20)
    p_ss = np.zeros(np.shape(wth_vec))

    for idx, wth in enumerate(wth_vec):

        n_th = 1.0 / (np.exp(1.0 / wth) - 1)  # bath temperature
        c_op_list = []
        rate = gamma1 * (1 + n_th)
        c_op_list.append(np.sqrt(rate) * sm)
        rate = gamma1 * n_th
        c_op_list.append(np.sqrt(rate) * sm.dag())
        rho_ss = steadystate(H, c_op_list, method='iterative-gmres')
        p_ss[idx] = expect(sm.dag() * sm, rho_ss)

    p_ss_analytic = np.exp(-1.0 / wth_vec) / (1 + np.exp(-1.0 / wth_vec))
    delta = sum(abs(p_ss_analytic - p_ss))
    assert_equal(delta < 1e-5, True)


def test_qubit_bicgstab():
    "Steady state: Thermal qubit - iterative-bicgstab solver"
    # thermal steadystate of a qubit: compare numerics with analytical formula
    sz = sigmaz()
    sm = destroy(2)

    H = 0.5 * 2 * np.pi * sz
    gamma1 = 0.05

    wth_vec = np.linspace(0.1, 3, 20)
    p_ss = np.zeros(np.shape(wth_vec))

    for idx, wth in enumerate(wth_vec):

        n_th = 1.0 / (np.exp(1.0 / wth) - 1)  # bath temperature
        c_op_list = []
        rate = gamma1 * (1 + n_th)
        c_op_list.append(np.sqrt(rate) * sm)
        rate = gamma1 * n_th
        c_op_list.append(np.sqrt(rate) * sm.dag())
        rho_ss = steadystate(H, c_op_list, method='iterative-bicgstab')
        p_ss[idx] = expect(sm.dag() * sm, rho_ss)

    p_ss_analytic = np.exp(-1.0 / wth_vec) / (1 + np.exp(-1.0 / wth_vec))
    delta = sum(abs(p_ss_analytic - p_ss))
    assert_equal(delta < 1e-5, True)


def test_ho_direct():
    "Steady state: Thermal HO - direct solver"
    # thermal steadystate of an oscillator: compare numerics with analytical
    # formula
    a = destroy(40)
    H = 0.5 * 2 * np.pi * a.dag() * a
    gamma1 = 0.05

    wth_vec = np.linspace(0.1, 3, 20)
    p_ss = np.zeros(np.shape(wth_vec))

    for idx, wth in enumerate(wth_vec):

        n_th = 1.0 / (np.exp(1.0 / wth) - 1)  # bath temperature
        c_op_list = []
        rate = gamma1 * (1 + n_th)
        c_op_list.append(np.sqrt(rate) * a)
        rate = gamma1 * n_th
        c_op_list.append(np.sqrt(rate) * a.dag())
        rho_ss = steadystate(H, c_op_list, method='direct')
        p_ss[idx] = np.real(expect(a.dag() * a, rho_ss))

    p_ss_analytic = 1.0 / (np.exp(1.0 / wth_vec) - 1)
    delta = sum(abs(p_ss_analytic - p_ss))
    assert_equal(delta < 1e-3, True)


def test_ho_eigen():
    "Steady state: Thermal HO - eigen solver"
    # thermal steadystate of an oscillator: compare numerics with analytical
    # formula
    a = destroy(40)
    H = 0.5 * 2 * np.pi * a.dag() * a
    gamma1 = 0.05

    wth_vec = np.linspace(0.1, 3, 20)
    p_ss = np.zeros(np.shape(wth_vec))

    for idx, wth in enumerate(wth_vec):

        n_th = 1.0 / (np.exp(1.0 / wth) - 1)  # bath temperature
        c_op_list = []
        rate = gamma1 * (1 + n_th)
        c_op_list.append(np.sqrt(rate) * a)
        rate = gamma1 * n_th
        c_op_list.append(np.sqrt(rate) * a.dag())
        rho_ss = steadystate(H, c_op_list, method='eigen')
        p_ss[idx] = np.real(expect(a.dag() * a, rho_ss))

    p_ss_analytic = 1.0 / (np.exp(1.0 / wth_vec) - 1)
    delta = sum(abs(p_ss_analytic - p_ss))
    assert_equal(delta < 1e-3, True)


def test_ho_power():
    "Steady state: Thermal HO - power solver"
    # thermal steadystate of an oscillator: compare numerics with analytical
    # formula
    a = destroy(40)
    H = 0.5 * 2 * np.pi * a.dag() * a
    gamma1 = 0.05

    wth_vec = np.linspace(0.1, 3, 20)
    p_ss = np.zeros(np.shape(wth_vec))

    for idx, wth in enumerate(wth_vec):

        n_th = 1.0 / (np.exp(1.0 / wth) - 1)  # bath temperature
        c_op_list = []
        rate = gamma1 * (1 + n_th)
        c_op_list.append(np.sqrt(rate) * a)
        rate = gamma1 * n_th
        c_op_list.append(np.sqrt(rate) * a.dag())
        rho_ss = steadystate(H, c_op_list, method='power', mtol=1e-5)
        p_ss[idx] = np.real(expect(a.dag() * a, rho_ss))

    p_ss_analytic = 1.0 / (np.exp(1.0 / wth_vec) - 1)
    delta = sum(abs(p_ss_analytic - p_ss))
    assert_equal(delta < 1e-3, True)

def test_ho_power_gmres():
    "Steady state: Thermal HO - power-gmres solver"
    # thermal steadystate of an oscillator: compare numerics with analytical
    # formula
    a = destroy(40)
    H = 0.5 * 2 * np.pi * a.dag() * a
    gamma1 = 0.05

    wth_vec = np.linspace(0.1, 3, 20)
    p_ss = np.zeros(np.shape(wth_vec))

    for idx, wth in enumerate(wth_vec):

        n_th = 1.0 / (np.exp(1.0 / wth) - 1)  # bath temperature
        c_op_list = []
        rate = gamma1 * (1 + n_th)
        c_op_list.append(np.sqrt(rate) * a)
        rate = gamma1 * n_th
        c_op_list.append(np.sqrt(rate) * a.dag())
        rho_ss = steadystate(H, c_op_list, method='power-gmres', mtol=1e-1,
                             use_precond=1)
        p_ss[idx] = np.real(expect(a.dag() * a, rho_ss))

    p_ss_analytic = 1.0 / (np.exp(1.0 / wth_vec) - 1)
    delta = sum(abs(p_ss_analytic - p_ss))
    assert_equal(delta < 1e-3, True)


def test_ho_power_bicgstab():
    "Steady state: Thermal HO - power-bicgstab solver"
    # thermal steadystate of an oscillator: compare numerics with analytical
    # formula
    a = destroy(40)
    H = 0.5 * 2 * np.pi * a.dag() * a
    gamma1 = 0.05

    wth_vec = np.linspace(0.1, 3, 20)
    p_ss = np.zeros(np.shape(wth_vec))

    for idx, wth in enumerate(wth_vec):

        n_th = 1.0 / (np.exp(1.0 / wth) - 1)  # bath temperature
        c_op_list = []
        rate = gamma1 * (1 + n_th)
        c_op_list.append(np.sqrt(rate) * a)
        rate = gamma1 * n_th
        c_op_list.append(np.sqrt(rate) * a.dag())
        rho_ss = steadystate(H, c_op_list, method='power-bicgstab',use_precond=1)
        p_ss[idx] = np.real(expect(a.dag() * a, rho_ss))

    p_ss_analytic = 1.0 / (np.exp(1.0 / wth_vec) - 1)
    delta = sum(abs(p_ss_analytic - p_ss))
    assert_equal(delta < 1e-3, True)


def test_ho_gmres():
    "Steady state: Thermal HO - iterative-gmres solver"
    # thermal steadystate of an oscillator: compare numerics with analytical
    # formula
    a = destroy(40)
    H = 0.5 * 2 * np.pi * a.dag() * a
    gamma1 = 0.05

    wth_vec = np.linspace(0.1, 3, 20)
    p_ss = np.zeros(np.shape(wth_vec))

    for idx, wth in enumerate(wth_vec):

        n_th = 1.0 / (np.exp(1.0 / wth) - 1)  # bath temperature
        c_op_list = []
        rate = gamma1 * (1 + n_th)
        c_op_list.append(np.sqrt(rate) * a)
        rate = gamma1 * n_th
        c_op_list.append(np.sqrt(rate) * a.dag())
        rho_ss = steadystate(H, c_op_list, method='iterative-gmres')
        p_ss[idx] = np.real(expect(a.dag() * a, rho_ss))

    p_ss_analytic = 1.0 / (np.exp(1.0 / wth_vec) - 1)
    delta = sum(abs(p_ss_analytic - p_ss))
    assert_equal(delta < 1e-3, True)


def test_ho_bicgstab():
    "Steady state: Thermal HO - iterative-bicgstab solver"
    # thermal steadystate of an oscillator: compare numerics with analytical
    # formula
    a = destroy(40)
    H = 0.5 * 2 * np.pi * a.dag() * a
    gamma1 = 0.05

    wth_vec = np.linspace(0.1, 3, 20)
    p_ss = np.zeros(np.shape(wth_vec))

    for idx, wth in enumerate(wth_vec):

        n_th = 1.0 / (np.exp(1.0 / wth) - 1)  # bath temperature
        c_op_list = []
        rate = gamma1 * (1 + n_th)
        c_op_list.append(np.sqrt(rate) * a)
        rate = gamma1 * n_th
        c_op_list.append(np.sqrt(rate) * a.dag())
        rho_ss = steadystate(H, c_op_list, method='iterative-bicgstab')
        p_ss[idx] = np.real(expect(a.dag() * a, rho_ss))

    p_ss_analytic = 1.0 / (np.exp(1.0 / wth_vec) - 1)
    delta = sum(abs(p_ss_analytic - p_ss))
    assert_equal(delta < 1e-3, True)



def test_driven_cavity_direct():
    "Steady state: Driven cavity - direct solver"

    N = 30
    Omega = 0.01 * 2 * np.pi
    Gamma = 0.05

    a = destroy(N)
    H = Omega * (a.dag() + a)
    c_ops = [np.sqrt(Gamma) * a]

    rho_ss = steadystate(H, c_ops, method='direct')
    rho_ss_analytic = coherent_dm(N, -1.0j * (Omega)/(Gamma/2))

    assert_((rho_ss - rho_ss_analytic).norm() < 1e-4)


def test_driven_cavity_eigen():
    "Steady state: Driven cavity - eigen solver"

    N = 30
    Omega = 0.01 * 2 * np.pi
    Gamma = 0.05

    a = destroy(N)
    H = Omega * (a.dag() + a)
    c_ops = [np.sqrt(Gamma) * a]

    rho_ss = steadystate(H, c_ops, method='eigen')
    rho_ss_analytic = coherent_dm(N, -1.0j * (Omega)/(Gamma/2))

    assert_((rho_ss - rho_ss_analytic).norm() < 1e-4)


def test_driven_cavity_power():
    "Steady state: Driven cavity - power solver"

    N = 30
    Omega = 0.01 * 2 * np.pi
    Gamma = 0.05

    a = destroy(N)
    H = Omega * (a.dag() + a)
    c_ops = [np.sqrt(Gamma) * a]

    rho_ss = steadystate(H, c_ops, method='power', mtol=1e-5,)
    rho_ss_analytic = coherent_dm(N, -1.0j * (Omega)/(Gamma/2))

    assert_((rho_ss - rho_ss_analytic).norm() < 1e-4)


def test_driven_cavity_power_gmres():
    "Steady state: Driven cavity - power-gmres solver"

    N = 30
    Omega = 0.01 * 2 * np.pi
    Gamma = 0.05

    a = destroy(N)
    H = Omega * (a.dag() + a)
    c_ops = [np.sqrt(Gamma) * a]
    M = build_preconditioner(H, c_ops, method='power')
    rho_ss = steadystate(H, c_ops, method='power-gmres', M=M, mtol=1e-1,
                         use_precond=1)
    rho_ss_analytic = coherent_dm(N, -1.0j * (Omega)/(Gamma/2))
    assert_((rho_ss - rho_ss_analytic).norm() < 1e-4)



def test_driven_cavity_power_bicgstab():
    "Steady state: Driven cavity - power-bicgstab solver"

    N = 30
    Omega = 0.01 * 2 * np.pi
    Gamma = 0.05

    a = destroy(N)
    H = Omega * (a.dag() + a)
    c_ops = [np.sqrt(Gamma) * a]
    M = build_preconditioner(H, c_ops, method='power')
    rho_ss = steadystate(H, c_ops, method='power-bicgstab', M=M, use_precond=1)
    rho_ss_analytic = coherent_dm(N, -1.0j * (Omega)/(Gamma/2))
    assert_((rho_ss - rho_ss_analytic).norm() < 1e-4)


def test_driven_cavity_gmres():
    "Steady state: Driven cavity - iterative-gmres solver"

    N = 30
    Omega = 0.01 * 2 * np.pi
    Gamma = 0.05

    a = destroy(N)
    H = Omega * (a.dag() + a)
    c_ops = [np.sqrt(Gamma) * a]

    rho_ss = steadystate(H, c_ops, method='iterative-gmres')
    rho_ss_analytic = coherent_dm(N, -1.0j * (Omega)/(Gamma/2))

    assert_((rho_ss - rho_ss_analytic).norm() < 1e-4)


def test_driven_cavity_bicgstab():
    "Steady state: Driven cavity - iterative-bicgstab solver"

    N = 30
    Omega = 0.01 * 2 * np.pi
    Gamma = 0.05

    a = destroy(N)
    H = Omega * (a.dag() + a)
    c_ops = [np.sqrt(Gamma) * a]

    rho_ss = steadystate(H, c_ops, method='iterative-bicgstab')
    rho_ss_analytic = coherent_dm(N, -1.0j * (Omega)/(Gamma/2))

    assert_((rho_ss - rho_ss_analytic).norm() < 1e-4)



if __name__ == "__main__":
    run_module_suite()
