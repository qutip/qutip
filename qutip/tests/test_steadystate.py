import numpy as np
from numpy.testing import assert_, assert_equal, run_module_suite

from qutip import (sigmaz, destroy, steadystate, steadystate_floquet,
		   expect, coherent_dm, fock, mesolve, build_preconditioner)


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


def test_steadystate_floquet_sparse():
    """
    Test the steadystate solution for a periodically
    driven system.
    """
    N_c = 20

    a = destroy(N_c)
    a_d = a.dag()
    X_c = a + a_d

    w_c = 1

    A_l = 0.001
    w_l = w_c
    gam = 0.01

    H = w_c * a_d * a

    H_t = [H, [X_c, lambda t, args: args["A_l"] * np.cos(args["w_l"] * t)]]

    psi0 = fock(N_c, 0)

    args = {"A_l": A_l, "w_l": w_l}

    c_ops = []
    c_ops.append(np.sqrt(gam) * a)

    t_l = np.linspace(0, 20 / gam, 2000)

    expect_me = mesolve(H_t, psi0, t_l,
                        c_ops, [a_d * a], args=args).expect[0]

    rho_ss = steadystate_floquet(H, c_ops,
                                 A_l * X_c, w_l, n_it=3, sparse=True)
    expect_ss = expect(a_d * a, rho_ss)

    np.testing.assert_allclose(expect_me[-20:], expect_ss, atol=1e-3)


def test_steadystate_floquet_dense():
    """
    Test the steadystate solution for a periodically
    driven system.
    """
    N_c = 20

    a = destroy(N_c)
    a_d = a.dag()
    X_c = a + a_d

    w_c = 1

    A_l = 0.001
    w_l = w_c
    gam = 0.01

    H = w_c * a_d * a

    H_t = [H, [X_c, lambda t, args: args["A_l"] * np.cos(args["w_l"] * t)]]

    psi0 = fock(N_c, 0)

    args = {"A_l": A_l, "w_l": w_l}

    c_ops = []
    c_ops.append(np.sqrt(gam) * a)

    t_l = np.linspace(0, 20 / gam, 2000)

    expect_me = mesolve(H_t, psi0, t_l,
                        c_ops, [a_d * a], args=args).expect[0]

    rho_ss = steadystate_floquet(H, c_ops,
                                 A_l * X_c, w_l, n_it=3, sparse=False)
    expect_ss = expect(a_d * a, rho_ss)

    np.testing.assert_allclose(expect_me[-20:], expect_ss, atol=1e-3)


if __name__ == "__main__":
    run_module_suite()
