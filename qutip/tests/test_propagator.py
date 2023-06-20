import numpy as np
from numpy.testing import assert_, assert_equal
from qutip import *


def testPropHO():
    "Propagator: HO ('single mode')"
    a = destroy(5)
    H = a.dag()*a
    U = propagator(H,1, unitary_mode='single')
    U2 = (-1j*H).expm()
    assert_(np.abs((U-U2).full()).max() < 1e-4)

def testPropHOB():
    "Propagator: HO ('batch mode')"
    a = destroy(5)
    H = a.dag()*a
    U = propagator(H,1)
    U2 = (-1j*H).expm()
    assert_(np.abs((U-U2).full()).max() < 1e-4)

def testPropHOPar():
    "Propagator: HO parallel"
    a = destroy(5)
    H = a.dag()*a
    U = propagator(H,1, parallel=True)
    U2 = (-1j*H).expm()
    assert_(np.abs((U-U2).full()).max() < 1e-4)


def testPropHOStrTd():
    "Propagator: str td format"
    a = destroy(5)
    H = a.dag()*a
    H = [H,[H,'cos(t)']]
    U = propagator(H,1, unitary_mode='single')
    U2 = propagator(H,1, parallel=True)
    U3 = propagator(H,1)
    assert_(np.abs((U-U2).full()).max() < 1e-4)
    assert_(np.abs((U-U3).full()).max() < 1e-4)


def func(t,*args):
    return np.cos(t)

def testPropHOFuncTd():
    "Propagator: func td format"
    a = destroy(5)
    H = a.dag()*a
    H = [H,[H,func]]
    U = propagator(H,1, unitary_mode='single')
    U2 = propagator(H,1, parallel=True)
    U3 = propagator(H,1)
    assert_(np.abs((U-U2).full()).max() < 1e-4)
    assert_(np.abs((U-U3).full()).max() < 1e-4)


def testPropHOSteady():
    "Propagator: steady state"
    a = destroy(5)
    H = a.dag()*a
    c_op_list = []
    kappa = 0.1
    n_th = 2
    rate = kappa * (1 + n_th)
    c_op_list.append(np.sqrt(rate) * a)
    rate = kappa * n_th
    c_op_list.append(np.sqrt(rate) * a.dag())
    U = propagator(H,2*np.pi,c_op_list)
    rho_prop = propagator_steadystate(U)
    rho_ss = steadystate(H,c_op_list)
    assert_(np.abs((rho_prop-rho_ss).full()).max() < 1e-4)


def testPropHOSteadyPar():
    "Propagator: steady state parallel"
    a = destroy(5)
    H = a.dag()*a
    c_op_list = []
    kappa = 0.1
    n_th = 2
    rate = kappa * (1 + n_th)
    c_op_list.append(np.sqrt(rate) * a)
    rate = kappa * n_th
    c_op_list.append(np.sqrt(rate) * a.dag())
    U = propagator(H,2*np.pi,c_op_list, parallel=True)
    rho_prop = propagator_steadystate(U)
    rho_ss = steadystate(H,c_op_list)
    assert_(np.abs((rho_prop-rho_ss).full()).max() < 1e-4)

def testPropHDims():
    "Propagator: preserve H dims (unitary_mode='single', parallel=False)"
    H = tensor([qeye(2),qeye(2)])
    U = propagator(H,1, unitary_mode='single')
    assert_equal(U.dims,H.dims)


def testPropHSuperWithoutCops():
    "Propagator: super operator without collapse operators"
    H = tensor(sigmaz(), qeye(2))
    H = liouvillian(H)
    tlist = np.linspace(0, 10, 11)
    Fs = propagator(H, tlist)
    rho0 = qeye([[2, 2], [2, 2]])
    expected_Fs = mesolve(H, rho0, tlist).states
    assert Fs == expected_Fs


def testPropHSuperWithoutCopsParallel():
    "Propagator: super operator without collapse operators using parallel"
    H = tensor(sigmaz(), qeye(2))
    H = liouvillian(H)
    tlist = np.linspace(0, 10, 11)
    Fs = propagator(H, tlist, parallel=True)
    rho0 = qeye([[2, 2], [2, 2]])
    expected_Fs = mesolve(H, rho0, tlist).states
    for k, _ in enumerate(tlist):
        assert (Fs[k] - expected_Fs[k]).norm() < 1e-3


def testPropHWithCops():
    "Propagator: with collapse operators"
    H = tensor(sigmaz(), qeye(2))
    c_ops = [np.sqrt(1) * tensor(sigmam(), qeye(2))]
    tlist = np.linspace(0, 10, 11)
    Fs = propagator(H, tlist, c_op_list=c_ops)
    rho0 = ket2dm(tensor(basis(2, 0), basis(2, 0))).unit()
    rho_fs = [vector_to_operator(F * operator_to_vector(rho0)) for F in Fs]
    expected_rho_fs = mesolve(H, rho0, tlist, c_ops=c_ops).states
    assert rho_fs == expected_rho_fs


def testPropHWithCopsParallel():
    "Propagator: with collapse operators in parallel"
    H = tensor(sigmaz(), qeye(2))
    c_ops = [np.sqrt(1) * tensor(sigmam(), qeye(2))]
    tlist = np.linspace(0, 10, 11)
    Fs = propagator(H, tlist, c_op_list=c_ops, parallel=True)
    rho0 = ket2dm(tensor(basis(2, 0), basis(2, 0))).unit()
    rho_fs = [vector_to_operator(F * operator_to_vector(rho0)) for F in Fs]
    expected_rho_fs = mesolve(H, rho0, tlist, c_ops=c_ops).states
    assert rho_fs == expected_rho_fs
