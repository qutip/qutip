#!/usr/bin/python
"""
Plot the process tomography matrices for some 1, 2, and 3-qubit qubit gates.
"""
from qutip import *

rcParams['text.usetex'] = True
rcParams['font.size'] = 16
rcParams['font.family'] = 'serif'

## cnot gate 
U_psi = cnot()
U_rho = spre(U_psi) * spost(U_psi.dag())

op_basis = [[qeye(2), sigmax(), sigmay(), sigmaz()], [qeye(2), sigmax(), sigmay(), sigmaz()]]
op_label = [["i", "x", "y", "z"], ["i", "x", "y", "z"]]
chi = qpt(U_rho, op_basis)
qpt_plot_combined(chi, op_label, "cnot")

## swap gate 
U_psi = swap()
U_rho = spre(U_psi) * spost(U_psi.dag())

op_basis = [[qeye(2), sigmax(), sigmay(), sigmaz()] for i in range(2)]
op_label = [["i", "x", "y", "z"] for i in range(2)]
chi = qpt(U_rho, op_basis)
qpt_plot_combined(chi, op_label, "swap")

## sqrt(i swap) gate 
U_psi = Qobj(array([[1,0,0,0], [0, 1/sqrt(2), -1j/sqrt(2), 0], [0, -1j/sqrt(2), 1/sqrt(2), 0], [0, 0, 0, 1]]))
U_rho = spre(U_psi) * spost(U_psi.dag())

op_basis = [[qeye(2), sigmax(), sigmay(), sigmaz()] for i in range(2)]
op_label = [["i", "x", "y", "z"] for i in range(2)]
chi = qpt(U_rho, op_basis)
qpt_plot_combined(chi, op_label, "sqrt iSWAP")

## i swap gate 
U_psi = Qobj(array([[1,0,0,0], [0,0,1j,0], [0,1j,0,0], [0,0,0,1]]))
U_rho = spre(U_psi) * spost(U_psi.dag())

op_basis = [[qeye(2), sigmax(), sigmay(), sigmaz()] for i in range(2)]
op_label = [["i", "x", "y", "z"] for i in range(2)]
chi = qpt(U_rho, op_basis)
qpt_plot_combined(chi, op_label, "iSWAP")

## snot gate 
U_psi = snot()
U_rho = spre(U_psi) * spost(U_psi.dag())

op_basis = [[qeye(2), sigmax(), sigmay(), sigmaz()]]
op_label = [["i", "x", "y", "z"]]
chi = qpt(U_rho, op_basis)
qpt_plot_combined(chi, op_label, "snot")


## phase gate
theta = pi/2
U_psi = phasegate(theta)
U_rho = spre(U_psi) * spost(U_psi.dag())

op_basis = [[qeye(2), sigmax(), sigmay(), sigmaz()]]
op_label = [["i", "x", "y", "z"]]
chi = qpt(U_rho, op_basis)
qpt_plot_combined(chi, op_label, "phasegate(theta=%.2f)" % theta)

## toffoli gate 
U_psi = toffoli()
U_rho = spre(U_psi) * spost(U_psi.dag())

op_basis = [[qeye(2), sigmax(), sigmay(), sigmaz()] for i in range(3)] 
op_label = [["i", "x", "y", "z"] for i in range(3)]
chi = qpt(U_rho, op_basis)
qpt_plot_combined(chi, op_label, "toffoli")

## fredkin gate 
U_psi = fredkin()
U_rho = spre(U_psi) * spost(U_psi.dag())

op_basis = [[qeye(2), sigmax(), sigmay(), sigmaz()] for i in range(3)] 
op_label = [["i", "x", "y", "z"] for i in range(3)]
chi = qpt(U_rho, op_basis)
qpt_plot_combined(chi, op_label, "fredkin")


show()
