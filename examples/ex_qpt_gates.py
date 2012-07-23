#!/usr/bin/python
"""
Plot the process tomography matrices for some 1, 2, and 3-qubit qubit gates.
"""
from qutip import *

# select smaller than usual font size to fit all the axis labels
rcParams['text.usetex'] = True
rcParams['font.size'] = 14
rcParams['font.family'] = 'serif'

gate_list = [['C-NOT', cnot()],
             ['SWAP', swap()],
             ['$i$SWAP', iswap()],
             ['$\sqrt{i\mathrm{SWAP}}$', sqrtiswap()],
             ['S-NOT', snot()],
             ['$\pi/2$ phase gate', phasegate(pi/2)],
             ['Toffoli', toffoli()],
             ['Fredkin', fredkin()]]

# loop though the gate list and plot the gates
for gate in gate_list:

	name  = gate[0]
	U_psi = gate[1]

	N = len(U_psi.dims[0]) # number of qubits

	# create a superoperator for the density matrix
	# transformation rho = U_psi * rho_0 * U_psi.dag()
	U_rho = spre(U_psi) * spost(U_psi.dag())

	# operator basis for the process tomography
	op_basis = [[qeye(2), sigmax(), sigmay(), sigmaz()] for i in range(N)]

	# labels for operator basis
	op_label = [["i", "x", "y", "z"] for i in range(N)]

	# calculate the chi matrix
	chi = qpt(U_rho, op_basis)

	# visualize the chi matrix
	qpt_plot_combined(chi, op_label, name)

#tight_layout()
show()
