# 
# Copyright (C) 2011, Paul D. Nation & Robert J. Johansson
# 
# QuTiP: An open-source Python framework for the dynamics of open quantum systems
#
# Appendix C: Dissipative $i$-SWAP gate}\label{sec:iswap_code}

from qutip import *
g  = 1.0 * 2 * pi   # coupling strength
g1 = 0.75           # relaxation rate
g2 = 0.05           # dephasing rate
n_th = 0.75         # bath temperature
T  = pi/(4*g) 
H = g * (tensor(sigmax(), sigmax()) +
         tensor(sigmay(), sigmay()))
psi0 = tensor(basis(2,1), basis(2,0))     
c_op_list = []
## qubit 1 collapse operators ## 
sm1 = tensor(sigmam(), qeye(2))
sz1 = tensor(sigmaz(), qeye(2))
c_op_list.append(sqrt(g1 * (1+n_th)) * sm1)
c_op_list.append(sqrt(g1 * n_th) * sm1.dag())
c_op_list.append(sqrt(g2) * sz1)
## qubit 2 collapse operators ## 
sm2 = tensor(qeye(2), sigmam())
sz2 = tensor(qeye(2), sigmaz())
c_op_list.append(sqrt(g1 * (1+n_th)) * sm2)
c_op_list.append(sqrt(g1 * n_th) * sm2.dag())
c_op_list.append(sqrt(g2) * sz2)
## evolve the system ## 
tlist = linspace(0, T, 100)
rho_list  = odesolve(H, psi0, tlist, c_op_list, [])
rho_final = rho_list[-1]
## calculate expectation values ## 
n1 = expect(sm1.dag() * sm1, rho_list)
n2 = expect(sm2.dag() * sm2, rho_list)     
## calculate the fidelity ## 
U = (-1j * H * pi / (4*g)).expm()
psi_ideal = U * psi0
rho_ideal = psi_ideal * psi_ideal.dag()
f = fidelity(rho_ideal, rho_final) 


# ------------------------------------------------------------------------------
# Plot the results (omitted from the code listing in the appendix in the paper)
#
from pylab import *
plot(tlist / T, n1, 'r')
plot(tlist / T, n2, 'b')
xlabel('t/T', fontsize=18)
ylabel('Occupation probability', fontsize=18)
#title('Two-qubit i-SWAP gate', fontsize=18)
#figtext(0.65, 0.6, "Fidelity = %.3f" % (f,), fontsize=18)
#savefig("jc_model_rabi_oscillations.png")
show()
