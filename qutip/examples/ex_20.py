# 
# Dissipative $i$-SWAP gate vs ideal gate.
# Accuracy of gate given by Fidelity of
# final state and ideal final state.
#
from qutip import *
from pylab import *


def run():
    # setup system parameters
    g  = 1.0 * 2 * pi   # coupling strength
    g1 = 0.75           # relaxation rate
    g2 = 0.05           # dephasing rate
    n_th = 0.75         # bath temperature
    T  = pi/(4*g) 
    
    # construct Hamiltonian
    H = g * (tensor(sigmax(), sigmax()) +
             tensor(sigmay(), sigmay()))
    # construct inital state
    psi0 = tensor(basis(2,1), basis(2,0))     
    
    #construct collapse operators
    c_op_list = []
    ## qubit 1 collapse operators
    sm1 = tensor(sigmam(), qeye(2))
    sz1 = tensor(sigmaz(), qeye(2))
    c_op_list.append(sqrt(g1 * (1+n_th)) * sm1)
    c_op_list.append(sqrt(g1 * n_th) * sm1.dag())
    c_op_list.append(sqrt(g2) * sz1)
    ## qubit 2 collapse operators
    sm2 = tensor(qeye(2), sigmam())
    sz2 = tensor(qeye(2), sigmaz())
    c_op_list.append(sqrt(g1 * (1+n_th)) * sm2)
    c_op_list.append(sqrt(g1 * n_th) * sm2.dag())
    c_op_list.append(sqrt(g2) * sz2)
    
    # evolve the dissipative system
    tlist = linspace(0, T, 100)
    rho_list  = odesolve(H, psi0, tlist, c_op_list, [])
    rho_final = rho_list[-1]
    
    # calculate expectation values 
    n1 = expect(sm1.dag() * sm1, rho_list)
    n2 = expect(sm2.dag() * sm2, rho_list)     
    
    # calculate the ideal evolution 
    psi_list_ideal= odesolve(H, psi0, tlist, [], [])
    n1_ideal = expect(sm1.dag() * sm1, psi_list_ideal)
    n2_ideal = expect(sm2.dag() * sm2, psi_list_ideal)
    # get last ket vector for comparision with dissipative model
    # output is ket since no collapse operators.
    psi_ideal=psi_list_ideal[-1]
    rho_ideal=ket2dm(psi_ideal)
    
    # calculate the fidelity of final states
    F = fidelity(rho_ideal, rho_final) 
    
    # plot the results
    plot(tlist / T, n1, 'r',tlist / T, n2, 'b',lw=2)
    plot(tlist / T, n1_ideal, 'r--',tlist / T, n2_ideal, 'b--',lw=1)
    xlabel('t/T', fontsize=16)
    ylabel('Occupation probability', fontsize=16)
    figtext(0.65, 0.6, "Fidelity = %.3f" % F, fontsize=16)
    title("Dissipative i-Swap Gate vs. Ideal Gate (dashed)")
    ylim([0,1])
    show()


if __name__=='__main__':
    run()