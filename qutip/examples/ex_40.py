#
# Rabi oscillations of qubit subject to a classical driving field.
#
from qutip.states import *
from qutip.Qobj import *
from qutip.tensor import *
from qutip.ptrace import *
from qutip.operators import *
from qutip.expect import *
from qutip.correlation import *

from pylab import *

def run():

    #
    # problem parameters:
    #
    delta = 0.0 * 2 * pi   # qubit sigma_x coefficient
    eps0  = 1.0 * 2 * pi   # qubit sigma_z coefficient
    A     = 0.25 * 2 * pi  # driving amplitude (reduce to make the RWA more accurate)
    w     = 1.0 * 2 * pi   # driving frequency
    gamma1 = 0.0           # relaxation rate
    n_th   = 0.0           # average number of excitations ("temperature")
    psi0 = basis(2,1)      # initial state

    #
    # Hamiltonian
    #
    sx = sigmax()
    sz = sigmaz()
    sm = destroy(2)

    H0 = - delta/2.0 * sx - eps0/2.0 * sz
    H1 = - A * sx
        
    # define the time-dependence of the hamiltonian using the list-string format
    args = {'w':  w}    
    Ht = [H0, [H1, 'sin(w*t)']]

    #
    # collapse operators
    #
    c_op_list = []
    
    rate = gamma1 * (1 + n_th)
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm)       # relaxation
    
    rate = gamma1 * n_th
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm.dag()) # excitation

    #
    # evolve and system subject to the time-dependent hamiltonian
    #
    tlist = linspace(0, 5.0 * 2 * pi / A, 500)
    output1 = mesolve(Ht, psi0, tlist, c_op_list, [sm.dag() * sm], args)  

    # Alternative: write the hamiltonian in a rotating frame, and neglect the
    # the high frequency component (rotating wave approximation), so that the
    # resulting Hamiltonian is time-independent.
    H_rwa = - delta/2.0 * sx - A * sx / 2
    output2 = mesolve(H_rwa, psi0, tlist, c_op_list, [sm.dag() * sm])  

    #
    # Plot the solution
    #
    plot(tlist, real(output1.expect[0]), 'b', tlist, real(output2.expect[0]), 'r')
    xlabel('Time')
    ylabel('Occupation probability')
    title('Excitation probabilty of qubit')
    legend(("Time-dependent Hamiltonian", "Corresponding RWA"))
    show()

if __name__=='__main__':
    run()

