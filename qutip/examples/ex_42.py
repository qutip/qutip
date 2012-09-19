#
# Landau-Zener transitions in a quantum two-level system
#
from qutip import *
from pylab import *

def run():

    def hamiltonian_t(t, args):
        """ evaluate the hamiltonian at time t. """
        H0 = args[0]
        H1 = args[1]
        return H0 + t * H1

    # 
    # set up the parameters
    #
    delta = 0.5 * 2 * pi   # qubit sigma_x coefficient
    eps0  = 0.0 * 2 * pi   # qubit sigma_z coefficient
    A     = 2.0 * 2 * pi   # sweep rate
    gamma1 = 0.0           # relaxation rate
    n_th = 0.0             # average number of thermal photons
    psi0 = basis(2,0)      # initial state

    # 
    # Hamiltonian
    #
    sx = sigmax()
    sz = sigmaz()
    sm = destroy(2)

    H0 = - delta/2.0 * sx - eps0/2.0 * sz
    H1 = - A/2.0 * sz   
    args = (H0, H1)

    # 
    # collapse operators, only active if gamma1 > 0
    #
    c_ops = []

    rate = gamma1 * (1 + n_th)
    if rate > 0.0:
        c_ops.append(sqrt(rate) * sm)       # relaxation

    rate = gamma1 * n_th
    if rate > 0.0:
        c_ops.append(sqrt(rate) * sm.dag()) # excitation

    # 
    # evolve and calculate expectation values
    #
    tlist = linspace(-10.0, 10.0, 1500)
    output = mesolve(hamiltonian_t, psi0, tlist, c_ops, [sm.dag() * sm], args)  
 
    # 
    # Plot the results
    #    
    plot(tlist, real(output.expect[0]), 'b', tlist, real(1-output.expect[0]), 'r')
    plot(tlist, 1 - exp( - pi * delta **2 / (2 * A)) * ones(shape(tlist)), 'k')
    xlabel('Time')
    ylabel('Occupation probability')
    title('Excitation probabilty the two-level system')
    legend(("Excited state", "Ground state", "Landau-Zener formula"), loc=0)
    show()

if __name__=='__main__':
    run()

