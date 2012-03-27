#
# Vacuum Rabi oscillations in the Jaynes-Cummings model with dissipation
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

    # Configure parameters
    wc = 1.0  * 2 * pi  # cavity frequency
    wa = 1.0  * 2 * pi  # atom frequency
    g  = 0.05 * 2 * pi  # coupling strength
    kappa = 0.005       # cavity dissipation rate
    gamma = 0.05        # atom dissipation rate    
    N = 5               # number of cavity fock states
    use_rwa = True
    
    # intial state
    psi0 = tensor(basis(N,0), basis(2,1))    # start with an excited atom 

    # Hamiltonian
    a  = tensor(destroy(N), qeye(2))
    sm = tensor(qeye(N), destroy(2))
    
    if use_rwa: 
        # use the rotating wave approxiation
        H = wc * a.dag() * a + wa * sm.dag() * sm + g * (a.dag() * sm + a * sm.dag())
    else:
        H = wc * a.dag() * a + wa * sm.dag() * sm + g * (a.dag() + a) * (sm + sm.dag())
                    
    # collapse operators
    c_op_list = []

    n_th_a = 0.0 # zero temperature
    rate = kappa * (1 + n_th_a)
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * a)

    rate = kappa * n_th_a
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * a.dag())

    rate = gamma
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm)

    # evolve and calculate expectation values
    tlist = linspace(0,25,100)
    nc, na = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a, sm.dag() * sm])  

    # plot the results
    plot(tlist, nc)
    plot(tlist, na)
    legend(("Cavity", "Atom excited state"))
    xlabel('Time')
    ylabel('Occupation probability')
    title('Vacuum Rabi oscillations')
    show()

if __name__=='main()':
    run()

