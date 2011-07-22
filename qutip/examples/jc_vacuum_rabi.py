from ..states import *
from ..Qobj import *
from ..tensor import *
from ..ptrace import *
from ..operators import *
from ..expect import *
from ..correlation import *
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from termpause import termpause


#
# run the example
#
def jc_vacuum_rabi():

    print "== Illustrates the vacuum Rabi oscillations in the Jaynes-Cumming model =="

    # --------------------------------------------------------------------------
    termpause()
    print("""
    #
    # Configure parameters
    #
    wc = 1.0  * 2 * pi  # cavity frequency
    wa = 1.0  * 2 * pi  # atom frequency
    g  = 0.05 * 2 * pi  # coupling strength
    kappa = 0.005       # cavity dissipation rate
    gamma = 0.05        # atom dissipation rate    
    N = 5               # number of cavity fock states
    """)
    wc = 1.0  * 2 * pi  # cavity frequency
    wa = 1.0  * 2 * pi  # atom frequency
    g  = 0.05 * 2 * pi  # coupling strength
    kappa = 0.005       # cavity dissipation rate
    gamma = 0.05        # atom dissipation rate    
    N = 5               # number of cavity fock states
    
    # --------------------------------------------------------------------------
    termpause()
    print("""
    # intial state
    psi0 = tensor(basis(N,0), basis(2,1))    # start with an excited atom 

    # Hamiltonian
    a  = tensor(destroy(N), qeye(2))
    sm = tensor(qeye(N), destroy(2))
    H = wc * a.dag() * a + wa * sm.dag() * sm + g * (a.dag() * sm + a * sm.dag())
    """)
    # intial state
    psi0 = tensor(basis(N,0), basis(2,1))    # start with an excited atom 

    # Hamiltonian
    a  = tensor(destroy(N), qeye(2))
    sm = tensor(qeye(N), destroy(2))
    H = wc * a.dag() * a + wa * sm.dag() * sm + g * (a.dag() * sm + a * sm.dag())
        
    # --------------------------------------------------------------------------
    termpause()
    print("""
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
    """)
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

    # --------------------------------------------------------------------------
    termpause()
    print("""
    # evolve and calculate expectation values
    tlist = linspace(0,25,100)
    nc, na = odesolve(H, psi0, tlist, c_op_list, [a.dag() * a, sm.dag() * sm])  
    """)
    # evolve and calculate expectation values
    tlist = linspace(0,25,100)
    nc, na = odesolve(H, psi0, tlist, c_op_list, [a.dag() * a, sm.dag() * sm])  

    # --------------------------------------------------------------------------
    termpause()
    print("""
    #
    # plot the results
    #
    plot(tlist, nc)
    plot(tlist, na)
    legend(("Cavity", "Atom excited state"))
    xlabel('Time')
    ylabel('Occupation probability')
    title('Vacuum Rabi oscillations')
    show()
    """)
    plot(tlist, nc)
    plot(tlist, na)
    legend(("Cavity", "Atom excited state"))
    xlabel('Time')
    ylabel('Occupation probability')
    title('Vacuum Rabi oscillations')
    show()


if __name__=='main()':
    jc_vacuum_rabi()

