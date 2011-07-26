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
def td_rabi():

    print "== Illustrates the Rabi oscillations due to a time-dependent driving field =="

    # --------------------------------------------------------------------------
    termpause()
    print("""
    #
    # Define a function that describes the time-dependence of the Hamiltonian
    #
    def hamiltonian_t(t, args):
        H0 = args[0]
        H1 = args[1]
        w  = args[2]

        return H0 + H1 * sin(w * t)
    """)
    def hamiltonian_t(t, args):
        H0 = args[0]
        H1 = args[1]
        w  = args[2]

        return H0 + H1 * sin(w * t)

   
    # --------------------------------------------------------------------------
    termpause()
    print("""
    #
    # set up the calculation
    #
    delta = 0.0 * 2 * pi   # qubit sigma_x coefficient
    eps0  = 1.0 * 2 * pi   # qubit sigma_z coefficient
    A     = 0.25 * 2 * pi  # driving amplitude (reduce to make the RWA more accurate)
    w     = 1.0 * 2 * pi   # driving frequency
    gamma1 = 0.0           # relaxation rate
    n_th   = 0.0           # average number of excitations ("temperature")
    psi0 = basis(2,1)      # initial state
    """)
    delta = 0.0 * 2 * pi   # qubit sigma_x coefficient
    eps0  = 1.0 * 2 * pi   # qubit sigma_z coefficient
    A     = 0.25 * 2 * pi  # driving amplitude (reduce to make the RWA more accurate)
    w     = 1.0 * 2 * pi   # driving frequency
    gamma1 = 0.0           # relaxation rate
    n_th   = 0.0           # average number of excitations ("temperature")
    psi0 = basis(2,1)      # initial state

    # --------------------------------------------------------------------------
    termpause()
    print("""
    #
    # Hamiltonian
    #
    sx = sigmax()
    sz = sigmaz()
    sm = destroy(2)

    H0 = - delta/2.0 * sx - eps0/2.0 * sz
    H1 = - A * sx
        
    H_args = (H0, H1, w)
    """)
    sx = sigmax()
    sz = sigmaz()
    sm = destroy(2)

    H0 = - delta/2.0 * sx - eps0/2.0 * sz
    H1 = - A * sx
        
    H_args = (H0, H1, w)

    # --------------------------------------------------------------------------
    termpause()
    print("""
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
    """)
    c_op_list = []
    
    rate = gamma1 * (1 + n_th)
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm)       # relaxation
    
    rate = gamma1 * n_th
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm.dag()) # excitation

    # --------------------------------------------------------------------------
    termpause()
    print("""
    #
    # evolve and system subject to the time-dependent hamiltonian
    #
    tlist = linspace(0, 5.0 * 2 * pi / A, 500)
    expt_list1 = odesolve(hamiltonian_t, psi0, tlist, c_op_list, [sm.dag() * sm], H_args)  

    # Alternative: write the hamiltonian in a rotating frame, and neglect the
    # the high frequency component (rotating wave approximation), so that the
    # resulting Hamiltonian is time-independent.
    H_rwa = - delta/2.0 * sx - A * sx / 2
    expt_list2 = odesolve(H_rwa, psi0, tlist, c_op_list, [sm.dag() * sm])  
    """)
    tlist = linspace(0, 5.0 * 2 * pi / A, 500)
    expt_list1 = odesolve(hamiltonian_t, psi0, tlist, c_op_list, [sm.dag() * sm], H_args)  
    H_rwa = - delta/2.0 * sx - A * sx / 2
    expt_list2 = odesolve(H_rwa, psi0, tlist, c_op_list, [sm.dag() * sm])  

    # --------------------------------------------------------------------------
    termpause()
    print("""
    #
    # Plot the solution
    #
    plot(tlist, real(expt_list1[0]), 'b', tlist, real(expt_list2[0]), 'r.')
    xlabel('Time')
    ylabel('Occupation probability')
    title('Excitation probabilty of qubit')
    legend(("Time-dependent Hamiltonian", "Corresponding RWA"))
    show()
    """)
    plot(tlist, real(expt_list1[0]), 'b', tlist, real(expt_list2[0]), 'r.')
    xlabel('Time')
    ylabel('Occupation probability')
    title('Excitation probabilty of qubit')
    legend(("Time-dependent Hamiltonian", "Corresponding RWA"))
    show()


if __name__=='main()':
    td_rabi()

