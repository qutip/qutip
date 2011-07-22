from ..states import *
from ..Qobj import *
from ..tensor import *
from ..ptrace import *
from ..operators import *
from ..expect import *
from ..wigner import *
from ..odesolve import odesolve
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from termpause import termpause


#
# run the example
#
def jc_lasing():

    print "== Illustrates single-atom lasing in the Jaynes-Cumming model with a incorently pumped atom =="

    # --------------------------------------------------------------------------
    termpause()
    print("""
    #
    # Configure parameters
    #
    N = 12          # number of cavity fock states
    wc = 2*pi*1.0   # cavity frequency
    wa = 2*pi*1.0   # atom frequency
    g  = 2*pi*0.1   # coupling strength
    kappa = 0.05    # cavity dissipation rate
    gamma = 0.0     # atom dissipation rate
    pump  = 0.4     # atom pump rate
    psi0  = tensor(basis(N,0), basis(2,0))    # start without any excitations
    tlist = linspace(0, 200, 500)
    """)
    N = 12          # number of cavity fock states
    wc = 2*pi*1.0   # cavity frequency
    wa = 2*pi*1.0   # atom frequency
    g  = 2*pi*0.1   # coupling strength
    kappa = 0.05    # cavity dissipation rate
    gamma = 0.0     # atom dissipation rate
    pump  = 0.4     # atom pump rate
    psi0  = tensor(basis(N,0), basis(2,0))    # start without any excitations
    tlist = linspace(0, 200, 500)

    # --------------------------------------------------------------------------
    termpause()
    print("""
    #
    # Hamiltonian
    #
    a  = tensor(destroy(N), qeye(2))
    sm = tensor(qeye(N), destroy(2))
    H = wc * a.dag() * a + wa * sm.dag() * sm + g * (a.dag() * sm + a * sm.dag()) 
    """)
    a  = tensor(destroy(N), qeye(2))
    sm = tensor(qeye(N), destroy(2))
    H = wc * a.dag() * a + wa * sm.dag() * sm + g * (a.dag() * sm + a * sm.dag()) 

    # --------------------------------------------------------------------------
    termpause()
    print("""
    #           
    # collapse operators
    #
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

    rate = pump
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm.dag())
    """)
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

    rate = pump
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm.dag())

    # --------------------------------------------------------------------------
    termpause()
    print("""
    #
    # evolve the system
    #
    rho_list = odesolve(H, psi0, tlist, c_op_list, [])  

    # calculate expectation values
    nc = expect(a.dag()  *  a, rho_list) 
    na = expect(sm.dag() * sm, rho_list)
    """)
    rho_list = odesolve(H, psi0, tlist, c_op_list, [])  

    # calculate expectation values
    nc = expect(a.dag()  *  a, rho_list) 
    na = expect(sm.dag() * sm, rho_list)

    # --------------------------------------------------------------------------
    termpause()
    print("""
    #
    # plot the time-evolution of the cavity and atom occupation
    #
    figure(1)
    plot(tlist, real(nc), 'r.-',   tlist, real(na), 'b.-')
    xlabel('Time');
    ylabel('Occupation probability');
    legend(("Cavity occupation", "Atom occupation"))
    show()
    """)
    figure(1)
    plot(tlist, real(nc), 'r.-',   tlist, real(na), 'b.-')
    xlabel('Time');
    ylabel('Occupation probability');
    legend(("Cavity occupation", "Atom occupation"))
    show()

    # --------------------------------------------------------------------------
    termpause()
    print("""
    #
    # plot the final photon distribution in the cavity
    #
    rho_final  = rho_list[-1]
    rho_cavity = ptrace(rho_final, 0)

    figure(2)
    bar(range(0, N), real(rho_cavity.diag()))
    xlabel("Photon number")
    ylabel("Occupation probability")
    title("Photon distribution in the cavity")
    show()
    """)
    rho_final  = rho_list[-1]
    rho_cavity = ptrace(rho_final, 0)

    figure(2)
    bar(range(0, N), real(rho_cavity.diag()))
    xlabel("Photon number")
    ylabel("Occupation probability")
    title("Photon distribution in the cavity")
    show()

    # --------------------------------------------------------------------------
    termpause()
    print("""
    #
    # plot the wigner function
    #
    xvec = linspace(-5, 5, 100)
    W = wigner(rho_cavity, xvec, xvec)
    X,Y = meshgrid(xvec, xvec)
    fig=figure(3)
    contourf(X, Y, W, 100)
    colorbar()
    show()
    """)
    xvec = linspace(-5, 5, 100)
    W = wigner(rho_cavity, xvec, xvec)
    X,Y = meshgrid(xvec, xvec)
    fig=figure(3)
    contourf(X, Y, W, 100)
    colorbar()
    show()


if __name__=='main()':
    jc_lasing()

