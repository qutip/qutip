#
# Single-atom lasing in a Jaynes-Cumming-like system
#
from qutip.Qobj import *
from qutip.tensor import *
from qutip.ptrace import *
from qutip.operators import *
from qutip.expect import *
from qutip.wigner import *
from qutip.odesolve import mesolve
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def run():

    # Configure parameters
    N = 12          # number of cavity fock states
    wc = 2*pi*1.0   # cavity frequency
    wa = 2*pi*1.0   # atom frequency
    g  = 2*pi*0.1   # coupling strength
    kappa = 0.05    # cavity dissipation rate
    gamma = 0.0     # atom dissipation rate
    pump  = 0.4     # atom pump rate
    use_rwa = True
    
    # start without any excitations
    psi0  = tensor(basis(N,0), basis(2,0))

    # Hamiltonian
    a  = tensor(destroy(N), qeye(2))
    sm = tensor(qeye(N), destroy(2))
    
    if use_rwa: # use the rotating wave approxiation
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

    rate = pump
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm.dag())

    # evolve the system
    tlist = linspace(0, 200, 500)
    rho_list = mesolve(H, psi0, tlist, c_op_list, [])  

    # calculate expectation values
    nc = expect(a.dag()  *  a, rho_list) 
    na = expect(sm.dag() * sm, rho_list)

    #
    # plot the time-evolution of the cavity and atom occupation
    #
    figure(1)
    plot(tlist, real(nc), 'r-',   tlist, real(na), 'b-')
    xlabel('Time');
    ylabel('Occupation probability');
    legend(("Cavity occupation", "Atom occupation"))

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

    #
    # plot the wigner function
    #
    xvec = linspace(-5, 5, 100)
    W = wigner(rho_cavity, xvec, xvec)
    X,Y = meshgrid(xvec, xvec)
    figure(3)
    contourf(X, Y, W, 100)
    colorbar()
    show()

if __name__=='__main__':
    run()

