#
# Dynamics of the Wigner distributions for the Jaynes-Cummings model
#
from qutip import *
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def run():

    # Configure parameters
    N = 10          # number of cavity fock states
    wc = 2*pi*1.0   # cavity frequency
    wa = 2*pi*1.0   # atom frequency
    g  = 2*pi*0.1   # coupling strength
    kappa = 0.05    # cavity dissipation rate
    gamma = 0.15    # atom dissipation rate
    use_rwa = True
    
    # a coherent initial state the in cavity
    psi0 = tensor(coherent(N,1.5), basis(2,0))   

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

    # evolve the system
    tlist = linspace(0, 10, 100)
    output = mesolve(H, psi0, tlist, c_op_list, [])  

    # calculate the wigner function  
    xvec = linspace(-5.,5.,100)
    X,Y = meshgrid(xvec, xvec)

    #for idx, rho in enumerate(output.states): # suggestion: try to loop over all rho
    for idx, rho in enumerate([output.states[44]]): # for a selected time t=4.4
           
        rho_cavity = ptrace(rho, 0)
        W = wigner(rho_cavity, xvec, xvec)
    
        # plot the wigner function
        fig = figure(figsize=(9, 6))
        ax = Axes3D(fig, azim=-107, elev=49)
        ax.set_xlim3d(-5, 5)
        ax.set_ylim3d(-5, 5)
        ax.set_zlim3d(-0.30, 0.30)
        surf=ax.plot_surface(X, Y, W, rstride=1, cstride=1, cmap=cm.jet, alpha=1.0, linewidth=0.05, vmax=0.4, vmin=-0.4)
        fig.colorbar(surf, shrink=0.65, aspect=20)
        #savefig("jc_model_wigner_"+str(idx)+".png")
        
    show()

if __name__=='__main__':
    run()

