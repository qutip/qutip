#
# Textbook example: Groundstate properties of an 
# ultra-strongly coupled atom-cavity system.
# 
#
from qutip.expect import *
from qutip.Qobj import *
from qutip.operators import *
from qutip.states import *
from qutip.tensor import *
from qutip.wigner import *
import time
from pylab import *
from mpl_toolkits.mplot3d import Axes3D

def compute(N, wc, wa, glist, use_rwa):

    # Pre-compute operators for the hamiltonian
    a  = tensor(destroy(N), qeye(2))
    sm = tensor(qeye(N), destroy(2))
    nc = a.dag() * a
    na = sm.dag() * sm
        
    idx = 0
    na_expt = zeros(len(glist))
    nc_expt = zeros(len(glist))
    grnd_kets= zeros(len(glist),dtype=object)
    for g in glist:

        # recalculate the hamiltonian for each value of g in glist
        # use non-RWA Hamiltonian 
        if use_rwa: 
            H = wc * nc + wa * na + g * (a.dag() * sm + a * sm.dag())
        else:
            H = wc * nc + wa * na + g * (a.dag() + a) * (sm + sm.dag())

        # find the groundstate of the composite system
        gval,grndstate=H.groundstate()
        #ground state expectation values for qubit and cavity occupation number
        na_expt[idx] = expect(na, grndstate)
        nc_expt[idx] = expect(nc, grndstate)
        grnd_kets[idx]=grndstate
        idx += 1

    return nc_expt, na_expt,grnd_kets
    

def run():
    #
    # set up the calculation
    #
    wc = 1.0 * 2 * pi   # cavity frequency
    wa = 1.0 * 2 * pi   # atom frequency
    N = 25              # number of cavity fock states
    use_rwa = False     # Set to True to see that non-RWA is necessary
    
    # array of coupling strengths to calcualate ground state
    glist = linspace(0, 2.5, 50) * 2 * pi

    #run computation
    start_time = time.time()
    nc, na, grnd_kets = compute(N, wc, wa, glist, use_rwa)
    print('time elapsed = ' +str(time.time() - start_time))

    #
    # plot the cavity and atom occupation numbers as a function of 
    # coupling strength
    figure(1)
    plot(glist/(2*pi), nc,lw=2)
    plot(glist/(2*pi), na,lw=2)
    legend(("Cavity", "Atom excited state"),loc=0)
    xlabel('Coupling strength (g)')
    ylabel('Occupation Number')
    title('# of Photons in the Groundstate')
    show()
    
    #partial trace over qubit
    rho_cavity=ptrace(grnd_kets[-1],0)
    
    #calculate Wigner function for cavity mode
    xvec = linspace(-7.5,7.5,150)
    X,Y = meshgrid(xvec, xvec)
    W = wigner(rho_cavity, xvec, xvec)
    
    #plot Wigner function
    fig=figure()
    ax = Axes3D(fig, azim=-61, elev=43)
    surf=ax.plot_surface(X,Y,W,rstride=1,cstride=1,cmap=cm.jet,linewidth=0.1,vmax=0.15,vmin=-0.05)
    title("Wigner Function for the Cavity Ground State at g= "+str(1./(2*pi)*glist[-1]))
    ax.set_xlabel('Position')
    ax.set_ylabel('Momentum')
    show()


if __name__ == "__main__":
    run()

