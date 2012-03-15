#
# Textbook example: groundstate properties of an ultra-strongly coupled atom-cavity system.
# 
#
from ..Qobj import *
from ..tensor import *
from ..expect import *
from ..operators import *
from ..states import *
from ..wigner import *
from pylab import *
import time
from mpl_toolkits.mplot3d import Axes3D

def compute(N, wc, wa, glist, use_rwa):

    # Pre-compute operators for the hamiltonian
    a  = tensor(destroy(N), qeye(2))
    sm = tensor(qeye(N), destroy(2))
    nc = a.dag() * a
    na = sm.dag() * sm
        
    idx = 0
    na_expt = zeros(shape(glist))
    nc_expt = zeros(shape(glist))
    for g in glist:

        # recalculate the hamiltonian for each value of g
        if use_rwa: 
            H = wc * nc + wa * na + g * (a.dag() * sm + a * sm.dag())
        else:
            H = wc * nc + wa * na + g * (a.dag() + a) * (sm + sm.dag())

        # find the groundstate of the composite system
        ekets, evals = H.eigenstates()
        psi_gnd = ekets[0]
        na_expt[idx] = expect(na, psi_gnd)
        nc_expt[idx] = expect(nc, psi_gnd)

        idx += 1

    return nc_expt, na_expt, ket2dm(psi_gnd)
    

def run():
    #
    # set up the calculation
    #
    wc = 1.0 * 2 * pi   # cavity frequency
    wa = 1.0 * 2 * pi   # atom frequency
    N = 20              # number of cavity fock states
    use_rwa = False     # Set to True to see that non-RWA is necessary in this regime

    glist = linspace(0, 2.5, 50) * 2 * pi # coupling strength vector

    start_time = time.time()
    nc, na, rhoss_final = compute(N, wc, wa, glist, use_rwa)
    print 'time elapsed = ' +str(time.time() - start_time) 

    #
    # plot the cavity and atom occupation numbers as a function of 
    #
    figure(1)
    plot(glist/(2*pi), nc,lw=2)
    plot(glist/(2*pi), na,lw=2)
    legend(("Cavity", "Atom excited state"),loc=0)
    xlabel('Coupling strength (g)')
    ylabel('Occupation Number')
    title('# of Photons in the Groundstate')
    show()


    #
    # plot the cavity wigner function for the cavity state (final coupling strenght)
    #
    fig = plt.figure(2, figsize=(9, 6))
    rho_cavity = ptrace(rhoss_final, 0)
    xvec = linspace(-7.5,7.5,200)
    X,Y = meshgrid(xvec, xvec)
    W = wigner(rho_cavity, xvec, xvec)
    ax = Axes3D(fig, azim=-107, elev=49)
    surf=ax.plot_surface(X, Y, W, rstride=2, cstride=2, cmap=cm.jet, alpha=1.0, linewidth=0.05, vmax=0.20, vmin=-0.1)
    ax.set_xlim3d(-7.5, 7.5)
    ax.set_ylim3d(-7.5, 7.5)
    fig.colorbar(surf, shrink=0.65, aspect=20)
    title("Wigner Function for the Cavity Groundstate (Ultra-Strong Coupling to Qubit)")
    show()




