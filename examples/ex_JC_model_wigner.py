#
# Textbook example: Rabi oscillation in the dissipative Jaynes-Cummings model.
# 
#
import matplotlib
matplotlib.use('AGG')
from qutip import *
#import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from pylab import *


def jc_integrate(N, wc, wa, g, kappa, gamma, psi0, use_rwa, tlist):

    # Hamiltonian
    idc = qeye(N)
    ida = qeye(2)

    a  = tensor(destroy(N), ida)
    sm = tensor(idc, destroy(2))

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
    output = mesolve(H, psi0, tlist, c_op_list, [])  


    xvec = linspace(-5.,5.,100)
    X,Y = meshgrid(xvec, xvec)

    i = 0

    if not os.path.exists("jc_animation"):
        os.mkdir("jc_animation")

    for wf in output.states:

        # trace out the atom
        rho_cavity = ptrace(wf, 0)

        W = wigner(rho_cavity, xvec, xvec)
        
        fig = plt.figure(figsize=(9, 6))
        ax = Axes3D(fig, azim=-107, elev=49)
        surf=ax.plot_surface(X, Y, W, rstride=1, cstride=1, cmap=cm.jet, alpha=1.0, linewidth=0.05, vmax=0.25, vmin=-0.25)
        ax.set_xlim3d(-5, 5)
        ax.set_ylim3d(-5, 5)
        ax.set_zlim3d(-0.25, 0.25)
        fig.colorbar(surf, shrink=0.65, aspect=20)
        savefig("jc_animation/jc_model_wigner_%.3d.png" % i)       
        close(fig)
        print i
        i = i + 1

    return 0
    
#
# set up the calculation
#
wc = 1.0 * 2 * pi   # cavity frequency
wa = 1.0 * 2 * pi   # atom frequency
g  = 0.05 * 2 * pi  # coupling strength

kappa = 0.05       # cavity dissipation rate
gamma = 0.15        # atom dissipation rate

N = 10               # number of cavity fock states

use_rwa = True

# intial state
psi0 = tensor(basis(N,0),    basis(2,1))    # start with an excited atom 
#psi0 = tensor(coherent(N,1.5), basis(2,0))   # or a coherent state the in cavity
#psi0 = tensor((coherent(N,2.0)+coherent(N,-2.0)).unit(), basis(2,0))   # or a superposition of coherent states the in cavity

tlist = linspace(0,40,500)

start_time = time.time()
jc_integrate(N, wc, wa, g, kappa, gamma, psi0, use_rwa, tlist)
print 'time elapsed = ' +str(time.time() - start_time) 


