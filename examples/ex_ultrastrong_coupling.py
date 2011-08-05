#
# Textbook example: Rabi oscillation in the dissipative Jaynes-Cummings model.
# 
#
from qutip import *
from pylab import *
import time
from mpl_toolkits.mplot3d import Axes3D

def jc_solve(N, wc, wa, glist, kappa, gamma, psi0, use_rwa):

    # Hamiltonian
    idc = qeye(N)
    ida = qeye(2)

    a  = tensor(destroy(N), ida)
    sm = tensor(idc, destroy(2))
    nc = a.dag() * a
    na = sm.dag() * sm
        
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

    idx = 0

    na_expt = zeros(shape(glist))
    nc_expt = zeros(shape(glist))
    for g in glist:

        # recalculate the hamiltonian for each value of g
        if use_rwa: 
            H = wc * nc + wa * na + g * (a.dag() * sm + a * sm.dag())
        else:
            H = wc * nc + wa * na + g * (a.dag() + a) * (sm + sm.dag())

        # find the steady state of the composite system
        rhoss = steadystate(H, c_op_list)

        na_expt[idx] = expect(na, rhoss)
        nc_expt[idx] = expect(nc, rhoss)

        idx += 1

    return nc_expt, na_expt, rhoss
    
#
# set up the calculation
#
wc = 1.0 * 2 * pi   # cavity frequency
wa = 1.0 * 2 * pi   # atom frequency

kappa = 0.25        # cavity dissipation rate
gamma = 0.25        # atom dissipation rate

N = 10              # number of cavity fock states

use_rwa = False

# intial state
psi0 = tensor(basis(N,0),    basis(2,0))    # start with an excited atom 

glist = linspace(0,1.5,50) * 2 * pi # coupling strength

start_time = time.time()
nc, na, rhoss_final = jc_solve(N, wc, wa, glist, kappa, gamma, psi0, use_rwa)
print 'time elapsed = ' +str(time.time() - start_time) 


#
# plot the cavity and atom occupation numbers as a function of 
#
figure(1)
plot(glist/(2*pi), nc)
plot(glist/(2*pi), na)
legend(("Cavity", "Atom excited state"))
xlabel('g - coupling strength')
ylabel('Occupation probability')
title('# photons in the ground state')
savefig("ultra-strong-coupling-occupation-numbers.png")

#
# plot the cavity wigner function for the cavity state (final coupling strenght)
#
fig = plt.figure(2, figsize=(9, 6))
rho_cavity = ptrace(rhoss_final, 0)
xvec = linspace(-5.,5.,100)
X,Y = meshgrid(xvec, xvec)
W = wigner(rho_cavity, xvec, xvec)
ax = Axes3D(fig, azim=-107, elev=49)
surf=ax.plot_surface(X, Y, W, rstride=1, cstride=1, cmap=cm.jet, alpha=1.0, linewidth=0.05, vmax=0.25, vmin=-0.25)
ax.set_xlim3d(-5, 5)
ax.set_ylim3d(-5, 5)
#ax.set_zlim3d(-0.25, 0.25)
fig.colorbar(surf, shrink=0.65, aspect=20)
savefig("ultra-strong-coupling-wigner.png")
show()

