#
# Test of QuTiP dynamics for cases with exact analytical solution
#
from qutip import *
from pylab import *
import time
import math

def jc_steadystate(N, wc, wa, g, kappa, gamma, pump, psi0, use_rwa, tlist):

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
    #if rate > 0.0:
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


    # find the steady state
    rho_ss = steadystate(H, c_op_list)

    return expect(a.dag() * a, rho_ss), expect(sm.dag() * sm, rho_ss)

def jc_integrate(N, wc, wa, g, kappa, gamma, pump, psi0, use_rwa, tlist):

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

    rate = pump
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm.dag())


    # evolve, and return the density matrix at each time
    rho_list = odesolve(H, psi0, tlist, c_op_list, [])  

    # calculate expectation values
    nc_list = expect(a.dag()  *  a, rho_list) 
    na_list = expect(sm.dag() * sm, rho_list)

    return na_list, nc_list, rho_list
    
#
# 
#
print "==="
print "A single-atom lasing example"
print

use_rwa = True
N = 12          # number of cavity fock states
wc = 2*pi*1.0   # cavity frequency
wa = 2*pi*1.0   # atom frequency
g  = 2*pi*0.1   # coupling strength
kappa = 0.05    # cavity dissipation rate
gamma = 0.0     # atom dissipation rate
pump  = 0.4     # atom pump rate
psi0  = tensor(basis(N,0), basis(2,0))    # start without any excitations
tlist = linspace(0, 200, 500)

#
# evolve the system
#
start_time = time.time()
nc, na, rho_list = jc_integrate(N, wc, wa, g, kappa, gamma, pump, psi0, use_rwa, tlist)
print 'time elapsed = ' +str(time.time() - start_time) 

#
# plot the time-evolution of the cavity and atom occupation
#
figure(1)
plot(tlist, real(nc), 'r.-',   tlist, real(na), 'b.-')
xlabel('Time');
ylabel('Occupation probability');
legend(("Cavity occupation", "Atom occupation"))
show()

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

#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
#fig =figure(figsize=(9, 6))
#ax = Axes3D(fig,azim=-30,elev=73)
#surf=ax.plot_surface(X, Y, W, rstride=1, cstride=1, cmap=cm.copper, alpha=1,linewidth=0.1)
#ax.set_zlim3d(-0.2,0.2)
##remove z-axis tick labels
#for a in ax.w_zaxis.get_ticklines()+ax.w_zaxis.get_ticklabels():
#    a.set_visible(False)
#fig.colorbar(surf,shrink=0.65,aspect=20)
#show()


