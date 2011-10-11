# 
# Copyright (C) 2011, Paul D. Nation & Robert J. Johansson
# 
# QuTiP: An open-source Python framework for the dynamics of open quantum systems
#
# Appendix B: QuTiP code for non-RWA Jaynes-Cummings Model
#

from qutip import *
## set up the calculation ## 
wc = 1.0 * 2 * pi # cavity frequency
wa = 1.0 * 2 * pi # atom frequency
N = 20            # number of cavity states
g = linspace(0, 2.5, 50)*2*pi # coupling strength vector
## create operators ## 
a  = tensor(destroy(N), qeye(2))
sm = tensor(qeye(N), destroy(2))
nc = a.dag() * a
na = sm.dag() * sm
## initialize output arrays ##
na_expt = zeros(len(g))
nc_expt = zeros(len(g))
## run calculation ## 
for k in range(len(g)):
    ## recalculate the hamiltonian for each value of g ## 
    H = wc*nc+wa*na+g[k]*(a.dag()+a)*(sm+sm.dag())
    ## find the groundstate ## 
    ekets, evals = H.eigenstates()
    psi_gnd = ekets[0]
    ## expectation values ## 
    na_expt[k] = expect(na, psi_gnd) # qubit occupation
    nc_expt[k] = expect(nc, psi_gnd) # cavity occupation 
## Calculate Wigner function for coupling g=2.5 ## 
rho_cavity = ptrace(psi_gnd,0) # trace out qubit
xvec = linspace(-7.5,7.5,200)
## Wigner function ## 
W = wigner(rho_cavity, xvec, xvec)

# ------------------------------------------------------------------------------
# Plot the results: this code was omitted from the code listing in appendix B.
#
from pylab import *
from mpl_toolkits.mplot3d import Axes3D

# plot the cavity/atom groundstate occupation numbers as a function coupling strength
figure(1)
plot(g/(2*pi), nc_expt)
plot(g/(2*pi), na_expt)
legend(("Cavity", "Atom"))
xlabel('Coupling strength g')
ylabel('Occupation number')

# plot the cavity wigner function for the cavity state (final coupling strenght)
fig = plt.figure(2, figsize=(9, 6))
X,Y = meshgrid(xvec, xvec)
ax = Axes3D(fig, azim=-107, elev=49)
surf=ax.plot_surface(X, Y, W, rstride=1, cstride=1, cmap=cm.jet, alpha=1.0, linewidth=0.05, vmax=0.25, vmin=-0.25)
ax.set_xlim3d(-7.5, 7.5)
ax.set_ylim3d(-7.5, 7.5)
xlabel('position')
ylabel('momentum')
fig.colorbar(surf, shrink=0.65, aspect=20)

show()

