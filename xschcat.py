
# xschcat dispalyes the Wigner and Q functions for a Schrodinger cat state comprised of coherent states
#
# Derived from xschcat.m from the Quantum Optics toolbox by Sze M. Tanfrom 
#from scipy import *
from qutip import *
from pylab import *

N = 40;
#amplitudes of coherent states
alpha1=-1-1j
alpha2=1+1j
#define ladder oeprators
a = destroy(N);
#define displacement oeprators
D1=(alpha1*dag(a)-conj(alpha1)*a).expm()
D2=(alpha2*dag(a)-conj(alpha2)*a).expm()
#sum of coherent states
psi = sqrt(2)**-1*(D1+D2)*basis(N,0); # Apply to vacuum state
g=2.
#calculate Wigner function
xvec = arange(-100.,100.)*5./100
yvec=xvec
W=wigner(psi,xvec,yvec)
pcolor(xvec,yvec,real(W))
xlim([-5,5])
ylim([-5,5])
title('Wigner function of Schrodinger cat')
colorbar()



#X,Y = meshgrid(xvec, xvec)
#fig1 = plt.figure()
#ax = Axes3D(fig1)
#ax.plot_surface(X, Y, W, rstride=3, cstride=3, cmap=cm.jet, alpha=0.9)
#ax.contour(X, Y, W, 15,zdir='x', offset=-6)
#ax.contour(X, Y, W, 15,zdir='y', offset=6)
#ax.contour(X, Y, W, 15,zdir='z', offset=-0.3)
#ax.set_xlim3d(-5,5)
#ax.set_xlim3d(-5,5)
#ax.set_zlim3d(-0.2,0.2)
#title('Wigner Function of Cat-State');

show()
#calculate Q function
Q=qfunc(psi,xvec,yvec)
pcolor(xvec,yvec,real(Q))
xlim([-5,5])
ylim([-5,5])
title('Q function of Schrodinger cat')
colorbar()
show()
