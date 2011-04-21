
# xschcat dispalyes the Wigner and Q functions for a Schrodinger cat state comprised of coherent states
#
# Derived from xschcat.m from the Quantum Optics toolbox by Sze M. Tanfrom 
#from scipy import *
from qutip import *
from pylab import *

N = 20;
#amplitudes of coherent states
alpha1=-2-1j
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
xvec = arange(-40.,40.)*5./40
yvec=xvec
W=wigner(psi,xvec,yvec)
pcolor(xvec,yvec,real(W))
xlim([-5,5])
ylim([-5,5])
title('Wigner function of Schrodinger cat')
colorbar()
show()

#calculate Q function
Q=qfunc(psi,xvec,yvec)
pcolor(xvec,yvec,real(Q))
xlim([-5,5])
ylim([-5,5])
title('Q function of Schrodinger cat')
colorbar()
show()
