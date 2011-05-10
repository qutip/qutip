from qutip import *
from pylab import *

N = 20;
#amplitudes of coherent states
alpha1=-2.0-2j
alpha2=2.0+2j
#define ladder oeprators
a = destroy(N);
#define displacement oeprators
D1=(alpha1*dag(a)-conj(alpha1)*a).expm()
D2=(alpha2*dag(a)-conj(alpha2)*a).expm()
#sum of coherent states
psi = sqrt(2)**-1*(D1+D2)*basis(N,0); # Apply to vacuum state
#calculate Wigner function
xvec = arange(-100.,100.)*5./100
yvec=xvec
g=2.
W=wigner(psi,xvec,yvec)
plt=contourf(xvec,yvec,real(W),100)
xlim([-5,5])
ylim([-5,5])
title('Wigner function of Schrodinger cat')
cbar=colorbar(plt)#create colorbar
cbar.ax.set_ylabel('Pseudoprobability')#set colorbar label
show()
#calculate Q function
Q=qfunc(psi,xvec,yvec)
qplt=contourf(xvec,yvec,real(Q),100)
xlim([-5,5])
ylim([-5,5])
title('Q function of Schrodinger cat')
cbar=colorbar(qplt)
cbar.ax.set_ylabel('Probability')
show()
