from qutip import *
from pylab import *
import time

kappa=2.0
gamma=0.2
g=1
wc=0
w0=0
wl=0
E=0.5
N=4
tlist=linspace(0,10,101)

# Hamiltonian
ida=qeye(N)
idatom=qeye(2)
a=tensor(destroy(N),idatom)
sm=tensor(ida,sigmam())

H=(w0-wl)*sm.dag()*sm+(wc-wl)*a.dag()*a+1j*g*(a.dag()*sm-sm.dag()*a)+E*(a.dag()+a)

#collapse operators
C1=sqrt(2*kappa)*a
C2=sqrt(gamma)*sm
C1dC1=C1.dag()*C1
C2dC2=C2.dag()*C2


#intial state
psi0=tensor(basis(N,0),basis(2,1))

Heff=H-0.5j*(C1dC1+C2dC2)

ntraj=100

start_time=time.time()
ops=mcsolve(Heff,psi0,tlist,ntraj,[C1,C2],[C1dC1,C2dC2])
finish_time=time.time()
print 'time elapsed = ',finish_time-start_time


avg=sum(ops,axis=0)/ntraj


plot(tlist,avg[0],tlist,avg[1])
show()
    


