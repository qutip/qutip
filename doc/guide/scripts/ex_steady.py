from qutip import *
from pylab import *


# Define paramters
N=20 #number of basis states to consider
a=destroy(N)
H=a.dag()*a
psi0=basis(N,10) #initial state
kappa=0.1 #coupling to oscillator

# collapse operators
c_op_list = []
n_th_a = 2 # temperature with average of 2 excitations
rate = kappa * (1 + n_th_a)
if rate > 0.0:
    c_op_list.append(sqrt(rate) * a) #decay operators
rate = kappa * n_th_a
if rate > 0.0:
    c_op_list.append(sqrt(rate) * a.dag()) #excitation operators

#find steady-state solution
final_state=steadystate(H,c_op_list)
#find expectation value for particle number in steady state
fexpt=expect(a.dag()*a,final_state)

tlist=linspace(0,50,100)
#monte-carlo
mcdata = mcsolve(H,psi0,tlist,c_op_list, [a.dag()*a],ntraj=100)
#master eq.
medata = mesolve(H,psi0,tlist,c_op_list, [a.dag()*a])

plot(tlist,mcdata.expect[0],tlist,medata.expect[0],lw=2)
#plot steady-state expt. value as horizontal line (should be = 2)
axhline(y=fexpt,color='r',lw=1.5)
ylim([0,10])
xlabel('Time',fontsize=14)
ylabel('Number of excitations',fontsize=14)
legend(('Monte-Carlo','Master Equation','Steady State'))
title('Decay of Fock state $\left|10\\rangle\\right.$'+ 
    ' in a thermal environment with $\langle n\\rangle=2$')
show()