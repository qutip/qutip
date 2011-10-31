from qutip import *
import matplotlib.pyplot as plt
#number of states for each mode
N0=6
N1=6
N2=6
#damping rates
gamma0=0.1
gamma1=0.4
gamma2=0.1
alpha=sqrt(2)#initial coherent state param for mode 0
tlist=linspace(0,4,200)
ntraj=250#number of trajectories
#define operators
a0=tensor(destroy(N0),qeye(N1),qeye(N2))
a1=tensor(qeye(N0),destroy(N1),qeye(N2))
a2=tensor(qeye(N0),qeye(N1),destroy(N2))
#number operators for each mode
num0=a0.dag()*a0
num1=a1.dag()*a1
num2=a2.dag()*a2
#dissipative operators for zero-temp. baths
C0=sqrt(2.0*gamma0)*a0
C1=sqrt(2.0*gamma1)*a1
C2=sqrt(2.0*gamma2)*a2
#initial state: coherent mode 0 & vacuum for modes #1 & #2
psi0=tensor(coherent(N0,alpha),basis(N1,0),basis(N2,0))
#trilinear Hamiltonian
H=1j*(a0*a1.dag()*a2.dag()-a0.dag()*a1*a2)

#run Monte-Carlo with three return values
avg,times,which=mcsolve(H,psi0,tlist,ntraj,[C1,C2],[num1,num2]) #<--- important line!

fig = plt.figure(figsize=[6,4])
ax = fig.add_subplot(111)
cs=['b','r']

for k in xrange(ntraj):
    if len(times[k])>0:
        colors=[cs[j] for j in which[k]]
        xdat=[k for x in xrange(len(times[k]))]
        ax.scatter(xdat,times[k],marker='o',c=colors)

ax.set_xlim([-1,ntraj+1])
ax.set_ylim([0,tlist[-1]])
ax.set_xlabel('Trajectory')
ax.set_ylabel('Collpase Time')
ax.set_title('Blue = C1, Red = C2')
plt.savefig('examples-collapsetimesmonte.png')
close(fig)