from qutip import *
from pylab import *
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#number of states for each mode
N0=6
N1=6
N2=6

#define operators
a0=tensor(destroy(N0),qeye(N1),qeye(N2))
a1=tensor(qeye(N0),destroy(N1),qeye(N2))
a2=tensor(qeye(N0),qeye(N1),destroy(N2))

#number operators for each mode
num0=a0.dag()*a0
num1=a1.dag()*a1
num2=a2.dag()*a2

#initial state: coherent mode 0 & vacuum for modes #1 & #2
alpha=sqrt(2)#initial coherent state param for mode 0
initial=tensor(coherent(N0,alpha),basis(N1,0),basis(N2,0))
psi0=initial

#trilinear Hamiltonian
H=1.0j*(a0*a1.dag()*a2.dag()-a0.dag()*a1*a2)

#run Monte-Carlo
tlist=linspace(0,2.5,50)
states=mcsolve(H,psi0,tlist,1,[],[])

mode1=[ptrace(k,1) for k in states]
diags1=[real(k.diag()) for k in mode1]
num1=[expect(num1,k) for k in states]
thermal=[thermal_dm(N1,k).diag() for k in num1]

colors=['m', 'g','orange','b', 'y','pink']
x=range(N1)
#set plotting parameters
params = {'axes.labelsize': 14,'text.fontsize': 14,'legend.fontsize': 12,'xtick.labelsize': 14,'ytick.labelsize': 14}
rcParams.update(params)
fig = plt.figure(figsize=(6, 4))
ax = Axes3D(fig)
for j in range(5):
    ax.bar(x, diags1[10*j], zs=tlist[10*j], zdir='y',color=colors[j],linewidth=1.0,alpha=0.6,align='center')
    ax.plot(x,thermal[10*j],zs=tlist[10*j],zdir='y',color='r',linewidth=3,alpha=1)
ax.set_zlabel(r'Probability')
ax.set_xlabel(r'Number State')
ax.set_ylabel(r'Time')
ax.set_zlim3d(0,1)
savefig('examples-thermalmonte.png')
close(fig)



