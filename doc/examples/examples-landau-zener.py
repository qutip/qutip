#
# Landau-Zener transitions in a quantum two-level system.
#
from qutip import *
from pylab import *
import time

def hamiltonian_t(t, args):
    """ evaluate the hamiltonian at time t. """
    H0 = args[0]
    H1 = args[1]

    return H0 + t * H1

def qubit_integrate(delta, eps0, A, gamma1, gamma2, psi0, tlist):

    # Hamiltonian
    sx = sigmax()
    sz = sigmaz()
    sm = destroy(2)

    H0 = - delta/2.0 * sx - eps0/2.0 * sz
    H1 = - A/2.0 * sz
        
    H_args = (H0, H1)

    # collapse operators
    c_op_list = []

    n_th = 0.0 # zero temperature

    # relaxation
    rate = gamma1 * (1 + n_th)
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm)

    # excitation
    rate = gamma1 * n_th
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm.dag())

    # dephasing 
    rate = gamma2
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sz)

    # evolve and calculate expectation values
    expt_list = odesolve(hamiltonian_t, psi0, tlist, c_op_list, [sm.dag() * sm], H_args)  

    return expt_list[0]
    
#
# set up the calculation
#
delta = 0.5 * 2 * pi   # qubit sigma_x coefficient
eps0  = 0.0 * 2 * pi   # qubit sigma_z coefficient
A     = 2.0 * 2 * pi   # sweep rate
gamma1 = 0.0           # relaxation rate
gamma2 = 0.0           # dephasing  rate
psi0 = basis(2,0)      # initial state

tlist = linspace(-10.0, 10.0, 1500)

start_time = time.time()
p_ex = qubit_integrate(delta, eps0, A, gamma1, gamma2, psi0, tlist)
print 'time elapsed = ' + str(time.time() - start_time) 

plot(tlist, real(p_ex), 'b', tlist, real(1-p_ex), 'r')
plot(tlist, 1 - exp( - pi * delta **2 / (2 * A)) * ones(shape(tlist)), 'k')
xlabel('Time')
ylabel('Occupation probability')
title('Landau-Zener transition')
legend(("Excited state", "Ground state", "Landau-Zener formula"), loc=0)
show()
#
# Plot LZ transition dynamics on Bloch sphere with time labeled by color
#
step=1
b=Bloch()
nrm=mpl.colors.Normalize(-2,10)
colors=cm.jet(nrm(tlist[0:-1:step]))
b.add_points([p_ex[1][0:-1:step],p_ex[2][0:-1:step],-p_ex[3][0:-1:step]],'m')
b.point_color=list(colors)
b.point_marker=['o']
b.point_size=[8]
b.view=[-9,11]
b.zlpos=[1.05,-1.2]
b.zlabel=['$\left|0\\right>$','$\left|1\\right>$']
b.sphere_alpha=0.1
b.show()
#
# Create colorbar corresponding to Bloch colors
#
fig = figure(figsize=(7,3))
ax1 = fig.add_axes([0.05, 0.5, 0.7, 0.05])
cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cm.jet,norm=nrm,orientation='horizontal')
cb1.set_label('Time',fontsize=16)
for t in cb1.ax.get_yticklabels():
     t.set_fontsize(16)
show()

