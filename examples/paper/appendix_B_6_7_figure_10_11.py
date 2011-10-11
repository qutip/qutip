#This file is part of QuTIP.
#
#    QuTIP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#    QuTIP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTIP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011, Paul D. Nation & Robert J. Johansson
#
###########################################################################

"""
Python script for generating Figures 10 & 11 from the 
QuTiP manuscript.

Here we calculate the occupation of a qubit after going through
an avoided level crossing and compare to the Landau-Zener formula.
The Bloch sphere representation is also given, where the data point 
color indicates the amount of elapsed time.

"""
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'
rcParams['legend.fontsize'] = 16
from qutip import *
from pylab import *

def hamiltonian_t(t, args):
    """ evaluate the hamiltonian at time t. """
    H0 = args[0]
    H1 = args[1]

    return H0 + t * H1

def qubit_integrate(delta, eps0, A, gamma1, gamma2, psi0, tlist):

    # Hamiltonian
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()
    sm = sigmam()

    H0 = - delta/2.0 * sx + eps0/2.0 * sz
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
    expt_list = odesolve(hamiltonian_t, psi0, tlist, c_op_list, [sm.dag() * sm,sx,sy,sz], H_args)  

    return expt_list
    
#
# set up the calculation
#
delta = 0.5 * 2 * pi   # qubit sigma_x coefficient
eps0  = 0.0 * 2 * pi   # qubit sigma_z coefficient
A     = 2.0 * 2 * pi   # sweep rate
gamma1 = 0.0           # relaxation rate
gamma2 = 0.0           # dephasing  rate
psi0 = basis(2,0)      # initial state

tlist = linspace(-10, 10.0, 10000)

p_ex = qubit_integrate(delta, eps0, A, gamma1, gamma2, psi0, tlist)
fig=figure()
ax=subplot(111)
ax.plot(tlist, p_ex[0], 'r', tlist, 1-p_ex[0], 'b',lw=2)
ax.plot(tlist, 1 - exp( - pi * delta **2 / (2 * A)) * ones(shape(tlist)), 'k',lw=1.5)
xlabel(r'Time',fontsize=18)
ylabel(r'Occupation probability',fontsize=18)
ylim([-0.1,1.1])
#title(r'Landau-Zener transition')
ax.legend((r"Ground state $\left|1\right>$", r"Excited state $\left|0\right>$", r"Landau-Zener"), loc=0)
for tick in ax.yaxis.get_ticklabels()+ax.xaxis.get_ticklabels():
    tick.set_fontsize(16)
show()
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

fig = figure(figsize=(7,3))
ax1 = fig.add_axes([0.05, 0.5, 0.7, 0.05])
cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cm.jet,norm=nrm,orientation='horizontal')
cb1.set_label('Time',fontsize=16)
for t in cb1.ax.get_yticklabels():
     t.set_fontsize(16)
show()




