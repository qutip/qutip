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

# ------------------------------------------------------------------------------
# Appendix B.6
#
from qutip import *
## callback function for time-dependence ## 
def hamiltonian_t(t, args):
    H0 = args[0]
    H1 = args[1]
    return H0 + t * H1
delta = 0.5 * 2 * pi 
v = 2.0 * 2 * pi # sweep rate
## arguments for Hamiltonian ## 
H0 = delta/2.0 * sigmax()
H1 = v/2.0 * sigmaz()
H_args = (H0, H1)
psi0 = basis(2,0)
## expectation operators ## 
sm = destroy(2)
sx=sigmax();sy=sigmay();sz=sigmaz()
expt_op_list = [sm.dag() * sm, sx, sy, sz]
## evolve the system ## 
tlist = linspace(-10.0, 10.0, 5000)
expt_list = odesolve(hamiltonian_t, psi0, tlist, 
                     [], expt_op_list, H_args)  



# ------------------------------------------------------------------------------
# Plot the results (omitted from the code listing in the appendix in the paper)
#
from pylab import *

fig=figure(figsize=[6,4])
ax=subplot(111)
ax.plot(tlist, expt_list[0], 'r', tlist, 1-expt_list[0], 'b',lw=2)
ax.plot(tlist, 1 - exp( - pi * delta **2 / (2 * v)) * ones(shape(tlist)), 'k',lw=1.5)
xlabel(r'Time',fontsize=12)
ylabel(r'Occupation probability',fontsize=12)
ylim([-0.1,1.1])
#title(r'Landau-Zener transition')
ax.legend((r"Ground state $\left|1\right>$", r"Excited state $\left|0\right>$", r"Landau-Zener"), loc=0)

for tick in ax.yaxis.get_ticklabels()+ax.xaxis.get_ticklabels():
    tick.set_fontsize(12)
savefig('examples-paperfig10_11_1.png')
show()

# ------------------------------------------------------------------------------
# Appendix B.7
#
b=Bloch()
## normalize colors to times in tlist ## 
nrm=mpl.colors.Normalize(-2,10)
colors=cm.jet(nrm(tlist))
## add data points from expectation values ## 
b.add_points([expt_list[1], expt_list[2], -expt_list[3]],'m')
## customize sphere properties ## 
b.point_color=list(colors)
b.point_marker=['o']
b.point_size=[8]
b.view=[-9,11]
b.zlpos=[1.1,-1.2]
b.size=[4,4]
b.zlabel=['$\left|0\\right>_{f}$','$\left|1\\right>_{f}$']
b.font_size=16
## plot sphere ## 
b.save('examples-paperfig10_11_2.png')
b.show()

fig = figure(figsize=[4,2])
ax1 = fig.add_axes([0.05, 0.5, 0.7, 0.05])
cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cm.jet,norm=nrm,orientation='horizontal')
cb1.set_label('Time',fontsize=12)
for t in cb1.ax.get_yticklabels():
     t.set_fontsize(12)
savefig('examples-paperfig10_11_3.png')
show()
