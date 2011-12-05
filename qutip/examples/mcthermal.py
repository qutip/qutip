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
# Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson
#
###########################################################################
from ..states import *
from ..Qobj import *
from ..tensor import *
from ..ptrace import *
from ..operators import *
from ..expect import *
from ..mcsolve import *
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from .termpause import termpause
def mcthermal():
    print('-'*80)
    print('Highlights the deviation from a thermal')
    print('spectrum seen in the trilinear Hamiltonian')
    print('driven by a harmonic oscillator in a')
    print('coherent state.')
    print('-'*80)
    termpause()
    
    print('#number of states for each mode')
    print('N0=6')
    print('N1=6')
    print('N2=6')
    #number of states for each mode
    N0=6
    N1=6
    N2=6

    print('')
    print('#define operators')
    print('a0=tensor(destroy(N0),qeye(N1),qeye(N2))')
    print('a1=tensor(qeye(N0),destroy(N1),qeye(N2))')
    print('a2=tensor(qeye(N0),qeye(N1),destroy(N2))')
    #define operators
    a0=tensor(destroy(N0),qeye(N1),qeye(N2))
    a1=tensor(qeye(N0),destroy(N1),qeye(N2))
    a2=tensor(qeye(N0),qeye(N1),destroy(N2))

    print('')
    print('#number operators for each mode')
    print('num0=a0.dag()*a0')
    print('num1=a1.dag()*a1')
    print('num2=a2.dag()*a2')
    #number operators for each mode
    num0=a0.dag()*a0
    num1=a1.dag()*a1
    num2=a2.dag()*a2

    print('')
    print('#initial state: coherent mode 0 & vacuum for modes #1 & #2')
    print('alpha=sqrt(2)#initial coherent state param for mode 0')
    print('psi0=tensor(coherent(N0,alpha),basis(N1,0),basis(N2,0))')
    #initial state: coherent mode 0 & vacuum for modes #1 & #2
    alpha=sqrt(2)#initial coherent state param for mode 0
    psi0=tensor(coherent(N0,alpha),basis(N1,0),basis(N2,0))

    print('')
    print('#trilinear Hamiltonian')
    print('H=1.0j*(a0*a1.dag()*a2.dag()-a0.dag()*a1*a2)')
    #trilinear Hamiltonian
    H=1.0j*(a0*a1.dag()*a2.dag()-a0.dag()*a1*a2)

    print('')
    print('#run Monte-Carlo to get state vectors')
    print('tlist=linspace(0,2.5,50)')
    print('mcdata=mcsolve(H,psi0,tlist,1,[],[])')
    #run Monte-Carlo
    tlist=linspace(0,2.5,50)
    mcdata=mcsolve(H,psi0,tlist,1,[],[])

    print('')
    print('extract mode 1 only')
    print('mode1=[ptrace(k,1) for k in mcdata.states]')
    mode1=[ptrace(k,1) for k in mcdata.states]
    print('get diagonal elements from density matricies')
    print('diags1=[real(k.diag()) for k in mode1]')
    diags1=[real(k.diag()) for k in mode1]
    print('calculate expectation values for mode 1 particles')
    print('num1=[expect(num1,k) for k in states]')
    num1=[expect(num1,k) for k in mcdata.states]
    print('calculate thermal number state probabilites from num1')
    print('thermal=[thermal_dm(N1,k).diag() for k in num1]')
    thermal=[thermal_dm(N1,k).diag() for k in num1]

    print('')
    print('Plot results...')
    termpause()
    
    print("colors=['m', 'g','orange','b', 'y','pink']")
    print('x=range(N1)')
    print("params = {'axes.labelsize': 14,'text.fontsize': 14,'legend.fontsize': 12,'xtick.labelsize': 14,'ytick.labelsize': 14}")
    print('rcParams.update(params)')
    print('fig = plt.figure()')
    print('ax = Axes3D(fig)')
    print('for j in range(5):')
    print("    ax.bar(x, diags1[10*j], zs=tlist[10*j], zdir='y',color=colors[j],linewidth=1.0,alpha=0.6,align='center')")
    print("    ax.plot(x,thermal[10*j],zs=tlist[10*j],zdir='y',color='r',linewidth=3,alpha=1)")
    print("ax.set_zlabel(r'Probability')")
    print("ax.set_xlabel(r'Number State')")
    print("ax.set_ylabel(r'Time')")
    print('ax.set_zlim3d(0,1)')
    print('show()')
    
    colors=['m', 'g','orange','b', 'y','pink']
    x=range(N1)
    params = {'axes.labelsize': 14,'text.fontsize': 14,'legend.fontsize': 12,'xtick.labelsize': 14,'ytick.labelsize': 14}
    rcParams.update(params)
    fig = plt.figure()
    ax = Axes3D(fig)
    for j in range(5):
        ax.bar(x, diags1[10*j], zs=tlist[10*j], zdir='y',color=colors[j],linewidth=1.0,alpha=0.6,align='center')
        ax.plot(x,thermal[10*j],zs=tlist[10*j],zdir='y',color='r',linewidth=3,alpha=1)
    ax.set_zlabel(r'Probability')
    ax.set_xlabel(r'Number State')
    ax.set_ylabel(r'Time')
    ax.set_zlim3d(0,1)
    show()
    print('')
    print('DEMO FINISHED...')

if __name__=='main()':
    mcthermal()

