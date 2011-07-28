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
from scipy import *
from ..states import *
from ..operators import *
from ..odesolve import *
from ..Bloch import *
from termpause import termpause

def blochdemo():
    print '-'*80
    print 'Plots the relaxation in a zero temperature bath'
    print 'of a qubit tilled-off the z-axis by an angle 0.2*pi.'
    print '-'*80
    from pylab import plot,show
    from matplotlib import mpl,cm #need to import colormap (cm)
    termpause()
    print '# set up the calculation'
    print 'w = 1.0 * 2 * pi # qubit angular frequency'
    print 'theta = 0.2 * pi # qubit angle from sigma_z axis (toward sigma_x axis)'
    print 'gamma1 = 0.05    # qubit relaxation rate'
    print 'gamma2 = 1.0     # qubit dephasing rate'
    
    # set up the calculation
    #
    w = 1.0 * 2 * pi # qubit angular frequency
    theta = 0.2 * pi # qubit angle from sigma_z axis (toward sigma_x axis)
    gamma1 = 0.05    # qubit relaxation rate
    gamma2 = 1.0     # qubit dephasing rate

    print ''
    print '# Hamiltonian'
    print 'sx = sigmax()'
    print 'sy = sigmay()'
    print 'sz = sigmaz()'
    print 'sm = sigmam()'
    print 'H = w * (cos(theta) * sz + sin(theta) * sx)'
    
    
    # Hamiltonian
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()
    sm = sigmam()
    H = w * (cos(theta) * sz + sin(theta) * sx)
    
    print ''
    print '# collapse operators'
    print 'c_op_list = []'
    print 'n_th = 0 # zero temperature'
    print 'rate = gamma1 * (n_th + 1)'
    print 'if rate > 0.0:'
    print '    c_op_list.append(sqrt(rate) * sm)'
    print 'rate = gamma1 * n_th'
    print 'if rate > 0.0:'
    print '    c_op_list.append(sqrt(rate) * sm.dag())'
    print 'rate = gamma2'
    print 'if rate > 0.0:'
    print '    c_op_list.append(sqrt(rate) * sz)'
    
    # collapse operators
    c_op_list = []
    n_th = 0 # zero temperature
    rate = gamma1 * (n_th + 1)
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm)
    rate = gamma1 * n_th
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm.dag())
    rate = gamma2
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sz)
    
    print ''
    print '# initial state, pointed toward |x>'
    print ' a = .5'
    print 'psi0 = (a* basis(2,0) + (1-a)*basis(2,1)).unit()'
    print 'tlist = linspace(0,3,500)'
    # initial state
    a = .5
    psi0 = (a* basis(2,0) + (1-a)*basis(2,1)).unit()
    tlist = linspace(0,3,500)
    
    print ''
    print '# evolve and calculate expectation values'
    print 'expt = odesolve(H, psi0, tlist, c_op_list, [sx, sy, sz])'
    print 'sx=expt[0];sy=expt[1];sz=expt[2]'
    # evolve and calculate expectation values
    expt = odesolve(H, psi0, tlist, c_op_list, [sx, sy, sz])
    sx=expt[0];sy=expt[1];sz=expt[2]

    print ''
    print 'Plot the Bloch sphere...'
    termpause()
    print 'sphere=Bloch()'
    print 'sphere.add_points([sx,sy,sz])'
    print "sphere.point_color=['r']"
    print "sphere.vector_color = ['b']"
    print 'sphere.add_vectors([sin(theta),0,cos(theta)])'
    print 'sphere.view=[-43,23]'
    print 'sphere.show()'
    sphere=Bloch()
    sphere.add_points([sx,sy,sz])
    sphere.point_color=['r']
    sphere.vector_color = ['b']
    sphere.add_vectors([sin(theta),0,cos(theta)])
    sphere.view=[-43,23]
    sphere.show()
    
    print ''
    print 'Color the Bloch points as a function of time...'
    termpause()
    
    print 'sphere.clear() #clear the previous data'
    print 'nrm=mpl.colors.Normalize(0,3)#normalize colors to tlist range'
    print 'colors=cm.jet(nrm(tlist)) #make list of colors, one for each time in tlist'
    print 'sphere.point_color=list(colors) #define sphere point colors'
    print "sphere.add_points([sx,sy,sz],'m')#add points as 'multi' colored points"
    print "sphere.point_marker=['o'] #make all markers same 'circle' shape"
    print 'sphere.point_size=[25] #same point sizes'
    print 'sphere.view=[-7,7] #change viewing angle to see all the colors'
    print 'sphere.zlpos=[1.1,-1.2] #reposition z-axis labels'
    print 'sphere.show()'
    sphere.clear() #clear the previous data
    nrm=mpl.colors.Normalize(0,3)#normalize colors to tlist range
    colors=cm.jet(nrm(tlist)) #make list of colors, one for each time in tlist
    sphere.point_color=list(colors) #define sphere point colors
    sphere.add_points([sx,sy,sz],'m')#add points as 'multi' colored points
    sphere.point_marker=['o'] #make all markers same 'circle' shape
    sphere.point_size=[25] #same point sizes
    sphere.view=[-7,7] #change viewing angle to see all the colors
    sphere.zlpos=[1.1,-1.2]
    sphere.show()


if __name__=='main()':
	blochdemo()




