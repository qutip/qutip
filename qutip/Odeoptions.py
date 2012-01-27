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
##
#Class of options for ODE solvers.
#
import os
class Odeoptions():
    """
    Class of options for ODE solver used by 'odesolve' and 'mcsolve'
    """
    def __init__(self):
        #: Absolute tolerance (default = 1e-8)
        self.atol=1e-8
        #: Relative tolerance (default = 1e-6)
        self.rtol=1e-6
        #: Integration method (default = 'adams', for stiff 'bdf')
        self.method='adams'
        #: Max. number of internal steps/call
        self.nsteps=1000
        #: Size of initial step (0 = determined by solver)
        self.first_step=0
        #: Minimal step size (0 = determined by solver)
        self.min_step=0
        #: Max step size (0 = determined by solver)
        self.max_step=0
        #: Maximum order used by integrator (<=12 for 'adams', <=5 for 'bdf')
        self.order=12
        #: Average expectation values over trajectories (default = True) 
        self.expect_avg=True
        #: tidyup Hamiltonian before calculation (default = True)
        self.tidy=True
        #: Number of processors to use (mcsolve only)
        self.num_cpus=int(os.environ['NUM_THREADS'])
        #: Use preexisting RHS function for time-dependent solvers
        self.rhs_reuse=False
        #: Use filename for preexisting RHS function (will default to last compiled function if None & rhs_exists=True)
        self.rhs_filename=None
        #: Use Progressbar (mcsolve only)
        if os.environ["QUTIP_GUI"]=="NONE" or os.environ["QUTIP_GRAPHICS"]=="NO":
            self.gui=False
        else:
            self.gui=True
    def __str__(self):
        print("Odeoptions properties:")
        print("----------------------")
        print("atol:         ",self.atol)
        print('rtol:         ',self.rtol)
        print('method:       ',self.method)
        print('order:        ',self.order)
        print('nsteps:       ',self.nsteps)
        print('first_step:   ',self.first_step)
        print('min_step:     ',self.min_step)
        print('max_step:     ',self.max_step)
        print('tidy:         ',self.tidy)
        print('expect_avg:   ',self.expect_avg)
        print('num_cpus:     ',self.num_cpus)
        print('rhs_reuse:    ',self.rhs_reuse)
        print('rhs_filename: ',self.rhs_filename)
        return ''

