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
##
#Class of options for ODE solvers.
#
class Odeoptions():
    """
    @brief Class of options for ODE solver used by 'odesolve' and 'mcsolve'
    """
    def __init__(self):
        ##Absolute tolerance (default = 1e-8)
        self.atol=1e-8
        ##Relative tolerance (default = 1e-6)
        self.rtol=1e-6
        ##Integration method (default = 'adams', for stiff 'bdf')
        self.method='adams'
        ##Max. number of internal steps/call
        self.nsteps=1000
        ##Size of initial step (0 = determined by solver)
        self.first_step=0
        ##Minimal step size (0 = determined by solver)
        self.min_step=0
        ##Max step size (0 = determined by solver)
        self.max_step=0
        ##Maximum order used by integrator (<=12 for 'adams', <=5 for 'bdf')
        self.order=12
        ## tidyup Hamiltonian before calculation (default = True)
        self.tidy=True
    def __str__(self):
        print "Odeoptions properties:"
        print "----------------------"
        print 'atol:       ',self.atol
        print 'rtol:       ',self.rtol
        print 'method:     ',self.method
        print 'order:      ',self.order
        print 'nsteps:     ',self.nsteps
        print 'first_step: ',self.first_step
        print 'min_step:   ',self.min_step
        print 'max_step:   ',self.max_step
        return ''


if __name__ == "__main__":
    print Odeoptions()
