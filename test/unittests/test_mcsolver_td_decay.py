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

from qutip import *
from qutip.odechecks import _ode_checks
from numpy import allclose
import unittest

kappa=0.2
def sqrt_kappa(t,args):
    return sqrt(kappa*exp(-t))

#average error for failure
mc_error=5e-2 #5% for ntraj=500

class TestODESolversTDDecay(unittest.TestCase):

    """
    A test class for the time-dependent odes.  Comparing to analytic answer
    
    N(t)=9 * exp[ -kappa*( 1-exp(-t) ) ]
    
    """

    def setUp(self):
        """
        setup
        """
        
    def testMCSimpleTDDecayAsFuncList(self):
        print ""
        N=10 #number of basis states to consider
        a=destroy(N)
        H=a.dag()*a
        psi0=basis(N,9) #initial state
        kappa=0.2 #coupling to oscillator
        c_op_list=[[a,sqrt_kappa]]
        tlist=linspace(0,10,100)
        mcdata=mcsolve(H,psi0,tlist,c_op_list,[a.dag()*a],options=Odeoptions(gui=False))
        expt=mcdata.expect[0]
        actual_answer=9.0*exp(-kappa*(1.0-exp(-tlist)))
        avg_diff=mean(abs(actual_answer-expt)/actual_answer)
        self.assertTrue(avg_diff<mc_error)
    
    
    def testMCSimpleTDDecayAsStrList(self):
        print ""
        N=10 #number of basis states to consider
        a=destroy(N)
        H=a.dag()*a
        psi0=basis(N,9) #initial state
        kappa=0.2 #coupling to oscillator
        c_op_list=[[a,'sqrt(k*exp(-t))']]
        args={'k':kappa}
        tlist=linspace(0,10,100)
        mcdata=mcsolve(H,psi0,tlist,c_op_list,[a.dag()*a],args=args,options=Odeoptions(gui=False))
        expt=mcdata.expect[0]
        actual_answer=9.0*exp(-kappa*(1.0-exp(-tlist)))
        avg_diff=mean(abs(actual_answer-expt)/actual_answer)
        self.assertTrue(avg_diff<mc_error)
    
    
    
      
if __name__ == '__main__':
    unittest.main()
