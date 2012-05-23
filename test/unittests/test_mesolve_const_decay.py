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
    return sqrt(kappa)

#percent error for failure
mc_error=5e-2
me_error=1e-2

class TestMESolverConstDecay(unittest.TestCase):

    """
    A test class for the time-dependent ode check function.
    """

    def setUp(self):
        """
        setup
        """
    def testMESimpleConstDecay(self):
        N=10 #number of basis states to consider
        a=destroy(N)
        H=a.dag()*a
        psi0=basis(N,9) #initial state
        kappa=0.2 #coupling to oscillator
        c_op_list=[sqrt(kappa)*a]
        tlist=linspace(0,10,100)
        medata=mesolve(H,psi0,tlist,c_op_list,[a.dag()*a])
        expt=medata.expect[0]
        actual_answer=9.0*exp(-kappa*tlist)
        avg_diff=mean(abs(actual_answer-expt)/actual_answer)
        print avg_diff
        self.assertTrue(avg_diff<me_error)
    
    def testMCSimpleConstDecay(self):
        print ""
        N=10 #number of basis states to consider
        a=destroy(N)
        H=a.dag()*a
        psi0=basis(N,9) #initial state
        kappa=0.2 #coupling to oscillator
        c_op_list=[sqrt(kappa)*a]
        tlist=linspace(0,10,100)
        mcdata=mcsolve(H,psi0,tlist,c_op_list,[a.dag()*a],options=Odeoptions(gui=False))
        expt=mcdata.expect[0]
        actual_answer=9.0*exp(-kappa*tlist)
        avg_diff=mean(abs(actual_answer-expt)/actual_answer)
        self.assertTrue(avg_diff<mc_error)
     
    def testMESimpleConstDecayAsFuncList(self):
        N=10 #number of basis states to consider
        a=destroy(N)
        H=a.dag()*a
        psi0=basis(N,9) #initial state
        kappa=0.2 #coupling to oscillator
        c_op_list=[[a,sqrt_kappa]]
        tlist=linspace(0,10,100)
        medata=mesolve(H,psi0,tlist,c_op_list,[a.dag()*a])
        expt=medata.expect[0]
        actual_answer=9.0*exp(-kappa*tlist)
        avg_diff=mean(abs(actual_answer-expt)/actual_answer)
        self.assertTrue(avg_diff<me_error)
    # 
    def testMCSimpleConstDecayAsFuncList(self):
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
        actual_answer=9.0*exp(-kappa*tlist)
        avg_diff=mean(abs(actual_answer-expt)/actual_answer)
        self.assertTrue(avg_diff<mc_error)
    # 
    def testMESimpleConstDecayAsStrList(self):
        N=10 #number of basis states to consider
        a=destroy(N)
        H=a.dag()*a
        psi0=basis(N,9) #initial state
        kappa=0.2 #coupling to oscillator
        c_op_list=[[a,'sqrt(k)']]
        args={'k':kappa}
        tlist=linspace(0,10,100)
        medata=mesolve(H,psi0,tlist,c_op_list,[a.dag()*a],args=args)
        expt=medata.expect[0]
        actual_answer=9.0*exp(-kappa*tlist)
        avg_diff=mean(abs(actual_answer-expt)/actual_answer)
        self.assertTrue(avg_diff<me_error)
        
    # 
    def testMCSimpleConstDecayAsStrList(self):
        print ''
        N=10 #number of basis states to consider
        a=destroy(N)
        H=a.dag()*a
        psi0=basis(N,9) #initial state
        kappa=0.2 #coupling to oscillator
        c_op_list=[[a,'sqrt(k)']]
        args={'k':kappa}
        tlist=linspace(0,10,100)
        mcdata=mcsolve(H,psi0,tlist,c_op_list,[a.dag()*a],args=args,options=Odeoptions(gui=False))
        expt=mcdata.expect[0]
        actual_answer=9.0*exp(-kappa*tlist)
        avg_diff=mean(abs(actual_answer-expt)/actual_answer)
        self.assertTrue(avg_diff<mc_error)
    
      
if __name__ == '__main__':
    unittest.main()
