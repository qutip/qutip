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

def sqrt_kappa2(t,args):
    return sqrt(kappa*exp(-t))

def const_H1_coeff(t,args):
    return 0.0

#average error for failure
mc_error=5e-2 #5% for ntraj=500

try:
    import Cython
except:
    Cython_found=0
else:
    Cython_found=1


class TestMCSolverConstDecay(unittest.TestCase):

    """
    Test the mcsolver for the decy of a fock state |9>
    into a zero-temp bath.
    """

    def setUp(self):
        """
        setup
        """
    
    
    def testMCNoCollExpt(self):
        """
        Constant H with no collapse ops (returns expect)
        """
        print ""
        error=1e-8
        N=10 #number of basis states to consider
        a=destroy(N)
        H=a.dag()*a
        psi0=basis(N,9) #initial state
        kappa=0.2 #coupling to oscillator
        c_op_list=[]
        tlist=linspace(0,10,100)
        mcdata=mcsolve(H,psi0,tlist,c_op_list,[a.dag()*a],options=Odeoptions(gui=False))
        expt=mcdata.expect[0]
        actual_answer=9.0*ones(len(tlist))
        diff=mean(abs(actual_answer-expt)/actual_answer)
        self.assertTrue(diff<error)
    
    def testMCNoCollStates(self):
        """
        Constant H with no collapse ops (returns states)
        """
        print ""
        error=1e-8
        N=10 #number of basis states to consider
        a=destroy(N)
        H=a.dag()*a
        psi0=basis(N,9) #initial state
        kappa=0.2 #coupling to oscillator
        c_op_list=[]
        tlist=linspace(0,10,100)
        mcdata=mcsolve(H,psi0,tlist,c_op_list,[],options=Odeoptions(gui=False))
        states=mcdata.states
        expt=expect(a.dag()*a,states)
        actual_answer=9.0*ones(len(tlist))
        diff=mean(abs(actual_answer-expt)/actual_answer)
        self.assertTrue(diff<error)
    
    
    def testMCNoCollStrExpt(self):
        """
        Constant H but str based (args=0) with no collapse ops (returns expect)
        """
        print ""
        error=1e-8
        N=10 #number of basis states to consider
        a=destroy(N)
        H=[a.dag()*a,[a.dag()*a,'a']]
        psi0=basis(N,9) #initial state
        kappa=0.2 #coupling to oscillator
        c_op_list=[]
        tlist=linspace(0,10,100)
        mcdata=mcsolve(H,psi0,tlist,c_op_list,[a.dag()*a],args={'a':0.0},options=Odeoptions(gui=False))
        expt=mcdata.expect[0]
        actual_answer=9.0*ones(len(tlist))
        diff=mean(abs(actual_answer-expt)/actual_answer)
        self.assertTrue(diff<error)
    
    
    
    def testMCSimpleConst(self):
        """
        Collapse terms are constants.
        """
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
     
    def testMCSimpleConstFunc(self):
        """
        Collapse terms are constant, but written in time-dependent
        function format with a constant coefficient (should yield same result).
        """
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
        
    @unittest.skipIf(Cython.__version__ < 0.14 or Cython_found==0,"Cython module not found")
    def testMCSimpleConstStr(self):
        """
        Collapse terms are constant, but written in time-dependent
        string format with a constant coefficient (should yield same result).
        """
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
    
    
    def testTDFunc(self):
        """
        Comparing to analytic answer

        N(t)=9 * exp[ -kappa*( 1-exp(-t) ) ]
        """
        print ""
        error=5e-2
        N=10 #number of basis states to consider
        a=destroy(N)
        H=a.dag()*a
        psi0=basis(N,9) #initial state
        kappa=0.2 #coupling to oscillator
        c_op_list=[[a,sqrt_kappa2]]
        tlist=linspace(0,10,100)
        mcdata=mcsolve(H,psi0,tlist,c_op_list,[a.dag()*a],options=Odeoptions(gui=False))
        expt=mcdata.expect[0]
        actual_answer=9.0*exp(-kappa*(1.0-exp(-tlist)))
        diff=mean(abs(actual_answer-expt)/actual_answer)
        self.assertTrue(diff<error)
        
    
    @unittest.skipIf(Cython.__version__ < 0.14 or Cython_found==0,"Cython module not found")
    def testTDStr(self):
        """
        Comparing to analytic answer

        N(t)=9 * exp[ -kappa*( 1-exp(-t) ) ]
        """
        print ""
        error=5e-2
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
        actual=9.0*exp(-kappa*(1.0-exp(-tlist)))
        diff=mean(abs(actual-expt)/actual)
        self.assertTrue(diff<error)
      
if __name__ == '__main__':
    unittest.main()
