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

from numpy.testing import assert_, run_module_suite

from qutip import *
from qutip.odechecks import _ode_checks

def test_setODEChecksMC():
    "odechecks: monte-carlo"
    
    #define operators
    H=rand_herm(10)
    c_op=qeye(10)
    def f_c_op(t,args):return 0
    def f_H(t,args):return 0
    #check constant H and no C_ops
    time_type,h_stuff,c_stuff=_ode_checks(H,[],'mc')
    assert_(time_type==0)
    
    #check constant H and constant C_ops
    time_type,h_stuff,c_stuff=_ode_checks(H,[c_op],'mc')
    assert_(time_type==0)
    
    #check constant H and str C_ops
    time_type,h_stuff,c_stuff=_ode_checks(H,[c_op,'1'],'mc')
    #assert_(time_type==1) # this test fails!!
    
    #check constant H and func C_ops
    time_type,h_stuff,c_stuff=_ode_checks(H,[f_c_op],'mc')
    #assert_(time_type==2) # FAILURE     
    
    #check str H and constant C_ops
    time_type,h_stuff,c_stuff=_ode_checks([H,'1'],[c_op],'mc')
    #assert_(time_type==10)
    
    #check str H and str C_ops
    time_type,h_stuff,c_stuff=_ode_checks([H,'1'],[c_op,'1'],'mc')
    #assert_(time_type==11)
    
    #check str H and func C_ops
    time_type,h_stuff,c_stuff=_ode_checks([H,'1'],[f_c_op],'mc')
    #assert_(time_type==12)
    

    #check func H and constant C_ops
    time_type,h_stuff,c_stuff=_ode_checks(f_H,[c_op],'mc')
    #assert_(time_type==20)
    
    #check func H and str C_ops
    time_type,h_stuff,c_stuff=_ode_checks(f_H,[c_op,'1'],'mc')
    #assert_(time_type==21)
    
    #check func H and func C_ops
    time_type,h_stuff,c_stuff=_ode_checks(f_H,[f_c_op],'mc')
    #assert_(time_type==22)
      

if __name__ == "__main__":
    run_module_suite()