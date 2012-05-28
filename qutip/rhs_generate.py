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
from cyQ.codegen import Codegen
import odeconfig,os,platform,numpy
from _reset import _reset_odeconfig
from qutip.Odeoptions import Odeoptions
from scipy import ndarray, array
from qutip.odechecks import _ode_checks
from qutip.mcsolve import _mc_data_config
import qutip.settings
def rhs_generate(H,psi0,tlist,c_ops,e_ops,ntraj=500,args={},options=Odeoptions(),solver='me',name=None):
    """
    Used to generate the Cython functions for solving the dynamics of a
    given system before using the parfor function.  
    
    Parameters
    ----------
    H : qobj/list
        Hamiltonian for system.
    
    Other Parameters
    ----------------
    
    """
    _reset_odeconfig() #clear odeconfig
    #------------------------
    # GENERATE MCSOLVER DATA
    #------------------------
    if solver=='mc':
        odeconfig.tlist=tlist
        if isinstance(ntraj,(list,ndarray)):
            odeconfig.ntraj=sort(ntraj)[-1]
        else:
            odeconfig.ntraj=ntraj
        #check for type of time-dependence (if any)
        time_type,h_stuff,c_stuff=_ode_checks(H,c_ops,'mc')
        h_terms=len(h_stuff[0])+len(h_stuff[1])+len(h_stuff[2])
        c_terms=len(c_stuff[0])+len(c_stuff[1])+len(c_stuff[2])
        #set time_type for use in multiprocessing
        odeconfig.tflag=time_type
        #check for collapse operators
        if c_terms>0:
            odeconfig.cflag=1
        else:
            odeconfig.cflag=0
        #Configure data
        _mc_data_config(H,psi0,h_stuff,c_ops,c_stuff,args,e_ops,options)
        os.environ['CFLAGS'] = '-w'
        import pyximport
        pyximport.install(setup_args={'include_dirs':[numpy.get_include()]})
        if odeconfig.tflag in array([1,11]):
            code = compile('from '+odeconfig.tdname+' import cyq_td_ode_rhs,col_spmv,col_expect', '<string>', 'exec')
            exec(code)
            odeconfig.tdfunc=cyq_td_ode_rhs
            odeconfig.colspmv=col_spmv
            odeconfig.colexpect=col_expect
        else:
            code = compile('from '+odeconfig.tdname+' import cyq_td_ode_rhs', '<string>', 'exec')
            exec(code)
            odeconfig.tdfunc=cyq_td_ode_rhs
        try:
            os.remove(odeconfig.tdname+".pyx")
        except:
            print("Error removing pyx file.  File not found.")
        
    #------------------------
    # GENERATE MESOLVER STUFF
    #------------------------
    elif solver=='me':
        
        odeconfig.tdname="rhs"+str(odeconfig.cgen_num)
        cgen=Codegen(h_terms=n_L_terms,h_tdterms=Lcoeff, args=args)
        cgen.generate(odeconfig.tdname+".pyx")
        os.environ['CFLAGS'] = '-O3 -w'
        import pyximport
        pyximport.install(setup_args={'include_dirs':[numpy.get_include()]})
        code = compile('from '+odeconfig.tdname+' import cyq_td_ode_rhs', '<string>', 'exec')
        exec(code)
        odeconfig.tdfunc=cyq_td_ode_rhs
            
            
            
            
            
            
            
            
            
            
            
            
            