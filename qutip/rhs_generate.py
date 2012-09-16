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
from qutip.cyQ.codegen import Codegen
import os,platform,numpy
from qutip._reset import _reset_odeconfig
from qutip.Odeoptions import Odeoptions
from scipy import ndarray, array
from qutip.odechecks import _ode_checks
import qutip.settings
import qutip.odeconfig as odeconfig
from types import FunctionType
from qutip.Qobj import Qobj
from superoperator import spre,spost

def rhs_generate(H,c_ops,args={},options=Odeoptions(),name=None):
    """
    Generates the Cython functions needed for solving the dynamics of a
    given system using the mesolve function inside a parfor loop.  
    
    Parameters
    ----------
    H : qobj
        System Hamiltonian.
    c_ops : list
        ``list`` of collapse operators.
    args : dict
        Arguments for time-dependent Hamiltonian and collapse operator terms.
    
    Other Parameters
    ----------------
    options : Odeoptions
        Instance of ODE solver options.
    name: str
        Name of generated RHS
    
    Note
    ----
    Using this function with any solver other than the mesolve function
    will result in an error.
    
    """
    _reset_odeconfig() #clear odeconfig
    if name:
        odeconfig.tdname=name
    else:
        odeconfig.tdname="rhs"+str(odeconfig.cgen_num)
    
    n_op = len(c_ops)

    Lconst = 0        

    Ldata = []
    Linds = []
    Lptrs = []
    Lcoeff = []
    
    # loop over all hamiltonian terms, convert to superoperator form and 
    # add the data of sparse matrix represenation to 
    for h_spec in H:
        if isinstance(h_spec, Qobj):
            h = h_spec
            Lconst += -1j*(spre(h) - spost(h)) 
        
        elif isinstance(h_spec, list): 
            h = h_spec[0]
            h_coeff = h_spec[1]

            L = -1j*(spre(h) - spost(h))

            Ldata.append(L.data.data)
            Linds.append(L.data.indices)
            Lptrs.append(L.data.indptr)
            Lcoeff.append(h_coeff)
            
        else:
            raise TypeError("Incorrect specification of time-dependent " + 
                             "Hamiltonian (expected string format)")
    
    # loop over all collapse operators        
    for c_spec in c_ops:
        if isinstance(c_spec, Qobj):
            c = c_spec
            cdc = c.dag() * c
            Lconst += spre(c) * spost(c.dag()) - 0.5 * spre(cdc) - 0.5 * spost(cdc) 

        elif isinstance(c_spec, list): 
            c = c_spec[0]
            c_coeff = c_spec[1]

            cdc = c.dag() * c
            L = spre(c) * spost(c.dag()) - 0.5 * spre(cdc) - 0.5 * spost(cdc) 

            Ldata.append(L.data.data)
            Linds.append(L.data.indices)
            Lptrs.append(L.data.indptr)
            Lcoeff.append("("+c_coeff+")**2")

        else:
            raise TypeError("Incorrect specification of time-dependent " + 
                             "collapse operators (expected string format)")

     # add the constant part of the lagrangian
    if Lconst != 0:
        Ldata.append(Lconst.data.data)
        Linds.append(Lconst.data.indices)
        Lptrs.append(Lconst.data.indptr)
        Lcoeff.append("1.0")


    # the total number of liouvillian terms (hamiltonian terms + collapse operators)      
    n_L_terms = len(Ldata)
    
    cgen=Codegen(h_terms=n_L_terms,h_tdterms=Lcoeff, args=args)
    cgen.generate(odeconfig.tdname+".pyx")
    os.environ['CFLAGS'] = '-O3 -w'
    import pyximport
    pyximport.install(setup_args={'include_dirs':[numpy.get_include()]})
    code = compile('from '+odeconfig.tdname+' import cyq_td_ode_rhs', '<string>', 'exec')
    exec(code)
    odeconfig.tdfunc=cyq_td_ode_rhs
    try:
        os.remove(odeconfig.tdname+".pyx")
    except:
        pass
            
            
            
            
            
            
            
            
            
            
            
            
