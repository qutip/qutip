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
"""
This module resets the global properties in qutip.settings and the odeconfig parameters.
"""

def _reset():
    import os
    import qutip.settings
    qutip.settings.qutip_graphics=os.environ['QUTIP_GRAPHICS']
    qutip.settings.qutip_gui=os.environ['QUTIP_GUI']
    qutip.settings.auto_tidyup=True
    qutip.settings.auto_tidyup_atol=1e-15
    qutip.settings.num_cpus=int(os.environ['NUM_THREADS'])


def _reset_odeconfig():
    import qutip.odeconfig as odeconfig
    #flags for setting time-dependence, collapse ops, and number of times codegen has been run
    odeconfig.cflag=0             #Flag signaling collapse operators
    odeconfig.tflag=0             #Flag signaling time-dependent problem
    odeconfig.cgen_num=0          #Number of times codegen function has been called in current Python session.

    #time-dependent function stuff
    odeconfig.tdfunc=None         #Placeholder for time-dependent RHS function.
    odeconfig.colspmv=None        #Placeholder for time-dependent col-spmv function.
    odeconfig.colexpect=None      #Placeholder for time-dependent col_expect function.
    odeconfig.string=None         #Holds string of variables to be passed onto time-depdendent ODE solver.
    odeconfig.tdname=None         #Name of td .pyx file (used in parallel mc code)

    #Initial state stuff
    odeconfig.psi0=None   
    odeconfig.psi0_dims=None 
    odeconfig.psi0_shape=None 
    #Hamiltonian stuff
    odeconfig.h_td_inds=[]        #indicies of time-dependent Hamiltonian operators
    odeconfig.h_data=None          #List of sparse matrix data
    odeconfig.h_inds=None          #List of sparse matrix indices
    odeconfig.h_ptrs=None          #List of sparse matrix ptrs

    #Expectation operator stuff
    odeconfig.e_num=0             #number of expect ops
    odeconfig.e_ops_data=[]       #expect op data
    odeconfig.e_ops_ind=[]        #expect op indices
    odeconfig.e_ops_ptr=[]        #expect op indptrs
    odeconfig.e_ops_isherm=[]     #expect op isherm

    #Collapse operator stuff
    odeconfig.c_num=0             #number of collapse ops
    odeconfig.c_const_inds=[]     #indicies of constant collapse operators
    odeconfig.c_td_inds=[]        #indicies of time-dependent collapse operators
    odeconfig.c_ops_data=[]       #collapse op data
    odeconfig.c_ops_ind=[]        #collapse op indices
    odeconfig.c_ops_ptr=[]        #collapse op indptrs

    #Norm collapse operator stuff
    odeconfig.n_ops_data=[]       #norm collapse op data
    odeconfig.n_ops_ind=[]        #norm collapse op indices
    odeconfig.n_ops_ptr=[]        #norm collapse op indptrs
    
    #executable string stuff
    odeconfig.col_expect_code=None
    odeconfig.col_spmv_code=None
    
    #python function stuff
    odeconfig.c_funcs=None
    odeconfig.c_func_args=None