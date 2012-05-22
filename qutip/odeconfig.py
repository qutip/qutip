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

#flags for setting time-dependence, collapse ops, and number of times codegen has been run
cflag=0             #Flag signaling collapse operators
tflag=0             #Flag signaling time-dependent problem
cgen_num=0          #Number of times codegen function has been called in current Python session.

#time-dependent function stuff
tdfunc=None         #Placeholder for time-dependent RHS function.
colspmv=None        #Placeholder for time-dependent col-spmv function.
colexpect=None      #Placeholder for time-dependent col_expect function.
string=None         #Holds string of variables to be passed onto time-depdendent ODE solver.
tdname=None         #Name of td .pyx file (used in parallel mc code)

#Initial state stuff
psi0=None   
psi0_dims=None 
psi0_shape=None 

#Hamiltonian stuff
h_td_inds=[]        #indicies of time-dependent Hamiltonian operators
h_data=None          #List of sparse matrix data
h_ind=None          #List of sparse matrix indices
h_ptr=None          #List of sparse matrix ptrs

#Expectation operator stuff
e_num=0             #number of expect ops
e_ops_data=[]       #expect op data
e_ops_ind=[]        #expect op indices
e_ops_ptr=[]        #expect op indptrs
e_ops_isherm=[]     #expect op isherm

#Collapse operator stuff
c_num=0             #number of collapse ops
c_const_inds=[]     #indicies of constant collapse operators
c_td_inds=[]        #indicies of time-dependent collapse operators
c_ops_data=[]       #collapse op data
c_ops_ind=[]        #collapse op indices
c_ops_ptr=[]        #collapse op indptrs
c_args=[]           #store args for time-dependent collapse functions
#Norm collapse operator stuff
n_ops_data=[]       #norm collapse op data
n_ops_ind=[]        #norm collapse op indices
n_ops_ptr=[]        #norm collapse op indptrs

#holds executable strings for time-dependent collapse evaluation
col_expect_code=None
col_spmv_code=None

#hold stuff for function list based time dependence
h_td_inds=[]
h_td_data=[]
h_td_ind=[]
h_td_ptr=[]
h_funcs=None
h_func_args=None
c_funcs=None
c_func_args=None









