# This file is part of QuTIP.
#
#    QuTIP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    QuTIP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTIP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011-2013, Paul D. Nation & Robert J. Johansson
#
###########################################################################

import inspect

class Odeconfig():

    def __init__(self):

        self.cgen_num = 0

        self.reset()

    def reset(self):

        # General stuff
        self.tlist = None       # evaluations times
        self.ntraj = None       # number / list of trajectories
        self.options = None     # options for odesolvers
        self.norm_tol = None    # tolerance for wavefunction norm
        self.norm_steps = None  # max. number of steps to take in finding wavefunction
                                # norm within tolerance norm_tol.
        # Initial state stuff
        self.psi0 = None        # initial state
        self.psi0_dims = None   # initial state dims
        self.psi0_shape = None  # initial state shape

        # flags for setting time-dependence, collapse ops, and number of times
        # codegen has been run
        self.cflag = 0     # Flag signaling collapse operators
        self.tflag = 0     # Flag signaling time-dependent problem
 
        # time-dependent function stuff
        self.tdfunc = None     # Placeholder for time-dependent RHS function.
        self.tdname = None     # Name of td .pyx file (used in parallel mc code)
        self.colspmv = None    # Placeholder for time-dependent col-spmv function.
        self.colexpect = None  # Placeholder for time-dependent col_expect function.
        self.string = None     # Holds string of variables to be passed onto
                               # time-depdendent ODE solver.

        self.soft_reset()
    
    def soft_reset(self):


        # Hamiltonian stuff
        self.h_td_inds = []  # indicies of time-dependent Hamiltonian operators
        self.h_data = None   # List of sparse matrix data
        self.h_ind = None    # List of sparse matrix indices
        self.h_ptr = None    # List of sparse matrix ptrs

        # Expectation operator stuff
        self.e_num = 0        # number of expect ops
        self.e_ops_data = []  # expect op data
        self.e_ops_ind = []   # expect op indices
        self.e_ops_ptr = []   # expect op indptrs
        self.e_ops_isherm = []  # expect op isherm

        # Collapse operator stuff
        self.c_num = 0          # number of collapse ops
        self.c_const_inds = []  # indicies of constant collapse operators
        self.c_td_inds = []     # indicies of time-dependent collapse operators
        self.c_ops_data = []    # collapse op data
        self.c_ops_ind = []     # collapse op indices
        self.c_ops_ptr = []     # collapse op indptrs
        self.c_args = []        # store args for time-dependent collapse functions

        # Norm collapse operator stuff
        self.n_ops_data = []  # norm collapse op data
        self.n_ops_ind = []   # norm collapse op indices
        self.n_ops_ptr = []   # norm collapse op indptrs

        # holds executable strings for time-dependent collapse evaluation
        self.col_expect_code = None
        self.col_spmv_code = None


        # hold stuff for function list based time dependence
        self.h_td_inds = []
        self.h_td_data = []
        self.h_td_ind = []
        self.h_td_ptr = []
        self.h_funcs = None
        self.h_func_args = None
        self.c_funcs = None
        self.c_func_args = None   

# 
# create a global instance of the Odeconfig class
#
odeconfig = Odeconfig()

