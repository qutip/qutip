# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without 
#    modification, are permitted provided that the following conditions are 
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice, 
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT 
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY 
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################


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
        self.norm_steps = None  # max. number of steps to take in finding
                                # wavefunction norm within tolerance norm_tol.
        # Initial state stuff
        self.psi0 = None        # initial state
        self.psi0_dims = None   # initial state dims
        self.psi0_shape = None  # initial state shape

        # flags for setting time-dependence, collapse ops, and number of times
        # codegen has been run
        self.cflag = 0     # Flag signaling collapse operators
        self.tflag = 0     # Flag signaling time-dependent problem

        # time-dependent (TD) function stuff
        self.tdfunc = None     # Placeholder for TD RHS function.
        self.tdname = None     # Name of td .pyx file
        self.colspmv = None    # Placeholder for TD col-spmv function.
        self.colexpect = None  # Placeholder for TD col_expect function.
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
        self.c_args = []        # store args for time-dependent collapse
                                # functions

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
