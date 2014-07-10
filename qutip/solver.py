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

from __future__ import print_function
import os
import warnings
from qutip import __version__


class Options():
    """
    Class of options for evolution solvers such as :func:`qutip.mesolve` and
    :func:`qutip.mcsolve`. Options can be specified either as arguments to the
    constructor::

        opts = Options(order=10, ...)

    or by changing the class attributes after creation::

        opts = Options()
        opts.order = 10

    Returns options class to be used as options in evolution solvers.

    Attributes
    ----------

    atol : float {1e-8}
        Absolute tolerance.
    rtol : float {1e-6}
        Relative tolerance.
    method : str {'adams','bdf'}
        Integration method.
    order : int {12}
        Order of integrator (<=12 'adams', <=5 'bdf')
    nsteps : int {2500}
        Max. number of internal steps/call.
    first_step : float {0}
        Size of initial step (0 = automatic).
    min_step : float {0}
        Minimum step size (0 = automatic).
    max_step : float {0}
        Maximum step size (0 = automatic)
    tidy : bool {True,False}
        Tidyup Hamiltonian and initial state by removing small terms.
    num_cpus : int
        Number of cpus used by mcsolver (default = # of cpus).
    norm_tol : float
        Tolerance used when finding wavefunction norm in mcsolve.
    norm_steps : int
        Max. number of steps used to find wavefunction norm to within norm_tol
        in mcsolve.
    average_states : bool {True, False}
        Avg. expectation values in mcsolver.
    ntraj : int {500}
        Number of trajectories in stochastic solvers.
    rhs_reuse : bool {False,True}
        Reuse Hamiltonian data.
    rhs_with_state : bool {False,True}
        Whether or not to include the state in the Hamiltonian function
        callback signature.
    rhs_filename : str
        Name for compiled Cython file.
    store_final_state : bool {False, True}
        Whether or not to store the final state of the evolution in the
        result class.
    store_states : bool {False, True}
        Whether or not to store the state vectors or density matrices in the
        result class, even if expectation values operators are given. If no
        expectation are provided, then states are stored by default and this
        option has no effect.
    """
    def __init__(self, atol=1e-8, rtol=1e-6, method='adams', order=12,
                 nsteps=1000, first_step=0, max_step=0, min_step=0,
                 average_expect=True, average_states=False, tidy=True,
                 num_cpus=0, norm_tol=1e-3, norm_steps=5, rhs_reuse=False,
                 rhs_filename=None, ntraj=500, gui=False, rhs_with_state=False,
                 store_final_state=False, store_states=False, seeds=None,
                 steady_state_average=False):
        # Absolute tolerance (default = 1e-8)
        self.atol = atol
        # Relative tolerance (default = 1e-6)
        self.rtol = rtol
        # Integration method (default = 'adams', for stiff 'bdf')
        self.method = method
        # Max. number of internal steps/call
        self.nsteps = nsteps
        # Size of initial step (0 = determined by solver)
        self.first_step = first_step
        # Minimal step size (0 = determined by solver)
        self.min_step = min_step
        # Max step size (0 = determined by solver)
        self.max_step = max_step
        # Maximum order used by integrator (<=12 for 'adams', <=5 for 'bdf')
        self.order = order
        # Average expectation values over trajectories (default = True)
        self.average_states = average_states
        # average expectation values
        self.average_expect = average_expect
        # Number of trajectories (default = 500)
        self.ntraj = ntraj
        # tidyup Hamiltonian before calculation (default = True)
        self.tidy = tidy
        # include the state in the function callback signature
        self.rhs_with_state = rhs_with_state
        # Use preexisting RHS function for time-dependent solvers
        self.rhs_reuse = rhs_reuse
        # Use filename for preexisting RHS function (will default to last
        # compiled function if None & rhs_exists=True)
        self.rhs_filename = rhs_filename
        # Number of processors to use (mcsolve only)
        if num_cpus:
            self.num_cpus = num_cpus
            if self.num_cpus > int(os.environ['NUM_THREADS']):
                message = ("Requested number of threads larger than number " +
                           "of CPUs (%s)." % os.environ['NUM_THREADS'])
                warnings.warn(message)
        else:
            self.num_cpus = 0
        # Tolerance for wavefunction norm (mcsolve only)
        self.norm_tol = norm_tol
        # Max. number of steps taken to find wavefunction norm to within
        # norm_tol (mcsolve only)
        self.norm_steps = norm_steps
        # store final state?
        self.store_final_state = store_final_state
        # store states even if expectation operators are given?
        self.store_states = store_states
        # extra solver parameters
        self.seeds = seeds
        # average mcsolver density matricies assuming steady state evolution
        self.steady_state_average = steady_state_average

    def __str__(self):
        s = ""
        s += "Options:\n"
        s += "-----------\n"
        s += "atol:              " + str(self.atol) + "\n"
        s += "rtol:              " + str(self.rtol) + "\n"
        s += "method:            " + str(self.method) + "\n"
        s += "order:             " + str(self.order) + "\n"
        s += "nsteps:            " + str(self.nsteps) + "\n"
        s += "first_step:        " + str(self.first_step) + "\n"
        s += "min_step:          " + str(self.min_step) + "\n"
        s += "max_step:          " + str(self.max_step) + "\n"
        s += "tidy:              " + str(self.tidy) + "\n"
        s += "num_cpus:          " + str(self.num_cpus) + "\n"
        s += "norm_tol:          " + str(self.norm_tol) + "\n"
        s += "norm_steps:        " + str(self.norm_steps) + "\n"
        s += "rhs_filename:      " + str(self.rhs_filename) + "\n"
        s += "rhs_reuse:         " + str(self.rhs_reuse) + "\n"
        s += "rhs_with_state:    " + str(self.rhs_with_state) + "\n"
        s += "average_expect:    " + str(self.average_expect) + "\n"
        s += "average_states:    " + str(self.average_states) + "\n"
        s += "ntraj:             " + str(self.ntraj) + "\n"
        s += "store_states:      " + str(self.store_states) + "\n"
        s += "store_final_state: " + str(self.store_final_state) + "\n"

        return s


class Result():
    """Class for storing simulation results from any of the dynamics solvers.

    Attributes
    ----------

    solver : str
        Which solver was used [e.g., 'mesolve', 'mcsolve', 'brmesolve', ...]
    times : list/array
        Times at which simulation data was collected.
    expect : list/array
        Expectation values (if requested) for simulation.
    states : array
        State of the simulation (density matrix or ket) evaluated at ``times``.
    num_expect : int
        Number of expectation value operators in simulation.
    num_collapse : int
        Number of collapse operators in simualation.
    ntraj : int/list
        Number of monte-carlo trajectories (if using mcsolve).  List indicates
        that averaging of expectation values was done over a subset of total
        number of trajectories.
    col_times : list
        Times at which state collpase occurred. Only for Monte Carlo solver.
    col_which : list
        Which collapse operator was responsible for each collapse in
        ``col_times``. Only for Monte Carlo solver.

    """
    def __init__(self):
        self.solver = None
        self.times = None
        self.states = []
        self.expect = []
        self.num_expect = 0
        self.num_collapse = 0
        self.ntraj = None
        self.col_times = None
        self.col_which = None

    def __str__(self):
        s = "Result object "
        if self.solver:
            s += "with " + self.solver + " data.\n"
        else:
            s += "missing solver information.\n"
        s += "-" * (len(s) - 1) + "\n"
        if self.states is not None and len(self.states) > 0:
            s += "states = True\n"
        elif self.expect is not None and len(self.expect) > 0:
            s += "expect = True\nnum_expect = " + str(self.num_expect) + ", "
        else:
            s += "states = True, expect = True\n" + \
                "num_expect = " + str(self.num_expect) + ", "
        s += "num_collapse = " + str(self.num_collapse)
        if self.solver == 'mcsolve':
            s += ", ntraj = " + str(self.ntraj)
        return s

    def __repr__(self):
        return self.__str__()

    def __getstate__(self):
        # defines what happens when Qobj object gets pickled
        self.__dict__.update({'qutip_version': __version__[:5]})
        return self.__dict__

    def __setstate__(self, state):
        # defines what happens when loading a pickled Qobj
        if 'qutip_version' in state.keys():
            del state['qutip_version']
        (self.__dict__).update(state)


class SolverConfiguration():

    def __init__(self):

        self.cgen_num = 0

        self.reset()

    def reset(self):

        # General stuff
        self.tlist = None       # evaluations times
        self.ntraj = None       # number / list of trajectories
        self.options = None     # options for solvers
        self.norm_tol = None    # tolerance for wavefunction norm
        self.norm_steps = None  # max. number of steps to take in finding

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
        self.string = None     # Holds string of variables passed to td solver

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
        self.c_args = []        # store args for time-dependent collapse func.

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
# create a global instance of the SolverConfiguration class
#
config = SolverConfiguration()

# for backwards compatibility
Odeoptions = Options
Odedata = Result
