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


class Odeoptions():
    """
    Class of options for evolution solvers such as :func:`qutip.mesolve` and
    :func:`qutip.mcsolve`. Options can be specified either as arguments to the
    constructor::

        opts = Odeoptions(gui=False, order=10, ...)

    or by changing the class attributes after creation::

        opts = Odeoptions()
        opts.gui = False
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
    norm_tol :float
        Tolerance used when finding wavefunction norm in mcsolve.
    norm_steps : int
        Max. number of steps used to find wavefunction norm to within norm_tol
        in mcsolve.
    gui : bool {True,False}
        Use progress bar GUI for mcsolver.
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
                 rhs_filename=None, gui=False, ntraj=500, rhs_with_state=False,
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
        # Use Progressbar (mcsolve only)
        self.gui = gui
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
        s += "Odeoptions:\n"
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
        s += "gui:               " + str(self.gui) + "\n"
        s += "average_expect:    " + str(self.average_expect) + "\n"
        s += "average_states:    " + str(self.average_states) + "\n"
        s += "ntraj:             " + str(self.ntraj) + "\n"
        s += "store_states:      " + str(self.store_states) + "\n"
        s += "store_final_state: " + str(self.store_final_state) + "\n"

        return s
