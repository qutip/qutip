# This file is part of QuTiP.
#
#    QuTiP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    QuTiP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTiP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011-2013, Paul D. Nation & Robert J. Johansson
#
###########################################################################

from __future__ import print_function
import os


class Odeoptions():
    """
    Class of options for ODE solver used by :func:`qutip.mesolve` and
    :func:`qutip.mcsolve`. Options can be changed either inline::

        opts=Odeoptions(gui=False,order=10,.....)

    or by changing the class properties after creation::

        opts=Odeoptions()
        opts.gui=False
        opts.order=10

    Returns options class to be used as options in :func:`qutip.mesolve`
    and :func:`qutip.mcsolve`.

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
    mc_avg : bool {True,False}
        Avg. expectation values in mcsolver.
    ntraj : int {500}
        Number of trajectories in mcsolver.
    rhs_reuse : bool {False,True}
        Reuse Hamiltonian data.
    rhs_filename : str
        Name for compiled Cython file.

    """
    def __init__(self, atol=1e-8, rtol=1e-6, method='adams', order=12,
                 nsteps=1000, first_step=0, max_step=0, min_step=0,
                 mc_avg=True, tidy=True, num_cpus=0, norm_tol=1e-3,
                 norm_steps=5, rhs_reuse=False, rhs_filename=None, gui=True,
                 ntraj=500):
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
        self.mc_avg = mc_avg
        # Number of trajectories (default = 500)
        self.ntraj = ntraj
        # tidyup Hamiltonian before calculation (default = True)
        self.tidy = tidy
        # Use preexisting RHS function for time-dependent solvers
        self.rhs_reuse = rhs_reuse
        # Use filename for preexisting RHS function (will default to last
        # compiled function if None & rhs_exists=True)
        self.rhs_filename = rhs_filename
        # Number of processors to use (mcsolve only)
        if num_cpus:
            self.num_cpus = num_cpus
            if self.num_cpus > int(os.environ['NUM_THREADS']):
                raise Exception("Requested number of CPU's too large. Max = " +
                                str(int(os.environ['NUM_THREADS'])))
        else:
            self.num_cpus = 0
        # Tolerance for wavefunction norm (mcsolve only)
        self.norm_tol = norm_tol
        # Max. number of steps taken to find wavefunction norm to within
        # norm_tol (mcsolve only)
        self.norm_steps = norm_steps
        # Use Progressbar (mcsolve only)
        self.gui = gui

    def __str__(self):
        s = ""
        s += "Odeoptions properties:\n"
        s += "----------------------\n"
        s += "atol:         " + str(self.atol) + "\n"
        s += "rtol:         " + str(self.rtol) + "\n"
        s += "method:       " + str(self.method) + "\n"
        s += "order:        " + str(self.order) + "\n"
        s += "nsteps:       " + str(self.nsteps) + "\n"
        s += "first_step:   " + str(self.first_step) + "\n"
        s += "min_step:     " + str(self.min_step) + "\n"
        s += "max_step:     " + str(self.max_step) + "\n"
        s += "tidy:         " + str(self.tidy) + "\n"
        s += "num_cpus:     " + str(self.num_cpus) + "\n"
        s += "norm_tol:     " + str(self.norm_tol) + "\n"
        s += "norm_steps:   " + str(self.norm_steps) + "\n"
        s += "rhs_filename: " + str(self.rhs_filename) + "\n"
        s += "rhs_reuse:    " + str(self.rhs_reuse) + "\n"
        s += "gui:          " + str(self.gui) + "\n"
        s += "mc_avg:       " + str(self.mc_avg) + "\n"
        s += "ntraj:        " + str(self.ntraj) + "\n"
        return s
