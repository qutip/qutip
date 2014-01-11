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
from qutip import  __version__


class Odedata():
    """Class for storing simulation results from any of the dynamics solvers.

    Attributes
    ----------

    solver : str
        Which solver was used ['mesolve','mcsolve','brsolve','floquet']
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
        Times at which state collpase occurred.  Only for Monte-Carlo solver.
    col_which : list
        Which collapse operator was responsible for each collapse in
        ``col_times``. mcsolver only.

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
        s = "Odedata object "
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
