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
        Number of monte-carlo trajectories (if using mcsolve).  List indicates that averaging of
        expectation values was done over a subset of total number of trajectories.
    col_times : list
        Times at which state collpase occurred.  Only for Monte-Carlo solver.
    col_which : list
        Which collapse operator was responsible for each collapse in ``col_times``. mcsolver only.
    
    """
    def __init__(self):
        self.solver=None
        self.times=None
        self.states=[]
        self.expect=[]
        self.num_expect=0
        self.num_collapse=0
        self.ntraj=None
        self.col_times=None
        self.col_which=None
    def __str__(self):
        s="Odedata object "
        if self.solver:
            s+="with "+self.solver+" data.\n"
        else:
            s+="missing solver information.\n"
        s+="-"*(len(s)-1)+"\n"
        if len(self.states)>0 and (not len(self.expect)>0):
            s+= "states = True\n"
        elif len(self.expect)>0 and (not len(self.states)>0):
            s+="expect = True\nnum_expect = "+str(self.num_expect)+", "
        else:
            s+= "states = True, expect = True\n"+"num_expect = "+str(self.num_expect)+", "
        s+="num_collapse = "+str(self.num_collapse)
        if self.solver=='mcsolve':
            s+=", ntraj = "+str(self.ntraj)
        return s
    def __repr__(self):
        return self.__str__()
        


