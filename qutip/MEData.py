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

#
# Class for storing data from mesolve.
#
class MEData():
    """
    Class for storing results from mesolve.
    """
    def __init__(self):
        #: Array of times at which state vector was evaluated.
        self.times=None
        #: Array of state vectors if mcsolve was run without expectation operators.
        self.states=None
        #: Array of expectation values if mcsolve was called with expectation operators.
        self.expect=None
        #: Number of expectation operators (if any).
        self.num_expect=None

    def __str__(self):
        s="MEData object: "
        if (not self.expect) and (not self.states):
            s+="Empty MEData object."
            return s
        if self.states and (not self.expect):
            s+= "states = True, "
        elif self.expect and (not self.states):
            s+="expect = True, num_expect = "+str(self.num_expect)+", "
        else:
            s+= "states = True, expect = True, num_expect = "+str(self.num_expect)

        return s
        
    def __repr__(self):
        return self.__str__()
        


