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
# Copyright (C) 2011 and later, Paul D. Nation & Robert J. Johansson
#
###########################################################################
import numpy as np
from qutip.quantum_info.utils import _reg_str2array

class Circuit():
    """A class for representing quantum circuits.
    """
    def __init__(self,width,initial_state=None,name=None):
        # check for circuit name
        if name==None:
            name='QuTiP Circuit'
        # check initial_state
        if initial_state==None:
            initial_state='0'*width
        if isinstance(initial_state,str):
            initial_state=_reg_str2array(initial_state,width)
        initial_state=np.asarray(initial_state).flatten()
        self.record={'name': name,'width': width, 'initial_state':initial_state}













