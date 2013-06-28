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


def _reg_str2array(string,width):
    """
    Takes a string of 'u','d','0',and '1' and returns
    a flattened array of 0's and 1's.
    
    Parameters
    ----------
    string: str
        Input string
    
    """
    if len(string)!=width:
        raise ValueError('Registers string length is not equal to the registers width.')
    mask=[]
    for kk in range(len(string)):
        if string[kk] in ['u','0']:
            mask+=[0]
        elif string[kk] in ['d','1']:
            mask+=[1]
        else:
            raise ValueError("String element '"+string[kk]+"' is not a valid label.")
    return np.array(mask)
