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
# Copyright (C) 2011, Paul D. Nation & Robert J. Johansson
#
###########################################################################
from scipy import finfo
import scipy.sparse as sp
from Qobj import *

def tidyup(op,Atol=1e-8):
    """
    Removes small elements from a Qobj
    
    Parameter op *Qobj* input quantum object
    Parameter Atol *float* absolute tolerance
    
    Returns *Qobj* with small elements removed
    """
    mx=max(abs(op.data.data))
    data=abs(op.data.data)
    outdata=op.data.copy()
    outdata.data[data<(Atol*mx+finfo(float).eps)]=0
    outdata.eliminate_zeros()
    return Qobj(outdata,dims=op.dims,shape=op.shape)




