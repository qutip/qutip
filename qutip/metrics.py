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
from Qobj import *
import scipy.linalg as la
from scipy import real,trace
##@package metrics
#Collection of metrics (distance measures) between density matricies
#@version 0.1
#@date 2011

def fidelity(A,B):
    """
    @brief Calculates the fidelity (pseudo-metric) between two density matricies.
        See: Nielsen & Chuang, "Quantum Computation and Quantum Information"
    
    @param  A *Qobj* density matrix
    @param  B *Qobj* density matrix with same dimensions as A
    
    @return  *float*  fidelity
    """
    if A.dims!=B.dims:
        raise TypeError('Density matricies do not have same dimensions.')
    else:
        A=A.sqrtm()
        A*(B*A)
        return float(real(trace((A*(B*A)).sqrtm().full())))


def tracedist(A,B):
    """
    @brief Calculates the trace distance between two density matricies.
        See: Nielsen & Chuang, "Quantum Computation and Quantum Information"
    
    @param   A  *Qobj* density matrix
    @param   B  *Qobj* density matrix with same dimensions as A
    
    @return  *float*  trace distance
    """
    if A.dims!=B.dims:
        raise TypeError('Density matricies do not have same dimensions.')
    else:
        diff=A-B
        diff=diff.dag()*diff
        out=diff.sqrtm().full()
        return float(real(0.5*trace(out)))
