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
###########################################################################
from Qobj import *
import scipy.linalg as la
from scipy import real,trace
from numpy import vectorize
def scalar_fidelity(A,B):
    """
    Calculates the fidelity (pseudo-metric) between two density matricies.
    
    @param A: density matrix
    @param B: density matrix with same dimensions as A
    
    See: Nielsen & Chuang, "Quantum Computation and Quantum Information"
    """
    if A.dims!=B.dims:
        raise TypeError('Density matricies do not have same dimensions.')
    else:
        A=A.sqrtm()
        A*(B*A)
        return real(trace((A*(B*A)).sqrtm().full()))

fidelity=vectorize(scalar_fidelity)


def scalar_trace_dist(A,B):
    """
    Calculates the trace distance between two density matricies.
    
    @param A: density matrix
    @param B: density matrix with same dimensions as A
    
    See: Nielsen & Chuang, "Quantum Computation and Quantum Information"
    """
    diff=A-B
    diff=diff.dag()*diff
    out=diff.sqrtm().full()
    return real(0.5*trace(out))

trace_dist=vectorize(scalar_trace_dist)



if __name__ == "__main__":
    from states import *
    x=coherent(10,3j)
    y=coherent(10,1+1.5j)
    r1=x*x.dag()
    r2=y*y.dag()
    print fidelity([r1,r2],r2)
    print trace_dist([r1,r2],r2)