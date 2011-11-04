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
from numpy import min,max,sqrt,arange
from scipy import factorial

def clebsch(j1,j2,j3,m1,m2,m3):
    """
    Calculates the Clebsch-Gordon coefficient
        for coupling (j1,m1) and (j2,m2) to give (j3,m3).
    
    Args:
        j1 (float): for total angular momentum 1.
        
        j2 (float): for total angular momentum 2.
        
        j3 (float): for total angular momentum 3.
        
        m1 (float): for z-component of angular momentum 1.
        
        m2 (float): for z-component of angular momentum 2.
        
        m3 (float): for z-component of angular momentum 3.
    
    Returns:
        float for requested Clebsch-Gordan coefficient.
    """
    if m3!=m1+m2:
        return 0
    vmin=int(max([-j1+j2+m3,-j1+m1,0]))
    vmax=int(min([j2+j3+m1,j3-j1+j2,j3+m3]))
    C=sqrt((2.0*j3+1.0)*factorial(j3+j1-j2)*factorial(j3-j1+j2)*factorial(j1+j2-j3)*factorial(j3+m3)*factorial(j3-m3)/(factorial(j1+j2+j3+1)*factorial(j1-m1)*factorial(j1+m1)*factorial(j2-m2)*factorial(j2+m2)))
    S=0
    for v in xrange(vmin,vmax+1):
        S+=(-1.0)**(v+j2+m2)/factorial(v)*factorial(j2+j3+m1-v)*factorial(j1-m1+v)/factorial(j3-j1+j2-v)/factorial(j3+m3-v)/factorial(v+j1-j2-m3)
    C=C*S
    return C


if __name__=='main()':
    print clebsch(0.5,0.5,1,1,0,1)


