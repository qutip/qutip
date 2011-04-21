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
from qutip import *

L=2

theta = linspace(0,   pi, 180)
phi   = linspace(0, 2*pi,  30)

lmax  = 10

print "-----------------------------------------------------------------------------"
print " XDIRECTION computes an approximation to a direction eigenket in the direct sum space"
print "  of angular-momentum spaces"
print "-----------------------------------------------------------------------------"

psi_list = []

for l in range(0,lmax+1):

    psi_list.append(sqrt((2*l + 1)/(4*pi)) * basis(2*l + 1, l))

psi = orbital(theta, phi, psi_list)
sphereplot(theta, phi, psi)


