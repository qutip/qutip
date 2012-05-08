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


# XPTRACE illustrates computation of partial traces

from qutip import *

## 
#
# Adapted from the quantom optics toolbox example xptrace.m

#echo on
#-----------------------------------------------------------------------------
print "XPTRACE illustrates calculation of partial traces"
#-----------------------------------------------------------------------------
print
up = basis(2,0); print "up ="
print up
dn = basis(2,1); print "dn ="
print dn
bell = (tensor([up,up])+tensor([dn,dn]))/sqrt(2);
print "bell ="
print bell.full()
#-----------------------------------------------------------------------------
print
print "PTRACE of a Bell state is the 50-50 mixture"
#-----------------------------------------------------------------------------
print "Either specify a state as a ket ..."
#-----------------------------------------------------------------------------
print "ptrace(bell,1) = "
print ptrace(bell, 1).full()
#-----------------------------------------------------------------------------
print
print "... or as a density matrix"
#-----------------------------------------------------------------------------
rho_bell = bell * bell.dag()
print "ptrace(bell*bell.dag(),1) = "
print ptrace(rho_bell,1).full()
#-----------------------------------------------------------------------------
print
print "Now consider measuring the second particle, and obtaining the result 'left'"
#-----------------------------------------------------------------------------
left = (up + dn)/sqrt(2);
print "left = "
print left.full()
#-----------------------------------------------------------------------------
print "Action of this measurement is to apply the projection operator Omegaleft"
#-----------------------------------------------------------------------------
Omegaleft = tensor(qeye(2),left*left.dag());
print "Omegaleft ="
print Omegaleft.full()
after = Omegaleft*bell;
print after
#-----------------------------------------------------------------------------
print "Probability of result"
#-----------------------------------------------------------------------------
prob = after.norm()**2
print "prob = ", prob
after = after/after.norm();
#-----------------------------------------------------------------------------
print "The find reduced density matrix of particle 1"
#-----------------------------------------------------------------------------
rho = ptrace(after,1)
print rho
#-----------------------------------------------------------------------------
print "Check that it is pure"
#-----------------------------------------------------------------------------
print "trace(rho^2) = ", (rho*rho).tr()
#-----------------------------------------------------------------------------
print "End of demonstration"
#-----------------------------------------------------------------------------
