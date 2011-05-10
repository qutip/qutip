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
import time


kappa=2
gamma=0.2
g=1
wc=0
w0=0
N=5
E=0.5
nloop=101
wlist=linspace(-5,5,nloop)

#define single-variable function for use in parfor
def func(wl):#function of wl only
	count1,count2,infield=probss(E,kappa,gamma,g,wc,w0,wl,N)
	return count1,count2,infield

start_time=time.time()
#run parallel for-loop over wlist
[count1,count2,infield] =parfor(func,wlist)
print 'time elapsed = ' +str(time.time()-start_time) 

plot(wlist,real(count1),wlist,real(count2))
xlabel('Detuning')
ylabel('Count rates')
show()

plot(wlist,180.0*angle(infield)/pi)
xlabel('Detuning')
ylabel('Intracavity phase shift')
show()
