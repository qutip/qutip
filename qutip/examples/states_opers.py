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
from ..Qobj import *
from ..states import *
from ..operators import *
from ..tensor import *
from ..expect import *
from ..metrics import *
from .termpause import termpause


def states_opers():
    print('')
    print('-'*80)
    print('A brief introduction into manipulating')
    print('quantum operators and states.')
    print('-'*80)
    termpause()
    
    print('vec=basis(5,0)')
    vec=basis(5,0)
    print('vec')
    print(vec)
    print('')
    print('a=destroy(5)')
    a=destroy(5)
    print('a')
    print(a)
    print('')
    print('a*vec')
    print(a*vec)
    print('a.dag()*vec')
    print(a.dag()*vec)
    c=create(5)
    print('c=create(5)')
    print('c')
    print(c)
    print('c*c*vec')
    print(c*c*vec)
    print('or...')
    print('c**2*vec')
    print(c**2*vec)
    print('')
    print('c*a*(c**2*vec)')
    print(c*a*(c**2*vec))
    print('')
    print('c*a*(c**2*vec).unit()')
    print(c*a*(c**2*vec).unit())
    print('')
    print('or...')
    print('n=num(5)')
    n=num(5)
    print('n')
    print(n)
    print('vec=basis(5,2)')
    vec=basis(5,2)
    print('n*vec')
    print(n*vec)
    
    print('')
    print('displacement and squeezing operators...')
    termpause()
    vec=basis(5,0)
    d=displace(5,1j)
    s=squeez(5,0.25+0.25j)
    print('')
    print('vec=basis(5,0)')
    print('d=displace(5,1j)')
    print('s=squeez(5,0.25+0.25j)')
    print('d')
    print(d)
    print('s')
    print(s)
    print('')
    print('d*vec')
    print(d*vec)
    print('')
    print('d*s*vec')
    print(d*s*vec)
    
    
    print('')
    print('metrics for density matricies...')
    termpause()
    print('#pure states')
    print('x=coherent_dm(5,1.25)')
    x=coherent_dm(5,1.25)
    print("y=coherent_dm(5,1.25j) #<-- note the 'j'")
    y=coherent_dm(5,1.25j) #<-- note the 'j'
    print('#mixed state')
    print('z=thermal_dm(5,0.125)')
    z=thermal_dm(5,0.125)
    print('')
    print('fidelity(x,x)')
    print(fidelity(x,x))
    print('tracedist(y,y)')
    print(tracedist(y,y))
    print('')
    print('#for two-pure states, the trace distance (T) and the fidelity (F)')
    print('#are related by T=sqrt(1-F**2)')
    print('tracedist(y,x)')
    print(tracedist(y,x))
    print('sqrt(1-fidelity(y,x)**2)')
    print(sqrt(1-fidelity(y,x)**2))
    print('')
    print('#For a pure state and a mixed state, 1-F**2<=T')
    print('1-fidelity(x,z)**2')
    print(1-fidelity(x,z)**2)
    print('tracedist(x,z)')
    print(tracedist(x,z))
    
    print('')
    print('two-level spin systems...')
    termpause()
    print('sigmaz()')
    print(sigmaz())
    print('spin=basis(2,0)')
    spin=basis(2,0)
    print('sigmaz()*spin')
    sigmaz()*spin
    print('spin2=basis(2,1)')
    spin2=basis(2,1)
    print('sigmaz()*spin')
    sigmaz()*spin
    print('sigmaz()*spin2')
    sigmaz()*spin2
    
    
    print('')
    print('expectation value of composite objects...')
    termpause()
    
    spin1=basis(2,0)
    spin2=basis(2,1)
    two_spins=tensor(spin1,spin2)
    sz1=tensor(sigmaz(),qeye(2))
    sz2=tensor(qeye(2),sigmaz())
    print('spin1=basis(2,0)')
    print('spin2=basis(2,1)')
    print('two_spins=tensor(spin1,spin2)')
    print('sz1=tensor(sigmaz(),qeye(2))')
    print('sz2=tensor(qeye(2),sigmaz())')
    print('expect(sz1,two_spins)')
    print(expect(sz1,two_spins))
    print('expect(sz2,two_spins)')
    print(expect(sz2,two_spins))
    termpause()
    print('')
    print('DEMO FINISHED...')


if __name__=='main()':
    states_opers()

