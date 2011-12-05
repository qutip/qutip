#This file is part of QuTiP.
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
# Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson
#
###########################################################################
from ..Qobj import *
from ..istests import *
from ..states import *
from ..operators import *
from scipy import *
from .termpause import termpause

def qobjbasics():
    print('-'*80)
    print('Details basic usage of the Qobj class...')
    print('-'*80)
    termpause()
    print("The basics of Qobj:")
    print('-------------------\n')
    print('Qobj()')
    print(Qobj())
    
    print('')
    print('Qobj([1,2,3,4,5])')
    print(Qobj([1,2,3,4,5]))
    
    print('')
    print('x=array([[1],[2],[3],[4],[5]])')
    print('Qobj(x)')
    x=array([[1],[2],[3],[4],[5]])
    print(Qobj(x))
    
    print('')
    print('r=random.random((4,4))')
    print('Qobj(r)')
    r=random.random((4,4))
    print(Qobj(r))
    
    print('')
    print('States and operators:')
    print('---------------------')
    termpause()
    print('basis(5,3)')
    print(basis(5,3))
    
    print('')
    print('coherent(5,0.5-0.5j)')
    print(coherent(5,0.5-0.5j))
    
    print('')
    print('destroy(4)')
    print(destroy(4))
    
    print('')
    print('sigmaz()')
    print(sigmaz())
    
    print('')
    print("jmat(5/2.0,'+')")
    print(jmat(5/2.0,'+'))
    
    print('')
    print('Qobj properties:')
    print('----------------')
    termpause() 
    print('q=destroy(4)')
    q=destroy(4)
    print('')
    print('q.dims')
    print(q.dims)
    
    print('')
    print('q.shape')
    print(q.shape)
    
    print('')
    print('q.type')
    print(q.type)
    
    print('')
    print('q.isherm')
    print(q.isherm)
    
    print('')
    print('q.data')
    print(q.data)
    
    print('')
    print('Qobj math:')
    print('----------')
    termpause()
    print('q=destroy(4)')
    q=destroy(4)
    print('x=sigmax()')
    x=sigmax()
    
    print('')
    print('q+5')
    print(q+5)
    
    print('')
    print('x*x')
    print(x*x)
    
    print('')
    print('q**3')
    print(q**3)
    
    print('')
    print('x/sqrt(2)')
    print(x/sqrt(2))
    
    print('')
    print('Functions operating on Qobj class:')
    print('----------------------------------')
    termpause()
    print('basis(5,3)')
    print(basis(5,3))
    
    print('')
    print('basis(5,3).dag()')
    print(basis(5,3).dag())
    
    print('')
    print('coherent_dm(5,1)')
    print(coherent_dm(5,1))
    
    print('')
    print('coherent_dm(5,1).diag()')
    print(coherent_dm(5,1).diag())
    
    print('')
    print('coherent_dm(5,1).full()')
    print(coherent_dm(5,1).full())
    
    print('')
    print('coherent_dm(5,1).norm()')
    print(coherent_dm(5,1).norm())
    
    print('')
    print('coherent_dm(5,1).sqrtm()')
    print(coherent_dm(5,1).sqrtm())
    
    print('')
    print('coherent_dm(5,1).tr()')
    print(coherent_dm(5,1).tr())
    
    print('')
    print('(basis(4,2)+basis(4,1)).unit()')
    print((basis(4,2)+basis(4,1)).unit())
    
    print('')
    print('DEMO FINISHED...')
    termpause()
#--------------------------------------------

if __name__=='main()':
    qobjbasics()












