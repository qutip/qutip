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
from scipy import sqrt
from qstate import qstate
from Qobj import *


def cnot():
    """
    Quantum object representing the CNOT gate.
    
    Returns:
        
        Qobj for quantum object representation of CNOT gate
    
    Example::
         
         >>> cnot()
         Quantum object: dims = [[2, 2], [2, 2]], shape = [4, 4], type = oper, isHerm = True
         Qobj data = 
             [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
              [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]
              [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]
              [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j]]
    
    """
    uu=qstate('uu')
    ud=qstate('ud')
    du=qstate('du')
    dd=qstate('dd')
    Q=dd*dd.dag()+du*du.dag()+uu*ud.dag()+ud*uu.dag()
    return Qobj(Q)


#------------------
def fredkin():
    """
    Quantum object representing the Fredkin gate.
    
    Returns:
    
        Qobj for quantum object representation of Fredkin gate.
    
    Example::
        
        >>> fredkin()
        Quantum object: dims = [[2, 2, 2], [2, 2, 2]], shape = [8, 8], type = oper, isHerm = True
            Qobj data = 
            [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
             [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
             [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
             [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
             [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j]
             [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j]
             [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j]
             [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j]]
         
         

    """
    uuu = qstate('uuu')
    uud = qstate('uud') 
    udu = qstate('udu')
    udd = qstate('udd')
    duu = qstate('duu') 
    dud = qstate('dud')
    ddu = qstate('ddu')
    ddd = qstate('ddd')
    Q = ddd*dag(ddd) + ddu*dag(ddu) + dud*dag(dud) + duu*dag(duu) + udd*dag(udd) + uud*dag(udu) + udu*dag(uud) + uuu*dag(uuu)
    return Qobj(Q)


#------------------
def toffoli():
    """
    Quantum object representing the Toffoli gate.
    
    Returns:
    
        Qobj quantum object representation of Toffoli gate.
    
    Example::
        
        >>> toffoli()
        Quantum object: dims = [[2, 2, 2], [2, 2, 2]], shape = [8, 8], type = oper, isHerm = True
        Qobj data = 
            [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
             [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
             [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
             [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
             [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j]
             [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j]
             [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j]
             [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j]]
         
         
    """
    uuu = qstate('uuu') 
    uud = qstate('uud') 
    udu = qstate('udu') 
    udd = qstate('udd')
    duu = qstate('duu') 
    dud = qstate('dud') 
    ddu = qstate('ddu') 
    ddd = qstate('ddd')
    Q = ddd*dag(ddd) + ddu*dag(ddu) + dud*dag(dud) + duu*dag(duu) + udd*dag(udd) + udu*dag(udu) + uuu*dag(uud) + uud*dag(uuu)
    return Qobj(Q)

#------------------
def swap():
    """
    Quantum object representing the SWAP gate.
    
    Returns:
    
        Qobj quantum object representation of SWAP gate
    
    Example::
    
        >>> swap()
        Quantum object: dims = [[2, 2], [2, 2]], shape = [4, 4], type = oper, isHerm = True
        Qobj data = 
        [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j]
         [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]]
         
    """
    uu=qstate('uu')
    ud=qstate('ud')
    du=qstate('du')
    dd=qstate('dd')
    Q=uu*uu.dag()+ud*du.dag()+ du*ud.dag()+dd*dd.dag()
    return Qobj(Q)



def snot():
    """
    Quantum object representing the SNOT (Hadamard) gate.
    
    Returns:
    
        Qobj quantum object representation of SNOT (Hadamard) gate.
    
    Example::
    
        >>> snot()
        Quantum object: dims = [[2], [2]], shape = [2, 2], type = oper, isHerm = True
        Qobj data = 
        [[ 0.70710678+0.j  0.70710678+0.j]
         [ 0.70710678+0.j -0.70710678+0.j]]
         
    """
    u=qstate('u')
    d=qstate('d')
    Q=1.0/sqrt(2.0)*(d*d.dag()+u*d.dag()+d*u.dag()-u*u.dag())
    return Qobj(Q)    


def phasegate(theta):
    """
    Returns quantum object representing the phase shift gate.
    
    Args:
        
        theta (float): Phase rotation angle.
    
    Returns:
    
        Qobj for quantum object representation of phase shift gate.
    
    Example::
        
        >>> phasegate(pi/4)
        Quantum object: dims = [[2], [2]], shape = [2, 2], type = oper, isHerm = False
        Qobj data = 
        [[ 1.00000000+0.j          0.00000000+0.j        ]
         [ 0.00000000+0.j          0.70710678+0.70710678j]]
        
    """
    u=qstate('u')
    d=qstate('d')
    Q=d*d.dag()+(exp(1.0j*theta)*u*u.dag())
    return Qobj(Q)


    
