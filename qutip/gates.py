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
# Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson
#
###########################################################################
from scipy import sqrt
from qutip.states import qstate, state_number_index, state_number_enumerate
from qutip.Qobj import Qobj

def cnot():
    """
    Quantum object representing the CNOT gate.
    
    Returns
    -------
    cnot_gate : qobj
        Quantum object representation of CNOT gate
    
    Examples
    --------  
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
    """Quantum object representing the Fredkin gate.
    
    Returns
    -------
    fred_gate : qobj
        Quantum object representation of Fredkin gate.
    
    Examples
    --------    
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
    """Quantum object representing the Toffoli gate.
    
    Returns
    -------
    toff_gate : qobj
        Quantum object representation of Toffoli gate.
    
    Examples
    --------    
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
def swap(mask=None):
    """Quantum object representing the SWAP gate.
    
    Returns
    -------
    swap_gate : qobj
        Quantum object representation of SWAP gate
    
    Examples
    --------
    >>> swap()
    Quantum object: dims = [[2, 2], [2, 2]], shape = [4, 4], type = oper, isHerm = True
    Qobj data = 
    [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j]
     [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]]
         
    """
    if mask is None:
        uu=qstate('uu')
        ud=qstate('ud')
        du=qstate('du')
        dd=qstate('dd')
        Q=uu*uu.dag()+ud*du.dag()+ du*ud.dag()+dd*dd.dag()
        return Qobj(Q)
    else:
        if sum(mask) != 2:
            raise ValueError("mask must only have two ones, rest zeros")
        
        dims = [2] * len(mask)
        idx, = where(mask)
        N = prod(dims)
        data = sp.lil_matrix((N,N))

        for s1 in state_number_enumerate(dims):
            i1 = state_number_index(dims, s1)

            if s1[idx[0]] == s1[idx[1]]:
                i2 = i1
            else:
                s2 = array(s1).copy()
                s2[idx[0]], s2[idx[1]] = s2[idx[1]], s2[idx[0]]
                i2 = state_number_index(dims, s2)

            data[i1,i2] = 1

        return Qobj(data, dims=[dims, dims], shape=[N, N])

def iswap(mask=None):
    """Quantum object representing the iSWAP gate.
    
    Returns
    -------
    iswap_gate : qobj
        Quantum object representation of iSWAP gate
    
    Examples
    --------
    >>> iswap()
    Quantum object: dims = [[2, 2], [2, 2]], shape = [4, 4], type = oper, isHerm = False
    Qobj data =
    [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j  0.+1.j  0.+0.j]
     [ 0.+0.j  0.+1.j  0.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]]
    """
    if mask is None:
        return Qobj(array([[1,0,0,0], [0,0,1j,0], [0,1j,0,0], [0,0,0,1]]), dims=[[2, 2], [2, 2]])
    else:
        if sum(mask) != 2:
            raise ValueError("mask must only have two ones, rest zeros")
        
        dims = [2] * len(mask)
        idx,  = where(mask)
        N = prod(dims)
        data = sp.lil_matrix((N,N),dtype=complex)

        for s1 in state_number_enumerate(dims):
            i1 = state_number_index(dims, s1)

            if s1[idx[0]] == s1[idx[1]]:
                i2 = i1
                val = 1.0
            else:
                s2 = s1.copy()
                s2[idx[0]], s2[idx[1]] = s2[idx[1]], s2[idx[0]]
                i2 = state_number_index(dims, s2)
                val = 1.0j

            data[i1,i2] = val

        return Qobj(data, dims=[dims, dims], shape=[N, N])


def sqrtiswap():
    """Quantum object representing the square root iSWAP gate.
    
    Returns
    -------
    sqrtiswap_gate : qobj
        Quantum object representation of square root iSWAP gate
    
    Examples
    --------
    >>> sqrtiswap()
    Quantum object: dims = [[2, 2], [2, 2]], shape = [4, 4], type = oper, isHerm = False
    Qobj data =
    [[ 1.00000000+0.j   0.00000000+0.j          0.00000000+0.j          0.00000000+0.j]
     [ 0.00000000+0.j   0.70710678+0.j          0.00000000-0.70710678j  0.00000000+0.j]
     [ 0.00000000+0.j   0.00000000-0.70710678j  0.70710678+0.j          0.00000000+0.j]
     [ 0.00000000+0.j   0.00000000+0.j          0.00000000+0.j          1.00000000+0.j]]    
    """
    return Qobj(array([[1,0,0,0], [0, 1/sqrt(2), -1j/sqrt(2), 0], [0, -1j/sqrt(2), 1/sqrt(2), 0], [0, 0, 0, 1]]), dims=[[2, 2], [2, 2]])


def snot():
    """Quantum object representing the SNOT (Hadamard) gate.
    
    Returns
    -------
    snot_gate : qobj
        Quantum object representation of SNOT (Hadamard) gate.
    
    Examples
    --------
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
    
    Parameters
    ----------
    theta : float
        Phase rotation angle.
    
    Returns
    -------
    phase_gate : qobj
        Quantum object representation of phase shift gate.
    
    Examples
    --------    
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


    
