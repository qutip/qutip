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

import sys
from qutip import *
from numpy.testing import assert_equal

def test_Transformation1():
    "Transform 2-level to eigenbasis and back"
    H1 = rand() * sigmax() + rand() * sigmay() + rand() * sigmaz()
    evals, ekets = H1.eigenstates()
    Heb = H1.transform(ekets)       # eigenbasis (should be diagonal)
    H2 = Heb.transform(ekets, True) # back to original basis
    assert_equal((H1 - H2).norm() < 1e-6,True)


def test_Transformation2():
    "Transform 10-level real-values to eigenbasis and back"
    N = 10
    H1 = Qobj((0.5-rand(N,N)))
    H1 = H1 + H1.dag()
    evals, ekets = H1.eigenstates()
    Heb = H1.transform(ekets)       # eigenbasis (should be diagonal)
    H2 = Heb.transform(ekets, True) # back to original basis
    assert_equal((H1 - H2).norm() < 1e-6,True)
    

def test_Transformation3():
    "Transform 10-level to eigenbasis and back"
    N = 10
    H1 = Qobj((0.5-rand(N,N)) + 1j*(0.5-rand(N,N)))
    H1 = H1 + H1.dag()
    evals, ekets = H1.eigenstates()
    Heb = H1.transform(ekets)       # eigenbasis (should be diagonal)
    H2 = Heb.transform(ekets, True) # back to original basis
    assert_equal((H1 - H2).norm() < 1e-6,True)    
    
        
def test_Transformation4():
    "Transform 10-level imag to eigenbasis and back"
    N = 10
    H1 = Qobj(1j*(0.5-rand(N,N)))
    H1 = H1 + H1.dag()
    evals, ekets = H1.eigenstates()
    Heb = H1.transform(ekets)       # eigenbasis (should be diagonal)
    H2 = Heb.transform(ekets, True) # back to original basis
    assert_equal((H1 - H2).norm() < 1e-6,True)    
     


