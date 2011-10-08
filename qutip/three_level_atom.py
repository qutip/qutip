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
'''
This module provides functions that are useful for simulating the
three level atom with QuTiP.  A three level atom has three states,
which are linked by dipole transitions so that 1 <-> 2 <-> 3.
Depending on there relative energies they are in the ladder, lambda or
vee configuration. The structure of the relevant operators is the same
for any of the three configurations.

Ladder:          Lambda:                 Vee:
                            |two>                       |three>
  -------|three>           -------                      -------
     |                       / \             |one>         /
     |                      /   \           -------       /
     |                     /     \             \         /
  -------|two>            /       \             \       /
     |                   /         \             \     /
     |                  /           \             \   /
     |                 /        --------           \ /
  -------|one>      -------      |three>         -------
                     |one>                       |two>

The naming of qutip operators follows the convention in the book "The
Theory of Coherent Atomic excitation" by B. W. Shore.

'''

from . import *

def three_level_basis():
    ''' Return basis states for three level atom
    '''
    return basis(3,0), basis(3,1), basis(3,2)

def three_level_operators():
    ''' Return operators for a three level system
    '''
    one, two, three = three_level_basis()
    sig11 = one * one.dag()
    sig22 = two * two.dag()
    sig33 = three * three.dag()
    sig12 = one * two.dag()
    sig32 = three * two.dag()
    return sig11, sig22, sig33, sig12, sig32
