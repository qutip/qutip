# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################
'''
This module provides functions that are useful for simulating the
three level atom with QuTiP.  A three level atom (qutrit) has three states,
which are linked by dipole transitions so that 1 <-> 2 <-> 3.
Depending on there relative energies they are in the ladder, lambda or
vee configuration. The structure of the relevant operators is the same
for any of the three configurations::

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

References
----------
The naming of qutip operators follows the convention in [1]_ .

.. [1] Shore, B. W., "The Theory of Coherent Atomic Excitation",
   Wiley, 1990.

Notes
-----
Contributed by Markus Baden, Oct. 07, 2011

'''

__all__ = ['three_level_basis', 'three_level_ops']

from qutip.states import qutrit_basis
from numpy import array


def three_level_basis():
    ''' Basis states for a three level atom.

    Returns
    -------
    states : array
        `array` of three level atom basis vectors.

    '''
    # A three level atom has the same representation as a qutrit, i.e.
    # three states
    return qutrit_basis()


def three_level_ops():
    ''' Operators for a three level system (qutrit)

    Returns
    --------
    ops : array
        `array` of three level operators.

    '''
    one, two, three = qutrit_basis()
    # Note that the three level operators are different
    # from the qutrit operators. A three level atom only
    # has transitions 1 <-> 2 <-> 3, so we define the
    # operators seperately from the qutrit code
    sig11 = one * one.dag()
    sig22 = two * two.dag()
    sig33 = three * three.dag()
    sig12 = one * two.dag()
    sig32 = three * two.dag()
    return array([sig11, sig22, sig33, sig12, sig32], dtype=object)
