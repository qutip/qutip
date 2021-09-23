r'''
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

import numpy as np
from qutip.states import qutrit_basis


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
    out = np.empty((5,), dtype=object)
    one, two, three = qutrit_basis()
    # Note that the three level operators are different
    # from the qutrit operators. A three level atom only
    # has transitions 1 <-> 2 <-> 3, so we define the
    # operators seperately from the qutrit code
    out[0] = one * one.dag()
    out[1] = two * two.dag()
    out[2] = three * three.dag()
    out[3] = one * two.dag()
    out[4] = three * two.dag()
    return out
