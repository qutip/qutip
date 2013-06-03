# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:23:46 2013

@author: dcriger
"""

# This file is part of QuTiP.
#
#    QuTiP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    QuTiP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTiP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011 and later, Paul D. Nation & Robert J. Johansson
#
###########################################################################

from numpy.linalg import norm
from numpy.testing import assert_, run_module_suite
import scipy

from qutip.qobj import Qobj
from qutip.operators import create, destroy, jmat
from qutip.propagator import propagator

from qutip.superop_reps import (super_to_choi, choi_to_kraus,
                                choi_to_super, kraus_to_choi)


class TestSuperopReps(object):
    """
    A test class for the QuTiP function for applying superoperators to
    subsystems.
    """


    def test_SuperChoiSuper(self):
        """
        Superoperator: Converting superoperator to Choi matrix and back.  
        """
        h_5 = scipy.rand(5, 5)
        h_5 = Qobj(inpt=h_5 * h_5.conj().T)
        superoperator = propagator(h_5, scipy.rand,
                                   [create(5), destroy(5), jmat(2,'z')])
        choi_matrix=super_to_choi(superoperator)
        test_supe=choi_to_super(choi_matrix)
        assert_(norm(test_supe - superoperator) == 0.0)


    def test_ChoiKrausChoi(self):
        """
        Superoperator: Converting superoperator to Choi matrix and back.  
        """
        h_5 = scipy.rand(5, 5)
        h_5 = Qobj(inpt=h_5 * h_5.conj().T)
        superoperator = propagator(h_5, scipy.rand,
                                   [create(5), destroy(5), jmat(2,'z')])
        choi_matrix=super_to_choi(superoperator)
        kraus_ops=choi_to_kraus(choi_matrix)
        test_choi=kraus_to_choi(kraus_ops)
        assert_(norm(test_choi - choi_matrix) == 0.0)

if __name__ == "__main__":
    run_module_suite()
