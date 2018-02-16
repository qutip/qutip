"""
Tests for the Hierarchy model
"""
from __future__ import division

import numpy as np
from numpy.testing import (assert_, assert_almost_equal,
                           run_module_suite, assert_equal,
                           assert_array_equal, assert_array_almost_equal)

from scipy.integrate import quad, IntegrationWarning

from qutip import Qobj, sigmaz, sigmax, basis, Options
from qutip.models.hierarchy import Heom, _heom_state_dictionaries


class TestHeom(object):
    """
    Tests for the Heirarchy equation of motion model
    """
    def test_initialization(self):
        """
        Tests the correct initialization of the model class
        and checks if the shapes of the rho term is constructed properly
        for all the density matrices.
        """
        # test if the expansion coefficients are correctly specified
        pass

    def test_normalize(self):
        """
        Tests the normalization of the auxillary density matrices
        """
        pass

    def test_deltak(self):
        """
        Test the `_deltak` function which gives the factor to be used
        for the truncation of the Matsubara terms using the Ishizaki
        Tanimura scheme
        """
        pass

    def test_t1(self):
        """
        Test the function `_t1` for calculating the first term in the
        gradient function
        """
        pass
    
    def test_grad(self):
        """
        Tests the computation of the gradient for all the auxillary density
        operators
        """
        pass
    
    def test_pop_he(self):
        """
        Test the filtering scheme which pops some of the Hierarchy elements
        as the system evolves
        """
        pass

    def test_hierarchy_idx(self):
        """
        Test the HEOM state and index generation function
        """
        pass

    def test_no_cut(self):
        """
        Test if letting the Matsubara cutoff as the total number of exponentials
        gives a correct value of the delta term
        """
        pass


if __name__ == "__main__":
    np.run_module_suite()
