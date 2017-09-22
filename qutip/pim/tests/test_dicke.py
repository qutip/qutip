"""
Tests for Permutation Invariance methods
"""
import numpy as np
from numpy.testing import assert_, run_module_suite, assert_raises, assert_array_equal

from qutip.pim.dicke import (num_dicke_states, num_two_level, irreducible_dim,
                             num_dicke_ladders, generate_dicke_space, isdicke,
                             get_j_m)

class TestPim:
    """
    A test class for the Permutation invariance matrix generation
    """
    def test_num_dicke_states(self):
        """
        Tests the `num_dicke_state` function
        """
        N_list = [1, 2, 3, 4, 5, 6, 9, 10, 20, 100, 123]
        dicke_states = [num_dicke_states(i) for i in N_list]

        assert_array_equal(dicke_states, [2, 4, 6, 9, 12, 16, 30, 36, 121, 2601, 3906])

        N = -1
        assert_raises(ValueError, num_dicke_states, N)

        N = 0.2
        assert_raises(ValueError, num_dicke_states, N)

    def test_num_two_level(self):
        """
        Tests the `num_two_level` function.
        """
        N_dicke = [2, 4, 6, 9, 12, 16, 30, 36, 121, 2601, 3906]
        N = [1, 2, 3, 4, 5, 6, 9, 10, 20, 100, 123]

        calculated_N = [num_two_level(i) for i in N_dicke]

        assert_array_equal(calculated_N, N)
    
    def test_num_dicke_ladders(self):
        """
        Tests the `num_dicke_ladders` function
        """
        ndl_true = [1, 2, 2, 3, 3, 4, 4, 5, 5]
        ndl = [num_dicke_ladders(N) for N in range (1, 10)]        
        assert_array_equal(ndl, ndl_true)    

    def test_irreducible_dim(self):
        """
        Test the irreducible dimension function
        """
        pass

    def test_isdicke(self):
        """
        Tests the `isdicke` function
        """
        N = 6
        test_indices = [(0, 0), (0, 1), (1, 0), (-1, -1), (2, -1)]
        dicke_labels = [isdicke(N, x) for x in test_indices]

        assert_array_equal(dicke_labels, [True, False, True, False, False])


if __name__ == "__main__":
    run_module_suite()