"""
Tests for Permutation Invariance methods
"""
import numpy as np
from numpy.testing import assert_, run_module_suite, assert_raises, assert_array_equal

from qutip.pim.dicke import (num_dicke_states, num_two_level, irreducible_dim,
                             num_dicke_ladders, generate_dicke_space, isdicke,
                             get_j_m, is_j_m, get_k, initial_dicke_state)

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

    def test_get_j_m(self):
        """
        Tests `get_j_m` function to get the (j, m) values.

        N = 6

        | 3, 3>
        | 2, 3>  | 2, 2>
        | 1, 3>  | 1, 2>  | 1,  1>
        | 0, 3>  | 0, 2>  | 0,  1> |0, 0>
        |-1, 3>  |-1, 2>  |-1,  1>
        |-2, 3>  |-2, 2>
        |-3, 3>

        """
        N = 6
        indices = [(0, 0), (0, 1), (0, 2), (0, 3),
                   (1, 0), (1, 1), (1, 2), (1, 3),
                   (2, 0), (2, 1), (2, 2), (2, 3),
                   (3, 0), (3, 1), (3, 2), (3, 3),
                   (4, 0), (4, 1), (4, 2), (4, 3),
                   (5, 0), (5, 1), (5, 2), (5, 3),
                   (6, 0), (6, 1), (6, 2), (6, 3)]

        jm_vals = [(3., 3.), False, False, False,
                   (2., 3.), (2., 2.),  False, False,
                   (1., 3.), (1., 2.), (1., 1.), False,
                   (0., 3.), (0., 2.), (0., 1.), (0., 0.),
                   (-1., 3.), (-1., 2.), (-1., 1.), False,
                   (-2., 3.), (-2., 2.), False, False,
                   (-3., 3.), False, False, False]

        predicted_jm = [get_j_m(N, index) for index in indices]
        assert_array_equal(predicted_jm, jm_vals)

    def test_is_j_m(self):
        """
        Tests to check validity of |j, m> given N
        """
        N = 6
        jm_vals = [(3., 3.), (3, 2), (3, 1), (3, 0),
                   (2., 3.), (2, 2.),  (2, 1), (2, 0),
                   (1., 3.), (1., 2.), (1., 1.), (1, 0),
                   (0., 3.), (0., 2.), (0., 1.), (0., 0.),
                   (-1., 3.), (-1., 2.), (-1., 1.), (-1, 0),
                   (-2., 3.), (-2., 2.), (-2, 1), (-2, 0),
                   (-3., 3.), (-3., 2), (-3, 1), (-3, 0)]

        valid_jm = [True, False, False, False,
                    True, True,  False, False,
                    True, True, True, False,
                    True, True, True, True,
                    True, True, True, False,
                    True, True, False, False,
                    True, False, False, False]

        jm_check = [is_j_m(N, jm) for jm in jm_vals]
        assert_array_equal(jm_check, valid_jm)

    def test_get_k(self):
        """
        Tests the calculation of row number for the dicke state element
        """
        N = 6
        indices = [(0, 0), (0, 1), (0, 2), (0, 3),
                   (1, 0), (1, 1), (1, 2), (1, 3),
                   (2, 0), (2, 1), (2, 2), (2, 3),
                   (3, 0), (3, 1), (3, 2), (3, 3),
                   (4, 0), (4, 1), (4, 2), (4, 3),
                   (5, 0), (5, 1), (5, 2), (5, 3),
                   (6, 0), (6, 1), (6, 2), (6, 3)]

        k_values = [0, False, False, False,
                    1, 7, False, False,
                    2, 8, 12, False,
                    3, 9, 13, 15,
                    4, 10, 14, False,
                    5, 11, False, False,
                    6, False, False, False]

        predicted_k = [get_k(N, index) for index in indices]
        assert_array_equal(predicted_k, k_values)

    def test_initial_dicke_state(self):
        """
        Test generation of an initial dicke state vector from the |jo. mo>
        value
        """
        pass


if __name__ == "__main__":
    run_module_suite()
