"""
Tests for Permutation Invariance methods
"""
import numpy as np
from numpy.testing import (assert_, run_module_suite, assert_raises,
                           assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal)

from qutip.pim.dicke import (num_dicke_states, num_two_level, irreducible_dim,
                             num_dicke_ladders, generate_dicke_space, initial_dicke_state,
                             Pim, _tau_column_index)

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

        model = Pim(N)
        test_indices = [(0, 0), (0, 1), (1, 0), (-1, -1), (2, -1)]
        dicke_labels = [model.isdicke(x, y) for (x, y) in test_indices]

        assert_array_equal(dicke_labels, [True, False, True, False, False])

    def test_get_j_m(self):
        """
        Tests `get_j_m` function to get the (j, m) values.

        For N = 6 |j, m> would be :

        | 3, 3>
        | 3, 2> | 2, 2>
        | 3, 1> | 2, 1> | 1, 1>
        | 3, 0> | 2, 0> | 1, 0> |0, 0>
        | 3,-1> | 2,-1> | 1,-1>
        | 3,-2> | 2,-2>
        | 3,-3>

        """
        N = 6

        model = Pim(N)

        indices = [(0, 0), (0, 1), (0, 2), (0, 3),
                   (1, 0), (1, 1), (1, 2), (1, 3),
                   (2, 0), (2, 1), (2, 2), (2, 3),
                   (3, 0), (3, 1), (3, 2), (3, 3),
                   (4, 0), (4, 1), (4, 2), (4, 3),
                   (5, 0), (5, 1), (5, 2), (5, 3),
                   (6, 0), (6, 1), (6, 2), (6, 3)]

        jm_vals = [(3., 3.), False, False, False,
                   (3., 2.), (2., 2.),  False, False,
                   (3., 1.), (2., 1.), (1., 1.), False,
                   (3., 0.), (2., 0.), (1., 0.), (0., 0.),
                   (3.,-1.), (2.,-1.), (1.,-1.), False,
                   (3.,-2.), (2.,-2.), False, False,
                   (3.,-3.), False, False, False]

        predicted_jm = [model.get_j_m(x, y) for (x, y) in indices]
        assert_array_equal(predicted_jm, jm_vals)

    def test_is_j_m(self):
        """
        Tests to check validity of |j, m> given N
        """
        N = 6
        model = Pim(N)

        jm_vals = [(3., 3.), (2., 3.), (1., 3.), (0., 3.),
                   (3., 2.), (2., 2.), (1., 2.), (0., 2.),
                   (3., 1.), (2., 1.), (1., 1.), (0., 1.),
                   (3., 0.), (2., 0.), (1., 0.), (0., 0.),
                   (3.,-1.), (2.,-1.), (1.,-1.), (0.,-1.),
                   (3.,-2.), (2.,-2.), (1.,-2.), (0.,-2.),
                   (3.,-3.), (2.,-3.), (1.,-3.), (0.,-3.)]

        valid_jm = [True, False, False, False,
                    True, True,  False, False,
                    True, True, True, False,
                    True, True, True, True,
                    True, True, True, False,
                    True, True, False, False,
                    True, False, False, False]

        jm_check = [model.is_j_m(x, y) for (x, y) in jm_vals]
        assert_array_equal(jm_check, valid_jm)

    def test_get_k(self):
        """
        Tests the calculation of row number for the dicke state element
        """
        N = 6

        model = Pim(N)
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

        predicted_k = [model.get_k(x, y) for (x, y) in indices]
        assert_array_equal(predicted_k, k_values)

    def test_initial_dicke_state(self):
        """
        Test generation of an initial dicke state vector from the |j0, m0>
        """
        N = 6

        jm_vals = [(3., 3.),
                   (3., 2.), (2., 2.),
                   (3., 1.), (2., 1.), (1., 1.),
                   (3., 0.), (2., 0.), (1., 0.), (0., 0.),
                   (3.,-1.), (2.,-1.), (1.,-1.),
                   (3.,-2.), (2.,-2.),
                   (3.,-3.)]

        k_values = [0,
                    1, 7,
                    2, 8, 12,
                    3, 9, 13, 15,
                    4, 10, 14,
                    5, 11,
                    6]

        rhos_calculated = [initial_dicke_state(N, jm0) for jm0 in jm_vals]

        i = 0

        for rho in rhos_calculated:
            rho_true = np.zeros(16)
            k = k_values[i]
            rho_true[k] = 1
            assert_array_equal(rho_true, rho)
            i += 1

        assert_raises(ValueError, initial_dicke_state, N, (2, 3))

    def test_taus(self):
        """
        Tests the calculation of various Tau values for a given system

        For N = 6 |j, m> would be :

        | 3, 3>
        | 3, 2> | 2, 2>
        | 3, 1> | 2, 1> | 1, 1>
        | 3, 0> | 2, 0> | 1, 0> |0, 0>
        | 3,-1> | 2,-1> | 1,-1>
        | 3,-2> | 2,-2>
        | 3,-3>
        """
        N = 6
        emission = 1.
        loss = 1.
        dephasing = 1.
        pumping = 1.
        collective_pumping = 1.

        model = Pim(N, emission, loss, dephasing, pumping, collective_pumping)

        tau_calculated = [model.tau3(3, 1), model.tau2(2, 1), model.tau4(1, 1),
                          model.tau5(3, 0), model.tau1(2, 0), model.tau6(1, 0),
                          model.tau7(3,-1), model.tau8(2,-1), model.tau9(1,-1)]

        tau_real = [2., 8., 0.333333,
                    1.5, -19.5, 0.666667,
                    2., 8., 0.333333]

        assert_array_almost_equal(tau_calculated, tau_real)

    def test_tau_column_index(self):
        """
        Tests calculation of the non-zero columns for a particular row
        """
        k, j = 0, 1

        taus = {'tau3': -5, 'tau2': -1, 'tau4': 1,
                'tau5': -4, 'tau1': 0, 'tau6': 2,
                'tau7': -3, 'tau8': 1, 'tau9': 3}

        for tau in taus:
            assert_(_tau_column_index(tau, k, j) == taus[tau])

        k, j = 2, 1

        taus = {'tau3': -3, 'tau2': 1, 'tau4': 3,
                'tau5': -2, 'tau1': 2, 'tau6': 4,
                'tau7': -1, 'tau8': 3, 'tau9': 5}

        for tau in taus:
            assert_(_tau_column_index(tau, k, j) == taus[tau])

    def test_generate_row(self):
        """
        Test generating one row of the matrix M
        """
        N = 2

        model = Pim(N)

        dicke_row, dicke_col = 0, 0

        k = model.get_k(dicke_row, dicke_col)

        row = {0: {3:1}}

        generated_row = model.generate_row(dicke_row, dicke_col)

        assert_array_equal(generated_row.keys(), row.keys())
        # ====================================================================
        # Needs more testing here
        # ====================================================================

    def test_generate_M_dict(self):
        """
        Test the function for generating the matrix dict
        """
        N = 2

        model = Pim(N)
        model.generate_M_dict()

        actual_M = {(0, 0): -4.0, (0, 1): 3.0, (0, 3): 1.0,
                    (1, 0): 3.0, (1, 1): -6.5, (1, 3): 0.5, (1, 2): 3.0,
                    (2, 1): 3.0, (2, 3): 1.0, (2, 2): -4.0,
                    (3, 0): 1.0, (3, 1): 0.5, (3, 3): -2.5, (3, 2): 1.0}

        for (row, col) in actual_M.keys():
            assert_almost_equal(model.M_dict[(row, col)], actual_M[(row, col)])

    def test_generate_M(self):
        """
        Test the function to generate the sparse matrix from the M dictionary
        """
        N = 6
        model = Pim(N)

        model.generate_M()
        #=====================================================================
        # Write test functions for the sparse matrix
        pass

if __name__ == "__main__":
    run_module_suite()
