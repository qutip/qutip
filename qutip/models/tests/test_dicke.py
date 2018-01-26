"""
Tests for Permutation Invariance methods
"""
import numpy as np
from numpy.testing import (assert_, run_module_suite, assert_raises,
                           assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal, assert_equal)

from piqs.dicke import (num_tls, Piqs)

from piqs.cy.dicke import (_get_blocks, _j_min, _j_vals, m_vals, _num_dicke_states,
                            _num_dicke_ladders, get_index, jmm1_dictionary)
from piqs.cy.dicke import Dicke as _Dicke
from qutip import Qobj


class TestPim:
    """
    A test class for the Permutation invariance matrix generation
    """
    def test_num_dicke_states(self):
        """
        Tests the `num_dicke_state` function
        """
        N_list = [1, 2, 3, 4, 5, 6, 9, 10, 20, 100, 123]
        dicke_states = [_num_dicke_states(i) for i in N_list]

        assert_array_equal(dicke_states, [2, 4, 6, 9, 12, 16, 30, 36, 121,
                                          2601, 3906])

        N = -1
        assert_raises(ValueError, _num_dicke_states, N)

        N = 0.2
        assert_raises(ValueError, _num_dicke_states, N)

    def test_num_tls(self):
        """
        Tests the `num_two_level` function.
        """
        N_dicke = [2, 4, 6, 9, 12, 16, 30, 36, 121, 2601, 3906]
        N = [1, 2, 3, 4, 5, 6, 9, 10, 20, 100, 123]

        calculated_N = [num_tls(i) for i in N_dicke]

        assert_array_equal(calculated_N, N)
    
    def test_num_dicke_ladders(self):
        """
        Tests the `_num_dicke_ladders` function
        """
        ndl_true = [1, 2, 2, 3, 3, 4, 4, 5, 5]
        ndl = [_num_dicke_ladders(N) for N in range (1, 10)]        
        assert_array_equal(ndl, ndl_true)    
     
    def test_j_min(self):
        """
        Test the `_j_min` function
        """
        even = [2, 4, 6, 8]
        odd = [1, 3, 5, 7]

        for i in even:
            assert_(_j_min(i) == 0)

        for i in odd:
            assert_(_j_min(i) == 0.5)

    def test_get_blocks(self):
        """
        Test the function to get blocks
        """
        N_list = [1, 2, 5, 7]
        blocks = [np.array([2]), np.array([3, 4]), np.array([ 6, 10, 12]),
                  np.array([ 8, 14, 18, 20])]
        calculated_blocks = [_get_blocks(i) for i in N_list]
        for (i, j) in zip(calculated_blocks, blocks):
            assert_array_equal(i, j)

    def test_j_vals(self):
        """
        Test calculation of j values for given N
        """
        N_list = [1, 2, 3, 4, 7]
        _j_vals_real = [np.array([ 0.5]), np.array([ 0.,  1.]),
                       np.array([ 0.5,  1.5]),
                       np.array([ 0.,  1.,  2.]),
                       np.array([ 0.5,  1.5,  2.5,  3.5])]
        _j_vals_calc = [_j_vals(i) for i in N_list]

        for (i, j) in zip(_j_vals_calc, _j_vals_real):
            assert_array_equal(i, j)

    def test_m_vals(self):
        """
        Test calculation of m values for a particular j
        """
        j_list = [0.5, 1, 1.5, 2, 2.5]
        m_real = [np.array([-0.5,  0.5]), np.array([-1,  0,  1]),
                  np.array([-1.5, -0.5,  0.5,  1.5]),
                  np.array([-2, -1,  0,  1,  2]),
                  np.array([-2.5, -1.5, -0.5,  0.5,  1.5,  2.5])]
        
        m_calc = [m_vals(i) for i in j_list]
        for (i, j) in zip(m_real, m_calc):
            assert_array_equal(i, j)

    def test_get_index(self):
        """
        Test the index fetching function for given j, m, m1 value
        """
        N = 1
        jmm1_list = [(0.5, 0.5, 0.5), (0.5, 0.5, -0.5), 
                     (0.5, -0.5, 0.5), (0.5, -0.5, -0.5)]
        indices = [(0, 0), (0, 1), (1, 0), (1, 1)]

        blocks = _get_blocks(N)
        calculated_indices = [get_index(N, jmm1[0], jmm1[1], jmm1[2], blocks) for jmm1 in jmm1_list]
        assert_array_almost_equal(calculated_indices, indices)

        N = 2
        blocks = _get_blocks(N)
        jmm1_list = [(1, 1, 1), (1, 1, 0), (1, 1, -1), 
                     (1, 0, 1), (1, 0, 0), (1, 0, -1),
                     (1, -1, 1), (1, -1, 0), (1, -1, -1),
                     (0, 0, 0)]
        
        indices = [(0, 0), (0, 1), (0, 2),
                    (1, 0), (1, 1), (1, 2),
                    (2, 0), (2, 1), (2, 2),
                    (3, 3)]

        calculated_indices = [get_index(N, jmm1[0], jmm1[1], jmm1[2], blocks) for jmm1 in jmm1_list]
        assert_array_almost_equal(calculated_indices, indices)

        N = 3
        blocks = _get_blocks(N)
        jmm1_list = [(1.5, 1.5, 1.5), (1.5, 1.5, 0.5), (1.5, 1.5, -0.5), (1.5, 1.5, -1.5),
                     (1.5, 0.5, 0.5), (1.5, -0.5, -0.5), (1.5, -1.5, -1.5), (1.5, -1.5, 1.5),
                     (0.5, 0.5, 0.5), (0.5, 0.5, -0.5),
                     (0.5, -0.5, 0.5), (0.5, -0.5, -0.5)]
        
        indices = [(0, 0), (0, 1), (0, 2), (0, 3),
                   (1, 1), (2, 2), (3, 3), (3, 0),
                   (4, 4), (4, 5),
                   (5, 4), (5, 5)]

        calculated_indices = [get_index(N, jmm1[0], jmm1[1], jmm1[2], blocks) for jmm1 in jmm1_list]
        assert_array_almost_equal(calculated_indices, indices)

    def test_jmm1_dictionary(self):
        """
        Test the function to generate the mapping from jmm1 to ik matrix
        """
        d1, d2, d3, d4 = jmm1_dictionary(1)

        d1_correct = {(0, 0): (0.5, 0.5, 0.5), (0, 1): (0.5, 0.5, -0.5),
                        (1, 0): (0.5, -0.5, 0.5), (1, 1): (0.5, -0.5, -0.5)}

        d2_correct = {(0.5, -0.5, -0.5): (1, 1),(0.5, -0.5, 0.5): (1, 0),
                            (0.5, 0.5, -0.5): (0, 1),
                            (0.5, 0.5, 0.5): (0, 0)}

        d3_correct = {0: (0.5, 0.5, 0.5), 1: (0.5, 0.5, -0.5),
                        2: (0.5, -0.5, 0.5),
                        3: (0.5, -0.5, -0.5)}

        d4_correct = {(0.5, -0.5, -0.5): 3, (0.5, -0.5, 0.5): 2, 
                        (0.5, 0.5, -0.5): 1, (0.5, 0.5, 0.5): 0}

        assert_equal(d1, d1_correct)
        assert_equal(d2, d2_correct)
        assert_equal(d3, d3_correct)
        assert_equal(d4, d4_correct)


        d1, d2, d3, d4 = jmm1_dictionary(2)

        d1_correct = {(3, 3): (0.0, -0.0, -0.0), (2, 2): (1.0, -1.0, -1.0),
                        (2, 1): (1.0, -1.0, 0.0), (2, 0): (1.0, -1.0, 1.0),
                        (1, 2): (1.0, 0.0, -1.0), (1, 1): (1.0, 0.0, 0.0),
                        (1, 0): (1.0, 0.0, 1.0), (0, 2): (1.0, 1.0, -1.0),
                        (0, 1): (1.0, 1.0, 0.0), (0, 0): (1.0, 1.0, 1.0)}

        d2_correct = {(0.0, -0.0, -0.0): (3, 3), (1.0, -1.0, -1.0): (2, 2),
                        (1.0, -1.0, 0.0): (2, 1), (1.0, -1.0, 1.0): (2, 0),
                        (1.0, 0.0, -1.0): (1, 2), (1.0, 0.0, 0.0): (1, 1),
                        (1.0, 0.0, 1.0): (1, 0), (1.0, 1.0, -1.0): (0, 2),
                        (1.0, 1.0, 0.0): (0, 1), (1.0, 1.0, 1.0): (0, 0)}

        d3_correct ={15: (0.0, -0.0, -0.0), 10: (1.0, -1.0, -1.0),
                        9: (1.0, -1.0, 0.0), 8: (1.0, -1.0, 1.0),
                        6: (1.0, 0.0, -1.0), 5: (1.0, 0.0, 0.0),
                        4: (1.0, 0.0, 1.0), 2: (1.0, 1.0, -1.0),
                        1: (1.0, 1.0, 0.0), 0: (1.0, 1.0, 1.0)}

        d4_correct = {(0.0, -0.0, -0.0): 15, (1.0, -1.0, -1.0): 10,
                        (1.0, -1.0, 0.0): 9, (1.0, -1.0, 1.0): 8,
                        (1.0, 0.0, -1.0): 6, (1.0, 0.0, 0.0): 5,
                        (1.0, 0.0, 1.0): 4, (1.0, 1.0, -1.0): 2,
                        (1.0, 1.0, 0.0): 1, (1.0, 1.0, 1.0): 0}
        
        assert_equal(d1, d1_correct)
        assert_equal(d2, d2_correct)
        assert_equal(d3, d3_correct)
        assert_equal(d4, d4_correct)
    
    def test_lindbladian(self):
        """
        Test the generation of the Lindbladian matrix
        """
        N = 1
        gCE = 0.5
        gCD = 0.5
        gCP = 0.5
        gE = 0.1
        gD = 0.1
        gP = 0.1

        system = Piqs(N = N, emission = gE, pumping = gP, dephasing = gD,
                        collective_emission = gCE, collective_pumping = gCP,
                        collective_dephasing = gCD)

        lindbladian = system.lindbladian()
        Ldata = np.zeros((4, 4), dtype="complex")
        Ldata[0] = [-0.6, 0, 0, 0.6]
        Ldata[1] = [0, -0.9, 0, 0]
        Ldata[2] = [0, 0, -0.9, 0]
        Ldata[3] = [0.6, 0, 0, -0.6]

        lindbladian_correct = Qobj(Ldata, dims= [[[2], [2]], [[2], [2]]],
                                   shape = (4, 4))

        assert_array_almost_equal(lindbladian.data.toarray(), Ldata)

        N = 2
        gCE = 0.5
        gCD = 0.5
        gCP = 0.5
        gE = 0.1
        gD = 0.1
        gP = 0.1

        system = Piqs(N = N, emission = gE, pumping = gP, dephasing = gD,
                        collective_emission = gCE, collective_pumping = gCP,
                        collective_dephasing = gCD)

        lindbladian = system.lindbladian()

        Ldata = np.zeros((16, 16), dtype="complex")

        Ldata[0][0], Ldata[0][5], Ldata[0][15] = -1.2, 1.1, 0.1
        Ldata[1, 1], Ldata[1, 6] = -2, 1.1
        Ldata[2, 2] = -2.2999999999999998
        Ldata[4, 4], Ldata[4, 9] = -2, 1.1
        Ldata[5, 0], Ldata[5, 5], Ldata[5, 10], Ldata[5, 15] = (1.1, -2.25,
                                                                1.1, 0.05)
        Ldata[6, 1], Ldata[6, 6] = 1.1, -2
        Ldata[8, 8] = -2.2999999999999998
        Ldata[9, 4], Ldata[9, 9] = 1.1, -2
        Ldata[10, 5], Ldata[10, 10], Ldata[10, 15] = 1.1, -1.2, 0.1
        Ldata[15, 0], Ldata[15, 5], Ldata[15, 10], Ldata[15, 15] = (0.1,
                                                                    0.05,
                                                                    0.1,
                                                                    -0.25)

        lindbladian_correct = Qobj(Ldata, dims= [[[4], [4]], [[4], [4]]],
                                    shape = (16, 16))

        assert_array_almost_equal(lindbladian.data.toarray(), Ldata)


    def test_gamma(self):
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
        collective_emission = 1.
        emission = 1.
        dephasing = 1.
        pumping = 1.
        collective_pumping = 1.

        model = _Dicke(N, collective_emission = collective_emission, emission = emission, dephasing = dephasing,
                      pumping = pumping, collective_pumping = collective_pumping)

        tau_calculated = [model.gamma3((3, 1, 1)), model.gamma2((2, 1, 1)), model.gamma4((1, 1, 1)),
                          model.gamma5((3, 0, 0)), model.gamma1((2, 0, 0)), model.gamma6((1, 0, 0)),
                          model.gamma7((3,-1, -1)), model.gamma8((2,-1, -1)), model.gamma9((1,-1, -1))]

        tau_real = [2., 8., 0.333333,
                    1.5, -19.5, 0.666667,
                    2., 8., 0.333333]

        assert_array_almost_equal(tau_calculated, tau_real)


if __name__ == "__main__":
    run_module_suite()
