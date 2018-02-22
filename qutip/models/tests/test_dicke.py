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
"""
Tests for Permutation Invariance methods
"""
import numpy as np
from numpy.testing import (assert_, run_module_suite, assert_raises,
                           assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal, assert_equal)
from qutip.cy.dicke import (_get_blocks, _j_min, _j_vals, m_vals,
                            _num_dicke_states, _num_dicke_ladders,
                            get_index, jmm1_dictionary)
from qutip.cy.dicke import Dicke as _Dicke
from qutip.models.dicke import (num_tls, Piqs)
from qutip import Qobj
from qutip.models.dicke import *


class TestPiqs:
    """
    A test class for the Permutational Invariant Quantum Solver
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
        ndl = [_num_dicke_ladders(N) for N in range(1, 10)]
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
        blocks = [np.array([2]), np.array([3, 4]), np.array([6, 10, 12]),
                  np.array([8, 14, 18, 20])]
        calculated_blocks = [_get_blocks(i) for i in N_list]
        for (i, j) in zip(calculated_blocks, blocks):
            assert_array_equal(i, j)

    def test_j_vals(self):
        """
        Test calculation of j values for given N
        """
        N_list = [1, 2, 3, 4, 7]
        _j_vals_real = [np.array([0.5]), np.array([0., 1.]),
                        np.array([0.5, 1.5]),
                        np.array([0., 1., 2.]),
                        np.array([0.5, 1.5, 2.5, 3.5])]
        _j_vals_calc = [_j_vals(i) for i in N_list]

        for (i, j) in zip(_j_vals_calc, _j_vals_real):
            assert_array_equal(i, j)

    def test_m_vals(self):
        """
        Test calculation of m values for a particular j
        """
        j_list = [0.5, 1, 1.5, 2, 2.5]
        m_real = [np.array([-0.5, 0.5]), np.array([-1, 0, 1]),
                  np.array([-1.5, -0.5, 0.5, 1.5]),
                  np.array([-2, -1, 0, 1, 2]),
                  np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])]

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
        calculated_indices = [
            get_index(
                N,
                jmm1[0],
                jmm1[1],
                jmm1[2],
                blocks) for jmm1 in jmm1_list]
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

        calculated_indices = [
            get_index(
                N,
                jmm1[0],
                jmm1[1],
                jmm1[2],
                blocks) for jmm1 in jmm1_list]
        assert_array_almost_equal(calculated_indices, indices)

        N = 3
        blocks = _get_blocks(N)
        jmm1_list = [(1.5, 1.5, 1.5), (1.5, 1.5, 0.5), (1.5, 1.5, -0.5),
                     (1.5, 1.5, -1.5), (1.5, 0.5, 0.5), (1.5, -0.5, -0.5),
                     (1.5, -1.5, -1.5), (1.5, -1.5, 1.5), (0.5, 0.5, 0.5),
                     (0.5, 0.5, -0.5), (0.5, - 0.5, 0.5),
                     (0.5, -0.5, -0.5)]

        indices = [(0, 0), (0, 1), (0, 2), (0, 3),
                   (1, 1), (2, 2), (3, 3), (3, 0),
                   (4, 4), (4, 5),
                   (5, 4), (5, 5)]

        calculated_indices = [
            get_index(
                N,
                jmm1[0],
                jmm1[1],
                jmm1[2],
                blocks) for jmm1 in jmm1_list]
        assert_array_almost_equal(calculated_indices, indices)

    def test_jmm1_dictionary(self):
        """
        Test the function to generate the mapping from jmm1 to ik matrix
        """
        d1, d2, d3, d4 = jmm1_dictionary(1)

        d1_correct = {(0, 0): (0.5, 0.5, 0.5), (0, 1): (0.5, 0.5, -0.5),
                      (1, 0): (0.5, -0.5, 0.5), (1, 1): (0.5, -0.5, -0.5)}

        d2_correct = {(0.5, -0.5, -0.5): (1, 1), (0.5, -0.5, 0.5): (1, 0),
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

        d3_correct = {15: (0.0, -0.0, -0.0), 10: (1.0, -1.0, -1.0),
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

        system = Piqs(N=N, emission=gE, pumping=gP, dephasing=gD,
                      collective_emission=gCE, collective_pumping=gCP,
                      collective_dephasing=gCD)

        lindbladian = system.lindbladian()
        Ldata = np.zeros((4, 4), dtype="complex")
        Ldata[0] = [-0.6, 0, 0, 0.6]
        Ldata[1] = [0, -0.9, 0, 0]
        Ldata[2] = [0, 0, -0.9, 0]
        Ldata[3] = [0.6, 0, 0, -0.6]

        lindbladian_correct = Qobj(Ldata, dims=[[[2], [2]], [[2], [2]]],
                                   shape=(4, 4))

        assert_array_almost_equal(lindbladian.data.toarray(), Ldata)

        N = 2
        gCE = 0.5
        gCD = 0.5
        gCP = 0.5
        gE = 0.1
        gD = 0.1
        gP = 0.1

        system = Piqs(N=N, emission=gE, pumping=gP, dephasing=gD,
                      collective_emission=gCE, collective_pumping=gCP,
                      collective_dephasing=gCD)

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

        lindbladian_correct = Qobj(Ldata, dims=[[[4], [4]], [[4], [4]]],
                                   shape=(16, 16))

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

        model = _Dicke(
            N,
            collective_emission=collective_emission,
            emission=emission,
            dephasing=dephasing,
            pumping=pumping,
            collective_pumping=collective_pumping)

        tau_calculated = [model.gamma3((3, 1, 1)),
                          model.gamma2((2, 1, 1)),
                          model.gamma4((1, 1, 1)),
                          model.gamma5((3, 0, 0)),
                          model.gamma1((2, 0, 0)),
                          model.gamma6((1, 0, 0)),
                          model.gamma7((3, -1, -1)),
                          model.gamma8((2, -1, -1)),
                          model.gamma9((1, -1, -1))]

        tau_real = [2., 8., 0.333333,
                    1.5, -19.5, 0.666667,
                    2., 8., 0.333333]

        assert_array_almost_equal(tau_calculated, tau_real)

    def test_j_algebra(self):
        """
        Test calculation of the j algebra relation for the total operators.

        The [jx, jy, jz, jp, jm] for a given N in the (j, m, m1) basis should
        follow the following algebra
        [jx, jy] == 1j * jz, [jp, jm] == 2 * jz, jx^2 + jy^2 + jz^2 == j2^2.
        """
        N_list = [1, 2, 3, 4, 7]

        for nn in N_list:

            # tests 1

            [jX, jY, jZ, jP, jM] = j_algebra(nn)

            test_jxjy = jX * jY - jY * jX
            true_jxjy = 1j * jZ

            test_jpjm = jP * jM - jM * jP
            true_jpjm = 2 * jZ

            test_j2 = jX**2 + jY**2 + jZ**2
            true_j2 = j2_op(nn)

            assert_array_equal(test_jxjy, true_jxjy)
            assert_array_equal(test_jpjm, true_jpjm)
            assert_array_equal(test_j2, true_j2)

            # tests 2

            [jX, jY, jZ, jP, jM] = j_algebra(nn)

            test_jxjy = jx_op(nn) * jy_op(nn) - jy_op(nn) * jx_op(nn)
            true_jxjy = 1j * jz_op(nn)

            test_jpjm = jp_op(nn) * jm_op(nn) - jm_op(nn) * jp_op(nn)
            true_jpjm = 2 * jz_op(nn)

            test_j2 = jx_op(nn)**2 + jy_op(nn)**2 + jz_op(nn)**2
            true_j2 = j2_op(nn)

            assert_array_equal(test_jxjy, true_jxjy)
            assert_array_equal(test_jpjm, true_jpjm)
            assert_array_equal(test_j2, true_j2)

            # tests 3

            [jX, jY, jZ, jP, jM] = j_algebra(nn)

            assert_array_equal(jX, jx_op(nn))
            assert_array_equal(jY, jy_op(nn))
            assert_array_equal(jZ, jz_op(nn))
            assert_array_equal(jP, jp_op(nn))
            assert_array_equal(jM, jm_op(nn))

    def test_j_min_(self):
        """
        Test the `j_min` function
        """
        even = [2, 4, 6, 8]
        odd = [1, 3, 5, 7]

        for i in even:
            assert_(_j_min(i) == 0)

        for i in odd:
            assert_(_j_min(i) == 0.5)

    def test_energy_degeneracy(self):
        """
        Test the energy degeneracy (m) of Dicke state | j, m >

        """

        true_en_deg = [1, 1, 1, 1, 1]
        true_en_deg_even = [2, 6, 20]
        true_en_deg_odd = [1, 1, 3, 3, 35, 35]

        test_en_deg = []
        test_en_deg_even = []
        test_en_deg_odd = []

        for nn in [1, 2, 3, 4, 7]:
            test_en_deg.append(energy_degeneracy(nn, nn / 2))

        for nn in [2, 4, 6]:
            test_en_deg_even.append(energy_degeneracy(nn, 0))

        for nn in [1, 3, 7]:
            test_en_deg_odd.append(energy_degeneracy(nn, 1 / 2))
            test_en_deg_odd.append(energy_degeneracy(nn, -1 / 2))

        assert_array_equal(test_en_deg, true_en_deg)
        assert_array_equal(test_en_deg_even, true_en_deg_even)
        assert_array_equal(test_en_deg_odd, true_en_deg_odd)

    def test_state_degeneracy(self):
        """
        Test the calculation of the degeneracy of the Dicke state |j, m>,
        state_degeneracy(N, j).
        """
        true_state_deg = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 14, 14, 42, 42]
        state_deg = []

        state_deg = []
        for nn in [1, 2, 3, 4, 7, 8, 9, 10]:
            state_deg.append(state_degeneracy(nn, nn / 2))
        for nn in [1, 2, 3, 4, 7, 8, 9, 10]:
            state_deg.append(state_degeneracy(nn, (nn / 2) % 1))

        assert_array_equal(state_deg, true_state_deg)

    def test_m_degeneracy(self):
        """
        Tests the degeneracy of TLS states with same m eigenvalue
        """
        true_m_deg = [1, 2, 2, 3, 4, 5, 5, 6]
        m_deg = []

        for nn in [1, 2, 3, 4, 7, 8, 9, 10]:
            m_deg.append(m_degeneracy(nn, -(nn / 2) % 1))

        assert_array_equal(m_deg, true_m_deg)

    def test_dicke_basis(self):
        """
        Test if the Dicke basis (j, m, m') is constructed correctly.
        We test the state with for N = 2,

        0   0   0.3 0
        0   0.5 0   0
        0.3 0   0   0
        0   0   0   0.5
        """
        N = 2

        true_dicke_basis = np.zeros((4, 4))
        true_dicke_basis[1, 1] = 0.5
        true_dicke_basis[-1, -1] = 0.5
        true_dicke_basis[0, 2] = 0.3
        true_dicke_basis[2, 0] = 0.3
        true_dicke_basis = Qobj(true_dicke_basis)

        jmm1_1 = {(N / 2, 0, 0): 0.5}
        jmm1_2 = {(0, 0, 0): 0.5}
        jmm1_3 = {(N / 2, N / 2, N / 2 - 2): 0.3}
        jmm1_4 = {(N / 2, N / 2 - 2, N / 2): 0.3}

        db1 = dicke_basis(2, jmm1_1)
        db2 = dicke_basis(2, jmm1_2)
        db3 = dicke_basis(2, jmm1_3)
        db4 = dicke_basis(2, jmm1_4)
        test_dicke_basis = db1 + db2 + db3 + db4

        assert_equal(test_dicke_basis, true_dicke_basis)

    def test_dicke_state(self):
        """
        Tests the calculation of the Dicke state as a pure state in
        the diagonal density matrix of the (j, m, m') basis of size
        (O(N^2), O(N^2)).

        For N = 2 we test that the following states are correctly initialized

        excited, (N, j, m) = (2, 1, 1)

        1 0 0 0
        0 0 0 0
        0 0 0 0
        0 0 0 0

        superradiant, (N, j, m) = (2, 1, 0)

        0 0 0 0
        0 1 0 0
        0 0 0 0
        0 0 0 0

        subradiant, (N, j, m) = (2, 0, 0)

        0 0 0 0
        0 0 0 0
        0 0 0 0
        0 0 0 1

        """
        true_excited = np.zeros((4, 4))
        true_excited[0, 0] = 1

        true_superradiant = np.zeros((4, 4))
        true_superradiant[1, 1] = 1

        true_subradiant = np.zeros((4, 4))
        true_subradiant[-1, -1] = 1

        test_excited = dicke_state(2, 1, 1)
        test_superradiant = dicke_state(2, 1, 0)
        test_subradiant = dicke_state(2, 0, 0)

        assert_equal(test_excited, Qobj(true_excited))
        assert_equal(test_superradiant, Qobj(true_superradiant))
        assert_equal(test_subradiant, Qobj(true_subradiant))

    def test_superradiant(self):
        """
        Tests the calculation of the superradiant state density matrix.
        The state is |N/2, 0> for N even and |N/2, 0.5> for N odd.
        The matrix has size (O(N^2), O(N^2)) in Dicke basis ('dicke').
        The matrix has size (2^N, 2^N) in the uncoupled basis ('uncoupled').
        """
        N = 3
        true_state = np.zeros((6, 6))
        true_state[1, 1] = 1
        true_state = Qobj(true_state)

        test_state = superradiant(N)
        assert_equal(test_state, true_state)

        N = 4
        true_state = np.zeros((9, 9))
        true_state[2, 2] = 1
        true_state = Qobj(true_state)

        test_state = superradiant(N)
        assert_equal(test_state, true_state)

    def test_ghz(self):
        """
        Tests the calculation of the density matrix of the GHZ state.
        Test for N = 2 in the 'dicke' and in the 'uncoupled' basis
        """
        ghz_dicke = Qobj([[0.5, 0, 0.5, 0], [0, 0, 0, 0],
                          [0.5, 0, 0.5, 0], [0, 0, 0, 0]])

        ghz_uncoupled = Qobj([[0.5, 0, 0, 0.5], [0, 0, 0, 0], [
                             0, 0, 0, 0], [0.5, 0, 0, 0.5]])
        ghz_uncoupled.dims = [[2, 2], [2, 2]]

        assert_equal(ghz(2), ghz_dicke)
        assert_equal(ghz(2, "uncoupled"), ghz_uncoupled)

    def test_uncoupled_identity(self):
        """
        Tests the calculation of the identity matrix in a 2**N dimensional
        Hilbert space.

        The space is a tensor product of N TLSs. Test performed for N = 2.
        """
        true_id = Qobj(np.diag([1, 1, 1, 1]))
        true_id.dims = [[2, 2], [2, 2]]
        assert_equal(true_id, uncoupled_identity(2))

    def test_c_ops_tls(self):
        """
        Tests the calculation of the correct collapse operators (c_ops) list.
        In the "uncoupled" basis of N two-level system (TLS).
        The test is performed for N = 2 and emission = 1.
        """
        c1 = Qobj([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0],
                   [0, 1, 0, 0]], dims=[[2, 2], [2, 2]])
        c2 = Qobj([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0],
                   [0, 0, 1, 0]], dims=[[2, 2], [2, 2]])
        true_c_ops = [c1, c2]

        assert_equal(true_c_ops, c_ops_tls(N=2, emission=1))

    def test_get_blocks(self):
        """
        Tests the calculation of list of cumulative elements at each block.

        For N = 4

        1 1 1 1 1
        1 1 1 1 1
        1 1 1 1 1
        1 1 1 1 1
        1 1 1 1 1
                1 1 1
                1 1 1
                1 1 1
                     1

        Thus, the blocks are [5, 8, 9] denoting that after the first block 5
        elements have been accounted for and so on.
        """
        trueb1 = [2]
        trueb2 = [3, 4]
        trueb3 = [4, 6]
        trueb4 = [5, 8, 9]

        test_b1 = _get_blocks(1)
        test_b2 = _get_blocks(2)
        test_b3 = _get_blocks(3)
        test_b4 = _get_blocks(4)

        assert_equal(test_b1, trueb1)
        assert_equal(test_b2, trueb2)
        assert_equal(test_b3, trueb3)

    def test_lindbladian_dims(self):
        """
        Tests the calculation of the lindbladian matrix
        """

        true_L = [[-4, 0, 0, 3], [0, -3.54999995, 0, 0],
                  [0, 0, -3.54999995, 0], [4, 0, 0, -3]]

        true_L = Qobj(true_L)
        true_L.dims = [[[2], [2]], [[2], [2]]]

        N = 1
        test_dicke = _Dicke(N=N, pumping=1, collective_pumping=2,
                            emission=1, collective_emission=3,
                            dephasing=0.1)
        test_L = test_dicke.lindbladian()
        assert_array_almost_equal(test_L.full(), true_L.full())
        assert_array_equal(test_L.dims, true_L.dims)

    def test_liouvillian(self):
        """
        Tests the calculation of the liouvillian matrix
        """

        true_L = [[-4, 0, 0, 3], [0, -3.54999995, 0, 0],
                  [0, 0, -3.54999995, 0], [4, 0, 0, -3]]

        true_L = Qobj(true_L)
        true_L.dims = [[[2], [2]], [[2], [2]]]

        true_H = [[1. + 0.j, 1. + 0.j], [1. + 0.j, -1. + 0.j]]
        true_H = Qobj(true_H)
        true_H.dims = [[[2], [2]]]

        true_liouvillian = [[-4, -1.j, 1.j, 3],
                            [-1.j, -3.54999995+2.j, 0, 1.j],
                            [1.j, 0, -3.54999995-2.j, -1.j],
                            [4, +1.j, -1.j, -3]]
        true_liouvillian = Qobj(true_liouvillian)
        true_liouvillian.dims = [[[2], [2]], [[2], [2]]]

        N = 1
        test_piqs = Piqs(hamiltonian=sigmaz() + sigmax(), N=N,
                         pumping=1, collective_pumping=2, emission=1,
                         collective_emission=3, dephasing=0.1)

        test_liouvillian = test_piqs.liouvillian()
        test_hamiltonian = test_piqs.hamiltonian

        assert_array_almost_equal(
            test_liouvillian.full(),
            true_liouvillian.full())
        assert_array_almost_equal(test_hamiltonian.full(), true_H.full())
        assert_array_equal(test_liouvillian.dims, test_liouvillian.dims)

    def test_gamma1(self):
        """
        Tests the calculation of gamma1.
        """

        true_gamma_1 = -2
        true_gamma_2 = -3
        true_gamma_3 = -7
        true_gamma_4 = -1
        true_gamma_5 = 0
        true_gamma_6 = 0

        N = 4
        test_dicke = _Dicke(N=N, collective_emission=1)
        test_gamma_1 = test_dicke.gamma1((1, 1, 1))
        test_dicke = _Dicke(N=N, emission=1)
        test_gamma_2 = test_dicke.gamma1((1, 1, 1))
        test_dicke = _Dicke(N=N, emission=1, collective_emission=2)
        test_gamma_3 = test_dicke.gamma1((1, 1, 1))
        test_dicke = _Dicke(N=N, dephasing=4)
        test_gamma_4 = test_dicke.gamma1((1, 1, 1))
        test_dicke = _Dicke(N=N, collective_pumping=2)
        test_gamma_5 = test_dicke.gamma1((1, 1, 1))
        test_dicke = _Dicke(N=N, collective_dephasing=2)
        test_gamma_6 = test_dicke.gamma1((1, 1, 1))

        assert_almost_equal(true_gamma_1, test_gamma_1)
        assert_almost_equal(true_gamma_2, test_gamma_2)
        assert_almost_equal(true_gamma_3, test_gamma_3)
        assert_almost_equal(true_gamma_4, test_gamma_4)
        assert_almost_equal(true_gamma_5, test_gamma_5)
        assert_almost_equal(true_gamma_6, test_gamma_6)

    def test_gamma2(self):
        """
        Tests the calculation of gamma2. Test performed for N = 4.
        """
        true_gamma_1 = 2
        true_gamma_2 = 1.5
        true_gamma_3 = 5.5
        true_gamma_4 = 0
        true_gamma_5 = 0
        true_gamma_6 = 0

        N = 4
        test_dicke = _Dicke(N=N, collective_emission=1)
        test_gamma_1 = test_dicke.gamma2((1, 1, 1))
        test_dicke = _Dicke(N=N, emission=1)
        test_gamma_2 = test_dicke.gamma2((1, 1, 1))
        test_dicke = _Dicke(N=N, emission=1, collective_emission=2)
        test_gamma_3 = test_dicke.gamma2((1, 1, 1))
        test_dicke = _Dicke(N=N, dephasing=4)
        test_gamma_4 = test_dicke.gamma2((1, 1, 1))
        test_dicke = _Dicke(N=N, collective_pumping=2)
        test_gamma_5 = test_dicke.gamma2((1, 1, 1))
        test_dicke = _Dicke(N=N, collective_dephasing=2)
        test_gamma_6 = test_dicke.gamma2((1, 1, 1))

        assert_almost_equal(true_gamma_1, test_gamma_1)
        assert_almost_equal(true_gamma_2, test_gamma_2)
        assert_almost_equal(true_gamma_3, test_gamma_3)
        assert_almost_equal(true_gamma_4, test_gamma_4)
        assert_almost_equal(true_gamma_5, test_gamma_5)
        assert_almost_equal(true_gamma_6, test_gamma_6)

    def test_gamma3(self):
        """
        Tests the calculation of gamma3. Test performed for N = 4.
        """
        true_gamma_1 = 0
        true_gamma_2 = 1.3333333730697632
        true_gamma_3 = 1.3333333730697632
        true_gamma_4 = 0
        true_gamma_5 = 0
        true_gamma_6 = 0

        N = 4
        test_dicke = _Dicke(N=N, collective_emission=1)
        test_gamma_1 = test_dicke.gamma3((1, 1, 1))
        test_dicke = _Dicke(N=N, emission=1)
        test_gamma_2 = test_dicke.gamma3((1, 1, 1))
        test_dicke = _Dicke(N=N, emission=1, collective_emission=2)
        test_gamma_3 = test_dicke.gamma3((1, 1, 1))
        test_dicke = _Dicke(N=N, dephasing=4)
        test_gamma_4 = test_dicke.gamma3((1, 1, 1))
        test_dicke = _Dicke(N=N, collective_pumping=2)
        test_gamma_5 = test_dicke.gamma3((1, 1, 1))
        test_dicke = _Dicke(N=N, collective_dephasing=2)
        test_gamma_6 = test_dicke.gamma3((1, 1, 1))
        #
        assert_almost_equal(true_gamma_1, test_gamma_1)
        assert_almost_equal(true_gamma_2, test_gamma_2)
        assert_almost_equal(true_gamma_3, test_gamma_3)
        assert_almost_equal(true_gamma_4, test_gamma_4)
        assert_almost_equal(true_gamma_5, test_gamma_5)
        assert_almost_equal(true_gamma_6, test_gamma_6)

    def test_gamma4(self):
        """
        Tests the calculation of gamma4. Test performed for N = 4.
        """

        true_gamma_1 = 0.1666666716337204
        true_gamma_2 = 2
        true_gamma_3 = 0
        true_gamma_4 = 0.40824830532073975

        N = 4
        test_dicke = _Dicke(N=N, emission=1, collective_emission=2)
        test_gamma_1 = test_dicke.gamma4((1, 1, 1))
        test_gamma_2 = test_dicke.gamma4((0, 0, 0))
        test_gamma_3 = test_dicke.gamma4((2, 1, 1))
        test_gamma_4 = test_dicke.gamma4((1, -1, 1))

        assert_almost_equal(true_gamma_1, test_gamma_1)
        assert_almost_equal(true_gamma_2, test_gamma_2)
        assert_almost_equal(true_gamma_3, test_gamma_3)
        assert_almost_equal(true_gamma_4, test_gamma_4)

    def test_gamma5(self):
        """
        Tests the calculation of gamma5. Test performed for N = 4.
        """

        true_gamma_1 = 0
        true_gamma_2 = 0
        true_gamma_3 = 0.75
        true_gamma_4 = 0

        N = 4
        test_dicke = _Dicke(N=N, dephasing=1)
        test_gamma_1 = test_dicke.gamma5((1, 1, 1))
        test_gamma_2 = test_dicke.gamma5((0, 0, 0))
        test_gamma_3 = test_dicke.gamma5((2, 1, 1))
        test_gamma_4 = test_dicke.gamma5((1, -1, 1))

        assert_almost_equal(true_gamma_1, test_gamma_1)
        assert_almost_equal(true_gamma_2, test_gamma_2)
        assert_almost_equal(true_gamma_3, test_gamma_3)
        assert_almost_equal(true_gamma_4, test_gamma_4)

    def test_gamma6(self):
        """
        Tests the calculation of gamma6. Test performed for N = 4.
        """
        true_gamma_1 = 0.25
        true_gamma_2 = 1
        true_gamma_3 = 0
        true_gamma_4 = 0.25

        N = 4
        test_dicke = _Dicke(N=N, dephasing=1)
        test_gamma_1 = test_dicke.gamma6((1, 1, 1))
        test_gamma_2 = test_dicke.gamma6((0, 0, 0))
        test_gamma_3 = test_dicke.gamma6((2, 1, 1))
        test_gamma_4 = test_dicke.gamma6((1, -1, 1))

        assert_almost_equal(true_gamma_1, test_gamma_1)
        assert_almost_equal(true_gamma_2, test_gamma_2)
        assert_almost_equal(true_gamma_3, test_gamma_3)
        assert_almost_equal(true_gamma_4, test_gamma_4)

    def test_gamma7(self):
        """
        Tests the calculation of gamma7. Test performed for N = 4.
        """
        true_gamma_1 = 0
        true_gamma_2 = 0.5
        true_gamma_3 = 0
        true_gamma_4 = 1.5

        N = 4
        test_dicke = _Dicke(N=N, pumping=1)
        test_gamma_1 = test_dicke.gamma7((1, 1, 1))
        test_gamma_2 = test_dicke.gamma7((2, 0, 0))
        test_gamma_3 = test_dicke.gamma7((1, 0, 0))
        test_gamma_4 = test_dicke.gamma7((2, -1, -1))

        assert_almost_equal(true_gamma_1, test_gamma_1)
        assert_almost_equal(true_gamma_2, test_gamma_2)
        assert_almost_equal(true_gamma_3, test_gamma_3)
        assert_almost_equal(true_gamma_4, test_gamma_4)

    def test_gamma8(self):
        """
        Tests the calculation of gamma8. Test performed for N = 4.
        """
        true_gamma_1 = 0
        true_gamma_2 = 13.5
        true_gamma_3 = 5.5
        true_gamma_4 = 13.5

        N = 4
        test_dicke = _Dicke(N=N, pumping=1, collective_pumping=2)
        test_gamma_1 = test_dicke.gamma8((1, 1, 1))
        test_gamma_2 = test_dicke.gamma8((2, 0, 0))
        test_gamma_3 = test_dicke.gamma8((1, 0, 0))
        test_gamma_4 = test_dicke.gamma8((2, -1, -1))

        assert_almost_equal(true_gamma_1, test_gamma_1)
        assert_almost_equal(true_gamma_2, test_gamma_2)
        assert_almost_equal(true_gamma_3, test_gamma_3)
        assert_almost_equal(true_gamma_4, test_gamma_4)

    def test_gamma9(self):
        """
        Tests the calculation of gamma9. Test performed for N = 4.
        """
        true_gamma_1 = 1
        true_gamma_2 = 0
        true_gamma_3 = 0.5
        true_gamma_4 = 0

        N = 4
        test_dicke = _Dicke(N=N, pumping=1, collective_pumping=2)
        test_gamma_1 = test_dicke.gamma9((1, 1, 1))
        test_gamma_2 = test_dicke.gamma9((2, 0, 0))
        test_gamma_3 = test_dicke.gamma9((1, 0, 0))
        test_gamma_4 = test_dicke.gamma9((2, -1, -1))

        assert_almost_equal(true_gamma_1, test_gamma_1)
        assert_almost_equal(true_gamma_2, test_gamma_2)
        assert_almost_equal(true_gamma_3, test_gamma_3)
        assert_almost_equal(true_gamma_4, test_gamma_4)


if __name__ == "__main__":
    run_module_suite()
