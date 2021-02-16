# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, The QuTiP Project.
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
import numpy as np
from qutip.lattice import *
from qutip import (Qobj, tensor, basis, qeye, isherm, sigmax)
# from numpy.testing import (assert_equal, assert_, assert_almost_equal,
#                            run_module_suite)
from numpy.testing import (assert_, run_module_suite)


class TestLattice:
    """
    Tests for `qutip.lattice` class.
    """
    def test_hamiltonian(self):
        """
        lattice: Test the method Lattice1d.Hamiltonian().
        """
        # num_cell = 1
        # Four different instances
        Periodic_Atom_Chain = Lattice1d(num_cell=1, boundary="periodic")
        Aperiodic_Atom_Chain = Lattice1d(num_cell=1, boundary="aperiodic")
        p_1223 = Lattice1d(num_cell=1, boundary="periodic",
                           cell_num_site=2, cell_site_dof=[2, 3])
        ap_1223 = Lattice1d(num_cell=1, boundary="aperiodic",
                            cell_num_site=2, cell_site_dof=[2, 3])

        # Their Hamiltonians
        pHamt1 = Periodic_Atom_Chain.Hamiltonian()
        apHamt1 = Aperiodic_Atom_Chain.Hamiltonian()
        pHamt1223 = p_1223.Hamiltonian()
        apHamt1223 = ap_1223.Hamiltonian()

        # Benchmark answers
        pHamt1_C = Qobj([[0.]], dims=[[1], [1]])
        apHamt1_C = Qobj([[0.]], dims=[[1], [1]])
        site1223 = np.diag(np.zeros(2-1)-1, 1) + np.diag(np.zeros(2-1)-1, -1)
        Rpap1223 = tensor(Qobj(site1223), qeye([2, 3]))
        pap1223 = Qobj(Rpap1223, dims=[[2, 2, 3], [2, 2, 3]])

        # Checks for num_cell = 1
        assert_(pHamt1 == pHamt1_C)
        assert_(apHamt1 == apHamt1_C)
        assert_(pHamt1223 == pap1223)    # for num_cell=1, periodic and
        assert_(apHamt1223 == pap1223)   # aperiodic B.C. have same Hamiltonian

        # num_cell = 2
        # Four different instances
        Periodic_Atom_Chain = Lattice1d(num_cell=2, boundary="periodic")
        Aperiodic_Atom_Chain = Lattice1d(num_cell=2, boundary="aperiodic")

        p_2222 = Lattice1d(num_cell=2, boundary="periodic",
                           cell_num_site=2, cell_site_dof=[2, 2])
        ap_2222 = Lattice1d(num_cell=2, boundary="aperiodic",
                            cell_num_site=2, cell_site_dof=[2, 2])

        # Their Hamiltonians
        pHamt2222 = p_2222.Hamiltonian()
        apHamt2222 = ap_2222.Hamiltonian()
        pHamt2 = Periodic_Atom_Chain.Hamiltonian()
        apHamt2 = Aperiodic_Atom_Chain.Hamiltonian()

        # Benchmark answers
        pHamt2_C = Qobj([[0., -1.],
                         [-1.,  0.]], dims=[[2], [2]])
        apHamt2_C = Qobj([[0., -1.],
                          [-1., 0.]], dims=[[2], [2]])

        pap_2222 = np.zeros((16, 16), dtype=complex)
        pap_2222[0:4, 4:8] = -np.eye(4)
        pap_2222[4:8, 0:4] = -np.eye(4)
        pap_2222[4:8, 8:12] = -np.eye(4)
        pap_2222[8:12, 4:8] = -np.eye(4)
        pap_2222[8:12, 12:16] = -np.eye(4)
        pap_2222[12:16, 8:12] = -np.eye(4)
        pap_2222 = Qobj(pap_2222, dims=[[2, 2, 2, 2], [2, 2, 2, 2]])

        # Checks for num_cell = 2
        assert_(pHamt2 == pHamt2_C)
        assert_(apHamt2 == apHamt2_C)
        assert_(pHamt2222 == pap_2222)   # for num_cell=2, periodic and
        assert_(apHamt2222 == pap_2222)  # aperiodic B.C. have same Hamiltonian

        # num_cell = 3   # checking any num_cell >= 3 is pretty much equivalent
        Periodic_Atom_Chain = Lattice1d(num_cell=3, boundary="periodic")
        Aperiodic_Atom_Chain = Lattice1d(num_cell=3, boundary="aperiodic")
        p_3122 = Lattice1d(num_cell=3, boundary="periodic",
                           cell_num_site=1, cell_site_dof=[2, 2])
        ap_3122 = Lattice1d(num_cell=3, boundary="aperiodic",
                            cell_num_site=1, cell_site_dof=[2, 2])

        # Their Hamiltonians
        pHamt3 = Periodic_Atom_Chain.Hamiltonian()
        apHamt3 = Aperiodic_Atom_Chain.Hamiltonian()
        pHamt3122 = p_3122.Hamiltonian()
        apHamt3122 = ap_3122.Hamiltonian()

        # Benchmark answers
        pHamt3_C = Qobj([[0., -1., -1.],
                         [-1.,  0., -1.],
                         [-1., -1.,  0.]], dims=[[3], [3]])
        apHamt3_C = Qobj([[0., -1.,  0.],
                          [-1.,  0., -1.],
                          [0., -1.,  0.]], dims=[[3], [3]])

        Hp_3122 = np.zeros((12, 12), dtype=complex)
        Hp_3122[0:8, 4:12] = Hp_3122[0:8, 4:12] - np.eye(8)
        Hp_3122[4:12, 0:8] = Hp_3122[4:12, 0:8] - np.eye(8)
        Hp_3122[0:4, 8:12] = Hp_3122[0:4, 8:12] - np.eye(4)
        Hp_3122[8:12, 0:4] = Hp_3122[8:12, 0:4] - np.eye(4)
        Hp_3122 = Qobj(Hp_3122, dims=[[3, 2, 2], [3, 2, 2]])

        Hap_3122 = np.zeros((12, 12), dtype=complex)
        Hap_3122[0:8, 4:12] = Hap_3122[0:8, 4:12] - np.eye(8)
        Hap_3122[4:12, 0:8] = Hap_3122[4:12, 0:8] - np.eye(8)
        Hap_3122 = Qobj(Hap_3122, dims=[[3, 2, 2], [3, 2, 2]])

        # Checks for num_cell = 3
        assert_(pHamt3 == pHamt3_C)
        assert_(apHamt3 == apHamt3_C)
        assert_(pHamt3122 == Hp_3122)
        assert_(apHamt3122 == Hap_3122)

    def test_cell_structures(self):
        """
        lattice: Test the method Lattice1d.cell_structures().
        """
        val_s = ['site0', 'site1']
        val_t = [' orb0', 'orb1']
        (H_cell_form, inter_cell_T_form, H_cell,
         inter_cell_T) = cell_structures(val_s, val_t)
        c_H_form = [['<site0 orb0 H site0 orb0>',
                     '<site0 orb0 H site0orb1>',
                     '<site0 orb0 H site1 orb0>',
                     '<site0 orb0 H site1orb1>'],
                    ['<site0orb1 H site0 orb0>',
                     '<site0orb1 H site0orb1>',
                     '<site0orb1 H site1 orb0>',
                     '<site0orb1 H site1orb1>'],
                    ['<site1 orb0 H site0 orb0>',
                     '<site1 orb0 H site0orb1>',
                     '<site1 orb0 H site1 orb0>',
                     '<site1 orb0 H site1orb1>'],
                    ['<site1orb1 H site0 orb0>',
                     '<site1orb1 H site0orb1>',
                     '<site1orb1 H site1 orb0>',
                     '<site1orb1 H site1orb1>']]

        i_cell_T_form = [['<cell(i):site0 orb0 H site0 orb0:cell(i+1) >',
                          '<cell(i):site0 orb0 H site0orb1:cell(i+1) >',
                          '<cell(i):site0 orb0 H site1 orb0:cell(i+1) >',
                          '<cell(i):site0 orb0 H site1orb1:cell(i+1) >'],
                         ['<cell(i):site0orb1 H site0 orb0:cell(i+1) >',
                          '<cell(i):site0orb1 H site0orb1:cell(i+1) >',
                          '<cell(i):site0orb1 H site1 orb0:cell(i+1) >',
                          '<cell(i):site0orb1 H site1orb1:cell(i+1) >'],
                         ['<cell(i):site1 orb0 H site0 orb0:cell(i+1) >',
                          '<cell(i):site1 orb0 H site0orb1:cell(i+1) >',
                          '<cell(i):site1 orb0 H site1 orb0:cell(i+1) >',
                          '<cell(i):site1 orb0 H site1orb1:cell(i+1) >'],
                         ['<cell(i):site1orb1 H site0 orb0:cell(i+1) >',
                          '<cell(i):site1orb1 H site0orb1:cell(i+1) >',
                          '<cell(i):site1orb1 H site1 orb0:cell(i+1) >',
                          '<cell(i):site1orb1 H site1orb1:cell(i+1) >']]
        c_H = np.zeros((4, 4), dtype=complex)
        i_cell_T = np.zeros((4, 4), dtype=complex)
        assert_(H_cell_form == c_H_form)
        assert_(inter_cell_T_form == i_cell_T_form)
        assert_((H_cell == c_H).all())
        assert_((inter_cell_T == i_cell_T).all())

    def test_basis(self):
        """
        lattice: Test the method Lattice1d.basis().
        """
        lattice_3242 = Lattice1d(num_cell=3, boundary="periodic",
                                 cell_num_site=2, cell_site_dof=[4, 2])
        psi0 = lattice_3242.basis(1, 0, [2, 1])
        psi0dag_a = np.zeros((1, 48), dtype=complex)
        psi0dag_a[0, 21] = 1
        psi0dag = Qobj(psi0dag_a, dims=[[1, 1, 1, 1], [3, 2, 4, 2]])
        assert_(psi0 == psi0dag.dag())

    def test_distribute_operator(self):
        """
        lattice: Test the method Lattice1d.distribute_operator().
        """
        lattice_412 = Lattice1d(num_cell=4, boundary="periodic",
                                cell_num_site=1, cell_site_dof=[2])
        op = Qobj(np.array([[0, 1], [1, 0]]))
        op_all = lattice_412.distribute_operator(op)
        sv_op_all = tensor(qeye(4), sigmax())
        assert_(op_all == sv_op_all)

    def test_operator_at_cells(self):
        """
        lattice: Test the method Lattice1d.operator_between_cells().
        """
        p_2222 = Lattice1d(num_cell=2, boundary="periodic",
                           cell_num_site=2, cell_site_dof=[2, 2])
        op_0 = basis(2, 0) * basis(2, 1).dag()
        op_c = tensor(op_0, qeye([2, 2]))
        OP = p_2222.operator_between_cells(op_c, 1, 0)
        T = basis(2, 1) * basis(2, 0).dag()
        QP = tensor(T, op_c)
        assert_(OP == QP)

    def test_operator_between_cells(self):
        """
        lattice: Test the method Lattice1d.operator_at_cells().
        """
        lattice_412 = Lattice1d(num_cell=4, boundary="periodic",
                                cell_num_site=1, cell_site_dof=[2])
        op = Qobj(np.array([[0, 1], [1, 0]]))
        op_sp = lattice_412.operator_at_cells(op, cells=[1, 2])

        aop_sp = np.zeros((8, 8), dtype=complex)
        aop_sp[2:4, 2:4] = sigmax()
        aop_sp[4:6, 4:6] = sigmax()
        sv_op_sp = Qobj(aop_sp, dims=[[4, 2], [4, 2]])
        assert_(op_sp == sv_op_sp)

    def test_x(self):
        """
        lattice: Test the method Lattice1d.x().
        """
        lattice_3223 = Lattice1d(num_cell=3, boundary="periodic",
                                 cell_num_site=2, cell_site_dof=[2, 3])
        R = lattice_3223.x()
        npR = R.full()
        # count the number of off-diagonal elements n_off which should be 0
        n_off = np.count_nonzero(npR - np.diag(np.diagonal(npR)))
        assert_(n_off == 0)
        assert_((np.diag(R) == np.kron(range(3), np.ones(12))).all())

    def test_k(self):
        """
        lattice: Test the method Lattice1d.k().
        """
        L = 7
        lattice_L123 = Lattice1d(num_cell=L, boundary="periodic",
                                 cell_num_site=1, cell_site_dof=[2, 3])
        kq = lattice_L123.k()
        kop = np.zeros((L, L), dtype=complex)
        for row in range(L):
            for col in range(L):
                if row == col:
                    kop[row, col] = (L-1)/2
                else:
                    kop[row, col] = 1 / (np.exp(2j * np.pi * (row - col)/L)-1)

        kt = np.kron(kop * 2 * np.pi / L, np.eye(6))
        dim_H = [[2, 2, 3], [2, 2, 3]]
        kt = Qobj(kt, dims=dim_H)

        [k_q, Vq] = kq.eigenstates()
        [k_t, Vt] = kt.eigenstates()
        k_tC = k_t - 2*np.pi/L*((L-1)//2)
        # k_ts = [(i-(L-1)//2)*2*np.pi/L for i in range(L)]
        # k_w = np.kron((np.array(k_ts)).T, np.ones((1,6)))
        assert_((np.abs(k_tC - k_q) < 1E-13).all())

    def test_get_dispersion(self):
        """
        lattice: Test the method Lattice1d.get_dispersion().
        """
        Periodic_Atom_Chain = Lattice1d(num_cell=8, boundary="periodic")
        [knxA, val_kns] = Periodic_Atom_Chain.get_dispersion()
        kB = np.array([[-3.14159265],
                       [-2.35619449],
                       [-1.57079633],
                       [-0.78539816],
                       [0.],
                       [0.78539816],
                       [1.57079633],
                       [2.35619449]])
        valB = np.array([[2., 1.41421356, 0., -1.41421356, -2.,
                          -1.41421356, 0., 1.41421356]])
        assert_(np.max(abs(knxA-kB)) < 1.0E-6)
        assert_(np.max(abs(val_kns-valB)) < 1.0E-6)

        # SSH model with num_cell = 4 and two orbitals, two spins
        # cell_site_dof = [2,2]
        t_intra = -0.5
        t_inter = -0.6
        H_cell = tensor(
                Qobj(np.array([[0, t_intra], [t_intra, 0]])), qeye([2, 2]))
        inter_cell_T = tensor(
                Qobj(np.array([[0, 0], [t_inter, 0]])), qeye([2, 2]))

        SSH_comp = Lattice1d(num_cell=6, boundary="periodic",
                             cell_num_site=2, cell_site_dof=[2, 2],
                             Hamiltonian_of_cell=H_cell, inter_hop=inter_cell_T)
        [kScomp, vcomp] = SSH_comp.get_dispersion()
        kS = np.array([[-3.14159265],
                       [-2.0943951],
                       [-1.04719755],
                       [0.],
                       [1.04719755],
                       [2.0943951]])
        Oband = np.array([-0.1, -0.55677644, -0.9539392, -1.1,
                          -0.9539392, -0.55677644])
        vS = np.array([Oband, Oband, Oband, Oband, -Oband, -Oband, -Oband,
                       -Oband])
        assert_(np.max(abs(kScomp-kS)) < 1.0E-6)
        assert_(np.max(abs(vcomp-vS)) < 1.0E-6)

    def test_cell_periodic_parts(self):
        """
        lattice: Test the method Lattice1d.array_of_unk().
        """
        # Coupled Resonator Optical Waveguide(CROW) Example(PhysRevB.99.224201)
        J = 2
        eta = np.pi/4
        H_cell = Qobj(np.array([[0, J*np.sin(eta)], [J*np.sin(eta), 0]]))
        inter_cell_T0 = (J/2) * Qobj(np.array(
                [[np.exp(eta * 1j), 0], [0, np.exp(-eta*1j)]]))
        inter_cell_T1 = (J/2) * Qobj(np.array([[0, 1], [1, 0]]))

        inter_cell_T = [inter_cell_T0, inter_cell_T1]

        CROW_lattice = Lattice1d(num_cell=4, boundary="periodic",
                                 cell_num_site=1, cell_site_dof=[2],
                                 Hamiltonian_of_cell=H_cell,
                                 inter_hop=inter_cell_T)
        (kxA, val_kns) = CROW_lattice.get_dispersion()
        (knxA, vec_kns) = CROW_lattice.cell_periodic_parts()
        (knxA, qH_ks) = CROW_lattice.bulk_Hamiltonians()
        for i in range(4):
            for j in range(2):
                if val_kns[j][i] == 0:
                    E_V = Qobj(vec_kns[i, j, :])
                    eE_V = qH_ks[i] * E_V
                    assert_(np.max(abs(eE_V)) < 1.0E-12)
                else:
                    E_V = Qobj(vec_kns[i, j, :])
                    eE_V = qH_ks[i] * E_V
                    qE_V = np.divide(eE_V.full(), E_V.full())
                    oE = val_kns[j][i] * np.ones((2, 1))
                    assert_(np.max(abs(oE-qE_V)) < 1.0E-12)

    def test_bulk_Hamiltonians(self):
        """
        lattice: Test the method Lattice1d.bulk_Hamiltonian_array().
        """
        # Coupled Resonator Optical Waveguide(CROW) Example(PhysRevB.99.224201)
        J = 2
        eta = np.pi/2
        H_cell = Qobj(np.array([[0, J*np.sin(eta)], [J*np.sin(eta), 0]]))
        inter_cell_T0 = (J/2)*Qobj(np.array([[np.exp(eta * 1j), 0],
                                   [0, np.exp(-eta*1j)]]))
        inter_cell_T1 = (J/2)*Qobj(np.array([[0, 1], [1, 0]]))
        inter_cell_T = [inter_cell_T0, inter_cell_T1]

        CROW_lattice = Lattice1d(num_cell=4, boundary="periodic",
                                 cell_num_site=1, cell_site_dof=[2],
                                 Hamiltonian_of_cell=H_cell,
                                 inter_hop=inter_cell_T)
        (knxA, qH_ks) = CROW_lattice.bulk_Hamiltonians()
        Hk0 = np.array([[0.+0.j, 0.+0.j],
                        [0.+0.j, 0.+0.j]])
        Hk1 = np.array([[2.+0.j, 2.+0.j],
                        [2.+0.j, -2.+0.j]])
        Hk2 = np.array([[0.+0.j, 4.+0.j],
                        [4.+0.j, 0.+0.j]])
        Hk3 = np.array([[-2.+0.j, 2.+0.j],
                        [2.+0.j, 2.+0.j]])
        qHks = np.array([None for i in range(4)])
        qHks[0] = Qobj(Hk0)
        qHks[1] = Qobj(Hk1)
        qHks[2] = Qobj(Hk2)
        qHks[3] = Qobj(Hk3)
        for i in range(4):
            np.testing.assert_array_almost_equal(qH_ks[i], qHks[i], decimal=8)

    def test_bloch_wave_functions(self):
        """
        lattice: Test the method Lattice1d.bloch_wave_functions().
        """
        # Coupled Resonator Optical Waveguide(CROW) Example(PhysRevB.99.224201)
        J = 2
        eta = np.pi/2
        H_cell = Qobj(np.array([[0, J*np.sin(eta)], [J*np.sin(eta), 0]]))
        inter_cell_T0 = (J/2)*Qobj(np.array([[np.exp(eta * 1j), 0], [0,
                                   np.exp(-eta*1j)]]))
        inter_cell_T1 = (J/2)*Qobj(np.array([[0, 1], [1, 0]]))

        inter_cell_T = [inter_cell_T0, inter_cell_T1]

        CROW_lattice = Lattice1d(num_cell=4, boundary="periodic",
                                 cell_num_site=1, cell_site_dof=[2],
                                 Hamiltonian_of_cell=H_cell,
                                 inter_hop=inter_cell_T)
        CROW_Haml = CROW_lattice.Hamiltonian()

        H_CROW = Qobj(np.array([[0.+0.j, 2.+0.j, 0.+1.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.-1.j, 1.+0.j],
                                [2.+0.j, 0.+0.j, 1.+0.j, 0.-1.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+1.j],
                                [0.-1.j, 1.+0.j, 0.+0.j, 2.+0.j, 0.+1.j, 1.+0.j, 0.+0.j, 0.+0.j],
                                [1.+0.j, 0.+1.j, 2.+0.j, 0.+0.j, 1.+0.j, 0.-1.j, 0.+0.j, 0.+0.j],
                                [0.+0.j, 0.+0.j, 0.-1.j, 1.+0.j, 0.+0.j, 2.+0.j, 0.+1.j, 1.+0.j],
                                [0.+0.j, 0.+0.j, 1.+0.j, 0.+1.j, 2.+0.j, 0.+0.j, 1.+0.j, 0.-1.j],
                                [0.+1.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.-1.j, 1.+0.j, 0.+0.j, 2.+0.j],
                                [1.+0.j, 0.-1.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+1.j, 2.+0.j, 0.+0.j]]),
                      dims=[[4, 2], [4, 2]])
        # Check for CROW with num_cell = 4
        assert_(np.max(abs(CROW_Haml-H_CROW)) < 1.0E-6)  # 1.0E-8 worked too
        eigen_states = CROW_lattice.bloch_wave_functions()
        for i in range(8):
            if eigen_states[i][0] == 0:
                E_V = eigen_states[i][1]
                eE_V = CROW_Haml * E_V
                assert_(np.max(abs(eE_V)) < 1.0E-10)
            else:
                E_V = eigen_states[i][1]
                eE_V = CROW_Haml * E_V
                qE_V = np.divide(eE_V.full(), E_V.full())
                oE = eigen_states[i][0] * np.ones((8, 1))
                assert_(np.max(abs(oE-qE_V)) < 1.0E-10)

    def test_CROW(self):
        """
        lattice: Test the methods of Lattice1d in a CROW model.
        """
        # Coupled Resonator Optical Waveguide(CROW) Example(PhysRevB.99.224201)
        J = 2
        eta = np.pi/4
        H_cell = Qobj(np.array([[0, J * np.sin(eta)], [J * np.sin(eta), 0]]))
        inter_cell_T0 = (J/2)*Qobj(np.array([[np.exp(eta * 1j), 0],
                                   [0, np.exp(-eta*1j)]]))
        inter_cell_T1 = (J/2)*Qobj(np.array([[0, 1], [1, 0]]))

        inter_cell_T = [inter_cell_T0, inter_cell_T1]

        CROW_lattice = Lattice1d(num_cell=4, boundary="periodic",
                                 cell_num_site=1, cell_site_dof=[2],
                                 Hamiltonian_of_cell=H_cell,
                                 inter_hop=inter_cell_T)
        CROW_Haml = CROW_lattice.Hamiltonian()

        # Benchmark answers
        H_CROW = Qobj([[0.+0.j, 1.41421356+0.j, 0.70710678+0.70710678j, 1.+0.j,
                        0.+0.j, 0.+0.j, 0.70710678-0.70710678j, 1.+0.j],
        [1.41421356+0.j, 0.+0.j, 1.+0.j,  0.70710678-0.70710678j,
         0.+0.j, 0.+0.j, 1.+0.j, 0.70710678+0.70710678j],
        [0.70710678-0.70710678j, 1.+0.j, 0.+0.j, 1.41421356+0.j,
         0.70710678+0.70710678j, 1.+0.j, 0.+0.j, 0.+0.j],
        [1.+0.j, 0.70710678+0.70710678j, 1.41421356+0.j, 0.+0.j,
         1.+0.j, 0.70710678-0.70710678j, 0.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 0.70710678-0.70710678j, 1.+0.j,
         0.+0.j, 1.41421356+0.j, 0.70710678+0.70710678j, 1.+0.j],
        [0.+0.j, 0.+0.j, 1.+0.j, 0.70710678+0.70710678j,
         1.41421356+0.j, 0.+0.j, 1.+0.j, 0.70710678-0.70710678j],
        [0.70710678+0.70710678j, 1.+0.j, 0.+0.j, 0.+0.j,
         0.70710678-0.70710678j, 1.+0.j, 0.+0.j, 1.41421356+0.j],
        [1.+0.j, 0.70710678-0.70710678j, 0.+0.j, 0.+0.j,
         1.+0.j, 0.70710678+0.70710678j, 1.41421356+0.j, 0.+0.j]],
         dims=[[4, 2], [4, 2]])

        # Check for CROW with num_cell = 4
        assert_(np.max(abs(CROW_Haml-H_CROW)) < 1.0E-6)  # 1.0E-8 worked too

        (kxA, val_kns) = CROW_lattice.get_dispersion()
        kCR = np.array([[-3.14159265],
                        [-1.57079633],
                        [0.],
                        [1.57079633]])
        vCR = np.array([[-2.        , -2.        , -2.        , -2.],
                        [-0.82842712, 2.         , 4.82842712 , 2.]])
        assert_(np.max(abs(kxA-kCR)) < 1.0E-6)
        assert_(np.max(abs(val_kns-vCR)) < 1.0E-6)

        eigen_states = CROW_lattice.bloch_wave_functions()
        for i in range(8):
            E_V = eigen_states[i][1]
            eE_V = CROW_Haml * E_V
            qE_V = np.divide(eE_V.full(), E_V.full())
            oE = eigen_states[i][0] * np.ones((8, 1))
            assert_(np.max(abs(oE-qE_V)) < 1.0E-10)
        (knxA, qH_ks) = CROW_lattice.bulk_Hamiltonians()
        (knxA, vec_kns) = CROW_lattice.cell_periodic_parts()

        Hk0 = np.array([[-1.41421356+0.j, -0.58578644+0.j],
                        [-0.58578644+0.j, -1.41421356+0.j]])
        Hk1 = np.array([[1.41421356+0.j, 1.41421356+0.j],
                        [1.41421356+0.j, -1.41421356+0.j]])
        Hk2 = np.array([[1.41421356+0.j, 3.41421356+0.j],
                        [3.41421356+0.j, 1.41421356+0.j]])
        Hk3 = np.array([[-1.41421356+0.j, 1.41421356+0.j],
                        [1.41421356+0.j, 1.41421356+0.j]])
        qHks = np.array([None for i in range(4)])
        qHks[0] = Qobj(Hk0)
        qHks[1] = Qobj(Hk1)
        qHks[2] = Qobj(Hk2)
        qHks[3] = Qobj(Hk3)
        for i in range(4):
            np.testing.assert_array_almost_equal(qH_ks[i], qHks[i], decimal=8)
        for i in range(4):
            for j in range(2):
                E_V = Qobj(vec_kns[i, j, :])
                eE_V = qH_ks[i] * E_V
                qE_V = np.divide(eE_V.full(), E_V.full())
                oE = val_kns[j][i]*np.ones((2, 1))
                assert_(np.max(abs(oE-qE_V)) < 1.0E-12)

        # A test on CROW lattice dispersion with a random number of cells and
        # random values of eta
        J = 1
        num_cell = np.random.randint(2, 60)
        eta = 2*np.pi*np.random.random()

        H_cell = Qobj(np.array([[0, J * np.sin(eta)], [J * np.sin(eta), 0]]))
        inter_cell_T0 = (J/2) * Qobj(np.array([[np.exp(eta * 1j), 0],
                                              [0, np.exp(-eta*1j)]]))
        inter_cell_T1 = (J/2) * Qobj(np.array([[0, 1], [1, 0]]))
        inter_cell_T = [inter_cell_T0, inter_cell_T1]

        CROW_Random = Lattice1d(num_cell=num_cell, boundary="periodic",
                                cell_num_site=1, cell_site_dof=[2],
                                Hamiltonian_of_cell=H_cell,
                                inter_hop=inter_cell_T)
        a = 1  # The unit cell length is always considered 1
        kn_start = 0
        kn_end = 2*np.pi/a
        Ana_val_kns = np.zeros((2, num_cell), dtype=float)
        knxA = np.zeros((num_cell, 1), dtype=float)

        for ks in range(num_cell):
            knx = kn_start + (ks*(kn_end-kn_start)/num_cell)
            if knx >= np.pi:
                knxA[ks, 0] = knx - 2 * np.pi
            else:
                knxA[ks, 0] = knx
        knxA = np.roll(knxA, np.floor_divide(num_cell, 2))

        for ks in range(num_cell):
            knx = knxA[ks, 0]
            # Ana_val_kns are the analytical bands
            val0 = np.cos(knx) * np.cos(eta) + np.sqrt(2 * np.sin(eta) ** 2
                          + (np.cos(knx) * np.cos(eta)) ** 2
                          + 2 * np.sin(eta) * np.cos(knx))
            val1 = np.cos(knx) * np.cos(eta) - np.sqrt(2 * np.sin(eta) ** 2
                          + (np.cos(knx) * np.cos(eta)) ** 2
                          + 2 * np.sin(eta) * np.cos(knx))
            vals = [val0, val1]
            Ana_val_kns[0, ks] = np.min(vals)
            Ana_val_kns[1, ks] = np.max(vals)

        (kxA, val_kns) = CROW_Random.get_dispersion()
        assert_(np.max(abs(kxA-knxA)) < 1.0E-8)
        assert_(np.max(abs(val_kns-Ana_val_kns)) < 1.0E-8)

    def test_SSH(self):
        """
        lattice: Test the methods of Lattice1d in a SSH model.
        """
        # SSH model with num_cell = 4 and two orbitals, two spins
        # cell_site_dof = [2,2]
        t_intra = -0.5
        t_inter = -0.6
        H_cell = Qobj(np.array([[0, t_intra], [t_intra, 0]]))
        inter_cell_T = Qobj(np.array([[0, 0], [t_inter, 0]]))

        SSH_lattice = Lattice1d(num_cell=5, boundary="periodic",
                             cell_num_site=2, cell_site_dof=[1],
                             Hamiltonian_of_cell=H_cell,
                             inter_hop=inter_cell_T)        
        Ham_Sc = SSH_lattice.Hamiltonian()

        Hin = Qobj([[0, -0.5], [-0.5, 0]])
        Ht = Qobj([[0, 0], [-0.6, 0]])
        D = qeye(5)
        T = np.diag(np.zeros(4)+1, 1)
        Tdag = np.diag(np.zeros(4)+1, -1)
        Tdag[0][4] = 1
        T[4][0] = 1
        T = Qobj(T)
        Tdag = Qobj(Tdag)
        H_Sc = tensor(D, Hin) + tensor(T, Ht) + tensor(Tdag, Ht.dag())
        # check for SSH model with num_cell = 5          
        assert_(Ham_Sc == H_Sc)

        (kxA,val_ks) = SSH_lattice.get_dispersion()
        kSSH = np.array([[-2.51327412],
                         [-1.25663706],
                         [ 0.        ],
                         [ 1.25663706],
                         [ 2.51327412]])
        vSSH = np.array([[-0.35297281, -0.89185772, -1.1, -0.89185772, -0.35297281],
                         [ 0.35297281,  0.89185772,  1.1, 0.89185772, 0.35297281]])
        assert_(np.max(abs(kxA-kSSH)) < 1.0E-6)
        assert_(np.max(abs(val_ks-vSSH)) < 1.0E-6)

        # A test on SSH lattice dispersion with a random number of cells and
        # random values of t_inter and t_intra
        num_cell = np.random.randint(2, 60)
        t_intra = -np.random.random()
        t_inter = -np.random.random()
        H_cell = Qobj(np.array([[0, t_intra], [t_intra, 0]]))
        inter_cell_T = Qobj(np.array([[0, 0], [t_inter, 0]]))
        SSH_Random = Lattice1d(num_cell=num_cell, boundary="periodic",
                             cell_num_site=2, cell_site_dof=[1],
                             Hamiltonian_of_cell=H_cell,
                             inter_hop=inter_cell_T)
        a = 1  # The unit cell length is always considered 1
        kn_start = 0
        kn_end = 2*np.pi/a
        Ana_val_kns = np.zeros((2, num_cell), dtype=float)
        knxA = np.zeros((num_cell, 1), dtype=float)

        for ks in range(num_cell):
            knx = kn_start + (ks * (kn_end-kn_start)/num_cell)
            if knx >= np.pi:
                knxA[ks, 0] = knx - 2 * np.pi
            else:
                knxA[ks, 0] = knx
        knxA = np.roll(knxA, np.floor_divide(num_cell, 2))

        for ks in range(num_cell):
            knx = knxA[ks, 0]
            # Ana_val_kns are the analytical bands
            Ana_val_kns[0, ks] = -np.sqrt(t_intra ** 2 + t_inter ** 2
                       + 2 * t_intra * t_inter * np.cos(knx))
            Ana_val_kns[1, ks] = np.sqrt(t_intra ** 2 + t_inter ** 2
                       + 2 * t_intra * t_inter * np.cos(knx))
        (kxA, val_kns) = SSH_Random.get_dispersion()
        assert_(np.max(abs(val_kns-Ana_val_kns)) < 1.0E-13)

if __name__ == "__main__":
    run_module_suite()
