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
import pytest
import qutip

_r2 = np.sqrt(2)


def _assert_angles_close(test, expected, atol):
    """Assert that two arrays of angles are within tolerance of each other."""
    np.testing.assert_allclose(test % (2*np.pi), expected % (2*np.pi),
                               atol=atol)


def _hamiltonian_expected(cells, periodic, sites_per_cell,
                          freedom):
    """Expected Hamiltonian for the simple 1D lattice model"""
    # Within the cell, adjacent sites hop to each other, and it's always
    # non-periodic.
    cell = qutip.qdiags([-np.ones(sites_per_cell - 1)]*2, [1, -1])
    # To hop to the cell one to the left, then you have to drop from the lowest
    # in-cell location to the highest in the other.
    hop_left = qutip.qdiags(np.ones(cells - 1), 1)
    if periodic and cells > 2:
        # If cells <= 2 then all cells border each other anyway.
        hop_left += qutip.qdiags([1], -(cells - 1))
    drop_site = -qutip.projection(sites_per_cell, sites_per_cell-1, 0)
    if np.prod(freedom) != 1:
        # Degenerate degrees of freedom per lattice site are unchanged.
        identity = qutip.qeye(freedom)
        cell = qutip.tensor(cell, identity)
        drop_site = qutip.tensor(drop_site, identity)
    out = (qutip.tensor(qutip.qeye(cells), cell)
           + qutip.tensor(hop_left, drop_site)
           + qutip.tensor(hop_left, drop_site).dag())
    # Contract scalar spaces.
    dims = [x for x in [cells, sites_per_cell] + freedom
            if x != 1]
    dims = dims or [1]
    out.dims = [dims, dims]
    return out


def _k_expected(cells):
    out = (2*np.pi/cells) * np.arange(-(cells//2), cells - (cells//2))
    return out.reshape((-1, 1))


def _ssh_lattice(intra, inter,
                 cells=5, boundary="periodic", sites_per_cell=2,
                 freedom=[1]):
    part_hamiltonian = qutip.Qobj(np.array([[0, intra], [intra, 0]]))
    free_identity = qutip.qeye(freedom)
    hamiltonian = qutip.tensor(part_hamiltonian, free_identity)
    hopping = qutip.tensor(qutip.Qobj(np.array([[0, 0], [inter, 0]])),
                           free_identity)
    return qutip.Lattice1d(num_cell=cells, boundary=boundary,
                           cell_num_site=sites_per_cell,
                           cell_site_dof=freedom,
                           Hamiltonian_of_cell=hamiltonian,
                           inter_hop=hopping)


# Energies for the SSH model with 4 cells, 2 orbitals and 2 spins, with
# intra=-0.5 and inter=-0.6.
_ssh_energies = np.array([0.1, 0.55677644, 0.9539392,
                          1.1, 0.9539392,  0.55677644])


def _crow_lattice(coupling, phase_delay,
                  cells=4, boundary="periodic", sites_per_cell=1,
                  freedom=[2]):
    r"""
    Return a `qutip.Lattice1d` of a "Coupled Resonator Optical Waveguide"
    (CROW) with the given coupling strength and phase delay.

    See for example: https://www.doi.org/10.1103/PhysRevB.99.224201
    where `coupling` is $J$ and `phase_delay` is $\eta$.
    """
    cell_hamiltonian = coupling * np.sin(phase_delay) * qutip.sigmax()
    phase_term = np.exp(1j * phase_delay)
    hopping = [
        0.5 * coupling * qutip.qdiags([phase_term, phase_term.conj()], 0),
        0.5 * coupling * qutip.sigmax(),
    ]
    return qutip.Lattice1d(num_cell=cells, boundary=boundary,
                           cell_num_site=sites_per_cell,
                           cell_site_dof=freedom,
                           Hamiltonian_of_cell=cell_hamiltonian,
                           inter_hop=hopping)


class TestLattice1d:
    @pytest.mark.parametrize("cells", [1, 2, 3])
    @pytest.mark.parametrize("periodic", [True, False],
                             ids=["periodic", "aperiodic"])
    @pytest.mark.parametrize("sites_per_cell", [1, 2])
    @pytest.mark.parametrize("freedom", [[1], [2, 3]],
                             ids=["simple chain", "onsite freedom"])
    def test_hamiltonian(self, cells, periodic, sites_per_cell, freedom):
        """Test that the lattice model produces the expected Hamiltonians."""
        boundary = "periodic" if periodic else "aperiodic"
        lattice = qutip.Lattice1d(num_cell=cells, boundary=boundary,
                                  cell_site_dof=freedom,
                                  cell_num_site=sites_per_cell)
        expected = _hamiltonian_expected(cells, periodic, sites_per_cell,
                                         freedom)
        assert lattice.Hamiltonian() == expected

    def test_cell_structures(self):
        val_s = ['site0', 'site1']
        val_t = [' orb0', 'orb1']
        H_cell_form, inter_cell_T_form, H_cell, inter_cell_T =\
            qutip.cell_structures(val_s, val_t)
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
        assert H_cell_form == c_H_form
        assert inter_cell_T_form == i_cell_T_form
        assert (H_cell == c_H).all()
        assert (inter_cell_T == i_cell_T).all()

    def test_basis(self):
        lattice_3242 = qutip.Lattice1d(num_cell=3, boundary="periodic",
                                       cell_num_site=2, cell_site_dof=[4, 2])
        psi0 = lattice_3242.basis(1, 0, [2, 1])
        psi0dag_a = np.zeros((1, 48), dtype=complex)
        psi0dag_a[0, 21] = 1
        psi0dag = qutip.Qobj(psi0dag_a, dims=[[1, 1, 1, 1], [3, 2, 4, 2]])
        assert psi0 == psi0dag.dag()

    def test_distribute_operator(self):
        lattice_412 = qutip.Lattice1d(num_cell=4, boundary="periodic",
                                      cell_num_site=1, cell_site_dof=[2])
        op = qutip.Qobj(np.array([[0, 1], [1, 0]]))
        op_all = lattice_412.distribute_operator(op)
        sv_op_all = qutip.tensor(qutip.qeye(4), qutip.sigmax())
        assert op_all == sv_op_all

    def test_operator_at_cells(self):
        p_2222 = qutip.Lattice1d(num_cell=2, boundary="periodic",
                                 cell_num_site=2, cell_site_dof=[2, 2])
        op_0 = qutip.projection(2, 0, 1)
        op_c = qutip.tensor(op_0, qutip.qeye([2, 2]))
        OP = p_2222.operator_between_cells(op_c, 1, 0)
        T = qutip.projection(2, 1, 0)
        QP = qutip.tensor(T, op_c)
        assert OP == QP

    def test_operator_between_cells(self):
        lattice_412 = qutip.Lattice1d(num_cell=4, boundary="periodic",
                                      cell_num_site=1, cell_site_dof=[2])
        op = qutip.sigmax()
        op_sp = lattice_412.operator_at_cells(op, cells=[1, 2])

        aop_sp = np.zeros((8, 8), dtype=complex)
        aop_sp[2:4, 2:4] = aop_sp[4:6, 4:6] = op.full()
        sv_op_sp = qutip.Qobj(aop_sp, dims=[[4, 2], [4, 2]])
        assert op_sp == sv_op_sp

    def test_x(self):
        lattice_3223 = qutip.Lattice1d(num_cell=3, boundary="periodic",
                                       cell_num_site=2, cell_site_dof=[2, 3])
        test = lattice_3223.x().full()
        expected = np.diag([n for n in range(3) for _ in [None]*12])
        np.testing.assert_allclose(test, expected, atol=1e-12)

    def test_k(self):
        L = 7
        lattice_L123 = qutip.Lattice1d(num_cell=L, boundary="periodic",
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
        kt = qutip.Qobj(kt, dims=dim_H)
        k_q = kq.eigenstates()[0]
        k_t = kt.eigenstates()[0]
        k_tC = k_t - (2*np.pi/L) * ((L-1) // 2)
        np.testing.assert_allclose(k_tC, k_q, atol=1e-12)

    @pytest.mark.parametrize(["lattice", "k_expected", "energies_expected"], [
        pytest.param(qutip.Lattice1d(num_cell=8),
                     _k_expected(8),
                     np.array([[2, np.sqrt(2), 0, -np.sqrt(2), -2,
                                -np.sqrt(2), 0, np.sqrt(2)]]),
                     id="atom chain"),
        pytest.param(_ssh_lattice(-0.5, -0.6,
                                  cells=6, sites_per_cell=2, freedom=[2, 2]),
                     _k_expected(6),
                     np.array([-_ssh_energies]*4 + [_ssh_energies]*4),
                     id="ssh model")
    ])
    def test_get_dispersion(self, lattice, k_expected, energies_expected):
        k_test, energies_test = lattice.get_dispersion()
        _assert_angles_close(k_test, k_expected, atol=1e-8)
        np.testing.assert_allclose(energies_test, energies_expected, atol=1e-8)

    def test_cell_periodic_parts(self):
        lattice = _crow_lattice(2, np.pi/4)
        eigensystems = zip(lattice.get_dispersion()[1].T,
                           lattice.cell_periodic_parts()[1])
        hamiltonians = lattice.bulk_Hamiltonians()[1]
        for hamiltonian, system in zip(hamiltonians, eigensystems):
            for eigenvalue, eigenstate in zip(*system):
                eigenstate = qutip.Qobj(eigenstate)
                np.testing.assert_allclose((hamiltonian * eigenstate).full(),
                                           (eigenvalue * eigenstate).full(),
                                           atol=1e-12)

    def test_bulk_Hamiltonians(self):
        test = _crow_lattice(2, np.pi/2).bulk_Hamiltonians()[1]
        expected = [[[0, 0], [0, 0]],
                    [[2, 2], [2, -2]],
                    [[0, 4], [4, 0]],
                    [[-2, 2], [2, 2]]]
        for test_hamiltonian, expected_hamiltonian in zip(test, expected):
            np.testing.assert_allclose(test_hamiltonian.full(),
                                       expected_hamiltonian,
                                       atol=1e-12)

    def test_bloch_wave_functions(self):
        lattice = _crow_lattice(2, np.pi/2)
        hamiltonian = lattice.Hamiltonian()
        expected = np.array([[0,    2,   1j,  1,   0,   0,  -1j,  1],
                             [2,    0,   1,  -1j,  0,   0,   1,   1j],
                             [-1j,  1,   0,   2,   1j,  1,   0,   0],
                             [1,    1j,  2,   0,   1,  -1j,  0,   0],
                             [0,    0,  -1j,  1,   0,   2,   1j,  1],
                             [0,    0,   1,   1j,  2,   0,   1,  -1j],
                             [1j,   1,   0,   0,  -1j,  1,   0,   2],
                             [1,   -1j,  0,   0,   1,   1j,  2,   0]])
        np.testing.assert_allclose(hamiltonian.full(), expected, atol=1e-8)
        for eigenvalue, eigenstate in lattice.bloch_wave_functions():
            np.testing.assert_allclose((hamiltonian * eigenstate).full(),
                                       (eigenvalue * eigenstate).full(),
                                       atol=1e-12)


class TestIntegration:
    def test_fixed_crow(self):
        lattice = _crow_lattice(2, np.pi/4, cells=4)
        hamiltonian = lattice.Hamiltonian()
        expected_hamiltonian = np.array([
            [0, _r2, (1+1j)/_r2, 1, 0, 0, (1-1j)/_r2, 1],
            [_r2, 0, 1, (1-1j)/_r2, 0, 0, 1, (1+1j)/_r2],
            [(1-1j)/_r2, 1, 0, _r2, (1+1j)/_r2, 1, 0, 0],
            [1, (1+1j)/_r2, _r2, 0, 1, (1-1j)/_r2, 0, 0],
            [0, 0, (1-1j)/_r2, 1, 0, _r2, (1+1j)/_r2, 1],
            [0, 0, 1, (1+1j)/_r2, _r2, 0, 1, (1-1j)/_r2],
            [(1+1j)/_r2, 1, 0, 0, (1-1j)/_r2, 1, 0, _r2],
            [1, (1-1j)/_r2, 0, 0, 1, (1+1j)/_r2, _r2, 0]])
        np.testing.assert_allclose(hamiltonian, expected_hamiltonian,
                                   atol=1e-12)

        test_k, test_energies = lattice.get_dispersion()
        expected_k = _k_expected(4)
        expected_energies = 2*np.array([[-1,    -1,    -1, -1],
                                        [1-_r2,  1, 1+_r2,  1]])
        _assert_angles_close(test_k, expected_k, atol=1e-12)
        np.testing.assert_allclose(test_energies, expected_energies,
                                   atol=1e-12)

        for value, state in lattice.bloch_wave_functions():
            np.testing.assert_allclose(hamiltonian*state, value*state,
                                       atol=1e-12)

        test_bulk_system = zip(test_energies.T,
                               lattice.cell_periodic_parts()[1])
        test_bulk_h = lattice.bulk_Hamiltonians()[1]
        expected_bulk_h = np.array([
            [[-_r2, _r2-2], [_r2-2, -_r2]],
            [[_r2,    _r2], [_r2,   -_r2]],
            [[_r2,  2+_r2], [2+_r2,  _r2]],
            [[-_r2,   _r2], [_r2,    _r2]]])
        for test, expected in zip(test_bulk_h, expected_bulk_h):
            np.testing.assert_allclose(test.full(), expected, atol=1e-8)
        for hamiltonian, system in zip(test_bulk_h, test_bulk_system):
            for eigenvalue, eigenstate in zip(*system):
                eigenstate = qutip.Qobj(eigenstate)
                np.testing.assert_allclose((hamiltonian * eigenstate).full(),
                                           (eigenvalue * eigenstate).full(),
                                           atol=1e-12)

    def test_random_crow(self):
        cells = np.random.randint(2, 60)
        phase = 2*np.pi * np.random.rand()
        lattice = _crow_lattice(1, phase, cells=cells)
        expected_k = _k_expected(cells)
        _s, _c = np.sin(phase), np.cos(phase)
        expected_energies = np.array([
            [cosk*_c + np.sqrt(2*_s*(_s+cosk) + (cosk*_c)**2),
             cosk*_c - np.sqrt(2*_s*(_s+cosk) + (cosk*_c)**2)]
            for cosk in np.cos(expected_k).flat])
        expected_energies = np.sort(expected_energies.T, axis=0)
        test_k, test_energies = lattice.get_dispersion()
        _assert_angles_close(test_k, expected_k, atol=1e-12)
        np.testing.assert_allclose(test_energies, expected_energies,
                                   atol=1e-12)

    def test_fixed_ssh(self):
        intra, inter = -0.5, -0.6
        cells = 5
        lattice = _ssh_lattice(intra, inter,
                               cells=cells, sites_per_cell=2, freedom=[1])
        Hin = qutip.Qobj([[0, intra], [intra, 0]])
        Ht = qutip.Qobj([[0, 0], [inter, 0]])
        D = qutip.qeye(cells)
        T = qutip.qdiags([np.ones(cells-1), [1]], [1, 1-cells])
        hopping = qutip.tensor(T, Ht)
        expected_hamiltonian = qutip.tensor(D, Hin) + hopping + hopping.dag()
        assert lattice.Hamiltonian() == expected_hamiltonian

        test_k, test_energies = lattice.get_dispersion()
        band = np.array([0.35297281, 0.89185772, 1.1, 0.89185772, 0.35297281])
        _assert_angles_close(test_k, _k_expected(cells), atol=1e-12)
        np.testing.assert_allclose(test_energies, np.array([-band, band]),
                                   atol=1e-8)

    def test_random_ssh(self):
        cells = np.random.randint(2, 60)
        intra, inter = -np.random.random(), -np.random.random()
        lattice = _ssh_lattice(intra, inter,
                               cells=cells, sites_per_cell=2, freedom=[1])
        expected_k = _k_expected(cells)
        band = np.sqrt(intra**2 + inter**2 + 2*intra*inter*np.cos(expected_k))
        band = np.array(band.flat)
        expected_energies = np.array([-band, band])
        test_k, test_energies = lattice.get_dispersion()
        _assert_angles_close(test_k, expected_k, atol=1e-12)
        np.testing.assert_allclose(test_energies, expected_energies,
                                   atol=1e-12)
