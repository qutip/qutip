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
from qutip import (Qobj, tensor, basis, qeye, isherm)
from numpy.testing import (assert_)

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
        Periodic_Atom_Chain = Lattice1d(num_cell=1, boundary = "periodic")
        Aperiodic_Atom_Chain = Lattice1d(num_cell=1, boundary = "aperiodic")        
        p_1223 = Lattice1d(num_cell=1, boundary = "periodic",
                                 cell_num_site = 2, cell_site_dof = [2,3])
        ap_1223 = Lattice1d(num_cell=1, boundary = "aperiodic",
                                 cell_num_site = 2, cell_site_dof = [2,3])

        # Their Hamiltonians
        pHamt1 = Periodic_Atom_Chain.Hamiltonian()
        apHamt1 = Aperiodic_Atom_Chain.Hamiltonian()
        pHamt1223 = p_1223.Hamiltonian()
        apHamt1223 = ap_1223.Hamiltonian()

        # Benchmark answers
        pHamt1_C = Qobj([[0.]],dims=[[1, 1, 1], [1, 1, 1]])
        apHamt1_C = Qobj([[0.]],dims=[[1, 1, 1], [1, 1, 1]])        
        site1223 = np.diag(np.zeros(2-1)-1, 1) + np.diag(np.zeros(2-1)-1, -1)
        Rpap1223 = tensor(Qobj(site1223), qeye([2, 3]))
        pap1223 = Qobj(Rpap1223, dims=[[1, 2, 2, 3], [1, 2, 2, 3]])
        
        # Checks for num_cell = 1
        assert_(pHamt1 == pHamt1_C)
        assert_(apHamt1 == apHamt1_C)
        assert_(pHamt1223 == pap1223)    # for num_cell=1, periodic and
        assert_(apHamt1223 == pap1223)   # aperiodic B.C. have same Hamiltonian
        
        # num_cell = 2
        # Four different instances
        Periodic_Atom_Chain = Lattice1d(num_cell=2, boundary = "periodic")
        Aperiodic_Atom_Chain = Lattice1d(num_cell=2, boundary = "aperiodic")        

        p_2222 = Lattice1d(num_cell=2, boundary = "periodic",
                                 cell_num_site = 2, cell_site_dof = [2,2])
        ap_2222 = Lattice1d(num_cell=2, boundary = "aperiodic",
                                 cell_num_site = 2, cell_site_dof = [2,2])

        # Their Hamiltonians
        pHamt2222 = p_2222.Hamiltonian()
        apHamt2222 = ap_2222.Hamiltonian()
        pHamt2 = Periodic_Atom_Chain.Hamiltonian()
        apHamt2 = Aperiodic_Atom_Chain.Hamiltonian()
        
        # Benchmark answers        
        pHamt2_C = Qobj([[ 0., -1.],
        [-1.,  0.]],dims = [[2, 1, 1], [2, 1, 1]])
        apHamt2_C = Qobj([[ 0., -1.],
        [-1.,  0.]],dims = [[2, 1, 1], [2, 1, 1]])
        pap_2222 = Qobj([[ 0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
        [-1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
        [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.]],
        dims = [[2, 2, 2, 2], [2, 2, 2, 2]])

        # Checks for num_cell = 2
        assert_(pHamt2 == pHamt2_C)
        assert_(apHamt2 == apHamt2_C)
        assert_(pHamt2222 == pap_2222)   # for num_cell=2, periodic and
        assert_(apHamt2222 == pap_2222)  # aperiodic B.C. have same Hamiltonian
          
        # num_cell = 3   # checking any num_cell >= 3 is pretty much equivalent
        Periodic_Atom_Chain = Lattice1d(num_cell=3, boundary = "periodic")
        Aperiodic_Atom_Chain = Lattice1d(num_cell=3, boundary = "aperiodic")        
        p_3122 = Lattice1d(num_cell=3, boundary = "periodic",
                                 cell_num_site = 1, cell_site_dof = [2,2])
        ap_3122 = Lattice1d(num_cell=3, boundary = "aperiodic",
                                 cell_num_site = 1, cell_site_dof = [2,2])

        # Their Hamiltonians
        pHamt3 = Periodic_Atom_Chain.Hamiltonian()
        apHamt3 = Aperiodic_Atom_Chain.Hamiltonian()
        pHamt3122 = p_3122.Hamiltonian()
        apHamt3122 = ap_3122.Hamiltonian()
       
        # Benchmark answers        
        pHamt3_C = Qobj([[ 0., -1., -1.],
        [-1.,  0., -1.,],
        [-1., -1.,  0.,]],dims = [[3, 1, 1], [3, 1, 1]])
        apHamt3_C = Qobj([[ 0., -1.,  0.],
        [-1.,  0., -1.],
        [ 0., -1.,  0.]],dims = [[3, 1, 1], [3, 1, 1]])

        Hp_3122 = Qobj([[ 0.,  0.,  0.,  0., -1.,  0.,  0.,  0., -1.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0., -1.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.],
        [ 0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.],
        [ 0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0., -1.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0., -1.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0., -1.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.]],
        dims = [[3, 1, 2, 2], [3, 1, 2, 2]])

        Hap_3122 = Qobj([[ 0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.],
        [-1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.],
        [ 0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.],
        [ 0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
        [ 0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.]],
        dims = [[3, 1, 2, 2], [3, 1, 2, 2]])

        # Checks for num_cell = 3        
        assert_(pHamt3 == pHamt3_C)
        assert_(apHamt3 == apHamt3_C)        
        assert_(pHamt3122 == Hp_3122)
        assert_(apHamt3122 == Hap_3122)        

        # Coupled Resonator Optical Waveguide(CROW) Example(PhysRevB.99.224201)
        J = 2
        eta = np.pi/4
        cell_H = Qobj( np.array( [[0,J*np.sin(eta)  ],[J*np.sin(eta), 0]]   ) )
        inter_cell_T0 = (J/2)*Qobj(  np.array( [[ np.exp(eta * 1j) , 0 ],[0,np.exp(-eta*1j)]] )  )
        inter_cell_T1 = (J/2)*Qobj(  np.array( [[ 0 , 1 ],[ 1 , 0 ]] )  )

        inter_cell_T = [inter_cell_T0, inter_cell_T1]

        CROW_lattice = Lattice1d(num_cell=4, boundary="periodic",
                    cell_num_site=1, cell_site_dof=[2],
                    cell_Hamiltonian=cell_H, inter_hop = inter_cell_T )
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
         dims = [[4, 1, 2], [4, 1, 2]])

        # Check for CROW with num_cell = 4
        assert_( np.max(abs(CROW_Haml-H_CROW)) < 1.0E-6 ) # 1.0E-8 worked too
        
        # SSH model with num_cell = 4 and two orbitals, two spins 
        # cell_site_dof = [2,2]
        t_intra = -0.5
        t_inter = -0.6
        cell_H = tensor(
                Qobj( np.array( [[ 0, t_intra ],[t_intra,0]] )), qeye([2,2]))
        inter_cell_T = tensor(
                Qobj( np.array( [[ 0, 0 ],[t_inter,0]] )), qeye([2,2]))

        SSH_comp = Lattice1d(num_cell=2, boundary = "aperiodic",
                             cell_num_site = 2, cell_site_dof = [2,2],
                             cell_Hamiltonian = cell_H, inter_hop = inter_cell_T )        
        Ham_Sc = SSH_comp.Hamiltonian()

        H_Sc = Qobj([[ 0., 0., 0., 0., -0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [ 0., 0., 0., 0., 0., -0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [ 0., 0., 0., 0., 0., 0., -0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [ 0., 0., 0., 0., 0., 0., 0., -0.5, 0., 0., 0., 0., 0., 0., 0., 0.],
                     [-0.5, 0., 0., 0., 0., 0., 0., 0., -0.6, 0., 0., 0., 0., 0., 0., 0.],
                     [ 0., -0.5, 0., 0., 0., 0., 0., 0., 0., -0.6, 0., 0., 0., 0., 0., 0.],
                     [ 0., 0., -0.5, 0., 0., 0., 0., 0., 0., 0., -0.6, 0., 0., 0., 0., 0.],
                     [ 0., 0., 0., -0.5, 0., 0., 0., 0., 0., 0., 0., -0.6, 0., 0., 0., 0.],
                     [ 0., 0., 0., 0., -0.6, 0., 0., 0., 0., 0., 0., 0., -0.5, 0., 0., 0.],
                     [ 0., 0., 0., 0., 0., -0.6, 0., 0., 0., 0., 0., 0., 0., -0.5, 0., 0.],
                     [ 0., 0., 0., 0., 0., 0., -0.6, 0., 0., 0., 0., 0., 0., 0., -0.5, 0.],
                     [ 0., 0., 0., 0., 0., 0., 0., -0.6, 0., 0., 0., 0., 0., 0., 0., -0.5],
                     [ 0., 0., 0., 0., 0., 0., 0., 0., -0.5, 0., 0., 0., 0., 0., 0., 0.],
                     [ 0., 0., 0., 0., 0., 0., 0., 0., 0., -0.5, 0., 0., 0., 0., 0., 0.],
                     [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -0.5, 0., 0., 0., 0., 0.],
                     [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -0.5, 0., 0., 0., 0.]],
        dims = [[2, 2, 2, 2], [2, 2, 2, 2]])
        # check for SSH model with num_cell = 4 and two orbitals, two spins         
        assert_(Ham_Sc == H_Sc)


if __name__ == "__main__":
    run_module_suite()
