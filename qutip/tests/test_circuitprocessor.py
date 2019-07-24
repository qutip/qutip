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
import os

from numpy.testing import assert_, run_module_suite, assert_allclose
import numpy as np

from qutip.qip.models.circuitprocessor import CircuitProcessor
from qutip.operators import *
from qutip.states import basis
from qutip.qip.gates import hadamard_transform
from qutip.tensor import tensor
from qutip.solver import Options
from qutip.random_objects import rand_ket
from qutip.qip.models.circuitnoise import DecoherenceNoise, WhiteNoise
from qutip.qip.qubits import qubit_states
from qutip.metrics import fidelity


class TestCircuitProcessor:
    def test_modify_hams(self):
        """
        Test for modifying Hamiltonian, add_ctrl, remove_ctrl
        """
        N = 2
        proc = CircuitProcessor(N=N)
        proc.ctrls
        proc.ctrls = [sigmaz()]
        assert_(tensor([sigmaz(), identity(2)]), proc.ctrls[0])
        proc.add_ctrl(sigmax(), expand_type='periodic')
        assert_allclose(len(proc.ctrls), 3)
        assert_allclose(tensor([sigmax(), identity(2)]), proc.ctrls[1])
        assert_allclose(tensor([identity(2), sigmax()]), proc.ctrls[2])
        proc.add_ctrl(sigmay(), targets=1)
        assert_allclose(tensor([identity(2), sigmay()]), proc.ctrls[3])
        proc.remove_ctrl([0, 1, 2])
        assert_allclose(tensor([identity(2), sigmay()]), proc.ctrls[0])
        proc.remove_ctrl(0)
        assert_allclose(len(proc.ctrls), 0)

    def test_save_read(self):
        """
        Test for saving and reading a pulse matrix
        """
        proc = CircuitProcessor(N=2)
        proc.add_ctrl(sigmaz(), expand_type='periodic')
        proc1 = CircuitProcessor(N=2)
        proc1.add_ctrl(sigmaz(), expand_type='periodic')
        proc2 = CircuitProcessor(N=2)
        proc2.add_ctrl(sigmaz(), expand_type='periodic')
        tlist = [0., 0.1, 0.2, 0.3, 0.4, 0.5]
        amp1 = np.arange(0, 5, 1)
        amp2 = np.arange(5, 0, -1)

        proc.tlist = tlist
        proc.amps = np.array([amp1, amp2])
        proc.save_amps("qutip_test_CircuitProcessor.txt")
        proc1.read_amps("qutip_test_CircuitProcessor.txt")
        os.remove("qutip_test_CircuitProcessor.txt")
        assert_allclose(proc1.amps, proc.amps)
        assert_allclose(proc1.tlist, proc.tlist)
        proc.save_amps("qutip_test_CircuitProcessor.txt", inctime=False)
        proc2.read_amps("qutip_test_CircuitProcessor.txt", inctime=False)
        os.remove("qutip_test_CircuitProcessor.txt")
        assert_allclose(proc2.amps, proc.amps)
        assert_(proc2.tlist is None)

    def test_id_evolution(self):
        """
        Test for identity evolution with external/internal tlist
        """
        N = 1
        proc = CircuitProcessor(N=N)
        rho0 = rand_ket(2)
        result = proc.run_state(
            rho0, tlist=[0., 1., 2.],
            options=Options(store_final_state=True))
        global_phase = rho0.data[0, 0]/result.final_state.data[0, 0]
        assert_allclose(global_phase*result.final_state, rho0)
        proc.tlist = [0., 1., 2.]
        result = proc.run_state(
            rho0, options=Options(store_final_state=True))
        global_phase = rho0.data[0, 0]/result.final_state.data[0, 0]
        assert_allclose(global_phase*result.final_state, rho0)

    def test_id_with_T1_T2(self):
        """
        Test for identity evolution with relaxation T1 and T2
        """
        # setup
        a = destroy(2)
        Hadamard = hadamard_transform(1)
        ex_state = basis(2, 1)
        mines_state = (basis(2, 1)-basis(2, 0)).unit()
        end_time = 2.
        tlist = np.arange(0, end_time + 0.02, 0.02)
        T1 = 1.
        T2 = 0.5

        # test T1
        test = CircuitProcessor(1, T1=T1)
        result = test.run_state(ex_state, e_ops=[a.dag()*a], tlist=tlist)

        assert_allclose(
            result.expect[0][-1], np.exp(-1./T1*end_time),
            rtol=1e-5, err_msg="Error in T1 time simulation")

        # test T2
        test = CircuitProcessor(1, T2=T2)
        result = test.run_state(
            rho0=mines_state, tlist=tlist, e_ops=[Hadamard*a.dag()*a*Hadamard])
        assert_allclose(
            result.expect[0][-1], np.exp(-1./T2*end_time)*0.5+0.5,
            rtol=1e-5, err_msg="Error in T2 time simulation")

        # test T1 and T2
        T1 = np.random.rand(1) + 0.5
        T2 = np.random.rand(1) * 0.5 + 0.5
        test = CircuitProcessor(1, T1=T1, T2=T2)
        result = test.run_state(
            rho0=mines_state, tlist=tlist, e_ops=[Hadamard*a.dag()*a*Hadamard])
        assert_allclose(
            result.expect[0][-1], np.exp(-1./T2*end_time)*0.5+0.5,
            rtol=1e-5,
            err_msg="Error in T1 & T2 simulation, "
                    "with T1={} and T2={}".format(T1, T2))

    def TestNoise(self):
        """
        Test for CircuitProcessor with noise
        """
        # setup and fidelity without noise
        rho0 = qubit_states(2, [0, 0, 0, 0])
        tlist = np.array([0., np.pi/2])
        a = destroy(2)
        proc = CircuitProcessor(N=2)
        proc.tlist = tlist
        proc.amps = np.array([1]).reshape((1, 1))
        proc.add_ctrl(sigmax(), targets=1)
        result = proc.run_state(rho0=rho0)
        assert_allclose(
            fidelity(result.states[-1], qubit_states(2, [0, 1, 0, 0])),
            1, rtol=1.e-7)

        # decoherence noise
        dec_noise = DecoherenceNoise([0.5*a], targets=1)
        proc.add_noise(dec_noise)
        result = proc.run_state(rho0=rho0)
        assert_allclose(
            fidelity(result.states[-1], qubit_states(2, [0, 1, 0, 0])),
            0.9303888423022834, rtol=1.e-3)

        # white noise with internal/external operators
        proc.noise = []
        white_noise = WhiteNoise(mean=0.1, std=0.1)
        proc.add_noise(white_noise)
        result = proc.run_state(rho0=rho0)

        proc.noise = []
        white_noise = WhiteNoise(mean=0.1, std=0.1, ops=sigmax(), targets=1)
        proc.add_noise(white_noise)
        result = proc.run_state(rho0=rho0)

    def MultiLevelSystem(self):
        """
        Test for processor with multi-level system
        """
        N = 2
        proc = CircuitProcessor(N=N, dims=[2, 3])
        proc.add_ctrl(tensor(sigmaz(), rand_dm(3, density=1.)))
        proc.amps = np.array([1, 2]).reshape((1,2))
        proc.tlist = np.array([0., 1., 2])
        proc.run_state(rho0=tensor([basis(2, 0), basis(3, 1)]))


if __name__ == "__main__":
    run_module_suite()
