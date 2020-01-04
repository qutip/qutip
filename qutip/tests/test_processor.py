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

from numpy.testing import (
    assert_, run_module_suite, assert_allclose, assert_equal)
import numpy as np

from qutip.qip.device.processor import Processor
from qutip.states import basis
from qutip.operators import sigmaz, sigmax, sigmay, identity, destroy
from qutip.qip.gates import hadamard_transform
from qutip.tensor import tensor
from qutip.solver import Options
from qutip.random_objects import rand_ket, rand_dm
from qutip.qip.device.noise import (
    DecoherenceNoise, RandomNoise, ControlAmpNoise)
from qutip.qip.qubits import qubit_states
from qutip.metrics import fidelity


class TestCircuitProcessor:
    def test_modify_ctrls(self):
        """
        Test for modifying Hamiltonian, add_ctrl, remove_ctrl
        """
        N = 2
        proc = Processor(N=N)
        proc.ctrls
        proc.add_ctrl(sigmaz())
        assert_(tensor([sigmaz(), identity(2)]), proc.ctrls[0])
        proc.add_ctrl(sigmax(), cyclic_permutation=True)
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
        proc = Processor(N=2)
        proc.add_ctrl(sigmaz(), cyclic_permutation=True)
        proc1 = Processor(N=2)
        proc1.add_ctrl(sigmaz(), cyclic_permutation=True)
        proc2 = Processor(N=2)
        proc2.add_ctrl(sigmaz(), cyclic_permutation=True)
        tlist = [0., 0.1, 0.2, 0.3, 0.4, 0.5]
        amp1 = np.arange(0, 5, 1)
        amp2 = np.arange(5, 0, -1)

        proc.tlist = tlist
        proc.coeffs = np.array([amp1, amp2])
        proc.save_coeff("qutip_test_CircuitProcessor.txt")
        proc1.read_coeff("qutip_test_CircuitProcessor.txt")
        os.remove("qutip_test_CircuitProcessor.txt")
        assert_allclose(proc1.coeffs, proc.coeffs)
        assert_allclose(proc1.tlist, proc.tlist)
        proc.save_coeff("qutip_test_CircuitProcessor.txt", inctime=False)
        proc2.read_coeff("qutip_test_CircuitProcessor.txt", inctime=False)
        os.remove("qutip_test_CircuitProcessor.txt")
        assert_allclose(proc2.coeffs, proc.coeffs)
        assert_(proc2.tlist is None)

    def test_id_evolution(self):
        """
        Test for identity evolution
        """
        N = 1
        proc = Processor(N=N)
        rho0 = rand_ket(2)
        proc.tlist = [0., 1., 2.]
        result = proc.run_state(
            rho0, options=Options(store_final_state=True))
        global_phase = rho0.data[0, 0]/result.final_state.data[0, 0]
        assert_allclose(global_phase*result.final_state, rho0)

    def test_id_with_T1_T2(self):
        """
        Test for identity evolution with relaxation t1 and t2
        """
        # setup
        a = destroy(2)
        Hadamard = hadamard_transform(1)
        ex_state = basis(2, 1)
        mines_state = (basis(2, 1)-basis(2, 0)).unit()
        end_time = 2.
        tlist = np.arange(0, end_time + 0.02, 0.02)
        t1 = 1.
        t2 = 0.5

        # test t1
        test = Processor(1, t1=t1)
        test.tlist = tlist
        result = test.run_state(ex_state, e_ops=[a.dag()*a])

        assert_allclose(
            result.expect[0][-1], np.exp(-1./t1*end_time),
            rtol=1e-5, err_msg="Error in t1 time simulation")

        # test t2
        test = Processor(1, t2=t2)
        test.tlist = tlist
        result = test.run_state(
            rho0=mines_state, e_ops=[Hadamard*a.dag()*a*Hadamard])
        assert_allclose(
            result.expect[0][-1], np.exp(-1./t2*end_time)*0.5+0.5,
            rtol=1e-5, err_msg="Error in t2 time simulation")

        # test t1 and t2
        t1 = np.random.rand(1) + 0.5
        t2 = np.random.rand(1) * 0.5 + 0.5
        test = Processor(1, t1=t1, t2=t2)
        test.tlist = tlist
        result = test.run_state(
            rho0=mines_state, e_ops=[Hadamard*a.dag()*a*Hadamard])
        assert_allclose(
            result.expect[0][-1], np.exp(-1./t2*end_time)*0.5+0.5,
            rtol=1e-5,
            err_msg="Error in t1 & t2 simulation, "
                    "with t1={} and t2={}".format(t1, t2))

    def TestPlot(self):
        try:
            import matplotlib.pyplot as plt
        except:
            return True
        # step_func
        tlist = np.linspace(0., 2*np.pi, 20)
        processor = Processor(N=1, spline_kind="step_func")
        processor.add_ctrl(sigmaz())
        processor.tlist = tlist
        processor.coeffs = np.array([[np.sin(t) for t in tlist]])
        processor.plot_pulses(noisy=False)
        plt.clf()

        # cubic spline
        tlist = np.linspace(0., 2*np.pi, 20)
        processor = Processor(N=1, spline_kind="cubic")
        processor.add_ctrl(sigmaz())
        processor.tlist = tlist
        processor.coeffs = np.array([[np.sin(t) for t in tlist]])
        processor.plot_pulses(noisy=False)
        plt.clf()

        # noisy
        processor = Processor(N=1)
        processor.add_ctrl(sigmaz(), targets=0)
        processor.add_ctrl(sigmay(), targets=0)
        processor.coeffs = np.array([[ 0.5, 0.,  0.5],
                                    [ 0. , 0.5, 0. ]])
        processor.tlist = np.array([0., np.pi/2., 2*np.pi/2, 3*np.pi/2])

        processor.plot_pulses(noisy=False)
        plt.clf()
        processor.plot_pulses(noisy=True)
        plt.clf()

    def TestSpline(self):
        tlist = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        coeffs = np.array([[1, 1, 1, 1, 1, 1]], dtype=float)
        processor = Processor(N=1)
        processor.add_ctrl(sigmaz())
        processor.tlist = tlist
        processor.coeffs = coeffs

        processor.spline_kind = "step_func"
        noisy_qobjevo, c_ops = processor.get_noisy_qobjevo()
        assert_(noisy_qobjevo.args["_step_func_coeff"])

        processor.spline_kind = "cubic"
        noisy_qobjevo, c_ops = processor.get_noisy_qobjevo()
        assert_(not noisy_qobjevo.args["_step_func_coeff"])

    def TestGetObjevo(self):
        tlist = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        coeffs = np.array([[1, 1, 1, 1, 1, 1]], dtype=float)
        processor = Processor(N=1)
        processor.add_ctrl(sigmaz())
        processor.tlist = tlist
        processor.coeffs = coeffs

        # without noise
        unitary_qobjevo = processor.get_unitary_qobjevo(args={"test": True})
        assert_allclose(unitary_qobjevo.ops[0].qobj, sigmaz())
        assert_allclose(unitary_qobjevo.tlist, tlist)
        assert_allclose(unitary_qobjevo.ops[0].coeff, coeffs[0])
        assert_(unitary_qobjevo.args["test"],
                msg="Arguments not correctly passed on")

        # with decoherence noise
        dec_noise = DecoherenceNoise(
            c_ops=sigmax(), coeffs=coeffs, tlist=tlist)
        processor.add_noise(dec_noise)
        assert_equal(unitary_qobjevo.to_list(),
                        processor.get_unitary_qobjevo().to_list())

        noisy_qobjevo, c_ops = processor.get_noisy_qobjevo(args={"test": True})
        assert_(noisy_qobjevo.args["_step_func_coeff"],
                msg="Spline type not correctly passed on")
        assert_(noisy_qobjevo.args["test"],
                msg="Arguments not correctly passed on")
        assert_(sigmaz() in [pair[0] for pair in noisy_qobjevo.to_list()])
        assert_equal(c_ops[0].ops[0].qobj, sigmax())
        assert_equal(c_ops[0].tlist, tlist)

        # with amplitude noise
        processor.spline_kind = "cubic"
        new_tlist = np.linspace(1, 6, int(5/0.2))
        new_coeffs = np.random.rand(1, len(new_tlist))
        # noise with a different operator
        amp_noise = ControlAmpNoise(ops=sigmax(), coeffs=coeffs, tlist=tlist)
        processor.add_noise(amp_noise)
        noisy_qobjevo, c_ops = processor.get_noisy_qobjevo(args={"test": True})
        assert_(not noisy_qobjevo.args["_step_func_coeff"],
                msg="Spline type not correctly passed on")
        assert_(sigmax() in [pair[0] for pair in noisy_qobjevo.to_list()])
        # noise with operators in the processor
        # Since the noise operator is also sigmaz,
        # it should be merged with the original operator
        amp_noise2 = ControlAmpNoise(coeffs=coeffs, tlist=tlist)
        processor.noise[1] = amp_noise2
        noisy_qobjevo, c_ops = processor.get_noisy_qobjevo(args={"test": True})
        assert_(not noisy_qobjevo.args["_step_func_coeff"],
                msg="Spline type not correctly passed on")
        assert_equal(len(noisy_qobjevo.ops), 1)
        assert_equal(sigmaz(), noisy_qobjevo.ops[0].qobj)
        assert_equal(coeffs[0] * 2, noisy_qobjevo.ops[0].coeff)

    def TestNoise(self):
        """
        Test for Processor with noise
        """
        # setup and fidelity without noise
        rho0 = qubit_states(2, [0, 0, 0, 0])
        tlist = np.array([0., np.pi/2.])
        a = destroy(2)
        proc = Processor(N=2)
        proc.tlist = tlist
        proc.coeffs = np.array([1]).reshape((1, 1))
        proc.add_ctrl(sigmax(), targets=1)
        result = proc.run_state(rho0=rho0)
        assert_allclose(
            fidelity(result.states[-1], qubit_states(2, [0, 1, 0, 0])),
            1, rtol=1.e-7)

        # decoherence noise
        dec_noise = DecoherenceNoise([0.25*a], targets=1)
        proc.add_noise(dec_noise)
        result = proc.run_state(rho0=rho0)
        assert_allclose(
            fidelity(result.states[-1], qubit_states(2, [0, 1, 0, 0])),
            0.981852, rtol=1.e-3)

        # white noise with internal/external operators
        proc.noise = []
        white_noise = RandomNoise(loc=0.1, scale=0.1)
        proc.add_noise(white_noise)
        result = proc.run_state(rho0=rho0)

        proc.noise = []
        white_noise = RandomNoise(loc=0.1, scale=0.1, ops=sigmax(), targets=1)
        proc.add_noise(white_noise)
        result = proc.run_state(rho0=rho0)

    def TestMultiLevelSystem(self):
        """
        Test for processor with multi-level system
        """
        N = 2
        proc = Processor(N=N, dims=[2, 3])
        proc.add_ctrl(tensor(sigmaz(), rand_dm(3, density=1.)))
        proc.coeffs = np.array([1, 2]).reshape((1, 2))
        proc.tlist = np.array([0., 1., 2])
        proc.run_state(rho0=tensor([basis(2, 0), basis(3, 1)]))


if __name__ == "__main__":
    run_module_suite()
