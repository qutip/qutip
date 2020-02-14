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

from numpy.testing import (assert_, run_module_suite, assert_allclose,
                           assert_equal)
import numpy as np

from qutip.qip.device.optpulseprocessor import OptPulseProcessor
from qutip.qip.operations.gates import expand_operator
from qutip.operators import sigmaz, sigmax, sigmay, identity, destroy
from qutip.qip.circuit import QubitCircuit
from qutip.qip.qubits import qubit_states
from qutip.metrics import fidelity
from qutip.qobj import Qobj
from qutip.tensor import tensor
from qutip.solver import Options
from qutip.qip.operations.gates import cnot, gate_sequence_product, hadamard_transform
from qutip.random_objects import rand_ket
from qutip.states import basis


class TestOptPulseProcessor:
    def test_simple_hadamard(self):
        """
        Test for optimizing a simple hadamard gate
        """
        N = 1
        H_d = sigmaz()
        H_c = sigmax()
        qc = QubitCircuit(N)
        qc.add_gate("SNOT", 0)

        # test load_circuit, with verbose info
        num_tslots = 10
        evo_time = 10
        test = OptPulseProcessor(N, drift=H_d)
        test.add_control(H_c, targets=0)
        tlist, coeffs = test.load_circuit(
            qc, num_tslots=num_tslots, evo_time=evo_time, verbose=True)

        # test run_state
        rho0 = qubit_states(1, [0])
        plus = (qubit_states(1, [0]) + qubit_states(1, [1])).unit()
        result = test.run_state(rho0)
        assert_allclose(fidelity(result.states[-1], plus), 1, rtol=1.0e-6)

        # test add/remove ctrl
        test.add_control(sigmay())
        test.remove_pulse(0)
        assert_(
            len(test.pulses) == 1,
            msg="Method of remove_pulse could be wrong.")
        assert_allclose(test.drift.drift_hamiltonians[0].qobj, H_d)
        assert_(
            sigmay() in test.ctrls,
            msg="Method of remove_pulse could be wrong.")

    def test_multi_qubits(self):
        """
        Test for multi-qubits system.
        """
        N = 3
        H_d = tensor([sigmaz()]*3)
        H_c = []

        # test empty ctrls
        num_tslots = 30
        evo_time = 10
        test = OptPulseProcessor(N)
        test.add_drift(H_d, [0, 1, 2])
        test.add_control(tensor([sigmax(), sigmax()]),
                          cyclic_permutation=True)
        # test periodically adding ctrls
        sx = sigmax()
        iden = identity(2)
        # print(test.ctrls)
        # print(Qobj(tensor([sx, iden, sx])))
        assert_(Qobj(tensor([sx, iden, sx])) in test.ctrls)
        assert_(Qobj(tensor([iden, sx, sx])) in test.ctrls)
        assert_(Qobj(tensor([sx, sx, iden])) in test.ctrls)
        test.add_control(sigmax(), cyclic_permutation=True)
        test.add_control(sigmay(), cyclic_permutation=True)

        # test pulse genration for cnot gate, with kwargs
        qc = [tensor([identity(2), cnot()])]
        test.load_circuit(qc, num_tslots=num_tslots,
                          evo_time=evo_time, min_fid_err=1.0e-6)
        rho0 = qubit_states(3, [1, 1, 1])
        rho1 = qubit_states(3, [1, 1, 0])
        result = test.run_state(
            rho0, options=Options(store_states=True))
        assert_(fidelity(result.states[-1], rho1) > 1-1.0e-6)

        # # test save and read coeffs
        # test.save_coeff("qutip_test_multi_qubits.txt")
        # test2 = OptPulseProcessor(N, H_d, H_c)
        # test2.drift = test.drift
        # test2.ctrls = test.ctrls
        # test2.read_coeff("qutip_test_multi_qubits.txt")
        # os.remove("qutip_test_multi_qubits.txt")
        # assert_(np.max((test.coeffs-test2.coeffs)**2) < 1.0e-13)
        # result = test2.run_state(rho0,)
        # assert_(fidelity(result.states[-1], rho1) > 1-1.0e-6)

    def test_multi_gates(self):
        N = 2
        H_d = tensor([sigmaz()]*2)
        H_c = []

        test = OptPulseProcessor(N)
        test.add_drift(H_d, [0, 1])
        test.add_control(sigmax(), cyclic_permutation=True)
        test.add_control(sigmay(), cyclic_permutation=True)
        test.add_control(tensor([sigmay(), sigmay()]))

        # qubits circuit with 3 gates
        setting_args = {"SNOT": {"num_tslots": 10, "evo_time": 1},
                        "SWAP": {"num_tslots": 30, "evo_time": 3},
                        "CNOT": {"num_tslots": 30, "evo_time": 3}}
        qc = QubitCircuit(N)
        qc.add_gate("SNOT", 0)
        qc.add_gate("SWAP", targets=[0, 1])
        qc.add_gate('CNOT', controls=1, targets=[0])
        test.load_circuit(qc, setting_args=setting_args,
                          merge_gates=False)

        rho0 = rand_ket(4)  # use random generated ket state
        rho0.dims = [[2, 2], [1, 1]]
        U = gate_sequence_product(qc.propagators())
        rho1 = U * rho0
        result = test.run_state(rho0)
        assert_(fidelity(result.states[-1], rho1) > 1-1.0e-6)


if __name__ == "__main__":
    run_module_suite()
