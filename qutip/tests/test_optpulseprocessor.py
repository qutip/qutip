import os

from numpy.testing import assert_, run_module_suite, assert_allclose
import numpy as np
import matplotlib.pyplot as plt

from qutip.qip.models.optpulseprocessor import OptPulseProcessor
from qutip.qip.gates import expand_oper
from qutip import *


class TestOptPulseProcessor:
    def test_simple_hadamard(self):
        N = 1
        H_d = sigmaz()
        H_c = [sigmax()]
        qc = QubitCircuit(N)
        qc.add_gate("SNOT", 0)

        # test load_circuit, with verbose info
        n_ts = 10
        evo_time = 10
        test = OptPulseProcessor(N, H_d, H_c)
        tlist, amps = test.load_circuit(
            qc, n_ts=n_ts, evo_time=evo_time, verbose=True)

        # test run_state
        rho0 = qubit_states(1, [0])
        plus = (qubit_states(1, [0]) + qubit_states(1, [1])).unit()
        result = test.run_state(rho0)
        assert_(fidelity(result.states[-1], plus) > 1-1.0e-6)

        # test add/remove ctrl
        test.add_ctrl(sigmay())
        test.remove_ctrl(1)
        assert_(len(test.ctrls) == 1)
        assert_allclose(test.drift, H_d)
        assert_(sigmay() in test.ctrls)

        # test plot
        test.plot_pulses()
        plt.clf()

    def test_multi_qubits(self):
        N = 3
        H_d = tensor([sigmaz()]*3)
        H_c = []

        # test empty ctrls
        n_ts = 30
        evo_time = 10
        test = OptPulseProcessor(N, H_d, H_c)
        test.add_ctrl(tensor([sigmax(), sigmax()]), expand_type="periodic")

        # test periodically adding ctrls
        sx = sigmax()
        iden = identity(2)
        assert_(Qobj(tensor([sx, iden, sx])) in test.ctrls)
        assert_(Qobj(tensor([iden, sx, sx])) in test.ctrls)
        assert_(Qobj(tensor([sx, sx, iden])) in test.ctrls)
        test.add_ctrl(sigmax(), expand_type='periodic')
        test.add_ctrl(sigmay(), expand_type='periodic')

        # test pulse genration for cnot gate, with kwargs
        qc = [tensor([identity(2), cnot()])]
        test.load_circuit(qc, n_ts=n_ts, evo_time=evo_time, min_fid_err=1.0e-6)
        rho0 = qubit_states(3, [1, 1, 1])
        rho1 = qubit_states(3, [1, 1, 0])
        result = test.run_state(
            rho0, options=Options(store_states=True))
        print(result.states[-1])
        print(rho1)
        assert_(fidelity(result.states[-1], rho1) > 1-1.0e-6)

        # test save and read amps
        test.save_amps("qutip_test_multi_qubits.txt")
        test2 = OptPulseProcessor(N, H_d, H_c)
        test2.drift = test.drift
        test2.ctrls = test.ctrls
        test2.read_amps("qutip_test_multi_qubits.txt")
        os.remove("qutip_test_multi_qubits.txt")
        assert_(np.max((test.amps-test2.amps)**2) < 1.0e-13)
        result = test2.run_state(rho0,)
        assert_(fidelity(result.states[-1], rho1) > 1-1.0e-6)

    def test_multi_gates(self):
        N = 2
        H_d = tensor([sigmaz()]*2)
        H_c = []

        test = OptPulseProcessor(N, H_d, H_c)
        test.add_ctrl(sigmax(), expand_type='periodic')
        test.add_ctrl(sigmay(), expand_type='periodic')
        test.add_ctrl(tensor([sigmay(), sigmay()]))

        # qubits circuit with 3 gates
        n_ts = [10, 30, 30]
        evo_time = [1, 3, 3]
        qc = QubitCircuit(N)
        qc.add_gate("SNOT", 0)
        qc.add_gate("SWAP", targets=[0, 1])
        qc.add_gate('CNOT', controls=1, targets=[0])
        test.load_circuit(qc, n_ts=n_ts, evo_time=evo_time)

        rho0 = rand_ket(4)  # use random generated ket state
        rho0.dims = [[2, 2], [1, 1]]
        U = gate_sequence_product(qc.propagators())
        rho1 = U * rho0
        result = test.run_state(rho0)
        assert_(fidelity(result.states[-1], rho1) > 1-1.0e-6)

    def test_T1_T2(self):
        # setup
        a = destroy(2)
        Hadamard = hadamard_transform(1)
        ex_state = basis(2, 1)
        mines_state = (basis(2, 1)-basis(2, 0)).unit()
        end_time = 2.
        tlist = np.arange(0, end_time + 0.02, 0.02)
        H_d = 10.*sigmaz()
        T1 = 1.
        T2 = 0.5

        # test T1
        test = OptPulseProcessor(1, drift=H_d, T1=T1)
        result = test.run_state(ex_state, e_ops=[a.dag()*a], tlist=tlist)

        assert_allclose(
            result.expect[0][-1], np.exp(-1./T1*end_time),
            rtol=1e-5, err_msg="Error in T1 time simulation")

        # test T2
        test = OptPulseProcessor(1, T2=T2)
        result = test.run_state(
            rho0=mines_state, tlist=tlist, e_ops=[Hadamard*a.dag()*a*Hadamard])
        assert_allclose(
            result.expect[0][-1], np.exp(-1./T2*end_time)*0.5+0.5,
            rtol=1e-5, err_msg="Error in T2 time simulation")

        # test T1 and T2
        T1 = np.random.rand(1) + 0.5
        T2 = np.random.rand(1) * 0.5 + 0.5
        test = OptPulseProcessor(1, T1=T1, T2=T2)
        result = test.run_state(
            rho0=mines_state, tlist=tlist, e_ops=[Hadamard*a.dag()*a*Hadamard])
        assert_allclose(
            result.expect[0][-1], np.exp(-1./T2*end_time)*0.5+0.5,
            rtol=1e-5,
            err_msg="Error in T1 & T2 simulation, "
                    "with T1={} and T2={}".format(T1, T2))

if __name__ == "__main__":
    run_module_suite()
