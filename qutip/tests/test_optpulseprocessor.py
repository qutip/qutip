import unittest
import os

import numpy as np
import matplotlib.pyplot as plt

from qutip.qip.models.optpulseprocessor import OptPulseProcessor, expand_oper
from qutip import *

def test_expand_oper():
    # random single qubit gate test
    r = rand_unitary(2)
    assert(expand_oper(r, 3, 0) == tensor([r, identity(2), identity(2)]))
    assert(expand_oper(r, 3, 1) == tensor([identity(2), r, identity(2)]))
    assert(expand_oper(r, 3, 2) == tensor([identity(2), identity(2), r]))

    # random 2qubits gate test
    r2 = rand_unitary(4)
    r2.dims = [[2,2],[2,2]]
    assert(expand_oper(r2, 3, [2,1]) == tensor(
        [identity(2), r2.permute([1,0])]))
    assert(expand_oper(r2, 3, [0,1]) == tensor(
        [r2, identity(2)]))
    assert(expand_oper(r2, 3, [0,2]) == tensor(
        [r2, identity(2)]).permute([0,2,1]))

    # cnot expantion, qubit 2 control qubit 0
    assert(expand_oper(cnot(), 3, [2,0]) == Qobj([
        [1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0.]]
    , dims=[[2,2,2],[2,2,2]]))

def test_simple_hadamard():
    N = 1
    H_d = sigmaz()
    H_c = [sigmax()]
    qc = QubitCircuit(N)
    qc.add_gate("SNOT", 0)

    # test load_circuit
    n_ts = 10
    evo_time = 10
    test = OptPulseProcessor(N, H_d, H_c)
    tlist, amps = test.load_circuit(qc, n_ts, evo_time,)

    # test run_state
    rho0 = qubit_states(1, [0])
    plus = (qubit_states(1, [0]) + qubit_states(1, [1])).unit()
    result = test.run_state(rho0, dt = 0.01)
    assert(fidelity(result.states[-1], plus) > 1-1.0e-4)

    # test add/remove ctrl
    test.add_ctrl(sigmay())
    test.remove_ctrl(0)
    assert(len(test.ctrls) == 1)
    assert(sigmay() in test.ctrls)

    # test plot
    test.plot_pulses()
    plt.clf()

def test_multi_qubits():
    N = 3
    H_d = tensor([sigmaz()]*3)
    H_c = []

    n_ts = 30
    evo_time = 10
    test = OptPulseProcessor(N, H_d, H_c)
    test.add_ctrl(tensor([sigmax(), sigmax()]), expand_type = "periodic")

    # test periodically adding ctrls
    sx = sigmax()
    iden = identity(2)
    assert(Qobj(tensor([sx, iden, sx])) in test.ctrls)
    assert(Qobj(tensor([iden, sx, sx])) in test.ctrls)
    assert(Qobj(tensor([sx, sx, iden])) in test.ctrls)
    test.add_ctrl(sigmax(), expand_type = 'periodic')
    test.add_ctrl(sigmay(), expand_type = 'periodic')

    # test pulse genration for cnot gate
    qc = [tensor([identity(2), cnot()])]
    test.load_circuit(qc, n_ts, evo_time)
    rho0 = qubit_states(3, [1,1,1])
    rho1 = qubit_states(3, [1,1,0])
    result = test.run_state(rho0, dt = 0.003)
    assert(fidelity(result.states[-1], rho1) > 1-1.0e-4)

    # test save and read amps
    test.save_amps("qutip_test_multi_qubits.txt")
    test2 = OptPulseProcessor(N, H_d, H_c)
    test2.ctrls = test.ctrls
    test2.read_amps("qutip_test_multi_qubits.txt")
    os.remove("qutip_test_multi_qubits.txt")
    assert(np.max((test.amps-test2.amps)**2) < 1.0e-13)
    result = test2.run_state(rho0, dt = 0.003)
    assert(fidelity(result.states[-1], rho1) > 1-1.0e-4)

def test_multi_gates():
    N = 2
    H_d = tensor([sigmaz()]*2)
    H_c = []

    sx = sigmax()
    test = OptPulseProcessor(N, H_d, H_c)
    test.add_ctrl(sigmax(), expand_type = 'periodic')
    test.add_ctrl(sigmay(), expand_type = 'periodic')
    test.add_ctrl(tensor([sigmay(),sigmay()]))

    # qubits circuit with 3 gates
    n_ts = [10,30,30]
    evo_time = [1,3,3]
    qc = QubitCircuit(N)
    qc.add_gate("SNOT", 0)
    qc.add_gate("SWAP", targets = [0, 1])
    qc.add_gate('CNOT', controls = 1, targets = [0])
    test.load_circuit(qc, n_ts, evo_time)

    rho0 = rand_ket(4) # use random generated ket state
    rho0.dims = [[2,2],[1,1]]
    U = gate_sequence_product(qc.propagators())
    rho1 = U * rho0
    result = test.run_state(rho0, dt = 0.003)
    assert(fidelity(result.states[-1], rho1) > 1-1.0e-4)

# TODO: Test for exceptions