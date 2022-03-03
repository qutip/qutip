import pytest
from copy import deepcopy

from qutip.qip.circuit import QubitCircuit
from qutip.qip.compiler import Instruction, Scheduler
from qutip.qip.operations.gates import gate_sequence_product
from qutip import process_fidelity, qeye, tracedist
from qutip.qip.circuit import Gate


def _circuit1():
    circuit1 = QubitCircuit(6)
    circuit1.add_gate("SNOT", 2)
    circuit1.add_gate("CNOT", 4, 2)
    circuit1.add_gate("CNOT", 3, 2)
    circuit1.add_gate("CNOT", 1, 2)
    circuit1.add_gate("CNOT", 5, 4)
    circuit1.add_gate("CNOT", 1, 5)
    circuit1.add_gate("SWAP", [0, 1])
    return circuit1

def _circuit2():
    circuit2 = QubitCircuit(8)
    circuit2.add_gate("SNOT", 1)
    circuit2.add_gate("SNOT", 2)
    circuit2.add_gate("SNOT", 3)
    circuit2.add_gate("CNOT", 4, 5)
    circuit2.add_gate("CNOT", 4, 6)
    circuit2.add_gate("CNOT", 3, 4)
    circuit2.add_gate("CNOT", 3, 5)
    circuit2.add_gate("CNOT", 3, 7)
    circuit2.add_gate("CNOT", 2, 4)
    circuit2.add_gate("CNOT", 2, 6)
    circuit2.add_gate("CNOT", 2, 7)
    circuit2.add_gate("CNOT", 1, 5)
    circuit2.add_gate("CNOT", 1, 6)
    circuit2.add_gate("CNOT", 1, 7)
    return circuit2

def _instructions1():
    circuit3 = QubitCircuit(6)
    circuit3.add_gate("SNOT", 0)
    circuit3.add_gate("SNOT", 1)
    circuit3.add_gate("CNOT", 2, 3)
    circuit3.add_gate("CNOT", 1, 2)
    circuit3.add_gate("CNOT", 1, 0)
    circuit3.add_gate("SNOT", 3)
    circuit3.add_gate("CNOT", 1, 3)
    circuit3.add_gate("CNOT", 1, 3)

    instruction_list = []
    for gate in circuit3.gates:
        if gate.name == "SNOT":
            instruction_list.append(Instruction(gate, duration=1))
        else:
            instruction_list.append(Instruction(gate, duration=2))

    return instruction_list


@pytest.mark.parametrize(
    "circuit, method, expected_length, random_shuffle, gates_schedule",
    [
        pytest.param(deepcopy(_circuit1()), "ASAP", 4, False, False, id="circuit1 ASAP)"),
        pytest.param(deepcopy(_circuit1()), "ALAP", 4, False, False, id="circuit1 ALAP)")
    ])
def test_scheduling_gates1(
        circuit, method, expected_length, random_shuffle, gates_schedule):
    if random_shuffle:
        repeat_num = 5
    else:
        repeat_num = 0
    result0 = gate_sequence_product(circuit.propagators())

    # run the scheduler
    scheduler = Scheduler(method)
    gate_cycle_indices = scheduler.schedule(
        circuit, gates_schedule=gates_schedule, repeat_num=repeat_num)

    # check if the scheduled length is expected
    assert(max(gate_cycle_indices) == expected_length)
    scheduled_gate = [[] for i in range(max(gate_cycle_indices)+1)]

    # check if the scheduled circuit is correct
    for i, cycles in enumerate(gate_cycle_indices):
        scheduled_gate[cycles].append(circuit.gates[i])
    circuit.gates = sum(scheduled_gate, [])
    result1 = gate_sequence_product(circuit.propagators())
    assert(tracedist(result0*result1.dag(), qeye(result0.dims[0])) < 1.0e-7)


# There is some problem with circuit2 on Mac.
# When I list all three tests together, a segment fault arise.
# I suspect it has something to do with pytest.
@pytest.mark.parametrize(
    "circuit, method, expected_length, random_shuffle, gates_schedule",
    [
        pytest.param(deepcopy(_circuit2()), "ASAP", 4, False, False, id="circuit2 ASAP"),
    ])
def test_scheduling_gates2(
        circuit, method, expected_length, random_shuffle, gates_schedule):
    if random_shuffle:
        repeat_num = 5
    else:
        repeat_num = 0
    result0 = gate_sequence_product(circuit.propagators())

    # run the scheduler
    scheduler = Scheduler(method)
    gate_cycle_indices = scheduler.schedule(
        circuit, gates_schedule=gates_schedule, repeat_num=repeat_num)

    # check if the scheduled length is expected
    assert(max(gate_cycle_indices) == expected_length)
    scheduled_gate = [[] for i in range(max(gate_cycle_indices)+1)]

    # check if the scheduled circuit is correct
    for i, cycles in enumerate(gate_cycle_indices):
        scheduled_gate[cycles].append(circuit.gates[i])
    circuit.gates = sum(scheduled_gate, [])
    result1 = gate_sequence_product(circuit.propagators())
    assert(tracedist(result0*result1.dag(), qeye(result0.dims[0])) < 1.0e-7)

@pytest.mark.parametrize(
    "circuit, method, expected_length, random_shuffle, gates_schedule",
    [
        pytest.param(deepcopy(_circuit2()), "ALAP", 5, False, False, id="circuit2 ALAP no shuffle")
    ])
def test_scheduling_gates3(
        circuit, method, expected_length, random_shuffle, gates_schedule):
    if random_shuffle:
        repeat_num = 5
    else:
        repeat_num = 0
    result0 = gate_sequence_product(circuit.propagators())

    # run the scheduler
    scheduler = Scheduler(method)
    gate_cycle_indices = scheduler.schedule(
        circuit, gates_schedule=gates_schedule, repeat_num=repeat_num)

    # check if the scheduled length is expected
    assert(max(gate_cycle_indices) == expected_length)
    scheduled_gate = [[] for i in range(max(gate_cycle_indices)+1)]

    # check if the scheduled circuit is correct
    for i, cycles in enumerate(gate_cycle_indices):
        scheduled_gate[cycles].append(circuit.gates[i])
    circuit.gates = sum(scheduled_gate, [])
    result1 = gate_sequence_product(circuit.propagators())
    assert(tracedist(result0*result1.dag(), qeye(result0.dims[0])) < 1.0e-7)


@pytest.mark.parametrize(
    "circuit, method, expected_length, random_shuffle, gates_schedule",
    [
        pytest.param(deepcopy(_circuit2()), "ALAP", 4, True, False, id="circuit2 ALAP shuffle"),  # with random shuffling
    ])
def test_scheduling_gates4(
        circuit, method, expected_length, random_shuffle, gates_schedule):
    if random_shuffle:
        repeat_num = 5
    else:
        repeat_num = 0
    result0 = gate_sequence_product(circuit.propagators())

    # run the scheduler
    scheduler = Scheduler(method)
    gate_cycle_indices = scheduler.schedule(
        circuit, gates_schedule=gates_schedule, repeat_num=repeat_num)

    # check if the scheduled length is expected
    assert(max(gate_cycle_indices) == expected_length)
    scheduled_gate = [[] for i in range(max(gate_cycle_indices)+1)]

    # check if the scheduled circuit is correct
    for i, cycles in enumerate(gate_cycle_indices):
        scheduled_gate[cycles].append(circuit.gates[i])
    circuit.gates = sum(scheduled_gate, [])
    result1 = gate_sequence_product(circuit.propagators())
    assert(tracedist(result0*result1.dag(), qeye(result0.dims[0])) < 1.0e-7)


@pytest.mark.parametrize(
    "instructions, method, expected_length, random_shuffle, gates_schedule",
    [
        pytest.param(deepcopy(_instructions1()), "ASAP", 7, False, False, id="circuit1 pulse ASAP)"),
        pytest.param(deepcopy(_instructions1()), "ALAP", 7, False, False, id="circuit1 pulse ALAP)"),
    ])
def test_scheduling_pulse(
        instructions, method, expected_length, random_shuffle, gates_schedule):
    circuit = QubitCircuit(4)
    for instruction in instructions:
        circuit.add_gate(
            Gate(instruction.name,
            instruction.targets,
            instruction.controls))

    if random_shuffle:
        repeat_num = 5
    else:
        repeat_num = 0
    result0 = gate_sequence_product(circuit.propagators())

    # run the scheduler
    scheduler = Scheduler(method)
    gate_cycle_indices = scheduler.schedule(
        instructions, gates_schedule=gates_schedule, repeat_num=repeat_num)

    # check if the scheduled length is expected
    assert(max(gate_cycle_indices) == expected_length)
    scheduled_gate = [[] for i in range(max(gate_cycle_indices)+1)]

    # check if the scheduled circuit is correct
    for i, cycles in enumerate(gate_cycle_indices):
        scheduled_gate[cycles].append(circuit.gates[i])
    circuit.gates = sum(scheduled_gate, [])
    result1 = gate_sequence_product(circuit.propagators())
    assert(tracedist(result0*result1.dag(), qeye(result0.dims[0])) < 1.0e-7)
