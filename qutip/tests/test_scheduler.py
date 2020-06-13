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
import pytest
from copy import deepcopy

from qutip.qip.circuit import QubitCircuit
from qutip.qip.scheduler import Instruction, Scheduler
from qutip.qip.operations.gates import gate_sequence_product
from qutip import process_fidelity, qeye, tracedist


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

def _circuit3():
    circuit3 = QubitCircuit(4)
    instruction_list1 = [
        Instruction("SNOT", [0], duration=1),
        Instruction("SNOT", [1], duration=1),
        Instruction("CNOT", [2], [3], duration=2),
        Instruction("CNOT", [1], [2], duration=2),
        Instruction("CNOT", [1], [0], duration=2),
        Instruction("SNOT", [3], duration=1),
        Instruction("CNOT", [1], [3], duration=2),
        Instruction("CNOT", [1], [3], duration=2)
    ]
    circuit3.gates = instruction_list1
    return circuit3


@pytest.mark.parametrize(
    "circuit, method, expected_length, random_shuffle, gates_schedule",
    [
        (deepcopy(_circuit1()), "ASAP", 4, False, False),
        (deepcopy(_circuit1()), "ALAP", 4, False, False),
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


@pytest.mark.parametrize(
    "circuit, method, expected_length, random_shuffle, gates_schedule",
    [
        (deepcopy(_circuit2()), "ASAP", 4, False, False),
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
        (deepcopy(_circuit2()), "ALAP", 5, False, False),
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
        (deepcopy(_circuit2()), "ALAP", 4, True,  False),  # with random shuffling
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
        (deepcopy(_circuit3()), "ASAP", 7, False, False),
        (deepcopy(_circuit3()), "ALAP", 7, False, False),
        (deepcopy(_circuit3()), "ASAP", 4, False, True),  # treat instructions as gates
        (deepcopy(_circuit3()), "ALAP", 4, False, True),  # treat instructions as gates
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
