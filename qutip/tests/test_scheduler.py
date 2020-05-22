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
from qutip.qip.circuit import QubitCircuit
from qutip.qip.scheduler import Instruction, Scheduler

def test_scheduling_gates():
    circuit = QubitCircuit(7)
    circuit.add_gate("SNOT", 3)
    circuit.add_gate("CZ", 5, 3)
    circuit.add_gate("CZ", 4, 3)
    circuit.add_gate("CZ", 2, 3)
    circuit.add_gate("CZ", 6, 5)
    circuit.add_gate("CZ", 2, 6)
    circuit.add_gate("SWAP", [0, 2])

    scheduler = Scheduler("ASAP")
    cycles_list = scheduler.schedule(circuit, gates_schedule=True)
    assert(cycles_list == [0, 1, 3, 2, 2, 3, 4])

    scheduler = Scheduler()
    cycles_list = scheduler.schedule(circuit)
    assert(cycles_list == [0, 1, 4, 2, 2, 3, 4])


def test_scheduling_pulses():
    instruction_list = [
        Instruction("H", [0], duration=1),
        Instruction("H", [1], duration=1),
        Instruction("CNOT", [2], [3], duration=2),
        Instruction("CNOT", [1], [2], duration=2),
        Instruction("CNOT", [1], [0], duration=2),
        Instruction("H", [3], duration=1),
        Instruction("CNOT", [1], [3], duration=2),
        Instruction("SWAP", [1], [3], duration=2),
    ]
    scheduler = Scheduler("ASAP")
    cycles_list = scheduler.schedule(instruction_list)
    assert(cycles_list == [0, 0, 0, 3, 1, 2, 5, 7])
    scheduler = Scheduler("ALAP")
    cycles_list = scheduler.schedule(instruction_list)
    assert(cycles_list == [0, 0, 1, 3, 1, 4, 5, 7])