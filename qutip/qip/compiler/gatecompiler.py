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
import numpy as np
from ..scheduler import Instruction


__all__ = ['GateCompiler']


class _PulseInstruction(Instruction):
    def __init__(self, gate, tlist, pulse_coeffs):
        super(_PulseInstruction, self).__init__(
            gate)
        self.pulse_coeffs = pulse_coeffs
        self.tlist = tlist
        self.duration = tlist[-1]
    
    @property
    def step_num(self):
        return len(self.tlist)


class GateCompiler(object):
    """
    Base class. It decomposes a :class:`qutip.QubitCircuit` into
    the pulse sequence for the processor.

    Parameters
    ----------
    N: int
        The number of the component systems.

    params: dict
        A Python dictionary contains the name and the value of the parameters,
        such as laser frequency, detuning etc.

    num_ops: int
        Number of control Hamiltonians in the processor.

    Attributes
    ----------
    N: int
        The number of the component systems.

    params: dict
        A Python dictionary contains the name and the value of the parameters,
        such as laser frequency, detuning etc.

    num_ops: int
        Number of control Hamiltonians in the processor.

    gate_decomps: dict
        The Python dictionary in the form of {gate_name: decompose_function}.
        It saves the decomposition scheme for each gate.
    """
    def __init__(self, N, params, num_ops):
        self.gate_decomps = {}
        self.N = N
        self.params = params
        self.num_ops = num_ops

    def decompose(self, gates):
        """
        Decompose the the elementary gates
        into control pulse sequence.

        Parameters
        ----------
        gates: list
            A list of elementary gates that can be implemented in this
            model. The gate names have to be in `gate_decomps`.

        Returns
        -------
        tlist: array_like
            A NumPy array specifies the time of each coefficient

        coeffs: array_like
            A 2d NumPy array of the shape ``(len(ctrls), len(tlist))``. Each
            row corresponds to the control pulse sequence for
            one Hamiltonian.

        global_phase: bool
            Recorded change of global phase.
        """
        instruction_list = []
        for gate in gates:
            if gate.name not in self.gate_decomps:
                raise ValueError("Unsupported gate %s" % gate.name)
            compilered_gate = self.gate_decomps[gate.name](gate)
            if compilered_gate is None:
                continue  # neglecting global phase gate
            instruction_list += compilered_gate

        max_step_num = sum([instruction.step_num for instruction in instruction_list])
        dt_list = np.zeros(max_step_num)
        coeff_list = np.zeros((self.num_ops, max_step_num))
        last_time_step = 0
        for instruction in instruction_list:
            for pulse_ind, coeff in instruction.pulse_coeffs:
                dt_list[last_time_step: last_time_step + instruction.step_num] = instruction.tlist
                coeff_list[pulse_ind, last_time_step: last_time_step + instruction.step_num] = coeff
            last_time_step += instruction.step_num
        coeffs = np.asarray(coeff_list)

        tlist = np.empty(len(dt_list))
        t = 0
        for i in range(len(dt_list)):
            t += dt_list[i]
            tlist[i] = t
        return np.hstack([[0], tlist]), coeffs
