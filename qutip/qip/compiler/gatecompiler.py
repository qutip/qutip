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
from .instruction import Instruction
from .scheduler import Scheduler
from ..circuit import QubitCircuit, Gate


__all__ = ['GateCompiler']


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
        Dictionary of pulse indices.

    Attributes
    ----------
    N: int
        The number of the component systems.

    params: dict
        A Python dictionary contains the name and the value of the parameters,
        such as laser frequency, detuning etc.

    pulse_dict: int
        Dictionary of pulse indices.

    gate_compiler: dict
        The Python dictionary in the form of {gate_name: decompose_function}.
        It saves the decomposition scheme for each gate.
    """
    def __init__(self, N, params, pulse_dict):
        self.gate_compiler = {}
        self.N = N
        self.params = params
        self.pulse_dict = pulse_dict
        self.gate_compiler = {"GLOBALPHASE": self.globalphase_compiler}
        self.args = {}

    def globalphase_compiler(self, gate, args):
        """
        Compiler for the GLOBALPHASE gate
        """
        pass

    def compile(self, circuit, schedule_mode=None):
        """
        Compile the the elementary gates
        into control pulse sequence.

        Parameters
        ----------
        circuit: :class:`QubitCircuit` or list of :class:`gate`
            A list of elementary gates that can be implemented in this
            model. The gate names have to be in `gate_compiler`.
        
        schedule_mode: str
            "ASAP" for "as soon as possible" or
            "ALAP" for "as late as possible"

        Returns
        -------
        tlist: array_like
            A NumPy array specifies the time of each coefficient

        coeffs: array_like
            A 2d NumPy array of the shape ``(len(ctrls), len(tlist))``. Each
            row corresponds to the control pulse sequence for
            one Hamiltonian.
        """
        if isinstance(circuit, QubitCircuit):
            gates = circuit.gates
        else:
            gates = circuit
        num_ops = len(self.pulse_dict)
        instruction_list = []
        for gate in gates:
            if gate.name not in self.gate_compiler:
                raise ValueError("Unsupported gate %s" % gate.name)
            compilered_gate = self.gate_compiler[gate.name](gate, self.args)
            if compilered_gate is None:
                continue  # neglecting global phase gate
            instruction_list += compilered_gate
        if not instruction_list:
            return None, None

        # check continuous or discrete pulse
        if np.isscalar(instruction_list[0].tlist):
            spline_kind = "step_func"
        elif (len(instruction_list[0].tlist) - 1 == \
                len(instruction_list[0].pulse_info[0][1])):
            spline_kind = "step_func"
        elif (len(instruction_list[0].tlist) == \
                len(instruction_list[0].pulse_info[0][1])):
            spline_kind = "cubic"
        else:
            raise ValueError(
                "The shape of the compiled pulse is not correct.")

        # scheduling
        if schedule_mode:
            scheduler = Scheduler(schedule_mode)
            scheduled_start_time = scheduler.schedule(instruction_list)
            time_ordered_pos = np.argsort(scheduled_start_time)
        else:  # no scheduling
            time_ordered_pos = list(range(0, len(instruction_list)))
            scheduled_start_time = [0.]
            for instruction in instruction_list:
                scheduled_start_time.append(instruction.duration + scheduled_start_time[-1])
            scheduled_start_time = scheduled_start_time[:-1]

        # compile
        tlist = [[[0.]] for tmp in range(num_ops)]
        coeffs = [[] for tmp in range(num_ops)]
        for ind in time_ordered_pos:
            instruction = instruction_list[ind]
            start_time = scheduled_start_time[ind]
            for pulse_name, coeff in instruction.pulse_info:
                pulse_ind = self.pulse_dict[pulse_name]
                if np.isscalar(instruction.tlist):
                    step_size = instruction.tlist
                    temp_tlist = np.array([instruction.tlist])
                    coeff = np.array([coeff])
                else:
                    step_size = instruction.tlist[1] - instruction.tlist[0]
                    if coeffs[pulse_ind]:  # not the first pulse
                        temp_tlist = instruction.tlist[1:]
                        if spline_kind == "cubic":
                            coeff = coeff[1:]
                    else:
                        temp_tlist = instruction.tlist
                if np.abs(start_time - tlist[pulse_ind][-1][-1]) > step_size * 1.0e-6:
                    tlist[pulse_ind].append([start_time])
                    coeffs[pulse_ind].append([0.])
                tlist[pulse_ind].append(temp_tlist + start_time)
                coeffs[pulse_ind].append(coeff)
        for i in range(num_ops):
            if not coeffs[i]:
                tlist[i] = None
                coeffs[i] = None
            else:
                tlist[i] = np.concatenate(tlist[i])
                coeffs[i] = np.concatenate(coeffs[i])
                #  remove the leading 0
                if spline_kind == "cubic":
                    tlist[i] = tlist[i][1:]
        return tlist, coeffs
