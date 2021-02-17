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
    Base class. It compiles a :class:`qutip.QubitCircuit` into
    the pulse sequence for the processor. The core member function
    `compile` calls compiling method from the sub-class and concatenate
    the compiled pulses.

    Parameters
    ----------
    N: int
        The number of the component systems.

    kwargs:
        Keyword arguments for the compiler.
        By default, the hardware parameters `Processor.params` defined
        in the processor and a map between the pulse label and the
        position `Processor.pulse_dict` is passed to the compiler
        when calling it.
        It adds more flexibility in customizing compiler.

    params: dict, optional
        A Python dictionary contains the name and the value of the parameters,
        such as laser frequency, detuning etc.
        It will be saved in the class attributes and can be used to calculate
        the control pulses.

    pulse_dict: dict, optional
        A map between the pulse label and its index in the pulse list.
        If given, the compiled pulse can be identified with
        ``(pulse_label, coeff)``, instead of ``(pulse_index, coeff)``.
        The number of key-value pairs should match the number of pulses
        in the processor.

    Attributes
    ----------
    gate_compiler: dict
        The Python dictionary in the form of {gate_name: compiler_function}.
        It saves the compiling routine for each gate. See sub-classes
        for implementation.
    """
    def __init__(self, N, params=None, pulse_dict=None):
        self.gate_compiler = {}
        self.N = N
        self.params = params if params is not None else {}
        self.pulse_dict = pulse_dict
        self.gate_compiler = {"GLOBALPHASE": self.globalphase_compiler}
        self.args = {"params": self.params}
        self.global_phase = 0.

    def globalphase_compiler(self, gate, args):
        """
        Compiler for the GLOBALPHASE gate
        """
        pass

    def compile(self, circuit, schedule_mode=None, args=None):
        """
        Compile the the native gates into control pulse sequence.
        It calls each compiling method and concatenates
        the compiled pulses.

        Parameters
        ----------
        circuit: :class:`QubitCircuit` or list of :class:`gate`
            A list of elementary gates that can be implemented in the
            corresponding hardware.
            The gate names have to be in `gate_compiler`.

        schedule_mode: str, optional
            ``"ASAP"`` for "as soon as possible" or
            ``"ALAP"`` for "as late as possible" or
            ``False`` or ``None`` for no schedule.
            Default is None.

        args: dict, optional
            A dictionary of arguments used in a specific gate compiler
            function.

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
        if args is not None:
            self.args.update(args)
        instruction_list = []
        for gate in gates:
            if gate.name not in self.gate_compiler:
                raise ValueError("Unsupported gate %s" % gate.name)
            instruction = self.gate_compiler[gate.name](gate, self.args)
            if instruction is None:
                continue  # neglecting global phase gate
            instruction_list += instruction
        if not instruction_list:
            return None, None
        if self.pulse_dict is not None:
            num_pulses = len(self.pulse_dict)
        else:  # if pulse_dict is not given, compute the number of pulses
            num_pulses = 0
            for instruction in instruction_list:
                for pulse_index in instruction_list.pulse_info:
                    num_pulses = max(num_pulses, pulse_index)
            num_pulses += 1

        # schedule
        # scheduled_start_time:
        #   An ordered list of the start_time for each pulse,
        #   corresponding to gates in the instruction_list.
        # instruction_list reordered according to the scheduled result
        if schedule_mode:
            scheduler = Scheduler(schedule_mode)
            scheduled_start_time = scheduler.schedule(instruction_list)
            time_ordered_pos = np.argsort(scheduled_start_time)
            instruction_list = [instruction_list[i] for i in time_ordered_pos]
            scheduled_start_time.sort()
        else:  # no scheduling
            time_ordered_pos = list(range(0, len(instruction_list)))
            scheduled_start_time = [0.]
            for instruction in instruction_list:
                scheduled_start_time.append(
                    instruction.duration + scheduled_start_time[-1])
            scheduled_start_time = scheduled_start_time[:-1]

        # compile
        # An instruction can be composed from several different pulse elements.
        # We separate them an assign them to each pulse index.
        pulse_instructions = [[] for tmp in range(num_pulses)]
        for instruction, start_time in \
                zip(instruction_list, scheduled_start_time):
            for pulse_name, coeff in instruction.pulse_info:
                pulse_ind = self.pulse_dict[pulse_name]
                pulse_instructions[pulse_ind].append(
                    (start_time, instruction.tlist, coeff))

        compiled_tlist = [[[0.]] for tmp in range(num_pulses)]
        compiled_coeffs = [[] for tmp in range(num_pulses)]
        for pulse_ind in range(num_pulses):
            for start_time, tlist, coeff in pulse_instructions[pulse_ind]:
                if np.isscalar(tlist):
                    # a single constant rectanglar pulse, where
                    # tlist and coeff are just float numbers
                    step_size = tlist
                    temp_tlist = np.array([tlist])
                    coeff = np.array([coeff])
                elif len(tlist) - 1 == len(coeff):
                    # discrete pulse
                    step_size = tlist[1] - tlist[0]
                    temp_tlist = np.asarray(tlist)[1:]
                    coeff = np.asarray(coeff)
                elif len(tlist) == len(coeff):
                    # continuos pulse
                    step_size = tlist[1] - tlist[0]
                    if compiled_coeffs[pulse_ind]:  # not first pulse
                        temp_tlist = np.asarray(tlist)[1:]
                        coeff = np.asarray(coeff)[1:]
                    else:  # first pulse
                        temp_tlist = np.asarray(tlist)[1:]
                        coeff = np.asarray(coeff)
                else:
                    raise ValueError(
                        "The shape of the compiled pulse is not correct.")
                # If there is a idling time between the last pulse and
                # the current one, we need to add zeros in between.
                # We add sufficient number of zeros at the begining
                # and the end of the idling to prevent wrong cubic spline.
                if np.abs(start_time - compiled_tlist[pulse_ind][-1][-1]) \
                        > step_size * 1.0e-6:
                    idling_time = (
                            start_time - compiled_tlist[pulse_ind][-1][-1]
                        )
                    if idling_time > 3 * step_size:
                        idling_tlist1 = np.linspace(
                            compiled_tlist[pulse_ind][-1][-1] + step_size/5,
                            compiled_tlist[pulse_ind][-1][-1] + step_size,
                            5
                        )
                        idling_tlist2 = np.linspace(
                            start_time - step_size,
                            start_time - step_size/5,
                            5
                        )
                        idling_tlist = np.concatenate(
                            [idling_tlist1, idling_tlist2])
                    else:
                        idling_tlist = np.arange(
                            compiled_tlist[pulse_ind][-1][-1] + step_size,
                            start_time - step_size, step_size
                        )
                    compiled_tlist[pulse_ind].append(idling_tlist)
                    compiled_coeffs[pulse_ind].append(
                        np.zeros(len(idling_tlist))
                    )
                    # The pulse should always start with 0
                    # For rectangular pulse, len(tlist) = 0 and the above
                    # idling_tlist is empty, hence this is necenssary.
                    compiled_tlist[pulse_ind].append([start_time])
                    compiled_coeffs[pulse_ind].append([0.])
                # add the compiled pulse coeffs to the list
                compiled_tlist[pulse_ind].append(temp_tlist + start_time)
                compiled_coeffs[pulse_ind].append(coeff)

        for i in range(num_pulses):
            if not compiled_coeffs[i]:
                compiled_tlist[i] = None
                compiled_coeffs[i] = None
            else:
                compiled_tlist[i] = np.concatenate(compiled_tlist[i])
                compiled_coeffs[i] = np.concatenate(compiled_coeffs[i])

        return compiled_tlist, compiled_coeffs
