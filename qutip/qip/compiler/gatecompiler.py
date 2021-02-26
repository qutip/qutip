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
    Base class. It compiles a :class:`.QubitCircuit` into
    the pulse sequence for the processor. The core member function
    `compile` calls compiling method from the sub-class and concatenate
    the compiled pulses.

    Parameters
    ----------
    N: int
        The number of the component systems.

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

    args: dict
        Arguments for individual compiling routines.
        It adds more flexibility in customizing compiler.
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
        circuit: :class:`.QubitCircuit` or list of
            :class:`.Gate`
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
            num_controls = len(self.pulse_dict)
        else:  # if pulse_dict is not given, compute the number of pulses
            num_controls = 0
            for instruction in instruction_list:
                for pulse_index, _ in instruction.pulse_info:
                    num_controls = max(num_controls, pulse_index)
            num_controls += 1

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
        pulse_instructions = [[] for tmp in range(num_controls)]
        for instruction, start_time in \
                zip(instruction_list, scheduled_start_time):
            for pulse_name, coeff in instruction.pulse_info:
                if self.pulse_dict is not None:
                    pulse_ind = self.pulse_dict[pulse_name]
                else:
                    pulse_ind = pulse_name
                pulse_instructions[pulse_ind].append(
                    (start_time, instruction.tlist, coeff))

        # Concatenate tlist and coeffs
        compiled_tlist = [[] for tmp in range(num_controls)]
        compiled_coeffs = [[] for tmp in range(num_controls)]
        for pulse_ind in range(num_controls):
            last_pulse_time = 0.
            for start_time, tlist, coeff in pulse_instructions[pulse_ind]:
                tlist_to_add, coeffs_to_add, last_pulse_time = \
                    self._concatenate_instruction(
                        start_time, tlist, coeff, last_pulse_time)
                compiled_tlist[pulse_ind].extend(tlist_to_add)
                compiled_coeffs[pulse_ind].extend(coeffs_to_add)

        for i in range(num_controls):
            if not compiled_coeffs[i]:
                compiled_tlist[i] = None
                compiled_coeffs[i] = None
            else:
                compiled_tlist[i] = np.concatenate(compiled_tlist[i])
                compiled_coeffs[i] = np.concatenate(compiled_coeffs[i])

        return compiled_tlist, compiled_coeffs


    def _concatenate_instruction(
            self, start_time, tlist, coeff, last_pulse_time):
        # compute the gate time, step size and coeffs
        if np.isscalar(tlist):
            pulse_mode = "discrete"
            # a single constant rectanglar pulse, where
            # tlist and coeff are just float numbers
            step_size = tlist
            coeff = np.array([coeff])
            if abs(start_time) < 1.0e-10:
                gate_tlist = np.array([0., tlist])  # first gate
            else:
                gate_tlist = np.array([tlist])
        elif len(tlist) - 1 == len(coeff):
            # discrete pulse
            pulse_mode = "discrete"
            step_size = tlist[1] - tlist[0]
            coeff = np.asarray(coeff)
            if abs(start_time) < step_size * 1.0e-6:
                gate_tlist = np.asarray(tlist)
            else:
                gate_tlist = np.asarray(tlist)[1:]
        elif len(tlist) == len(coeff):
            # continuos pulse
            pulse_mode = "continuous"
            step_size = tlist[1] - tlist[0]
            if abs(start_time) < step_size * 1.0e-6:
                coeff = np.asarray(coeff)
                gate_tlist = np.asarray(tlist)
            else:
                coeff = np.asarray(coeff)[1:]
                gate_tlist = np.asarray(tlist)[1:]
        else:
            raise ValueError(
                "The shape of the compiled pulse is not correct.")

        # consider the idling time and add to the resulting list
        tlist_to_add = []
        coeffs_to_add = []
        # If there is a idling time between the last pulse and
        # the current one, we need to add zeros in between.
        if np.abs(start_time - last_pulse_time) > step_size * 1.0e-6:
            idling_time = start_time - last_pulse_time
            if pulse_mode == "continuous":
                # We add sufficient number of zeros at the begining
                # and the end of the idling to prevent wrong cubic spline.
                if abs(last_pulse_time) < 1.e-14:
                    tlist_to_add.append(np.array([0.]))
                    coeffs_to_add.append(np.array([0.]))
                if idling_time > 3 * step_size:
                    idling_tlist1 = np.linspace(
                        last_pulse_time + step_size/5,
                        last_pulse_time + step_size,
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
                        last_pulse_time + step_size,
                        start_time - step_size, step_size
                    )
                tlist_to_add.append(idling_tlist)
                coeffs_to_add.append(np.zeros(len(idling_tlist)))
            elif pulse_mode == "discrete":
                if abs(last_pulse_time) < 1.e-14:
                    tlist_to_add.append([0.])
            # The pulse should always start with 0.
            # For discrete pulse, it is assumed implicitly;
            # for continuous pulse, the difference is negligible.
            tlist_to_add.append([start_time])
            coeffs_to_add.append([0.])
        # Add the gate time and coeffs to the list.
        execution_time = gate_tlist + start_time
        last_pulse_time = execution_time[-1]
        tlist_to_add.append(execution_time)
        coeffs_to_add.append(coeff)

        return tlist_to_add, coeffs_to_add, last_pulse_time