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

from qutip.qip.circuit import QubitCircuit, Gate
from qutip.qip.compiler.gatecompiler import GateCompiler, Instruction


__all__ = ['CavityQEDCompiler']


class CavityQEDCompiler(GateCompiler):
    """
    Decompose a :class:`qutip.QubitCircuit` into
    the pulse sequence for the processor.

    Parameters
    ----------
    N: int
        The number of qubits in the system.

    params: dict
        A Python dictionary contains the name and the value of the parameters,
        such as laser frequency, detuning etc.

    wq: list of float
        The frequency of the qubits calculated from
        eps and delta for each qubit.

    Delta: list of float
        The detuning with respect to w0 calculated
        from wq and w0 for each qubit.

    global_phase: bool
        Record of the global phase change and will be returned.

    pulse_dict: int
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
    def __init__(self, N, params, global_phase, pulse_dict):
        super(CavityQEDCompiler, self).__init__(
            N=N, params=params, pulse_dict=pulse_dict)
        self.gate_compiler.update({"ISWAP": self.iswap_compiler,
                             "SQRTISWAP": self.sqrtiswap_compiler,
                             "RZ": self.rz_compiler,
                             "RX": self.rx_compiler,
                             "GLOBALPHASE": self.globalphase_compiler
                             })
        self.wq = np.sqrt(self.params["eps"]**2 + self.params["delta"]**2)
        self.Delta = self.wq - self.params["w0"]
        self.global_phase = global_phase

    def rz_compiler(self, gate, args):
        """
        Compiler for the RZ gate
        """
        targets = gate.targets
        g = self.params["sz"][targets[0]]
        coeff = np.sign(gate.arg_value) * g
        tlist = abs(gate.arg_value) / (2 * g)
        pulse_info = [("sz" + str(targets[0]), coeff)]
        return [Instruction(gate, tlist, pulse_info)]

    def rx_compiler(self, gate, args):
        """
        Compiler for the RX gate
        """
        targets = gate.targets
        g = self.params["sx"][targets[0]]
        coeff = np.sign(gate.arg_value) * g
        tlist = abs(gate.arg_value) / (2 * g)
        pulse_info = [("sx" + str(targets[0]), coeff)]
        return [Instruction(gate, tlist, pulse_info)]

    def sqrtiswap_compiler(self, gate, args):
        """
        Compiler for the SQRTISWAP gate

        Notes
        -----
        This version of sqrtiswap_compiler has very low fidelity, please use
        iswap
        """
        # FIXME This decomposition has poor behaviour
        q1, q2 = gate.targets
        pulse_info = []
        pulse_name = "sz" + str(q1)
        coeff = self.wq[q1] - self.params["w0"]
        pulse_info += [(pulse_name, coeff)]
        pulse_name = "sz" + str(q1)
        coeff = self.wq[q2] - self.params["w0"]
        pulse_info += [(pulse_name, coeff)]
        pulse_name = "g" + str(q1)
        coeff = self.params["g"][q1]
        pulse_info += [(pulse_name, coeff)]
        pulse_name = "g" + str(q2)
        coeff = self.params["g"][q2]
        pulse_info += [(pulse_name, coeff)]

        J = self.params["g"][q1] * self.params["g"][q2] * (
            1 / self.Delta[q1] + 1 / self.Delta[q2]) / 2
        tlist = 0., (4 * np.pi / abs(J)) / 8
        instruction_list = [Instruction(gate, tlist, pulse_info)]

        # corrections
        gate1 = Gate("RZ", [q1], None, arg_value=-np.pi/4) 
        compiled_gate1 = self.rz_compiler(gate1, args)
        instruction_list += compiled_gate1
        gate2 = Gate("RZ", [q2], None, arg_value=-np.pi/4)
        compiled_gate2 = self.rz_compiler(gate2, args)
        instruction_list += compiled_gate2
        gate3 = Gate("GLOBALPHASE", None, None, arg_value=-np.pi/4)
        self.globalphase_compiler(gate3, args)
        return instruction_list

    def iswap_compiler(self, gate, args):
        """
        Compiler for the ISWAP gate
        """
        q1, q2 = gate.targets
        pulse_info = []
        pulse_name = "sz" + str(q1)
        coeff = self.wq[q1] - self.params["w0"]
        pulse_info += [(pulse_name, coeff)]
        pulse_name = "sz" + str(q2)
        coeff = self.wq[q2] - self.params["w0"]
        pulse_info += [(pulse_name, coeff)]
        pulse_name = "g" + str(q1)
        coeff = self.params["g"][q1]
        pulse_info += [(pulse_name, coeff)]
        pulse_name = "g" + str(q2)
        coeff = self.params["g"][q2]
        pulse_info += [(pulse_name, coeff)]

        J = self.params["g"][q1] * self.params["g"][q2] * (
            1 / self.Delta[q1] + 1 / self.Delta[q2]) / 2
        tlist = (4 * np.pi / abs(J)) / 4
        instruction_list = [Instruction(gate, tlist, pulse_info)]

        # corrections
        gate1 = Gate("RZ", [q1], None, arg_value=-np.pi/2.)
        compiled_gate1 = self.rz_compiler(gate1, args)
        instruction_list += compiled_gate1
        gate2 = Gate("RZ", [q2], None, arg_value=-np.pi/2)
        compiled_gate2 = self.rz_compiler(gate2, args)
        instruction_list += compiled_gate2
        gate3 = Gate("GLOBALPHASE", None, None, arg_value=-np.pi/2)
        self.globalphase_compiler(gate3, args)
        return instruction_list

    def globalphase_compiler(self, gate, args):
        """
        Compiler for the GLOBALPHASE gate
        """
        self.global_phase += gate.arg_value
