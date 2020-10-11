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
from qutip.qip.compiler.gatecompiler import GateCompiler, _PulseInstruction


__all__ = ['SpinChainCompiler']


class SpinChainCompiler(GateCompiler):
    """
    Compile a :class:`qutip.QubitCircuit` into
    the pulse sequence for the processor.

    Parameters
    ----------
    N: int
        The number of qubits in the system.

    params: dict
        A Python dictionary contains the name and the value of the parameters,
        such as laser frequency, detuning etc.

    setup: string
        "linear" or "circular" for two sub-classes.

    global_phase: bool
        Record of the global phase change and will be returned.

    pulse_dict: dict
        Dictionary of pulse indices.

    Attributes
    ----------
    N: int
        The number of the component systems.

    params: dict
        A Python dictionary contains the name and the value of the parameters,
        such as laser frequency, detuning etc.

    pulse_dict: dict
        Dictionary of pulse indices.

    gate_compiler: dict
        The Python dictionary in the form of {gate_name: decompose_function}.
        It saves the decomposition scheme for each gate.

    setup: string
        "linear" or "circular" for two sub-classes.

    global_phase: bool
        Record of the global phase change and will be returned.
    """
    def __init__(self, N, params, setup, global_phase, pulse_dict):
        super(SpinChainCompiler, self).__init__(
            N=N, params=params, pulse_dict=pulse_dict)
        self.gate_compiler = {"ISWAP": self.iswap_compiler,
                             "SQRTISWAP": self.sqrtiswap_compiler,
                             "RZ": self.rz_compiler,
                             "RX": self.rx_compiler,
                             "GLOBALPHASE": self.globalphase_compiler
                             }
        self.N = N
        self.global_phase = global_phase

    def compile(self, gates, schedule_mode=None):
        tlist, coeffs = super(SpinChainCompiler, self).compile(gates, schedule_mode=schedule_mode)
        return tlist, coeffs, self.global_phase

    def rz_compiler(self, gate):
        """
        Compiler for the RZ gate
        """
        targets = gate.targets
        g = self.params["sz"][targets[0]]
        coeff = np.array([np.sign(gate.arg_value) * g])
        tlist = np.array([abs(gate.arg_value) / (2 * g)])
        pulse_coeffs = [("sz" + str(targets[0]), coeff)]
        return [_PulseInstruction(gate, tlist, pulse_coeffs)]

    def rx_compiler(self, gate):
        """
        Compiler for the RX gate
        """
        targets = gate.targets
        g = self.params["sx"][targets[0]]
        coeff = np.array([np.sign(gate.arg_value) * g])
        tlist = np.array([abs(gate.arg_value) / (2 * g)])
        pulse_coeffs = [("sx" + str(targets[0]), coeff)]
        return [_PulseInstruction(gate, tlist, pulse_coeffs)]

    def iswap_compiler(self, gate):
        """
        Compiler for the ISWAP gate
        """
        targets = gate.targets
        q1, q2 = min(targets), max(targets)
        g = self.params["sxsy"][q1]
        coeff = np.array([-g])
        tlist = np.array([np.pi / (4 * g)])
        if self.N != 2 and q1 == 0 and q2 == self.N - 1:
            pulse_name = "g" + str(q2)
        else:
            pulse_name = "g" + str(q1)
        pulse_coeffs = [(pulse_name, coeff)]
        return [_PulseInstruction(gate, tlist, pulse_coeffs)]

    def sqrtiswap_compiler(self, gate):
        """
        Compiler for the SQRTISWAP gate
        """
        targets = gate.targets
        q1, q2 = min(targets), max(targets)
        g = self.params["sxsy"][q1]
        coeff = np.array([-g])
        tlist = np.array([np.pi / (8 * g)])
        if self.N != 2 and q1 == 0 and q2 == self.N - 1:
            pulse_name = "g" + str(q2)
        else:
            pulse_name = "g" + str(q1)
        pulse_coeffs = [(pulse_name, coeff)]
        return [_PulseInstruction(gate, tlist, pulse_coeffs)]

    def globalphase_compiler(self, gate):
        """
        Compiler for the GLOBALPHASE gate
        """
        self.global_phase += gate.arg_value
