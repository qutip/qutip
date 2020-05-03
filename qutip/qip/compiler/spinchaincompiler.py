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
from qutip.qip.compiler.gatecompiler import GateCompiler


__all__ = ['SpinChainCompiler']


class SpinChainCompiler(GateCompiler):
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

    setup: string
        "linear" or "circular" for two sub-calsses.

    global_phase: bool
        Record of the global phase change and will be returned.

    num_ops: int
        Number of Hamiltonians in the processor.

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

    setup: string
        "linear" or "circular" for two sub-calsses.

    global_phase: bool
        Record of the global phase change and will be returned.
    """
    def __init__(self, N, params, setup, global_phase, num_ops):
        super(SpinChainCompiler, self).__init__(
            N=N, params=params, num_ops=num_ops)
        self.gate_decomps = {"ISWAP": self.iswap_dec,
                             "SQRTISWAP": self.sqrtiswap_dec,
                             "RZ": self.rz_dec,
                             "RX": self.rx_dec,
                             "GLOBALPHASE": self.globalphase_dec
                             }
        self.N = N
        self._sx_ind = list(range(0, N))
        self._sz_ind = list(range(N, 2*N))
        if setup == "circular":
            self._sxsy_ind = list(range(2*N, 3*N))
        elif setup == "linear":
            self._sxsy_ind = list(range(2*N, 3*N-1))
        self.global_phase = global_phase

    def decompose(self, gates):
        tlist, coeffs = super(SpinChainCompiler, self).decompose(gates)
        return tlist, coeffs, self.global_phase

    def rz_dec(self, gate):
        """
        Compiler for the RZ gate
        """
        pulse = np.zeros(self.num_ops)
        q_ind = gate.targets[0]
        g = self.params["sz"][q_ind]
        pulse[self._sz_ind[q_ind]] = np.sign(gate.arg_value) * g
        t = abs(gate.arg_value) / (2 * g)
        self.dt_list.append(t)
        self.coeff_list.append(pulse)

    def rx_dec(self, gate):
        """
        Compiler for the RX gate
        """
        pulse = np.zeros(self.num_ops)
        q_ind = gate.targets[0]
        g = self.params["sx"][q_ind]
        pulse[self._sx_ind[q_ind]] = np.sign(gate.arg_value) * g
        t = abs(gate.arg_value) / (2 * g)
        self.dt_list.append(t)
        self.coeff_list.append(pulse)

    def iswap_dec(self, gate):
        """
        Compiler for the ISWAP gate
        """
        pulse = np.zeros(self.num_ops)
        q1, q2 = min(gate.targets), max(gate.targets)
        g = self.params["sxsy"][q1]
        if self.N != 2 and q1 == 0 and q2 == self.N - 1:
            pulse[self._sxsy_ind[self.N - 1]] = -g
        else:
            pulse[self._sxsy_ind[q1]] = -g
        t = np.pi / (4 * g)
        self.dt_list.append(t)
        self.coeff_list.append(pulse)

    def sqrtiswap_dec(self, gate):
        """
        Compiler for the SQRTISWAP gate
        """
        pulse = np.zeros(self.num_ops)
        q1, q2 = min(gate.targets), max(gate.targets)
        g = self.params["sxsy"][q1]
        if self.N != 2 and q1 == 0 and q2 == self.N - 1:
            pulse[self._sxsy_ind[self.N - 1]] = -g
        else:
            pulse[self._sxsy_ind[q1]] = -g
        t = np.pi / (8 * g)
        self.dt_list.append(t)
        self.coeff_list.append(pulse)

    def globalphase_dec(self, gate):
        """
        Compiler for the GLOBALPHASE gate
        """
        self.global_phase += gate.arg_value
