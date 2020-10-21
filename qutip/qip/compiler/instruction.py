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


__all__ = ['Instruction']


class Instruction():
    """
    The instruction that implements a quantum gate.

    Parameters
    ----------
    name: str
        Name of the gate.
    targets: list, optional
        The target qubits.
    controls: list, optional
        The control qubits.
    duration: list, optional
        The execution time needed for the instruction.

    Attributes
    ----------
    used_qubits: set
        Union of the control and target qubits.
    """
    def __init__(
            self, gate, tlist=None,
            pulse_info=(), duration=1):
        self.gate = gate
        if self.targets is not None:
            self.targets.sort()
        if self.controls is None:
            self.used_qubits = set(self.targets)
        else:
            self.controls.sort()
            self.used_qubits = set(self.controls).union(set(self.targets))
        self.tlist = tlist
        if self.tlist is not None:
            if np.isscalar(self.tlist):
                self.duration = self.tlist
            elif abs(self.tlist[0] - 0.) > 1.e-8:
                raise ValueError("Pulse time sequence must start from 0")
            else:
                self.duration = self.tlist[-1]
        else:
            self.duration = duration
        self.pulse_info = pulse_info

    @property
    def name(self):
        return self.gate.name

    @property
    def targets(self):
        return self.gate.targets

    @property
    def controls(self):
        return self.gate.controls

    @property
    def step_num(self):
        if np.isscalar(self.tlist):
            return 1
        else:
            return len(self.tlist) - 1
