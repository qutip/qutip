from copy import deepcopy
import numpy as np


__all__ = ['Instruction']


class Instruction():
    """
    The instruction that implements a quantum gate.
    It contains the control pulse required to implement the gate
    on a particular hardware model.

    Parameters
    ----------
    gate: :class:`.Gate`
        The quantum gate.
    duration: list, optional
        The execution time needed for the instruction.
    tlist: array_like, optional
        A list of time at which the time-dependent coefficients are
        applied. See :class:`.Pulse` for detailed information`
    pulse_info: list, optional
        A list of tuples, each tuple corresponding to a pair of pulse label
        and pulse coefficient, in the format (str, array_like).
        This pulses will implement the desired gate.

    Attributes
    ----------
    targets: list, optional
        The target qubits.
    controls: list, optional
        The control qubits.
    used_qubits: set
        Union of the control and target qubits.
    """
    def __init__(
            self, gate, tlist=None,
            pulse_info=(), duration=1):
        self.gate = deepcopy(gate)
        self.used_qubits = set()
        if self.targets is not None:
            self.targets.sort()  # Used when comparing the instructions
            self.used_qubits |= set(self.targets)
        if self.controls is not None:
            self.controls.sort()
            self.used_qubits |= set(self.controls)
        self.tlist = tlist
        if self.tlist is not None:
            if np.isscalar(self.tlist):
                self.duration = self.tlist
            elif abs(self.tlist[0]) > 1.e-8:
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
