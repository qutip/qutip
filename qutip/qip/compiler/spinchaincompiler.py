import numpy as np

from qutip.qip.circuit import QubitCircuit, Gate
from qutip.qip.compiler import GateCompiler, Instruction


__all__ = ['SpinChainCompiler']


class SpinChainCompiler(GateCompiler):
    """
    Compile a :class:`.QubitCircuit` into
    the pulse sequence for the processor.

    Parameters
    ----------
    N: int
        The number of qubits in the system.

    params: dict
        A Python dictionary contains the name and the value of the parameters.
        See :meth:`.SpinChain.set_up_params` for the definition.

    setup: string
        "linear" or "circular" for two sub-classes.

    global_phase: bool
        Record of the global phase change and will be returned.

    pulse_dict: dict, optional
        A map between the pulse label and its index in the pulse list.
        If given, the compiled pulse can be identified with
        ``(pulse_label, coeff)``, instead of ``(pulse_index, coeff)``.
        The number of key-value pairs should match the number of pulses
        in the processor.
        If it is empty, an integer ``pulse_index`` needs to be used
        in the compiling routine saved under the attributes ``gate_compiler``.

    Attributes
    ----------
    N: int
        The number of the component systems.

    params: dict
        A Python dictionary contains the name and the value of the parameters,
        such as laser frequency, detuning etc.

    pulse_dict: dict
        A map between the pulse label and its index in the pulse list.

    gate_compiler: dict
        The Python dictionary in the form of {gate_name: decompose_function}.
        It saves the decomposition scheme for each gate.

    setup: string
        "linear" or "circular" for two sub-classes.

    global_phase: bool
        Record of the global phase change and will be returned.
    """
    def __init__(self, N, params, pulse_dict, setup="linear", global_phase=0.):
        super(SpinChainCompiler, self).__init__(
            N=N, params=params, pulse_dict=pulse_dict)
        self.gate_compiler.update({
            "ISWAP": self.iswap_compiler,
            "SQRTISWAP": self.sqrtiswap_compiler,
            "RZ": self.rz_compiler,
            "RX": self.rx_compiler,
            "GLOBALPHASE": self.globalphase_compiler
            })
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

    def iswap_compiler(self, gate, args):
        """
        Compiler for the ISWAP gate
        """
        targets = gate.targets
        q1, q2 = min(targets), max(targets)
        g = self.params["sxsy"][q1]
        coeff = -g
        tlist = np.pi / (4 * g)
        if self.N != 2 and q1 == 0 and q2 == self.N - 1:
            pulse_name = "g" + str(q2)
        else:
            pulse_name = "g" + str(q1)
        pulse_info = [(pulse_name, coeff)]
        return [Instruction(gate, tlist, pulse_info)]

    def sqrtiswap_compiler(self, gate, args):
        """
        Compiler for the SQRTISWAP gate
        """
        targets = gate.targets
        q1, q2 = min(targets), max(targets)
        g = self.params["sxsy"][q1]
        coeff = -g
        tlist = np.pi / (8 * g)
        if self.N != 2 and q1 == 0 and q2 == self.N - 1:
            pulse_name = "g" + str(q2)
        else:
            pulse_name = "g" + str(q1)
        pulse_info = [(pulse_name, coeff)]
        return [Instruction(gate, tlist, pulse_info)]

    def globalphase_compiler(self, gate, args):
        """
        Compiler for the GLOBALPHASE gate
        """
        self.global_phase += gate.arg_value
