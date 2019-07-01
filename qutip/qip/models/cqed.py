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
import warnings
from qutip.operators import tensor, identity, destroy, sigmax, sigmaz
from qutip.states import basis
from qutip.qip.circuit import QubitCircuit, Gate
from qutip.qip.models.circuitprocessor import CircuitProcessor, ModelProcessor


class DispersivecQED(ModelProcessor):
    """
    The circuitprocessor based on the physical implementation of
    a dispersive cavity QED system.
    The available Hamiltonian of the system is predefined.
    For a given pulse amplitude matrix, the processor can
    calculate the state evolution under the given control pulse,
    either analytically or numerically.

    Parameters
    ----------
    N : int
        The number of qubits in the system.
    correct_global_phase : bool
        Whether the correct phase should be considered in analytical
        evolution.
    Nres: Integer
        The number of energy levels in the resonator.
    deltamax: Integer/List
        The sigma-x paraicient for each of the qubits in the system.
    epsmax: Integer/List
        The sigma-z paraicient for each of the qubits in the system.
    wo: Integer
        The base frequency of the resonator.
    eps: Integer/List
        The epsilon for each of the qubits in the system.
    delta: Integer/List
        The epsilon for each of the qubits in the system.
    g: Integer/List
        The interaction strength for each of the qubit with the resonator.
    T1 : list or float
        Characterize the decoherence of amplitude damping for
        each qubit.
    T2 : list of float
        Characterize the decoherence of dephasing relaxation for
        each qubit.

    Attributes
    ----------
    hams : list of :class:`Qobj`
        A list of Hamiltonians of the control pulse driving the evolution.
    tlist : array like
        A NumPy array specifies at which time the next amplitude of
        a pulse is to be applied.
    amps : array like
        The pulse matrix, a 2d NumPy array of the shape
        (len(ctrls), len(tlist)).
        Each row corresponds to the control pulse sequence for
        one Hamiltonian.
    paras : dict
        A Python dictionary contains the name and the value of the parameters
        of the physical realization, such as laser freqeuncy, detuning etc.
    sx_ops :
        A list of sigmax Hamiltonians for each qubit.
    sz_ops :
        A list of sigmaz Hamiltonians for each qubit.
    cavityqubit_ops :
        A list of interacting Hamiltonians between cavity and each qubit.
    sx_u : array like
        Pulse matrix for sigmax Hamiltonians.
    sz_u : array like
        Pulse matrix for sigmaz Hamiltonians.
    g_u : array like
        Pulse matrix for interacting Hamiltonians
        between cavity and each qubit.
    wq : list of float
        The frequency of the qubits calculated from
        eps and delta for each qubit.
    Delta : list of float
        The detuning with repect to w0 calculated
        from wq and w0 for each qubit.
    """

    def __init__(self, N, correct_global_phase=True, Nres=10, deltamax=1.0,
                 epsmax=9.5, w0=10., wq=None, eps=9.5,
                 delta=0.0, g=0.01, T1=None, T2=None):
        super(DispersivecQED, self).__init__(
            N, correct_global_phase=correct_global_phase, T1=T1, T2=T2)
        self.correct_global_phase = correct_global_phase
        self.Nres = Nres
        self._hams = []
        self._paras = {}
        self.set_up_paras(
            N=N, Nres=Nres, deltamax=deltamax,
            epsmax=epsmax, w0=w0, wq=wq, eps=eps,
            delta=delta, g=g)
        self.set_up_ops(N)

    def set_up_ops(self, N):
        """
        Genrate the Hamiltonians for the spinchain model and save them in the
        attribute `hams`.

        Parameters
        ----------
        N : int
            The number of qubits in the system.
        """
        # single qubit terms
        self.a = tensor([destroy(self.Nres)] + [identity(2) for n in range(N)])
        self._hams.append(self.a.dag() * self.a)
        self._hams += [tensor([identity(self.Nres)] +
                              [sigmax() if m == n else identity(2)
                               for n in range(N)])
                       for m in range(N)]
        self._hams += [tensor([identity(self.Nres)] +
                              [sigmaz() if m == n else identity(2)
                               for n in range(N)])
                       for m in range(N)]
        # interaction terms
        for n in range(N):
            sm = tensor([identity(self.Nres)] +
                        [destroy(2) if m == n else identity(2)
                         for m in range(N)])
            self._hams.append(self.a.dag() * sm + self.a * sm.dag())

        self.psi_proj = tensor([basis(self.Nres, 0)] +
                               [identity(2) for n in range(N)])

    def set_up_paras(
            self, N, Nres, deltamax,
            epsmax, w0, wq, eps, delta, g):
        """
        Save the parameters in the attribute `paras` and check the validity.

        Parameters
        ----------
        N : int
            The number of qubits in the system.
        Nres: Integer
            The number of energy levels in the resonator.
        deltamax: Integer/List
            The sigma-x paraicient for each of the qubits in the system.
        epsmax: Integer/List
            The sigma-z paraicient for each of the qubits in the system.
        wo: Integer
            The base frequency of the resonator.
        wq: Integer/List
            The frequency of the qubits.
        eps: Integer/List
            The epsilon for each of the qubits in the system.
        delta: Integer/List
            The delta for each of the qubits in the system.
        g: Integer/List
            The interaction strength for each of the qubit with the resonator.

        Note
        ----
        All parameters will be multiplied by 2*pi for simplicity
        """
        sx_para = super(DispersivecQED, self)._para_list(deltamax, N)
        self._paras["sx"] = sx_para
        sz_para = super(DispersivecQED, self)._para_list(epsmax, N)
        self._paras["sz"] = sz_para
        w0 = w0 * 2 * np.pi
        self._paras["w0"] = w0
        eps = super(DispersivecQED, self)._para_list(eps, N)
        self._paras["eps"] = eps
        delta = super(DispersivecQED, self)._para_list(delta, N)
        self._paras["delta"] = delta
        g = super(DispersivecQED, self)._para_list(g, N)
        self._paras["g"] = g

        # computed
        self.wq = [np.sqrt(eps[i] ** 2 + delta[i] ** 2) for i in range(N)]
        self.Delta = [self.wq[i] - w0 for i in range(N)]

        # rwa/dispersive regime tests
        if any([g[i] / (w0 - self.wq[i]) > 0.05 for i in range(N)]):
            warnings.warn("Not in the dispersive regime")

        if any([(w0-self.wq[i]) / (w0+self.wq[i]) > 0.05 for i in range(N)]):
            warnings.warn(
                "The rotating-wave approximation might not be valid.")

    @property
    def sx_ops(self):
        return self._hams[1: self.N+1]

    @property
    def sz_ops(self):
        return self._hams[self.N+1: 2*self.N+1]

    @property
    def cavityqubit_ops(self):
        return self._hams[2*self.N+1: 3*self.N+1]

    @property
    def sx_u(self):
        return self.amps[1: self.N+1]

    @property
    def sz_u(self):
        return self.amps[self.N+1: 2*self.N+1]

    @property
    def g_u(self):
        return self.amps[2*self.N+1: 3*self.N+1]

    def get_ops_labels(self):
        """
        Returns the Hamiltonian operators and corresponding values by stacking
        them together.
        """
        return ([r"$a^\dagger a$"] +
                [r"$\sigma_x^%d$" % n for n in range(self.N)] +
                [r"$\sigma_z^%d$" % n for n in range(self.N)] +
                [r"$g_{%d}$" % (n) for n in range(self.N)])

    def optimize_circuit(self, qc):
        """
        Function to take a quantum circuit/algorithm and convert it into the
        optimal form/basis for the desired physical system.

        Parameters
        ----------
        qc: :class:`qutip.QubitCircuit`
            Takes the quantum circuit to be implemented.

        Returns
        --------
        qc: :class:`qutip.QubitCircuit`
            The circuit representation with elementary gates
            that can be implemented in this model.
        """
        self.qc0 = qc
        self.qc1 = self.qc0.resolve_gates(basis=["ISWAP", "RX", "RZ"])
        return self.qc1

    def eliminate_auxillary_modes(self, U):
        """
        Eliminate the auxillary modes like the cavity modes in cqed.
        """
        return self.psi_proj.dag() * U * self.psi_proj

    def load_circuit(self, qc):
        """
        Decompose a :class:`qutip.QubitCircuit` in to the control
        amplitude generating the corresponding evolution.

        Parameters
        ----------
        qc: :class:`qutip.QubitCircuit`
            Takes the quantum circuit to be implemented.

        Returns
        -------
        tlist : array like
            A NumPy array specifies at which time the next amplitude of
            a pulse is to be applied.
        amps : array like
            A 2d NumPy array of the shape (len(ctrls), len(tlist)). Each
            row corresponds to the control pulse sequence for
            one Hamiltonian.
        """
        gates = self.optimize_circuit(qc).gates

        dec = CQEDGateDecomposer(
            self.N, self._paras, self.wq, self.Delta,
            global_phase=0., num_ops=len(self._hams))
        self.tlist, self.amps, self.global_phase = dec.decompose(gates)

        # TODO The amplitude of the first control a.dag()*a
        # was set to zero before I made this refactoring.
        # It is probably due to the fact that
        # it contributes only a constant (N) and can be neglected.
        # but change the below line to np.ones leads to test error.
        self.amps[0] = self._paras["w0"] * np.zeros((self.sx_u.shape[1]))
        return self.tlist, self.amps


class CQEDGateDecomposer(object):
    """
    The obejct that decompose a :class:`qutip.QubitCircuit` into
    the pulse sequence for the processor.

    Parameters
    ----------
    N : int
        The number of qubits in the system.
    paras : dict
        A Python dictionary contains the name and the value of the parameters
        of the physical realization, such as laser freqeuncy,detuning etc.
    wq : list of float
        The frequency of the qubits calculated from
        eps and delta for each qubit.
    Delta : list of float
        The detuning with repect to w0 calculated
        from wq and w0 for each qubit.
    global_phase : bool
        Record of the global phase change and will be returned.
    num_ops : int
        Number of Hamiltonians in the processor.

    Attributes
    ----------
    gate_decs : dict
        The Python dictionary in the form {gate_name: decompose_function}.
    sx_ind: Integer/List
        The list of indices in the Hamiltonian list of sigmax.
    sz_ind: Integer/List
        The list of indices in the Hamiltonian list of sigmaz.
    g_ind: Integer/List
        The list of indices in the Hamiltonian list of tensor(sigmax, sigmay).
    """
    def __init__(self, N, paras, wq, Delta, global_phase, num_ops):
        self.gate_decs = {"ISWAP": self.iswap_dec,
                          "SQRTISWAP": self.sqrtiswap_dec,
                          "RZ": self.rz_dec,
                          "RX": self.rx_dec,
                          "GLOBALPHASE": self.globalphase_dec
                          }
        self.sx_ind = list(range(1, N+1))
        self.sz_ind = list(range(N+1, 2*N+1))
        self.g_ind = list(range(2*N+1, 3*N+1))
        self.num_ops = num_ops
        self.paras = paras
        self.wq = wq
        self.Delta = Delta
        self.global_phase = global_phase

    def decompose(self, gates):
        """
        Decompose the the elementary gates
        into control pulse sequence.

        Parameters
        ----------
        gates : list
            A list of elementary gates that can be implemented in this
            model. The gate names have to be in `gate_decs`.

        Returns
        -------
        tlist : array like
            A NumPy array specifies at which time the next amplitude of
            a pulse is to be applied.
        amps : array like
            A 2d NumPy array of the shape (len(ctrls), len(tlist)). Each
            row corresponds to the control pulse sequence for
            one Hamiltonian.
        global_phase : bool
            Recorded change of global phase.
        """
        self.dt_list = []
        self.amps_list = []
        for gate in gates:
            if gate.name not in self.gate_decs:
                raise ValueError("Unsupported gate %s" % gate.name)
            self.gate_decs[gate.name](gate)
            amps = np.vstack(self.amps_list).T

        tlist = np.zeros(len(self.dt_list))
        t = 0
        for i in range(len(self.dt_list)):
            t += self.dt_list[i]
            tlist[i] = t
        return tlist, amps, self.global_phase

    def rz_dec(self, gate):
        """
        Decomposer for the RZ gate
        """
        pulse = np.zeros(self.num_ops)
        q_ind = gate.targets[0]
        g = self.paras["sz"][q_ind]
        pulse[self.sz_ind[q_ind]] = np.sign(gate.arg_value) * g
        t = abs(gate.arg_value) / (2 * g)
        self.dt_list.append(t)
        self.amps_list.append(pulse)

    def rx_dec(self, gate):
        """
        Decomposer for the RX gate
        """
        pulse = np.zeros(self.num_ops)
        q_ind = gate.targets[0]
        g = self.paras["sx"][q_ind]
        pulse[self.sx_ind[q_ind]] = np.sign(gate.arg_value) * g
        t = abs(gate.arg_value) / (2 * g)
        self.dt_list.append(t)
        self.amps_list.append(pulse)

    def sqrtiswap_dec(self, gate):
        """
        Decomposer for the SQRTISWAP gate

        Note
        ----
        This version of sqrtiswap_dec has very low fidelity, please use
        iswap
        """
        # FIXME This decomposition has poor behaviour
        pulse = np.zeros(self.num_ops)
        q1, q2 = gate.targets
        pulse[self.sz_ind[q1]] = self.wq[q1] - self.paras["w0"]
        pulse[self.sz_ind[q2]] = self.wq[q2] - self.paras["w0"]
        pulse[self.g_ind[q1]] = self.paras["g"][q1]
        pulse[self.g_ind[q2]] = self.paras["g"][q2]
        J = self.paras["g"][q1] * self.paras["g"][q2] * (
            1 / self.Delta[q1] + 1 / self.Delta[q2]) / 2
        t = (4 * np.pi / abs(J)) / 8
        self.dt_list.append(t)
        self.amps_list.append(pulse)

        # corrections
        gate1 = Gate("RZ", [q1], None, arg_value=-np.pi/4)
        self.rz_dec(gate1)
        gate2 = Gate("RZ", [q2], None, arg_value=-np.pi/4)
        self.rz_dec(gate2)
        gate3 = Gate("GLOBALPHASE", None, None, arg_value=-np.pi/4)
        self.globalphase_dec(gate3)

    def iswap_dec(self, gate):
        """
        Decomposer for the ISWAP gate
        """
        pulse = np.zeros(self.num_ops)
        q1, q2 = gate.targets
        pulse[self.sz_ind[q1]] = self.wq[q1] - self.paras["w0"]
        pulse[self.sz_ind[q2]] = self.wq[q2] - self.paras["w0"]
        pulse[self.g_ind[q1]] = self.paras["g"][q1]
        pulse[self.g_ind[q2]] = self.paras["g"][q2]
        J = self.paras["g"][q1] * self.paras["g"][q2] * (
            1 / self.Delta[q1] + 1 / self.Delta[q2]) / 2
        t = (4 * np.pi / abs(J)) / 4
        self.dt_list.append(t)
        self.amps_list.append(pulse)

        # corrections
        gate1 = Gate("RZ", [q1], None, arg_value=-np.pi/2.)
        self.rz_dec(gate1)
        gate2 = Gate("RZ", [q2], None, arg_value=-np.pi/2)
        self.rz_dec(gate2)
        gate3 = Gate("GLOBALPHASE", None, None, arg_value=-np.pi/2)
        self.globalphase_dec(gate3)

    def globalphase_dec(self, gate):
        """
        Decomposer for the GLOBALPHASE gate
        """
        self.global_phase += gate.arg_value
