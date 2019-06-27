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
from qutip import tensor, identity, destroy, sigmax, sigmaz, basis
from qutip.qip.circuit import QubitCircuit, Gate
from qutip.qip.models.circuitprocessor import CircuitProcessor, ModelProcessor


class DispersivecQED(ModelProcessor):
    """
    Representation of the physical implementation of a quantum
    program/algorithm on a dispersive cavity-QED system.
    """

    def __init__(self, N, correct_global_phase=True, Nres=10, deltamax=1.0,
                 epsmax=9.5, w0=10., wq=None, eps=9.5,
                 delta=0.0, g=0.01, T1=None, T2=None):
        """
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
        wq: Integer/List
            The frequency of the qubits.
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
        """

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
        Calculate the paraicients for this setup.
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

    def get_ops_and_u(self):
        return (self._hams, self.amps.T)

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
        self.qc0 = qc
        self.qc1 = self.qc0.resolve_gates(basis=["ISWAP", "RX", "RZ"])
        self.qc2 = self.dispersive_gate_correction(self.qc1)

        return self.qc2

    def eliminate_auxillary_modes(self, U):
        return self.psi_proj.dag() * U * self.psi_proj

    def dispersive_gate_correction(self, qc1, rwa=True):
        """
        Method to resolve ISWAP and SQRTISWAP gates in a cQED system by adding
        single qubit gates to get the correct output matrix.

        Parameters
        ----------
        qc: Qobj
            The circular spin chain circuit to be resolved

        rwa: Boolean
            Specify if RWA is used or not.

        Returns
        ----------
        qc: QubitCircuit
            Returns QubitCircuit of resolved gates for the qubit circuit in the
            desired basis.
        """
        qc = QubitCircuit(qc1.N, qc1.reverse_states)

        for gate in qc1.gates:
            qc.gates.append(gate)
            if rwa:
                if gate.name == "SQRTISWAP":
                    qc.gates.append(Gate("RZ", [gate.targets[0]], None,
                                         arg_value=-np.pi / 4,
                                         arg_label=r"-\pi/4"))
                    qc.gates.append(Gate("RZ", [gate.targets[1]], None,
                                         arg_value=-np.pi / 4,
                                         arg_label=r"-\pi/4"))
                    qc.gates.append(Gate("GLOBALPHASE", None, None,
                                         arg_value=-np.pi / 4,
                                         arg_label=r"-\pi/4"))
                elif gate.name == "ISWAP":
                    qc.gates.append(Gate("RZ", [gate.targets[0]], None,
                                         arg_value=-np.pi / 2,
                                         arg_label=r"-\pi/2"))
                    qc.gates.append(Gate("RZ", [gate.targets[1]], None,
                                         arg_value=-np.pi / 2,
                                         arg_label=r"-\pi/2"))
                    qc.gates.append(Gate("GLOBALPHASE", None, None,
                                         arg_value=-np.pi / 2,
                                         arg_label=r"-\pi/2"))

        return qc

    def load_circuit(self, qc):
        """
        Decompose a :class:`qutip.QubitCircuit` in to the control
        amplitude generating the corresponding evolution.
        """
        gates = self.optimize_circuit(qc).gates

        self.global_phase = 0
        self.amps = np.zeros([len(self._hams), len(gates)])
        dt_list = []

        n = 0
        phase_gate_num = 0
        for gate in gates:

            if gate.name == "ISWAP":
                t0, t1 = gate.targets[0], gate.targets[1]
                self.sz_u[t0, n] = self.wq[t0] - self._paras["w0"]
                self.sz_u[t1, n] = self.wq[t1] - self._paras["w0"]
                self.g_u[t0, n] = self._paras["g"][t0]
                self.g_u[t1, n] = self._paras["g"][t1]

                J = self._paras["g"][t0] * self._paras["g"][t1] * (
                    1 / self.Delta[t0] + 1 / self.Delta[t1]) / 2
                T = (4 * np.pi / abs(J)) / 4
                dt_list.append(T)
                n += 1

            elif gate.name == "SQRTISWAP":
                t0, t1 = gate.targets[0], gate.targets[1]
                self.sz_u[t0, n] = self.wq[t0] - self._paras["w0"]
                self.sz_u[t1, n] = self.wq[t1] - self._paras["w0"]
                self.g_u[t0, n] = self._paras["g"][t0]
                self.g_u[t1, n] = self._paras["g"][t1]

                J = self._paras["g"][t0] * self._paras["g"][t1] * (
                    1 / self.Delta[t0] + 1 / self.Delta[t1]) / 2
                T = (4 * np.pi / abs(J)) / 8
                dt_list.append(T)
                n += 1

            elif gate.name == "RZ":
                g = self._paras["sz"][gate.targets[0]]
                self.sz_u[gate.targets[0], n] = np.sign(gate.arg_value) * g
                T = abs(gate.arg_value) / (2 * g)
                dt_list.append(T)
                n += 1

            elif gate.name == "RX":
                g = self._paras["sx"][gate.targets[0]]
                self.sx_u[gate.targets[0], n] = np.sign(gate.arg_value) * g
                T = abs(gate.arg_value) / (2 * g)
                dt_list.append(T)
                n += 1

            elif gate.name == "GLOBALPHASE":
                self.global_phase += gate.arg_value
                phase_gate_num += 1

            else:
                raise ValueError("Unsupported gate %s" % gate.name)

        self.tlist = np.zeros(len(dt_list))
        t = 0
        for temp_ind in range(len(dt_list)):
            t += dt_list[temp_ind]
            self.tlist[temp_ind] = t
        self.amps = self.amps[:, :len(gates)-phase_gate_num]
        # TODO The amplitude of the first control a.dag()*a
        # was set to zero before I made this refactoring.
        # It is probably due to the fact that
        # it contributes only a constant (N) and can be neglected.
        # but change the below line to np.ones leads to test error.
        self.amps[0] = self._paras["w0"] * np.zeros((self.sx_u.shape[1]))
        return self.tlist, self.amps
