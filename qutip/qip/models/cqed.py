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

        return self.qc1

    def eliminate_auxillary_modes(self, U):
        return self.psi_proj.dag() * U * self.psi_proj

    def load_circuit(self, qc):
        """
        Decompose a :class:`qutip.QubitCircuit` in to the control
        amplitude generating the corresponding evolution.
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
        pulse = np.zeros(self.num_ops)
        q_ind = gate.targets[0]
        g = self.paras["sz"][q_ind]
        pulse[self.sz_ind[q_ind]] = np.sign(gate.arg_value) * g
        t = abs(gate.arg_value) / (2 * g)
        self.dt_list.append(t)
        self.amps_list.append(pulse)

    def rx_dec(self, gate):
        pulse = np.zeros(self.num_ops)
        q_ind = gate.targets[0]
        g = self.paras["sx"][q_ind]
        pulse[self.sx_ind[q_ind]] = np.sign(gate.arg_value) * g
        t = abs(gate.arg_value) / (2 * g)
        self.dt_list.append(t)
        self.amps_list.append(pulse)

    def sqrtiswap_dec(self, gate):
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
        self.global_phase += gate.arg_value
