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
import warnings

import numpy as np

from qutip.operators import tensor, identity, destroy, sigmax, sigmaz
from qutip.states import basis
from qutip.qip.circuit import QubitCircuit, Gate
from qutip.qip.models.circuitprocessor import CircuitProcessor
from qutip.qip.models.modelprocessor import ModelProcessor, GateDecomposer
from qutip.qobj import Qobj
from qutip.qobjevo import QobjEvo


__all__ = ['DispersivecQED', 'CQEDGateDecomposer']


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
    N: int
        The number of qubits in the system.

    correct_global_phase: float, optional
        Save the global phase, the analytical solution
        will track the global phase.
        It has no effect on the numerical solution.

    Nres: int, optional
        The number of energy levels in the resonator.

    deltamax: int or list, optional
        The sigma-x paraicient for each of the qubits in the system.

    epsmax: int or list, optional
        The sigma-z paraicient for each of the qubits in the system.

    wo: int, optional
    The base frequency of the resonator.

    eps: int or list, optional
        The epsilon for each of the qubits in the system.

    delta: int or list, optional
        The epsilon for each of the qubits in the system.

    g: int or list, optional
        The interaction strength for each of the qubit with the resonator.

    T1: list or float, optional
        Characterize the decoherence of amplitude damping for
        each qubit.

    T2: list of float, optional
        Characterize the decoherence of dephasing for
        each qubit.

    noise: :class:`qutip.qip.CircuitNoise`, optional
        A list of noise objects. They will be processed when creating the
        noisy :class:`qutip.QobjEvo` from the processor or run the simulation.
        Defaut is a empty list.

    Attributes
    ----------
    N: int
        The number of component system

    ctrls: list
        A list of the control Hamiltonians driving the evolution.

    tlist: array-like
        A NumPy array specifies the time of each coefficient.

    coeffs: array-like
        A 2d NumPy array of the shape, the length is dependent on the
        spline type

    T1: list
        Characterize the decoherence of amplitude damping for
        each qubit.

    T2: list
        Characterize the decoherence of dephasing for
        each qubit.

    noise: :class:`qutip.qip.CircuitNoise`, optional
        A list of noise objects. They will be processed when creating the
        noisy :class:`qutip.QobjEvo` from the processor or run the simulation.

    dims: list
        The dimension of each component system.
        If not given, it will be a
        qutbis system of dim=[2,2,2,...,2]

    spline_kind: str
        Type of the coefficient interpolation. Default is "step_func".
        Note that they have different requirements for the shape of
        :attr:`qutip.qip.circuitprocessor.coeffs`.

    sx_ops: list
        A list of sigmax Hamiltonians for each qubit.

    sz_ops: list
        A list of sigmaz Hamiltonians for each qubit.

    cavityqubit_ops: list
        A list of interacting Hamiltonians between cavity and each qubit.

    sx_u: array-like
        Pulse matrix for sigmax Hamiltonians.

    sz_u: array-like
        Pulse matrix for sigmaz Hamiltonians.

    g_u: array-like
        Pulse matrix for interacting Hamiltonians
        between cavity and each qubit.

    wq: list of float
        The frequency of the qubits calculated from
        eps and delta for each qubit.

    Delta: list of float
        The detuning with repect to w0 calculated
        from wq and w0 for each qubit.
    """

    def __init__(self, N, correct_global_phase=True, Nres=10, deltamax=1.0,
                 epsmax=9.5, w0=10., wq=None, eps=9.5,
                 delta=0.0, g=0.01, T1=None, T2=None, noise=None):
        super(DispersivecQED, self).__init__(
            N, correct_global_phase=correct_global_phase,
            T1=T1, T2=T2, noise=noise, dims=[Nres] + [2] * N)
        self.correct_global_phase = correct_global_phase
        self.spline_kind = "step_func"
        self.Nres = Nres
        self._paras = {}
        self.set_up_paras(
            N=N, Nres=Nres, deltamax=deltamax,
            epsmax=epsmax, w0=w0, wq=wq, eps=eps,
            delta=delta, g=g)
        self.set_up_ops(N)

    def set_up_ops(self, N):
        """
        Genrate the Hamiltonians for the spinchain model and save them in the
        attribute `ctrls`.

        Parameters
        ----------
        N: int
            The number of qubits in the system.
        """
        # single qubit terms
        self.a = tensor([destroy(self.Nres)] + [identity(2) for n in range(N)])
        self.ctrls.append(self.a.dag() * self.a)
        self.ctrls += [tensor([identity(self.Nres)] +
                              [sigmax() if m == n else identity(2)
                               for n in range(N)])
                       for m in range(N)]
        self.ctrls += [tensor([identity(self.Nres)] +
                              [sigmaz() if m == n else identity(2)
                               for n in range(N)])
                       for m in range(N)]
        # interaction terms
        for n in range(N):
            sm = tensor([identity(self.Nres)] +
                        [destroy(2) if m == n else identity(2)
                         for m in range(N)])
            self.ctrls.append(self.a.dag() * sm + self.a * sm.dag())

        self.psi_proj = tensor([basis(self.Nres, 0)] +
                               [identity(2) for n in range(N)])

    def set_up_paras(
            self, N, Nres, deltamax,
            epsmax, w0, wq, eps, delta, g):
        """
        Save the parameters in the attribute `paras` and check the validity.

        Parameters
        ----------
        N: int
            The number of qubits in the system.

        Nres: int
            The number of energy levels in the resonator.

        deltamax: list
            The sigma-x paraicient for each of the qubits in the system.

        epsmax: list
            The sigma-z paraicient for each of the qubits in the system.

        wo: int
            The base frequency of the resonator.

        wq: list
            The frequency of the qubits.

        eps: list
            The epsilon for each of the qubits in the system.

        delta: list
            The delta for each of the qubits in the system.

        g: list
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
        return self.ctrls[1: self.N+1]

    @property
    def sz_ops(self):
        return self.ctrls[self.N+1: 2*self.N+1]

    @property
    def cavityqubit_ops(self):
        return self.ctrls[2*self.N+1: 3*self.N+1]

    @property
    def sx_u(self):
        return self.coeffs[1: self.N+1]

    @property
    def sz_u(self):
        return self.coeffs[self.N+1: 2*self.N+1]

    @property
    def g_u(self):
        return self.coeffs[2*self.N+1: 3*self.N+1]

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
        self.qc1 = self.qc0.resolve_gates(
            basis=["SQRTISWAP","ISWAP", "RX", "RZ"])
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
        tlist: array-like
            A NumPy array specifies the time of each coefficients

        coeffs: array-like
            A 2d NumPy array of the shape (len(ctrls), len(tlist)). Each
            row corresponds to the control pulse sequence for
            one Hamiltonian.
        """
        gates = self.optimize_circuit(qc).gates

        dec = CQEDGateDecomposer(
            self.N, self._paras, self.wq, self.Delta,
            global_phase=0., num_ops=len(self.ctrls))
        self.tlist, self.coeffs, self.global_phase = dec.decompose(gates)

        # TODO The amplitude of the first control a.dag()*a
        # was set to zero before I made this refactoring.
        # It is probably due to the fact that
        # it contributes only a constant (N) and can be neglected.
        # but change the below line to np.ones leads to test error.
        self.coeffs[0] = self._paras["w0"] * np.zeros((self.sx_u.shape[1]))
        return self.tlist, self.coeffs


class CQEDGateDecomposer(GateDecomposer):
    """
    Decompose a :class:`qutip.QubitCircuit` into
    the pulse sequence for the processor.

    Parameters
    ----------
    N: int
        The number of qubits in the system.

    paras: dict
        A Python dictionary contains the name and the value of the parameters
        of the physical realization, such as laser frequency,detuning etc.

    wq: list of float
        The frequency of the qubits calculated from
        eps and delta for each qubit.

    Delta: list of float
        The detuning with repect to w0 calculated
        from wq and w0 for each qubit.

    global_phase: bool
        Record of the global phase change and will be returned.

    num_ops: int
        Number of Hamiltonians in the processor.

    Attributes
    ----------
    N: int
        The number of the component systems.

    paras: dict
        A Python dictionary contains the name and the value of the parameters
        of the physical realization, such as laser frequency,detuning etc.

    num_ops: int
        Number of control Hamiltonians in the processor.

    gate_decs: dict
        The Python dictionary in the form {gate_name: decompose_function}.
    """
    def __init__(self, N, paras, wq, Delta, global_phase, num_ops):
        super(CQEDGateDecomposer, self).__init__(
            N=N, paras=paras, num_ops=num_ops)
        self.gate_decs = {"ISWAP": self.iswap_dec,
                          "SQRTISWAP": self.sqrtiswap_dec,
                          "RZ": self.rz_dec,
                          "RX": self.rx_dec,
                          "GLOBALPHASE": self.globalphase_dec
                          }
        self._sx_ind = list(range(1, N+1))
        self._sz_ind = list(range(N+1, 2*N+1))
        self._g_ind = list(range(2*N+1, 3*N+1))
        self.wq = wq
        self.Delta = Delta
        self.global_phase = global_phase

    def decompose(self, gates):
        tlist, coeffs = super(CQEDGateDecomposer, self).decompose(gates)
        return tlist, coeffs, self.global_phase

    def rz_dec(self, gate):
        """
        Decomposer for the RZ gate
        """
        pulse = np.zeros(self.num_ops)
        q_ind = gate.targets[0]
        g = self.paras["sz"][q_ind]
        pulse[self._sz_ind[q_ind]] = np.sign(gate.arg_value) * g
        t = abs(gate.arg_value) / (2 * g)
        self.dt_list.append(t)
        self.coeff_list.append(pulse)

    def rx_dec(self, gate):
        """
        Decomposer for the RX gate
        """
        pulse = np.zeros(self.num_ops)
        q_ind = gate.targets[0]
        g = self.paras["sx"][q_ind]
        pulse[self._sx_ind[q_ind]] = np.sign(gate.arg_value) * g
        t = abs(gate.arg_value) / (2 * g)
        self.dt_list.append(t)
        self.coeff_list.append(pulse)

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
        pulse[self._sz_ind[q1]] = self.wq[q1] - self.paras["w0"]
        pulse[self._sz_ind[q2]] = self.wq[q2] - self.paras["w0"]
        pulse[self._g_ind[q1]] = self.paras["g"][q1]
        pulse[self._g_ind[q2]] = self.paras["g"][q2]
        J = self.paras["g"][q1] * self.paras["g"][q2] * (
            1 / self.Delta[q1] + 1 / self.Delta[q2]) / 2
        t = (4 * np.pi / abs(J)) / 8
        self.dt_list.append(t)
        self.coeff_list.append(pulse)

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
        pulse[self._sz_ind[q1]] = self.wq[q1] - self.paras["w0"]
        pulse[self._sz_ind[q2]] = self.wq[q2] - self.paras["w0"]
        pulse[self._g_ind[q1]] = self.paras["g"][q1]
        pulse[self._g_ind[q2]] = self.paras["g"][q2]
        J = self.paras["g"][q1] * self.paras["g"][q2] * (
            1 / self.Delta[q1] + 1 / self.Delta[q2]) / 2
        t = (4 * np.pi / abs(J)) / 4
        self.dt_list.append(t)
        self.coeff_list.append(pulse)

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
