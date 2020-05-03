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
from qutip.qip.device.processor import Processor
from qutip.qip.device.modelprocessor import ModelProcessor
from qutip.qip.operations import expand_operator
from qutip.qobj import Qobj
from qutip.qobjevo import QobjEvo
from qutip.qip.pulse import Pulse
from qutip.qip.compiler.gatecompiler import GateCompiler
from qutip.qip.compiler import CavityQEDCompiler


__all__ = ['DispersiveCavityQED']


class DispersiveCavityQED(ModelProcessor):
    """
    The processor based on the physical implementation of
    a dispersive cavity QED system.
    The available Hamiltonian of the system is predefined.
    For a given pulse amplitude matrix, the processor can
    calculate the state evolution under the given control pulse,
    either analytically or numerically.
    (Only additional attributes are documented here, for others please
    refer to the parent class :class:`qutip.qip.device.ModelProcessor`)

    Parameters
    ----------
    N: int
        The number of qubits in the system.

    correct_global_phase: float, optional
        Save the global phase, the analytical solution
        will track the global phase.
        It has no effect on the numerical solution.

    num_levels: int, optional
        The number of energy levels in the resonator.

    deltamax: int or list, optional
        The coefficients of sigma-x for each of the qubits in the system.

    epsmax: int or list, optional
        The coefficients of sigma-z for each of the qubits in the system.

    w0: int, optional
        The base frequency of the resonator.

    eps: int or list, optional
        The epsilon for each of the qubits in the system.

    delta: int or list, optional
        The epsilon for each of the qubits in the system.

    g: int or list, optional
        The interaction strength for each of the qubit with the resonator.

    t1: list or float
        Characterize the decoherence of amplitude damping for
        each qubit. A list of size `N` or a float for all qubits.

    t2: list of float
        Characterize the decoherence of dephasing for
        each qubit. A list of size `N` or a float for all qubits.

    Attributes
    ----------
    sx_ops: list
        A list of sigmax Hamiltonians for each qubit.

    sz_ops: list
        A list of sigmaz Hamiltonians for each qubit.

    cavityqubit_ops: list
        A list of interacting Hamiltonians between cavity and each qubit.

    sx_u: array_like
        Pulse matrix for sigmax Hamiltonians.

    sz_u: array_like
        Pulse matrix for sigmaz Hamiltonians.

    g_u: array_like
        Pulse matrix for interacting Hamiltonians
        between cavity and each qubit.

    wq: list of float
        The frequency of the qubits calculated from
        eps and delta for each qubit.

    Delta: list of float
        The detuning with respect to w0 calculated
        from wq and w0 for each qubit.
    """

    def __init__(self, N, correct_global_phase=True,
                 num_levels=10, deltamax=1.0,
                 epsmax=9.5, w0=10., wq=None, eps=9.5,
                 delta=0.0, g=0.01, t1=None, t2=None):
        super(DispersiveCavityQED, self).__init__(
            N, correct_global_phase=correct_global_phase,
            t1=t1, t2=t2)
        self.correct_global_phase = correct_global_phase
        self.spline_kind = "step_func"
        self.num_levels = num_levels
        self._params = {}
        self.set_up_params(
            N=N, num_levels=num_levels, deltamax=deltamax,
            epsmax=epsmax, w0=w0, wq=wq, eps=eps,
            delta=delta, g=g)
        self.set_up_ops(N)
        self.dims = [num_levels] + [2] * N

    def set_up_ops(self, N):
        """
        Generate the Hamiltonians for the spinchain model and save them in the
        attribute `ctrls`.

        Parameters
        ----------
        N: int
            The number of qubits in the system.
        """
        # single qubit terms
        for m in range(N):
            self.pulses.append(
                Pulse(sigmax(), [m+1], spline_kind=self.spline_kind))
        for m in range(N):
            self.pulses.append(
                Pulse(sigmaz(), [m+1], spline_kind=self.spline_kind))
        # coupling terms
        a = tensor(
            [destroy(self.num_levels)] +
            [identity(2) for n in range(N)])
        for n in range(N):
            sm = tensor([identity(self.num_levels)] +
                        [destroy(2) if m == n else identity(2)
                         for m in range(N)])
            self.pulses.append(
                Pulse(a.dag() * sm + a * sm.dag(),
                      list(range(N+1)), spline_kind=self.spline_kind))

    def set_up_params(
            self, N, num_levels, deltamax,
            epsmax, w0, wq, eps, delta, g):
        """
        Save the parameters in the attribute `params` and check the validity.

        Parameters
        ----------
        N: int
            The number of qubits in the system.

        num_levels: int
            The number of energy levels in the resonator.

        deltamax: list
            The coefficients of sigma-x for each of the qubits in the system.

        epsmax: list
            The coefficients of sigma-z for each of the qubits in the system.

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

        Notes
        -----
        All parameters will be multiplied by 2*pi for simplicity
        """
        sx_para = 2 * np.pi * self.to_array(deltamax, N)
        self._params["sx"] = sx_para
        sz_para = 2 * np.pi * self.to_array(epsmax, N)
        self._params["sz"] = sz_para
        w0 = 2 * np.pi * w0
        self._params["w0"] = w0
        eps = 2 * np.pi * self.to_array(eps, N)
        self._params["eps"] = eps
        delta = 2 * np.pi * self.to_array(delta, N)
        self._params["delta"] = delta
        g = 2 * np.pi * self.to_array(g, N)
        self._params["g"] = g

        # computed
        self.wq = np.sqrt(eps**2 + delta**2)
        self.Delta = self.wq - w0

        # rwa/dispersive regime tests
        if any(g / (w0 - self.wq) > 0.05):
            warnings.warn("Not in the dispersive regime")

        if any((w0 - self.wq)/(w0 + self.wq) > 0.05):
            warnings.warn(
                "The rotating-wave approximation might not be valid.")

    @property
    def sx_ops(self):
        return self.ctrls[0: self.N]

    @property
    def sz_ops(self):
        return self.ctrls[self.N: 2*self.N]

    @property
    def cavityqubit_ops(self):
        return self.ctrls[2*self.N: 3*self.N]

    @property
    def sx_u(self):
        return self.coeffs[: self.N]

    @property
    def sz_u(self):
        return self.coeffs[self.N: 2*self.N]

    @property
    def g_u(self):
        return self.coeffs[2*self.N: 3*self.N]

    def get_operators_labels(self):
        """
        Get the labels for each Hamiltonian.
        It is used in the method``plot_pulses``.
        It is a 2-d nested list, in the plot,
        a different color will be used for each sublist.
        """
        return ([[r"$\sigma_x^%d$" % n for n in range(self.N)],
                 [r"$\sigma_z^%d$" % n for n in range(self.N)],
                 [r"$g_{%d}$" % (n) for n in range(self.N)]])

    def optimize_circuit(self, qc):
        """
        Take a quantum circuit/algorithm and convert it into the
        optimal form/basis for the desired physical system.

        Parameters
        ----------
        qc: :class:`qutip.QubitCircuit`
            Takes the quantum circuit to be implemented.

        Returns
        -------
        qc: :class:`qutip.QubitCircuit`
            The circuit representation with elementary gates
            that can be implemented in this model.
        """
        self.qc0 = qc
        self.qc1 = self.qc0.resolve_gates(
            basis=["SQRTISWAP", "ISWAP", "RX", "RZ"])
        return self.qc1

    def eliminate_auxillary_modes(self, U):
        """
        Eliminate the auxillary modes like the cavity modes in cqed.
        """
        psi_proj = tensor(
            [basis(self.num_levels, 0)] +
            [identity(2) for n in range(self.N)])
        return psi_proj.dag() * U * psi_proj

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
        tlist: array_like
            A NumPy array specifies the time of each coefficient

        coeffs: array_like
            A 2d NumPy array of the shape (len(ctrls), len(tlist)). Each
            row corresponds to the control pulse sequence for
            one Hamiltonian.
        """
        gates = self.optimize_circuit(qc).gates
        compiler = CavityQEDCompiler(
            self.N, self._params,
            global_phase=0., num_ops=len(self.ctrls))
        tlist, self.coeffs, self.global_phase = compiler.decompose(gates)
        self.set_all_tlist(tlist)
        return tlist, self.coeffs
