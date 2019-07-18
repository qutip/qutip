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
from collections.abc import Iterable
import numbers
import inspect

import numpy as np

from qutip.qobj import Qobj
from qutip.qobjevo import QobjEvo
from qutip.operators import identity
from qutip.qip.gates import expand_oper, globalphase
from qutip.tensor import tensor
from qutip.mesolve import mesolve
from qutip.qip.circuit import QubitCircuit
from qutip.qip.models.circuitnoise import (
    CircuitNoise, RelaxationNoise, DecoherenceNoise,
    ControlAmpNoise, WhiteNoise, UserNoise)


__all__ = ['CircuitProcessor', 'ModelProcessor', 'GateDecomposer']


class CircuitProcessor(object):
    """
    The base class for a circuit processor,
    which is defined by the available Hamiltonian of the system
    and the decoherence time for each qubit.
    For a given pulse amplitude matrix, the processor can
    calculate the state evolution under the given control pulse,
    either analytically or numerically.
    In the subclasses, further methods are defined so that
    it can be used to find the driving pulse
    of a quantum circuit.

    Parameters
    ----------
    N : int
        The number of qubits in the system
    T1 : list or float
        Characterize the decoherence of amplitude damping for
        each qubit.
    T2 : list of float
        Characterize the decoherence of dephasing relaxation for
        each qubit.

    Attributes
    ----------
    ctrls : list of :class:`Qobj`
        A list of Hamiltonians of the control pulse driving the evolution.
    tlist : array like
        A NumPy array specifies at which time the next amplitude of
        a pulse is to be applied.
    amps : array like
        A 2d NumPy array of the shape (len(ctrls), len(tlist)). Each
        row corresponds to the control pulse sequence for
        one Hamiltonian.
    """
    def __init__(self, N, T1=None, T2=None, noise=None, spline_kind="step_func"):
        self.N = N
        self.tlist = None
        self.amps = None
        self.ctrls = []
        self.T1 = T1
        self.T2 = T2
        if noise is None:
            self.noise = []
        else:
            self.noise = noise
        self.dims = [2] * N
        self.spline_kind = "step_func"

    def add_ctrl(self, ctrl, targets=None, expand_type=None):
        """
        Add a control Hamiltonian to the processor

        Parameters
        ----------
        ctrl : :class:`qutip.Qobj`
            A hermitian Qobj representation of the driving Hamiltonian
        targets : list or int
            The indices of qubits that are acted on
        expand_type : string
            The type of expansion
            None - only expand for the given target qubits
            "periodic" - the Hamiltonian is to be expanded for
                all cyclic permutation of target qubits
        """
        # Check validity of ctrl
        if not isinstance(ctrl, Qobj):
            raise TypeError("The Hamiltonian must be a qutip.Qobj.")
        if not ctrl.isherm:
            raise ValueError("The Hamiltonian must be Hermitian.")

        num_qubits = len(ctrl.dims[0])
        if targets is None:
            targets = list(range(num_qubits))

        if expand_type is None:
            if num_qubits == self.N:
                self.ctrls.append(ctrl)
            else:
                self.ctrls.append(expand_oper(ctrl, self.N, targets))
        elif expand_type == "periodic":
            for i in range(self.N):
                new_targets = np.mod(np.array(targets)+i, self.N)
                self.ctrls.append(
                    expand_oper(ctrl, self.N, new_targets))
        else:
            raise ValueError(
                "expand_type can only be None or 'periodic', "
                "not {}".format(expand_type))

    def remove_ctrl(self, indices):
        """
        Remove the ctrl Hamiltonian with given indices.

        Parameters
        ----------
        indices : int or list or int
            The indices of the control Hamiltonians to be removed
        """
        if not isinstance(indices, Iterable):
            indices = [indices]
        indices.sort(reverse=True)
        for ind in indices:
            del self.ctrls[ind]

    def _is_time_amps_valid(self):
        """
        Check it the len(tlist) and amps.shape[1] are the same.
        """
        if self.tlist is None and self.amps is None:
            pass
        elif self.tlist is None:
            raise ValueError("`tlist` or `amps` is not given.")
        elif self.amps is None:
            pass  # evolution with identity
        else:
            amps_len = self.amps.shape[1]
            tlist_len = len(self.tlist)
            if amps_len != tlist_len-1:
                raise ValueError(
                    "`tlist` has the length of {} while amps "
                    "has {}.".format(tlist_len, amps_len))

    def _is_ctrl_amps_valid(self):
        """
        Check if the number of control Hamiltonians
        and amps.shape[0] are the same.
        """
        if self.amps is None and len(self.ctrls) != 0:
            raise ValueError(
                "The control amplitude is None while "
                "the number of ctrls is {}.".format(len(self.ctrls)))
        if self.amps is not None:
            if self.amps.shape[0] != len(self.ctrls):
                raise ValueError(
                    "The control amplitude matrix do not match the "
                    "number of ctrls  "
                    "#ctrls = {}  "
                    "#amps = {}.".format(len(self.ctrls), len(self.amps)))

    def _is_amps_valid(self):
        """
        Check if the attribute are in the corret shape.
        """
        self._is_time_amps_valid()
        self._is_ctrl_amps_valid()

    def save_amps(self, file_name, inctime=True):
        """
        Save a file with the current control amplitudes in each timeslot

        Parameters
        ----------
        file_name : string
            Name of the file.
        inctime : boolean
            True if the time list in included in the first column.
        """
        self._is_amps_valid()

        if inctime:
            shp = self.amps.T.shape
            data = np.empty([shp[0], shp[1] + 1], dtype=np.float)
            data[:, 0] = self.tlist[1:]
            data[:, 1:] = self.amps.T
        else:
            data = self.amps.T

        np.savetxt(file_name, data, delimiter='\t', fmt='%1.16f')

    def read_amps(self, file_name, inctime=True):
        """
        Read the pulse amplitude matrix and time list
        saved in the file by `save_amp`

        Parameters
        ----------
        file_name : string
            Name of the file.
        inctime : boolean
            True if the time list in included in the first column.

        Returns
        -------
        tlist : array like
            The time list read from the file.
        amps : array like
            The pulse matrix read from the file.
        """
        data = np.loadtxt(file_name, delimiter='\t')
        if not inctime:
            self.amps = data.T
        else:
            self.tlist = np.hstack([[0], data[:, 0]])
            self.amps = data[:, 1:].T
        return self.tlist, self.amps

    def get_qobjevo(self, tlist=None):
        """
        tlist : array like
            Used if there is no tlist defined in the processor but given as 
            keyword arguments
        """
        if tlist is None:
            tlist = self.tlist
        args = {}
        if self.spline_kind == "step_func" and self.amps is not None:
            amps = np.hstack([self.amps, self.amps[:, -1:]])
            args = {"_spline_kind": "previous"}
        else:
            amps = self.amps

        dummy_cte = tensor([Qobj(np.zeros((d, d))) for d in self.dims])
        H = [dummy_cte]
        for op_ind in range(len(self.ctrls)):
            H.append(
                [self.ctrls[op_ind], amps[op_ind]])
        return QobjEvo(H, tlist=tlist, args=args)

    def run_state(self, rho0, **kwargs):
        """
        Use mesolve to calculate the time of the state evolution
        and return the result. Other arguments of mesolve can be
        given as keyword arguments.

        Parameters
        ----------
        rho0 : Qobj
            Initial density matrix or state vector (ket).
        kwargs
            Keyword arguments for the qutip solver.
            (currently `qutip.mesolve`)
        Returns
        -------
        evo_result : :class:`qutip.Result`
            An instance of the class :class:`qutip.Result`, which contains
            either an *array* `result.expect` of expectation values for
            the times specified by `tlist`, or an *array* `result.states`
            of state vectors or density matrices corresponding to
            the times in `tlist` [if `e_ops` is
            an empty list], or nothing if a callback function was
            given in place of
            operators for which to calculate the expectation values.
        """
        # check validity
        self._is_amps_valid()
        if "H" in kwargs or "args" in kwargs:
            raise ValueError(
                "`H` and `args` are already specified by the processor "
                "and can not be given as a key word argument")

        # if no control pulse specified (e.g. drift or id evolution with c_ops)
        if self.tlist is None:
            if "tlist" in kwargs:
                tlist = kwargs["tlist"]
                del kwargs["tlist"]
            else:
                raise ValueError(
                    "`tlist` is not defined in the processor.")
        elif self.tlist is not None and "tlist" in kwargs:
            raise ValueError(
                "`tlist` is already specified by the processor, "
                "thus can not be given as a key word argument")
        else:
            tlist = self.tlist

        # contruct time-dependent Hamiltonian
        proc_qobjevo = self.get_qobjevo(tlist=tlist)
        ham_noise, sys_c_ops = self.process_noise(proc_qobjevo)
        if "c_ops" in kwargs:
            kwargs["c_ops"] += sys_c_ops
        else:
            kwargs["c_ops"] = sys_c_ops
        proc_qobjevo += ham_noise
        evo_result = mesolve(
            H=proc_qobjevo, rho0=rho0, tlist=tlist, **kwargs)

        return evo_result

    def optimize_circuit(self, qc):
        """
        Function to take a quantum circuit/algorithm and convert it into the
        optimal form/basis for the desired physical system.
        (Defined in subclasses)
        """
        raise NotImplementedError("Use the function in the sub-class")

    def load_circuit(self, qc):
        """
        Translates an abstract quantum circuit to its
        corresponding Hamiltonians.(Defined in subclasses)
        """
        raise NotImplementedError("Use the function in the sub-class")

    def get_ops_and_u(self):
        """
        Returns the Hamiltonian operators and the pulse matrix
        (Defined in subclasses)
        """
        raise NotImplementedError("Use the function in the sub-class")

    def eliminate_auxillary_modes(self, U):
        """
        Eliminate the auxillary modes like the cavity modes in cqed.
        (Defined in subclasses)
        """
        return U

    def plot_pulses(self, amps=None, tlist=None, **kwargs):
        """
        Plot the pulse amplitude

        Parameters
        ----------
        **kwargs
            Key word arguments for figure

        Returns
        -------
        fig : matplotlib.figure.Figure
            The `Figure` object for the plot.
        ax : matplotlib.axes._subplots.AxesSubplot
            The axes for the plot.
        kwargs
            Keyword arguments for `plt.subplot` or `as.step`.
        """
        import matplotlib.pyplot as plt
        if amps is None:
            amps = self.amps
        if tlist is None:
            tlist = self.tlist

        fig_keys = inspect.getfullargspec(plt.subplots)[0]
        fig, ax = plt.subplots(
            1, 1, **{key: kwargs[key] for key in kwargs if key in fig_keys})
        ax.set_ylabel("Control pulse amplitude")
        ax.set_xlabel("Time")
        amps = np.hstack([amps, amps[:, -1:]])
        plot_keys = inspect.getfullargspec(ax.step)[0]
        for i in range(amps.shape[0]):
            ax.step(tlist, amps[i], where='post',
                    **{key: kwargs[key] for key in kwargs if key in plot_keys})
        fig.tight_layout()
        return fig, ax

    def add_noise(self, noise):
        """
        Add a noise object to the processor

        Parameters
        ----------
        noise : :class:`qutip.qip.CircuitNoise`
            The noise object defined out side the processor
        """
        if isinstance(noise, CircuitNoise):
            self.noise.append(noise)
        else:
            raise TypeError("`noise` is not a CircuitNoise object.")

    def process_noise(self, unitary_qobjevo):
        """
        Call all the noise object saved in the processor and
        return the :class:`qutip.QobjEvo` and :class:`qutip.Qobj`
        representing the noise. 

        Parameters
        ----------
        unitary_qobjevo : :class:`qutip.qip.QobjEvo`
            The :class:`qutip.qip.QobjEvo` representing the unitary evolution
            in the noiseless processor.

        Returns
        -------
        H_noise : :class:`qutip.qip.QobjEvo`
            The :class:`qutip.qip.QobjEvo` representing the noise.
        c_ops : list
            A list of :class:`qutip.qip.QobjEvo` or :class:`qutip.qip.Qobj`,
            representing the time-(in)dependent collapse operators.
        """
        # TODO there is a bug when using +=QobjEvo()
        tlist = self.tlist
        evo_noise_list = QobjEvo(tensor([identity(2)]*self.N), tlist=tlist)
        c_ops = []
        evo_noise_list = []
        if (self.T1 is not None) or (self.T2 is not None):
            c_ops += RelaxationNoise(self.T1, self.T2).get_noise(
                N=self.N)
        for noise in self.noise:
            if isinstance(noise, (DecoherenceNoise, RelaxationNoise)):
                c_ops += noise.get_noise(
                    self.N, tlist)
            elif isinstance(noise, ControlAmpNoise):
                evo_noise_list.append(noise.get_noise(
                    self.N, tlist, unitary_qobjevo))
            elif isinstance(noise, UserNoise):
                noise_qobjevo, new_c_ops = noise.get_noise(
                    self.N, tlist, unitary_qobjevo)
                evo_noise_list.append(noise_qobjevo)
                c_ops += new_c_ops
            else:
                raise NotImplementedError(
                    "The noise type {} is not"
                    "implemented in the processor".format(
                        type(noise)))
        return sum(evo_noise_list), c_ops


class ModelProcessor(CircuitProcessor):
    """
    The base class for a circuit processor based on phyiscal hardwares,
    e.g ion trap, spinchain.
    The available Hamiltonian of the system is predefined.
    For a given pulse amplitude matrix, the processor can
    calculate the state evolution under the given control pulse,
    either analytically or numerically.
    In the subclasses, further methods are defined so that
    it can be used to find the driving pulse
    of a quantum circuit.

    Parameters
    ----------
    N : int
        The number of qubits in the system.
    T1 : list or float
        Characterize the decoherence of amplitude damping for
        each qubit.
    T2 : list of float
        Characterize the decoherence of dephasing relaxation for
        each qubit.
    correct_global_phase : boolean
        If the gloabl phase will be tracked when analytical method is choosen.

    Attributes
    ----------
    ctrls : list of :class:`Qobj`
        A list of Hamiltonians of the control pulse driving the evolution.
    tlist : array like
        A NumPy array specifies at which time the next amplitude of
        a pulse is to be applied.
    amps : array like
        A 2d NumPy array of the shape (len(ctrls), len(tlist)). Each
        row corresponds to the control pulse sequence for
        one Hamiltonian.
    paras : dict
        A Python dictionary contains the name and the value of the parameters
        in the physical realization, such as laser freqeuncy, detuning etc.
    """
    def __init__(self, N, correct_global_phase=True, T1=None, T2=None):
        super(ModelProcessor, self).__init__(N, T1=T1, T2=T2)
        self.correct_global_phase = correct_global_phase
        self.global_phase = 0.
        self._paras = {}
        self.ctrls = []

    def _para_list(self, para, N):
        """
        transfer a parameter to list form and multiplied by 2*pi.

        Parameters
        ----------
        para : float or list
            One parameter in the setup.
        N : int
            The number of qubits in the system.

        Returns
        -------
        paras : list
            The given parameter in a list for
        """
        if isinstance(para, numbers.Real):
            return [para * 2 * np.pi] * N
        elif isinstance(para, Iterable):
            return [c * 2 * np.pi for c in para]

    def set_up_paras(self):
        """
        Save the parameters in the attribute `paras` and check the validity.
        (Defined in subclasses)

        Note
        ----
        All parameters will be multiplied by 2*pi for simplicity
        """
        raise NotImplementedError("Parameters should be defined in subclass.")

    @property
    def paras(self):
        return self._paras

    @paras.setter
    def paras(self, par):
        self.set_up_paras(**par)

    def get_ops_and_u(self):
        """
        Returns the Hamiltonian operators and corresponding values by stacking
        them together.

        Returns
        -------
        ctrls : list
            The list of Hamiltonians
        amps : array like
            The transposed pulse matrix
        """
        return (self.ctrls, self.amps.T)

    def get_ops_labels(self):
        """
        Returns the Hamiltonian operators and corresponding labels by stacking
        them together. (Defined in subclasses)
        """
        raise NotImplementedError("Use the function in the sub-class")

    def run(self, qc=None):
        """
        Generates the pulse matrix by running the Hamiltonian for the
        appropriate time duration for the desired physical system
        analytically with U = exp(-iHt). This method won't consider
        any noise.

        Parameters
        ----------
        qc : :class:`qutip.qip.QubitCircuit`
            Takes the quantum circuit to be implemented.

        Returns
        --------
        U_list : list
            The propagator matrix obtained from the physical implementation.
        """
        if qc:
            self.load_circuit(qc)

        U_list = []
        tlist = self.tlist
        for n in range(len(tlist)-1):
            H = sum([self.amps[m, n] * self.ctrls[m]
                    for m in range(len(self.ctrls))])
            dt = tlist[n+1] - tlist[n]
            U = (-1j * H * dt).expm()
            U = self.eliminate_auxillary_modes(U)
            U_list.append(U)

        if self.correct_global_phase and self.global_phase != 0:
            U_list.append(globalphase(self.global_phase, N=self.N))

        return U_list

    def run_state(self, qc=None, rho0=None, states=None,
                  numerical=True, **kwargs):
        """
        Simulate the state evolution under the given `qutip.QubitCircuit`
        If `analytical` is false, it will generate
        the propagator matrix by running U = exp(-iHt)
        If `analytical` is true, it will use mesolve to
        calculate the time of the state evolution
        and return the result. Other arguments of mesolve can be
        given as kwargs.

        Parameters
        ----------
        qc : :class:`qutip.QubitCircuit`
            Takes the quantum circuit to be implemented.
        rho0 : :class:`qutip.Qobj`
            Initial state of the qubits in the register.
        numerical : boolean
            If qutip solver is used to calculate the evolution numerically.
            Noise will only be considered if this is true.
        kwargs
            Key word arguments for the qutip solver.
            (currently `qutip.mesolve`)
        states : :class:`qutip.Qobj`
            Old API, deprecated to be consistent with qutip solver.

        Returns
        --------
        U_list : list
            If the analytical method is used,
            the propagator matrix obtained from the physical implementation.
        evo_result : :class:`qutip.Result`
            If the numerical method is used, the result of the solver will
            be returned.
            An instance of the class :class:`qutip.Result`, which contains
            either an *array* `result.expect` of expectation values for
            the times specified by `tlist`, or an *array* `result.states`
            of state vectors or density matrices corresponding to
            the times in `tlist` [if `e_ops` is
            an empty list], or nothing if a callback function was
            given in place of
            operators for which to calculate the expectation values.
        """
        if rho0 is None and states is None:
            raise ValueError("Qubit state not defined.")
        elif rho0 is None:
            # just to keep the old prameters `states`, it is replaced by rho0
            rho0 = states
        if qc:
            self.load_circuit(qc)
            # TODO It noise exists, give warning
        if numerical:
            return super(ModelProcessor, self).run_state(rho0=rho0, **kwargs)

        U_list = [rho0]
        tlist = self.tlist
        for n in range(len(tlist)-1):
            H = sum([self.amps[m, n] * self.ctrls[m]
                    for m in range(len(self.ctrls))])
            dt = tlist[n+1] - tlist[n]
            U = (-1j * H * dt).expm()
            U = self.eliminate_auxillary_modes(U)
            U_list.append(U)

        if self.correct_global_phase and self.global_phase != 0:
            U_list.append(globalphase(self.global_phase, N=self.N))

        return U_list

    def pulse_matrix(self):
        """
        Generates the pulse matrix for the desired physical system.

        Returns
        --------
        t, u, labels:
            Returns the total time and label for every operation.
        """
        dt = 0.01
        H_ops, H_u = self.get_ops_and_u()

        diff_tlist = self.tlist - np.hstack([[0], self.tlist[:-1]])
        t_tot = sum(diff_tlist)
        n_t = int(np.ceil(t_tot / dt))
        n_ops = len(H_ops)

        t = np.linspace(0, t_tot, n_t)
        u = np.zeros((n_ops, n_t))

        t_start = 0
        for n in range(len(diff_tlist)):

            t_idx_len = int(np.floor(diff_tlist[n] / dt))

            mm = 0
            for m in range(len(H_ops)):
                u[mm, t_start:(t_start + t_idx_len)] = (np.ones(t_idx_len) *
                                                        H_u[n, m])
                mm += 1

            t_start += t_idx_len

        return t, u, self.get_ops_labels()

    def plot_pulses(self):
        """
        Maps the physical interaction between the circuit components for the
        desired physical system.

        Returns
        --------
        fig, ax: Figure
            Maps the physical interaction between the circuit components.
        """
        import matplotlib.pyplot as plt
        t, u, u_labels = self.pulse_matrix()
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        for n, uu in enumerate(u):
            ax.plot(t, u[n], label=u_labels[n])

        ax.axis('tight')
        ax.set_ylim(-1.5 * 2 * np.pi, 1.5 * 2 * np.pi)
        ax.legend(loc='center left',
                  bbox_to_anchor=(1, 0.5), ncol=(1 + len(u) // 16))
        fig.tight_layout()

        return fig, ax


class GateDecomposer(object):
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
    num_ops : int
        Number of Hamiltonians in the processor.

    Attributes
    ----------
    gate_decs : dict
        The Python dictionary in the form {gate_name: decompose_function}.
    """
    def __init__(self, N, paras, num_ops):
        self.gate_decs = {}
        self.N = N
        self.paras = paras
        self.num_ops = num_ops

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

        tlist = np.empty(len(self.dt_list))
        t = 0
        for i in range(len(self.dt_list)):
            t += self.dt_list[i]
            tlist[i] = t
        return np.hstack([[0], tlist]), amps
