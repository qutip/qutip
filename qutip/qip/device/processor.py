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
import warnings
from copy import deepcopy

import numpy as np
from scipy.interpolate import CubicSpline

from qutip.qobj import Qobj
from qutip.qobjevo import QobjEvo
from qutip.operators import identity
from qutip.qip.gates import expand_operator, globalphase
from qutip.tensor import tensor
from qutip.mesolve import mesolve
from qutip.qip.circuit import QubitCircuit
from qutip.qip.noise import (
    Noise, RelaxationNoise, DecoherenceNoise,
    ControlAmpNoise, RandomNoise, UserNoise, process_noise)
from qutip.qip.pulse import Pulse, Drift, _merge_qobjevo


__all__ = ['Processor']


class Processor(object):
    """
    A simulator of a quantum device based on the QuTiP solver
    :func:`qutip.mesolve`.
    It is defined by the available driving Hamiltonian and
    the decoherence time for each component systems.
    The processor can simulate the evolution under the given
    control pulses. Noisy evolution is supported by
    :class:`qutip.qip.Noise` and can be added to the processor.

    Parameters
    ----------
    N: int
        The number of component systems.

    ctrls: list of :class:`Qobj`
        A list of the control Hamiltonians driving the evolution.

    t1: list or float
        Characterize the decoherence of amplitude damping for
        each qubit. A list of size ``N`` or a float for all qubits.

    t2: list of float
        Characterize the decoherence of dephasing for
        each qubit. A list of size ``N`` or a float for all qubits.

    dims: list
        The dimension of each component system.
        Default value is a
        qubit system of ``dim=[2,2,2,...,2]``

    spline_kind: str, optional
        Type of the coefficient interpolation. Default is ``step_func``
        Note that they have different requirement for the length of ``coeffs``.

        -"step_func":
        The coefficient will be treated as a step function.
        E.g. ``tlist=[0,1,2]`` and ``coeffs=[3,2]``, means that the coefficient
        is 3 in t=[0,1) and 2 in t=[2,3). It requires
        ``coeffs.shape[1]=len(tlist)-1`` or ``coeffs.shape[1]=len(tlist)``, but
        in the second case the last element has no effect.

        -"cubic": Use cubic interpolation for the coefficient. It requires
        ``coeffs.shape[1]=len(tlist)``

    Attributes
    ----------
    N: int
        The number of component systems.

    ctrls: list
        A list of the control Hamiltonians driving the evolution.

    tlist: array_like
        A NumPy array specifies the time of each coefficient.

    coeffs: array_like
        A 2d NumPy array of the shape, the length is dependent on the
        spline type

    t1: list
        Characterize the decoherence of amplitude damping for
        each qubit.

    t2: list
        Characterize the decoherence of dephasing for
        each qubit.

    noise: :class:`qutip.qip.Noise`, optional
        A list of noise objects. They will be processed when creating the
        noisy :class:`qutip.QobjEvo` from the processor or run the simulation.

    dims: list
        The dimension of each component system.
        Default value is a
        qubit system of ``dim=[2,2,2,...,2]``

    spline_kind: str
        Type of the coefficient interpolation.
        "step_func" or "cubic"
    """
    def __init__(self, N, t1=None, t2=None,
                 dims=None, spline_kind="step_func"):
        self.N = N
        self.tlist_ = None
        self.ctrl_pulses = []
        self.t1 = t1
        self.t2 = t2
        self.noise = []
        self.drift = Drift()
        if dims is None:
            self.dims = [2] * N
        else:
            self.dims = dims
        self.spline_kind = spline_kind

    def add_drift_ham(self, ham, targets=None):
        # Check validity of ctrl
        if not isinstance(ham, Qobj):
            raise TypeError("The drift Hamiltonian must be a qutip.Qobj.")
        if not ham.isherm:
            raise ValueError("The drift Hamiltonian must be Hermitian.")
        
        num_qubits = len(ham.dims[0])
        if targets is None:
            targets = list(range(num_qubits))
        self.drift.add_ham(ham, targets)

    def add_ctrl_ham(self, ham, targets=None, cyclic_permutation=False):
        """
        Add a control Hamiltonian to the processor.

        Parameters
        ----------
        ctrl: :class:`qutip.Qobj`
            The control Hamiltonian to be added.

        targets: list or int, optional
            The indices of qubits that are acted on.

        cyclic_permutation: boolean, optional
            If true, the Hamiltonian will be expanded for
            all cyclic permutation of the target qubits.
        """
        # Check validity of ctrl
        if not isinstance(ham, Qobj):
            raise TypeError("The control Hamiltonian must be a qutip.Qobj.")
        if not ham.isherm:
            raise ValueError("The control Hamiltonian must be Hermitian.")

        num_qubits = len(ham.dims[0])
        if targets is None:
            targets = list(range(num_qubits))
        if not isinstance(targets, list):
            targets = [targets]
        if cyclic_permutation:
            for i in range(self.N):
                temp = [(t + i)%self.N for t in targets]
                self.ctrl_pulses.append(Pulse(ham, temp, spline_kind=self.spline_kind))
        else:
            self.ctrl_pulses.append(Pulse(ham, targets, spline_kind=self.spline_kind))

    @property
    def ctrls(self):
        result = []
        for pulse in self.ctrl_pulses:
            result.append(pulse.get_ideal_qobj(self.dims))
        return result

    @property
    def coeffs(self):
        if not self.ctrl_pulses:
            return None
        coeffs_list = [pulse.coeff for pulse in self.ctrl_pulses]
        return np.array(coeffs_list)

    @coeffs.setter
    def coeffs(self, coeffs_list):
        # check number of pulse
        if len(coeffs_list) != len(self.ctrl_pulses):
            raise ValueError("The rwo number of coeffs must be same "
                             "as the number of control pulses.")
        for i, coeff in enumerate(coeffs_list):
            self.ctrl_pulses[i].coeff = coeff

    @property
    def tlist(self):
        all_tlists = [pulse.tlist for pulse in self.ctrl_pulses if pulse.tlist is not None]
        if self.tlist_ is not None:
            all_tlists += [self.tlist_]
        if not all_tlists:
            return None
        return np.unique(np.sort(np.hstack(all_tlists)))

    @tlist.setter
    def tlist(self, x):
        self.tlist_ = x

    def remove_ctrl(self, indices):
        """
        Remove the control Hamiltonian with given indices.

        Parameters
        ----------
        indices: int or list of int
            The indices of the control Hamiltonians to be removed.
        """
        if not isinstance(indices, Iterable):
            indices = [indices]
        indices.sort(reverse=True)
        for ind in indices:
            del self.ctrl_pulses[ind]

    def _is_pulses_valid(self):
        """
        Check if the pulses are in the correct shape.

        Returns: boolean
            If they are valid or not
        """
        for i, pulse in enumerate(self.ctrl_pulses):
            if pulse.tlist is None:
                if pulse.coeff is None:
                    continue
                else:
                    raise ValueError("Pulse id={} is invalid, "
                                     "a tlist is required.".format(i))
            if pulse.tlist is not None and pulse.coeff is None:
                continue
            coeff_len = len(pulse.coeff)
            tlist_len = len(pulse.tlist)
            if pulse.spline_kind == "step_func":
                if coeff_len == tlist_len-1 or coeff_len == tlist_len:
                    pass
                else:
                    raise ValueError(
                        "The length of tlist and coeffs in pulse {} is not valid. "
                        "It's either len(tlist)=len(coeff) or "
                        "len(tlist)-1=len(coeff) for coefficients "
                        "as step function".format(i))
            elif pulse.spline_kind == "cubic":
                if coeff_len == tlist_len:
                    pass
                else:
                    raise ValueError(
                        "The length of tlist and coeffs pulse {} is not valid. "
                        "It should be either len(tlist)=len(coeff)".format(i))
            else:
                raise ValueError("Unknown spline_kind.")
        return True

    def save_coeff(self, file_name, inctime=True):
        """
        Save a file with the control amplitudes in each timeslot.

        Parameters
        ----------
        file_name: string
            Name of the file.

        inctime: boolean, optional
            True if the time list should be included in the first column.
        """
        self._is_pulses_valid()

        if inctime:
            shp = self.coeffs.T.shape
            data = np.empty([shp[0], shp[1] + 1], dtype=np.float)
            data[:, 0] = self.tlist[1:]
            data[:, 1:] = self.coeffs.T
        else:
            data = self.coeffs.T

        np.savetxt(file_name, data, delimiter='\t', fmt='%1.16f')

    def read_coeff(self, file_name, inctime=True):
        """
        Read the control amplitudes matrix and time list
        saved in the file by `save_amp`.

        Parameters
        ----------
        file_name: string
            Name of the file.

        inctime: boolean, optional
            True if the time list in included in the first column.

        Returns
        -------
        tlist: array_like
            The time list read from the file.

        coeffs: array_like
            The pulse matrix read from the file.
        """
        data = np.loadtxt(file_name, delimiter='\t')
        if not inctime:
            self.coeffs = data.T
        else:
            self.tlist = np.hstack([[0], data[:, 0]])
            self.coeffs = data[:, 1:].T
        return self.tlist, self.coeffs

    def get_dynamics(self, args=None, noisy=False):
        """
        Create a :class:`qutip.QobjEvo` without any noise that can be given to
        the QuTiP open system solver.

        Parameters
        ----------
        args: dict, optional
            Arguments for :class:`qutip.QobjEvo`

        Returns
        -------
        ideal_qobjevo: :class:`qutip.QobjEvo`
            The :class:`qutip.QobjEvo` representation of the unitary evolution.
        """
        # check validity
        self._is_pulses_valid()

        if args is None:
            args = {}
        else:
            args = args
        # set step function

        if not noisy:
            dynamics = self.ctrl_pulses
            c_ops = []
        else:
            dynamics, c_ops = process_noise(self.ctrl_pulses, self.noise, self.N, self.dims, t1=self.t1, t2=self.t2)
        dynamics = [self.drift] + dynamics

        qu_list = []
        for pulse in dynamics:
            if noisy:
                qu, new_c_ops = pulse.get_full_evo(dims=self.dims)
                c_ops += new_c_ops
            else:
                qu = pulse.get_ideal_evo(dims=self.dims)
            qu_list.append(qu)
        
        final_qu = _merge_qobjevo(qu_list)
        final_qu.args.update(args)

        if final_qu.tlist is None:
            # No time specified in Pulse, e.g. evolution under constant pulse.
            final_qu.tlist = self.tlist

        if noisy:
            return final_qu, c_ops
        else:
            return final_qu

    # def get_noisy_coeffs(self):
    #     """
    #     Create the array_like coefficients including the noise.

    #     Returns
    #     -------
    #     coeff_list: list
    #         A list of coefficient for each control Hamiltonian.

    #     tlist: np.array
    #         Time array for the coefficient.

    #     Notes
    #     -----
    #     Collapse operators are not included in this method,
    #     please use :meth:`qutip.qip.processor.get_full_evo`
    #     if they are needed.
    #     """
    #     fine_tlist = self.tlist
    #     noisy_pulses, c_ops = self._process_noise(self.ctrl_pulses)
    #     coeffs_list = []
    #     for pulse in noisy_pulses:
    #         coeff = _fill_coeff(pulse.ideal_pulse.coeff, pulse.ideal_pulse.tlist, fine_tlist, args={"_step_func_coeff": True})
    #         coeffs_list.append(coeff)

    #     ops_list = [(pulse.op, pulse.targets) for pulse in self.ctrl_pulses]
    #     print(coeffs_list)
    #     for pulse in noisy_pulses:
    #         for ele in pulse.coherent_noise:
    #             for p, (op, targets) in enumerate(ops_list):
    #                 print(targets,ele.targets)
    #                 if op==ele.op and targets==ele.targets:
    #                     coeffs_list[p] += _fill_coeff(ele.coeff, ele.tlist, fine_tlist, args={"_step_func_coeff": True})
    #     return coeffs_list, fine_tlist


    def run_analytically(self, rho0=None, qc=None):
        """
        Simulate the state evolution under the given `qutip.QubitCircuit`
        with matrice exponentiation. It will calculate the propagator
        with matrix exponentiation and return a list of :class:`qutip.Qobj`.
        This method won't include noise or collpase.

        Parameters
        ----------
        qc: :class:`qutip.qip.QubitCircuit`, optional
            Takes the quantum circuit to be implemented. If not given, use
            the quantum circuit saved in the processor by ``load_circuit``.

        rho0: :class:`qutip.Qobj`, optional
            The initial state of the qubits in the register.

        Returns
        -------
        evo_result: :class:`qutip.Result`
            An instance of the class
            :class:`qutip.Result` will be returned.
        """
        if rho0 is not None:
            U_list = [rho0]
        else:
            U_list = []
        tlist = self.tlist
        for n in range(len(tlist)-1):
            H = sum([self.coeffs[m, n] * self.ctrls[m]
                    for m in range(len(self.ctrls))])
            dt = tlist[n + 1] - tlist[n]
            U = (-1j * H * dt).expm()
            U = self.eliminate_auxillary_modes(U)
            U_list.append(U)

        try:  # correct_global_phase are defined for ModelProcessor
            if self.correct_global_phase and self.global_phase != 0:
                U_list.append(globalphase(self.global_phase, N=self.N))
        except AttributeError:
            pass

        return U_list

    def run(self, qc=None):
        """
        Calculate the propagator of the evolution by matrix exponentiation.
        This method won't include noise or collpase.

        Parameters
        ----------
        qc: :class:`qutip.qip.QubitCircuit`, optional
            Takes the quantum circuit to be implemented. If not given, use
            the quantum circuit saved in the processor by `load_circuit`.

        Returns
        -------
        U_list: list
            The propagator matrix obtained from the physical implementation.
        """
        if qc:
            self.load_circuit(qc)
        return self.run_analytically(qc=qc, rho0=None)

    def run_state(self, rho0=None, analytical=False, states=None,
                  **kwargs):
        """
        If `analytical` is False, use :func:`qutip.mesolve` to
        calculate the time of the state evolution
        and return the result. Other arguments of mesolve can be
        given as keyword arguments.
        If `analytical` is True, calculate the propagator
        with matrix exponentiation and return a list of matrices.

        Parameters
        ----------
        rho0: Qobj
            Initial density matrix or state vector (ket).

        analytical: boolean
            If True, calculate the evolution with matrices exponentiation.

        states: :class:`qutip.Qobj`, optional
            Old API, same as rho0.

        **kwargs
            Keyword arguments for the qutip solver.

        Returns
        -------
        evo_result: :class:`qutip.Result`
            If ``analytical`` is False,  an instance of the class
            :class:`qutip.Result` will be returned.

            If ``analytical`` is True, a list of matrices representation
            is returned.
        """
        if states is not None:
            warnings.warn("states will be deprecated and replaced by rho0"
                          "to be consistent with the QuTiP solver.",
                          DeprecationWarning)
        if rho0 is None and states is None:
            raise ValueError("Qubit state not defined.")
        elif rho0 is None:
            # just to keep the old prameters `states`, it is replaced by rho0
            rho0 = states
        if analytical:
            if kwargs or self.noise:
                raise ValueError("Analytical matrices exponentiation"
                                 "cannot process noise or"
                                 "keyword arguments.")
            return self.run_analytically(rho0=rho0)

        # kwargs can not contain H or tlist
        if "H" in kwargs or "tlist" in kwargs:
            raise ValueError(
                "`H` and `tlist` are already specified by the processor "
                "and can not be given as a keyword argument")

        # construct qobjevo for unitary evolution
        if "args" in kwargs:
            noisy_qobjevo, sys_c_ops = self.get_dynamics(args=kwargs["args"], noisy=True)
        else:
            noisy_qobjevo, sys_c_ops = self.get_dynamics(noisy=True)

        # add noise into kwargs
        if "c_ops" in kwargs:
            if isinstance(kwargs["c_ops"], (Qobj, QobjEvo)):
                kwargs["c_ops"] += [kwargs["c_ops"]] + sys_c_ops
            else:
                kwargs["c_ops"] += sys_c_ops
        else:
            kwargs["c_ops"] = sys_c_ops

        evo_result = mesolve(
            H=noisy_qobjevo, rho0=rho0, tlist=noisy_qobjevo.tlist, **kwargs)
        return evo_result

    def optimize_circuit(self, qc):
        """
        Take a quantum circuit/algorithm and convert it into the
        optimal form/basis for the desired physical system.
        (Defined in subclasses)
        """
        raise NotImplementedError("Use the function in the sub-class")

    def load_circuit(self, qc):
        """
        Translate an :class:`qutip.qip.QubitCircuit` to its
        corresponding Hamiltonians. (Defined in subclasses)
        """
        raise NotImplementedError("Use the function in the sub-class")

    def get_ops_and_u(self):
        """
        Return the Hamiltonian operators and the pulse matrix.
        (Defined in subclasses)
        """
        raise NotImplementedError("Use the function in the sub-class")

    def eliminate_auxillary_modes(self, U):
        """
        Eliminate the auxillary modes like the cavity modes in cqed.
        (Defined in subclasses)
        """
        return U

    def get_ops_labels(self):
        """
        Returns the Hamiltonian operators and corresponding labels by stacking
        them together. (Defined in subclasses)
        """
        raise NotImplementedError("Use the function in the sub-classes")

    def plot_pulses(self, title=None, figsize=None, dpi=None):
        """
        Plot the pulse amplitude

        Parameters
        ----------
        noisy: boolean, optional
            If true, plot the noisy pulses.

        title: str
            Title for the plot.

        figsize: tuple
            The size of the figure

        dpi: int
            The dpi of the figure

        Returns
        -------
        fig: matplotlib.figure.Figure
            The `Figure` object for the plot.

        ax: matplotlib.axes._subplots.AxesSubplot
            The axes for the plot.

        Notes
        -----
        ``plot_pulses`` only works for array_like coefficients
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        ax.set_ylabel("Control pulse amplitude")
        ax.set_xlabel("Time")
        if noisy:
            coeffs, tlist = self.get_noisy_coeffs()
        else:
            coeffs = [coeff for coeff in self.coeffs]
            tlist = self.tlist

        for i in range(len(coeffs)):
            if not isinstance(coeffs[i], (Iterable, np.ndarray)):
                raise ValueError(
                    "plot_pulse only accepts array_like coefficients.")
            if self.spline_kind == "step_func":
                if len(coeffs[i]) == len(tlist) - 1:
                    coeffs[i] = np.hstack(
                        [self.coeffs[i], self.coeffs[i, -1:]])
                else:
                    coeffs[i][-1] = coeffs[i][-2]
                ax.step(tlist, coeffs[i], where='post')
            elif self.spline_kind == "cubic":
                sp = CubicSpline(tlist, coeffs[i])
                t_line = np.linspace(tlist[0], tlist[-1], 200)
                c_line = [sp(t) for t in t_line]
                ax.plot(t_line, c_line)
        if title is not None:
            ax.set_title(title)
        fig.tight_layout()
        return fig, ax

    def add_noise(self, noise):
        """
        Add a noise object to the processor

        Parameters
        ----------
        noise: :class:`qutip.qip.Noise`
            The noise object defined outside the processor
        """
        if isinstance(noise, Noise):
            self.noise.append(noise)
        else:
            raise TypeError("Input is not a Noise object.")



def _dummy_qobjevo(dims, **kwargs):
    """
    Create a dummy :class":`qutip.QobjEvo` with
    a constant zero Hamiltonian. This is used since empty QobjEvo
    is not yet supported.
    """
    dummy = QobjEvo(tensor([identity(d) for d in dims]) * 0., **kwargs)
    return dummy
