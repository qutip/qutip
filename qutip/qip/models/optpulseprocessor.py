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
import numbers

import numpy as np

from qutip.qobj import Qobj
import qutip.control.pulseoptim as cpo
from qutip.operators import identity
from qutip.tensor import tensor
from qutip.mesolve import mesolve
from qutip.qip.circuit import QubitCircuit
from qutip.qip.models.circuitprocessor import CircuitProcessor


__all__ = ['OptPulseProcessor']


class OptPulseProcessor(CircuitProcessor):
    """
    A circuit processor, which takes the Hamiltonian available
    as dynamic generators, calls the `optimize_pulse` function
    to find an optimized pulse sequence. The processor can then
    calculate the state evolution under this defined dynamics
    using 'mesolve'.

    Parameters
    ----------
    N : int
        The number of qubits in the system.
    drift : :class:`Qobj`
        The drift Hamiltonian with no time-dependent amplitude.
    ctrls : list of :class:`Qobj`
        The control Hamiltonian whose amplitude will be optimized.
    T1 : list or float
        Characterize the decoherence of amplitude damping for
        each qubit. A list of size N or a float for all qubits
    T2 : list of float
        Characterize the decoherence of dephase relaxation for
        each qubit. A list of size N or a float for all qubits

    Attributes
    ----------
    tlist : array like
        A NumPy array specifies at which time the next amplitude of
        a pulse is to be applied.
    amps : array like
        A 2d NumPy array of the shape (len(ctrls), len(tlist)). Each
        row corresponds to the control pulse sequence for
        one Hamiltonian.
    """
    def __init__(self, N, drift=None, ctrls=None, T1=None, T2=None, dims=None):
        super(OptPulseProcessor, self).__init__(N, T1=T1, T2=T2, dims=dims)
        if drift is None:
            self.drift = tensor([identity(self.dims[i]) for i in range(N)])
        else:
            self.drift = drift
        if ctrls is not None:
            for H in ctrls:
                self.add_ctrl(H)

    def load_circuit(
            self, qc, n_ts, evo_time,
            min_fid_err=np.inf, verbose=False, **kwargs):
        """
        Translates an abstract quantum circuit to its corresponding
        Hamiltonian with `optimize_pulse_unitary`.

        Parameters
        ----------
        qc : :class:`qutip.QubitCircuit` or list of Qobj
            The quantum circuit to be translated.
        n_ts : int or list
            The number of time steps for each gate in `qc`
        evo_time : int or list
            The allowed evolution time for each gate in `qc`
        min_fid_err : float
            The minimal fidelity tolerance, if the fidelity error of any
            gate decomposition is higher, a warning will be given.
        verbose : boolean
            If true, the information for each decomposed gate
            will be shown.
        kwargs
            Key word arguments for `qutip.optimize_pulse_unitary`

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
        if isinstance(qc, QubitCircuit):
            props = qc.propagators()
        elif isinstance(qc, Iterable):
            props = qc
        else:
            raise ValueError(
                "qc should be a "
                "QubitCircuit or a list of Qobj")
        if isinstance(n_ts, numbers.Integral):
            n_ts = [n_ts] * len(props)
        if isinstance(evo_time, numbers.Integral):
            evo_time = [evo_time] * len(props)

        time_record = []  # a list for all the gates
        amps_record = []
        last_time = 0.  # used in concatenation of tlist
        for prop_ind in range(len(props)):
            U_targ = props[prop_ind]
            U_0 = identity(U_targ.dims[0])

            # TODO: different settings for different oper in qc? How?
            result = cpo.optimize_pulse_unitary(
                self.drift, self.ctrls, U_0,
                U_targ, n_ts[prop_ind], evo_time[prop_ind], **kwargs)

            # TODO: To prevent repitition, pulse for the same oper can
            # be cached and reused?

            if result.fid_err > min_fid_err:
                warnings.warn(
                    "The fidelity error of gate {} is higher "
                    "than required limit".format(prop_ind))

            time_record.append(result.time[1:] + last_time)
            last_time += result.time[-1]
            amps_record.append(result.final_amps.T)

            if verbose:
                print("********** Gate {} **********".format(prop_ind))
                print("Final fidelity error {}".format(result.fid_err))
                print("Final gradient normal {}".format(
                                                result.grad_norm_final))
                print("Terminated due to {}".format(result.termination_reason))
                print("Number of iterations {}".format(result.num_iter))
        self.tlist = np.hstack([[0.]]+time_record)
        self.amps = np.vstack(
            [np.hstack(amps_record)])
        return self.tlist, self.amps

    def get_qobjevo(self, **kwargs):
        proc_qobjevo = super(OptPulseProcessor, self).get_qobjevo(
            **kwargs)
        if self.drift is not None:
            return proc_qobjevo + self.drift
        else:
            return proc_qobjevo

    def plot_pulses(self, **kwargs):
        """
        Plot the pulse amplitude, the constant drift Hamiltonian is not shown.

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
        """
        # first row of amps is for drift Ham
        super(OptPulseProcessor, self).plot_pulses(
            amps=self.amps[1:], tlist=self.tlist)
