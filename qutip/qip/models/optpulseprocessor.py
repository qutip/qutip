from collections.abc import Iterable
import warnings
import numbers

import numpy as np
import matplotlib.pyplot as plt

from qutip.qobj import Qobj
import qutip.control.pulseoptim as cpo
from qutip.operators import identity, sigmax, sigmaz, destroy
from qutip.qip.gates import expand_oper
from qutip.tensor import tensor
from qutip.mesolve import mesolve
from qutip.qip.circuit import QubitCircuit
from qutip.qip.models.circuitprocessor import CircuitProcessor


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
        A NumPy array specifies the time steps.
    amps : array like
        A 2d NumPy array of the shape (len(ctrls), len(tlist)). Each
        row corresponds to the control pulse sequence for
        one Hamiltonian.
    """
    def __init__(self, N, drift=None, ctrls=None, T1=None, T2=None):
        super(OptPulseProcessor, self).__init__(N, T1=T1, T2=T2)
        if drift is None:
            self.ctrls.append(tensor([identity(2)] * N))
        else:
            self.add_ctrl(drift)
        if ctrls is not None:
            for H in ctrls:
                self.add_ctrl(H)

    @property
    def drift(self):
        return self.ctrls[0]

    def remove_ctrl(self, indices):
        """
        Remove the ctrl Hamiltonian with given indices

        Parameters
        ----------
        indices : int or list of int
        """
        if isinstance(indices, numbers.Integral) and indices != 0:
            pass
        elif isinstance(indices, Iterable) and 0 not in indices:
            pass
        else:
            raise ValueError("ctrls 0 is the "
                "drift Ham and cannot be removed")
        super(OptPulseProcessor, self).remove_ctrl(indices)

    def load_circuit(
            self, qc, n_ts, evo_time,
            min_fid_err=np.inf, verbose=False, **kwargs):
        """
        Translates an abstract quantum circuit to its corresponding
        Hamiltonian with `optimize_pulse_unitary`.

        Parameters
        ----------
        qc : QubitCircuit or list of Qobj
            The quantum circuit to be translated.
        n_ts : int or list
            The number of time steps for each gate in `qc`
        evo_time : int or list
            The allowed evolution time for each gate in `qc`
        **kwargs
            Key word arguments for `optimize_pulse_unitary`

        Returns
        -------
        tlist : array like
            A NumPy array specifies the time steps.
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
                self.ctrls[0], self.ctrls[1:], U_0,
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

        self.tlist = np.hstack(time_record)
        self.amps = np.vstack([np.ones(len(self.tlist)), np.hstack(amps_record)])
        return self.tlist, self.amps

    def _refine_step(self, tlist, amps, dt):
        """
        Refine the NumPy amplitude. It is useful if the amps
        is given by analytical solution or the step size is
        too large.
        """
        # For now it only works if tlist is equidistant
        origin_dt = tlist[1:]-tlist[:-1]
        if (max(origin_dt)-min(origin_dt)) < 1.0e-8:
            repeat_num = np.floor(origin_dt[0]/dt)
            new_dt = origin_dt[0]/repeat_num
            tlist = np.arange(tlist[0], tlist[-1]+origin_dt[0], new_dt)
            amps = np.repeat(amps, repeat_num, axis=1)
            return tlist, amps
        else:
            raise NotImplementedError(
                "Works only for tlist with equal distance.")

    def plot_pulses(self, **kwargs):
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
        """
        fig, ax = plt.subplots(1, 1, **kwargs)
        ax.set_ylabel("Control amplitude")
        ax.set_xlabel("Time")
        amps = np.hstack([self.amps, self.amps[:, -1:]])
        for i in range(1, self.amps.shape[0]):
            ax.step(np.hstack([[0],self.tlist]), amps[i], where='post')
        fig.tight_layout()
        return fig, ax
