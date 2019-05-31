from collections.abc import Iterable
import warnings

import numpy as np
import matplotlib.pyplot as plt

import qutip.control.pulseoptim as cpo
from qutip.operators import identity
from qutip.tensor import tensor
from qutip.mesolve import mesolve

class OptPulseProcessor(object):
    """
    A circuit processr, which takes the Hamiltonian available 
    as dynamic generators, calls the `optimize_pulse` function 
    to find an optimized pulse sequence. The proessor can then
    calculate the state evolution under this defined dynamics 
    using 'mesolve'.

    Parameters
    ----------
    N : int
        The number of qubits in the system.
    drift : Qobj
        The drift Hamiltonian with no time dependent amplitude.
    ctrls : list of Qobj
        The control Hamiltonian whose amplitude will be optimized.

    Attributes
    ----------
    tlist : array like
        A NumPy array specifies the time steps.
    amps : array like
        A 2d NumPy array of the shape (len(ctrls), len(tlist)). Each 
        row corresponds to the control pulse sequence for 
        one Hmiltonian.
    """
    def __init__(self, N, H_d, H_c):
        # TODO: drift can be time dependent with fixed amps
        self.N = N
        self.H_d = H_d
        self.H_c = H_c
        self.tlist = []
        self.amps = []

    def load_circuit(self, qc, **kwargs):
        """
        Translates an abstract quantum circuit to its corresponding 
        Hamiltonian with `optimize_pulse_unitary`.

        Parameters
        ----------
        qc : QubitCircuit
            The quantum circuit to be translated.
        **kwargs
            Key word arguments for `optimize_pulse_unitary`

        Returns
        -------
        tlist : array like
            A NumPy array specifies the time steps.
        amps : array like
            A 2d NumPy array of the shape (len(ctrls), len(tlist)). Each 
            row corresponds to the control pulse sequence for 
            one Hmiltonian.
        """
        prop = qc.propagators()
        time_record = []  # a list for all the gates
        amps_record = []
        last_time = 0.  # used in concatenation of tlist
        for gate_ind in range(len(prop)):
            U_targ = prop[gate_ind]
            U_0 = identity(U_targ.dims[0])

            # TODO: different settings for different gate type?
            # Number of time slots
            n_ts = 10
            # Time allowed for the evolution
            evo_time = 10
            
            # TODO: choose only those H that relevant for this gate
            result = cpo.optimize_pulse_unitary(self.H_d, self.H_c, U_0, 
                U_targ, n_ts, evo_time, **kwargs)

            # TODO: To prevent repitition, pulse for certain gates should 
            # be recorded in a dictionary (or maybe only the position in 
            # self.amps)

            # TODO: Give warning if the find pulse fails 
            # useful data:
            # result.termination_reason
            # result.fidelity
            # result.goal_achieved
            # result.final_amps
            # result.evo_full_final
            if gate_ind == 0:
                time_record.append(result.time[:-1])
            else:
                time_record.append(result.time[:-1] + last_time)
            
            last_time += result.time[-1]
            amps_record.append(result.final_amps.T)
        
        self.tlist = np.hstack(time_record)
        self.amps = np.hstack(amps_record)
        return self.tlist, self.amps

    def _denser_step(self, tlist, amps, dt):
        origin_dt = tlist[1:]-tlist[:-1]
        if max(origin_dt) == min(origin_dt):
            repeat_num = np.floor(origin_dt[0]/dt)
            new_dt = origin_dt[0]/repeat_num
            tlist = np.arange(tlist[0], tlist[-1]+origin_dt[0], new_dt)
            amps = np.repeat(amps, repeat_num, axis=1)
            return tlist, amps
        raise NotImplementedError("Works only for tlist with equal distance.")

    def run_state(self, state, dt=None, **kwargs):
        """
        Use mesolve to calculate the evolution.

        Parameters
        ----------
        rho0 : Qobj
            Initial density matrix or state vector (ket).

        dt : Qobj
            Time step for mesolve

        Returns
        -------
        evo_result: :class:`qutip.Result`
            An instance of the class :class:`qutip.Result`, which contains
            either an *array* `result.expect` of expectation values for 
            the times specified by `tlist`, or an *array* `result.states` 
            of state vectors or density matrices corresponding to 
            the times in `tlist` [if `e_ops` is
            an empty list], or nothing if a callback function was 
            given in place of
            operators for which to calculate the expectation values.
        """
        # tlist or amps is empty. This can happend when user loads their 
        # own Hamiltoian without calling self.load_circuit.
        if self.tlist.size==0 or self.amps.size==0:
            return state
        
        # refine the time slice for mesolve
        if dt!=None: 
            tlist, amps = self._denser_step(self.tlist, self.amps, dt)
        else:
            tlist, amps = self.tlist, self.amps

        H = [self.H_d]
        for i in range(len(self.H_c)):
            H.append([self.H_c[i], amps[i]])
        
        evo_result = mesolve(H, state, tlist, **kwargs)
        # TODO: Is it possible to only save the last state instead of all?
        return evo_result

    def plot_pulses(self):
        """
        Plot the pulse amplitude

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object for the plot.
        ax : matplotlib.axes._subplots.AxesSubplot
            The axes for the plot.
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.set_ylabel("Control amplitude")
        ax.set_xlabel("Time")
        y_line = np.hstack([self.amps[:,0:1], self.amps])
        for i in range(y_line.shape[0]):
            ax.step(self.tlist, y_line[i])
        fig.tight_layout()
        return fig, ax