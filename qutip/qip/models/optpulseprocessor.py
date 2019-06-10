from collections.abc import Iterable
import warnings
import numbers

import numpy as np
import matplotlib.pyplot as plt

from qutip.qobj import Qobj
import qutip.control.pulseoptim as cpo
from qutip.operators import identity
from qutip.tensor import tensor
from qutip.mesolve import mesolve
from qutip.qip.circuit import QubitCircuit
from qutip.qip.models.circuitprocessor import CircuitProcessor

def expand_oper(oper, N, targets):
    """
    Expand an operator to dimension N acting on the targeting qubits.

    Parameters
    ----------
    oper : `Qobj`
        An operator acts on qubits, the type of the `Qobj` 
        has to be an operator 
        and the dimension matches the tensored qubit Hilbertspace
        e.g. dims = [[2,2,2],[2,2,2]]
    N : int
        The number of qubits in the system.
    targets : int or list of int
        The indices of qubits that are acted on.

    Note
    ----
    This is equivalent to gate_expand_1toN, gate_expand_2toN,
    gate_expand_3toN in `qutip.qip.gate.py`, but works for any dimension.
    """
    req_num = len(oper.dims[0])
    if req_num > N:
        raise ValueError("The dimension of the operator "
            "cannot be larger than the system dimension "
            "N={}.".format(N))

    # Check validity of targets
    if targets is None:
        targets = list(range(len(oper.dims[0])))
    elif isinstance(targets, numbers.Integral):
        targets = [targets]
    elif isinstance(targets, Iterable):  # correct type
        if not all([isinstance(t, numbers.Integral) for t in targets]):
            raise ValueError("targets should be "
                "an integer or a list of integer, but {} "
                "was given.".format(targets))
        if len(targets) != req_num:  # correct number of targets
            raise ValueError("The given Hamiltonian needs {} "
                "target qutbis, "
                "but {} qubits was given.".format(
                    req_num, len(targets)))
    else:
        raise ValueError("targets should be "
            "an integer or a list of integer, but {} "
            "was given.".format(targets))

    # Generate the correct order for qubits permutation, 
    # eg. if N = 5, targets = [3,0], the order is [1,3,2,0,4]. 
    # If the operator is cnot, 
    # this order means that the 3rd qubit control the 0th qubit.
    new_order = list(range(N))
    old_ind = range(len(targets)) 
    for i in old_ind:
        new_order[targets[i]] = i
    old_ind_set = set(old_ind)
    targets_set = set(targets)
    mod_value = targets_set.difference(old_ind_set)
    mod_ind = old_ind_set.difference(targets_set)
    for ind, value in zip(mod_ind, mod_value):
        new_order[ind] = value
    return tensor([oper] + [identity(2)]*(N-len(targets))).permute(new_order)

class OptPulseProcessor(CircuitProcessor):
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
    def __init__(self, N, drift=None, ctrls=[]):
        self.N = N
        self.ctrls = []
        if drift == None:
            self.drift = tensor([identity(2)] * N)
        else:
            self.add_ctrl(drift) # just make use of existing method
            self.drift = self.ctrls.pop(0)
        for H in ctrls:
            self.add_ctrl(H)
        self.tlist = np.empty(0)
        self.amps = np.empty(0)

    def add_ctrl(self, ctrl, targets=None, expand_type=None):
        """
        Add a ctrl Hamiltonian to the processor

        Parameters
        ----------
        ctrl : Qobj
            A hermitian Qobj representation of the driving Hamiltonian
        targets : list of int
            The indices of qubits that are acted on.
        expand_type : string
            The tyoe of expantion
            None - only expand for the given target qubits
            "periodic" - the hamiltonian is to be expanded for 
                all cyclic permutation of target qubits
        """
        # Check validity of ctrl
        if not isinstance(ctrl, Qobj):
            raise TypeError("The Hamiltonian must be a qutip.Qobj.")
        if not ctrl.isherm:
            raise ValueError("The Hamiltonian must be hermitian.")

        d = len(ctrl.dims[0])
        if targets is None:
            targets = list(range(d))

        if expand_type is None: 
            if d == self.N:
                self.ctrls.append(ctrl)
            else:
                self.ctrls.append(expand_oper(ctrl, self.N, targets))
        elif expand_type == "periodic":
            for i in range(self.N):
                new_targets = np.mod(np.array(targets)+i, self.N)
                self.ctrls.append(
                    expand_oper(ctrl, self.N, new_targets))
        else:
            raise ValueError("expand_type can only be None or 'periodic', "
                "not {}".format(expand_type))

    def remove_ctrl(self, indices):
        """
        Remove the ctrl Hamiltonian with given indices

        Parameters
        ----------
        indices : int or list of int
        """
        if not isinstance(indices, Iterable):
            indices = [indices]
        for ind in indices:
            if not isinstance(ind, numbers.Integral):
                raise TypeError("Index must in an integer")
            else:
                del self.ctrls[ind]

    def _is_time_amps_valid(self):
        amps_len = self.amps.shape[1]
        tlist_len = self.tlist.shape[0]
        if amps_len != tlist_len:
            raise ValueError("tlist has length of {} while amps "
                "has {}".format(tlist_len, amps_len))

    def _is_ctrl_amps_valid(self):
        if self.amps.shape[0] != len(self.ctrls):
            raise ValueError("The control amplitudes does not match the " 
            "number of control Hamiltonians")

    def _is_amps_valid(self):
        self._is_time_amps_valid()
        self._is_ctrl_amps_valid()

    def save_amps(self, file_name, inctime=True):
        """
        Save a file with the current control amplitudes in each timeslot

        Parameters
        ----------
        file_name : string
            Name of the file
        inctime : boolean
            True if the time list in included in the first column
        """
        self._is_amps_valid()

        if inctime:
            shp = self.amps.T.shape
            data = np.empty([shp[0], shp[1] + 1], dtype=np.float)
            data[:, 0] = self.tlist
            data[:, 1:] = self.amps.T
        else:
            data = self.amps.T

        np.savetxt(file_name, data, delimiter='\t', fmt='%1.16f')

    def read_amps(self, file_name, inctime=True):
        """
        Read the pulse amplitudes save in the file by `save_amp`

        Parameters
        ----------
        file_name : string
            Name of the file
        inctime : boolean
            True if the time list in included in the first column
        """
        data = np.loadtxt(file_name, delimiter='\t')
        if not inctime:
            self.amps = data.T
        else:
            self.tlist = data[:,0]
            self.amps = data[:,1:].T
        try:
            self._is_amps_valid()
        except Exception as e:
            warnings.warn("{}".format(e))
        return self.tlist, self.amps

    def load_circuit(self, qc, n_ts, evo_time, min_fid_err=np.inf, verbose=False, **kwargs):
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
            one Hmiltonian.
        """
        if isinstance(qc, QubitCircuit):
            props = qc.propagators()
        elif isinstance(qc, Iterable):
            props = qc
        else:
            raise ValueError("qc should be a "
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

            # TODO: different settings for different gates? How?
            result = cpo.optimize_pulse_unitary(self.drift, self.ctrls, U_0, 
                U_targ, n_ts[prop_ind], evo_time[prop_ind], **kwargs)

            # TODO: To prevent repitition, pulse for the same can 
            # be cached and reused?

            if result.fid_err > min_fid_err:
                warnings.warn("The fidelity error of gate {} is higher "
                    "than required limit".format(prop_ind))

            # append time_record to tlist but not time_record[-1] 
            # since len(time_record) = len(amps_record) + 1
            if prop_ind == 0:
                time_record.append(result.time[:-1])
            else:
                time_record.append(result.time[:-1] + last_time)
            
            last_time += result.time[-1]
            amps_record.append(result.final_amps.T)

            if verbose:
                print("********** Gate {} **********".format(prop_ind))
                print("Final fidelity error {}".format(result.fid_err))
                print("Final gradient normal {}".format(result.grad_norm_final))
                print("Terminated due to {}".format(result.termination_reason))
                print("Number of iterations {}".format(result.num_iter))
        
        self.tlist = np.hstack(time_record)
        self.amps = np.hstack(amps_record)
        return self.tlist, self.amps

    def _refine_step(self, tlist, amps, dt):
        """
        Refine the NumPy amplitude. It is useful if the amps
        is given by analytical solution or the step size is 
        too large

        Parameters
        ----------
        amps = 
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

    def run_state(self, rho0, dt=None, **kwargs):
        """
        Use mesolve to calculate the time of the state evolution 
        and return the result. Other arguments of mesolve can be 
        given as kwargs.

        Parameters
        ----------
        rho0 : Qobj
            Initial density matrix or state vector (ket).
        dt : float
            Time step for mesolve
        **kwargs
            Key word arguments for `mesolve`

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
        self._is_amps_valid()
        if self.tlist.size==0:
            return rho0
        
        # refine the time slice for mesolve with time step dt
        if dt!=None: 
            tlist, amps = self._refine_step(self.tlist, self.amps, dt)
        else:
            tlist, amps = self.tlist, self.amps

        if self.drift is not None:
            H = [self.drift]
        else:
            H = []
        for i in range(len(self.ctrls)):
            H.append([self.ctrls[i], amps[i]])
        
        evo_result = mesolve(H, rho0, tlist, **kwargs)
        # TODO: Is it possible to only save the last state instead of all?
        return evo_result

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
            The Figure object for the plot.
        ax : matplotlib.axes._subplots.AxesSubplot
            The axes for the plot.
        """
        fig, ax = plt.subplots(1, 1, **kwargs)
        ax.set_ylabel("Control amplitude")
        ax.set_xlabel("Time")
        tlist = np.hstack([self.tlist, self.tlist[-1:0]])
        amps = np.hstack([self.amps, self.amps[:,-1:0]])
        for i in range(self.amps.shape[0]):
            ax.step(tlist, amps[i], where = 'post')
        fig.tight_layout()
        return fig, ax