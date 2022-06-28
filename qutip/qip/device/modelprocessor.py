from collections.abc import Iterable
import numbers

import numpy as np

from qutip.qobj import Qobj
from qutip.qobjevo import QobjEvo
from qutip.qip.operations.gates import globalphase
from qutip.tensor import tensor
from qutip.mesolve import mesolve
from qutip.qip.circuit import QubitCircuit
from qutip.qip.device.processor import Processor


__all__ = ['ModelProcessor']


class ModelProcessor(Processor):
    """
    The base class for a circuit processor simulating a physical device,
    e.g cavityQED, spinchain.
    The available Hamiltonian of the system is predefined.
    The processor can simulate the evolution under the given
    control pulses either numerically or analytically.
    It cannot be used alone, please refer to the sub-classes.
    (Only additional attributes are documented here, for others please
    refer to the parent class :class:`.Processor`)

    Parameters
    ----------
    N: int
        The number of component systems.

    correct_global_phase: boolean, optional
        If true, the analytical solution will track the global phase. It
        has no effect on the numerical solution.

    t1: list or float
        Characterize the decoherence of amplitude damping for
        each qubit. A list of size `N` or a float for all qubits.

    t2: list of float
        Characterize the decoherence of dephasing for
        each qubit. A list of size `N` or a float for all qubits.

    Attributes
    ----------
    params: dict
        A Python dictionary contains the name and the value of the parameters
        in the physical realization, such as laser frequency, detuning etc.

    correct_global_phase: float
        Save the global phase, the analytical solution
        will track the global phase.
        It has no effect on the numerical solution.
    """
    def __init__(self, N, correct_global_phase=True, t1=None, t2=None):
        super(ModelProcessor, self).__init__(N, t1=t1, t2=t2)
        self.correct_global_phase = correct_global_phase
        self.global_phase = 0.
        self._params = {}

    def to_array(self, params, N):
        """
        Transfer a parameter to an array.
        """
        if isinstance(params, numbers.Real):
            return np.asarray([params] * N)
        elif isinstance(params, Iterable):
            return np.asarray(params)

    def set_up_params(self):
        """
        Save the parameters in the attribute `params` and check the validity.
        (Defined in subclasses)

        Notes
        -----
        All parameters will be multiplied by 2*pi for simplicity
        """
        raise NotImplementedError("Parameters should be defined in subclass.")

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, par):
        self._params = par

    def run_state(self, init_state=None, analytical=False, qc=None,
                  states=None, **kwargs):
        """
        If `analytical` is False, use :func:`qutip.mesolve` to
        calculate the time of the state evolution
        and return the result. Other arguments of mesolve can be
        given as keyword arguments.
        If `analytical` is True, calculate the propagator
        with matrix exponentiation and return a list of matrices.

        Parameters
        ----------
        init_state: Qobj
            Initial density matrix or state vector (ket).

        analytical: boolean
            If True, calculate the evolution with matrices exponentiation.

        qc: :class:`.QubitCircuit`, optional
            A quantum circuit. If given, it first calls the ``load_circuit``
            and then calculate the evolution.

        states: :class:`qutip.Qobj`, optional
         Old API, same as init_state.

        **kwargs
           Keyword arguments for the qutip solver.

        Returns
        -------
        evo_result: :class:`qutip.solver.Result`
            If ``analytical`` is False,  an instance of the class
            :class:`qutip.solver.Result` will be returned.

            If ``analytical`` is True, a list of matrices representation
            is returned.
        """
        if qc is not None:
            self.load_circuit(qc)
        return super(ModelProcessor, self).run_state(
            init_state=init_state, analytical=analytical,
            states=states, **kwargs)

    def get_ops_and_u(self):
        """
        Get the labels for each Hamiltonian.

        Returns
        -------
        ctrls: list
            The list of Hamiltonians
        coeffs: array_like
            The transposed pulse matrix
        """
        return (self.ctrls, self.get_full_coeffs().T)

    def pulse_matrix(self, dt=0.01):
        """
        Generates the pulse matrix for the desired physical system.

        Returns
        -------
        t, u, labels:
            Returns the total time and label for every operation.
        """
        ctrls = self.ctrls
        coeffs = self.get_full_coeffs().T

        # FIXME This might becomes a problem if new tlist other than
        # int the default pulses are added.
        tlist = self.get_full_tlist()
        dt_list = tlist[1:] - tlist[:-1]
        t_tot = tlist[-1]
        num_step = int(np.ceil(t_tot / dt))

        t = np.linspace(0, t_tot, num_step)
        u = np.zeros((len(ctrls), num_step))

        t_start = 0
        for n in range(len(dt_list)):
            t_idx_len = int(np.floor(dt_list[n] / dt))
            mm = 0
            for m in range(len(ctrls)):
                u[mm, t_start:(t_start + t_idx_len)] = (np.ones(t_idx_len) *
                                                        coeffs[n, m])
                mm += 1
            t_start += t_idx_len

        return t, u, self.get_operators_labels()
