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
from qutip.qip.device.processor import Processor
from qutip.qip.operations.gates import gate_sequence_product


__all__ = ['OptPulseProcessor']


class OptPulseProcessor(Processor):
    """
    A processor, which takes the Hamiltonian available
    as dynamic generators, calls the
    `qutip.control.optimize_pulse_unitary` function
    to find an optimized pulse sequence for the desired quantum circuit.
    The processor can simulate the evolution under the given
    control pulses using :func:`qutip.mesolve`.
    (For attributes documentation, please
    refer to the parent class :class:`qutip.qip.device.Processor`)

    Parameters
    ----------
    N: int
        The number of component systems.

    drift: `:class:`qutip.Qobj`
        The drift Hamiltonian. The size must match the whole quantum system.

    t1: list or float
        Characterize the decoherence of amplitude damping for
        each qubit. A list of size `N` or a float for all qubits.

    t2: list of float
        Characterize the decoherence of dephasing for
        each qubit. A list of size `N` or a float for all qubits.

    dims: list
        The dimension of each component system.
        Default value is a
        qubit system of ``dim=[2,2,2,...,2]``
    """
    def __init__(self, N, drift=None, t1=None, t2=None, dims=None):
        super(OptPulseProcessor, self).__init__(
            N, t1=t1, t2=t2, dims=dims)
        if drift is not None:
            self.add_drift(drift, list(range(N)))
        self.spline_kind = "step_func"

    def load_circuit(self, qc, min_fid_err=np.inf, merge_gates=True,
                     setting_args=None, verbose=False, **kwargs):
        """
        Find the pulses realizing a given :class:`qutip.qip.Circuit` using
        `qutip.control.optimize_pulse_unitary`. Further parameter for
        for `qutip.control.optimize_pulse_unitary` needs to be given as
        keyword arguments. By default, it first merge all the gates
        into one unitary and then find the control pulses for it.
        It can be turned off and one can set different parameters
        for different gates. See examples for details.

        Examples
        --------
        # Same parameter for all the gates
        qc = QubitCircuit(N=1)
        qc.add_gate("SNOT", 0)

        num_tslots = 10
        evo_time = 10
        processor = OptPulseProcessor(N=1, drift=sigmaz(), ctrls=[sigmax()])
        # num_tslots and evo_time are two keyword arguments
        tlist, coeffs = processor.load_circuit(
            qc, num_tslots=num_tslots, evo_time=evo_time)

        # Different parameters for different gates
        qc = QubitCircuit(N=2)
        qc.add_gate("SNOT", 0)
        qc.add_gate("SWAP", targets=[0, 1])
        qc.add_gate('CNOT', controls=1, targets=[0])

        processor = OptPulseProcessor(N=2, drift=tensor([sigmaz()]*2))
        processor.add_control(sigmax(), cyclic_permutation=True)
        processor.add_control(sigmay(), cyclic_permutation=True)
        processor.add_control(tensor([sigmay(), sigmay()]))
        setting_args = {"SNOT": {"num_tslots": 10, "evo_time": 1},
                        "SWAP": {"num_tslots": 30, "evo_time": 3},
                        "CNOT": {"num_tslots": 30, "evo_time": 3}}
        tlist, coeffs = processor.load_circuit(qc, setting_args=setting_args,
                                               merge_gates=False)

        Parameters
        ----------
        qc: :class:`qutip.QubitCircuit` or list of Qobj
            The quantum circuit to be translated.

        min_fid_err: float, optional
            The minimal fidelity tolerance, if the fidelity error of any
            gate decomposition is higher, a warning will be given.
            Default is infinite.

        merge_gates: boolean, optimal
            If True, merge all gate/Qobj into one Qobj and then
            find the optimal pulses for this unitary matrix. If False,
            find the optimal pulses for each gate/Qobj.

        setting_args: dict, optional
            Only considered if merge_gates is False.
            It is a dictionary containing keyword arguments
            for different gates.

            E.g:
            setting_args = {"SNOT": {"num_tslots": 10, "evo_time": 1},
                            "SWAP": {"num_tslots": 30, "evo_time": 3},
                            "CNOT": {"num_tslots": 30, "evo_time": 3}}

        verbose: boolean, optional
            If true, the information for each decomposed gate
            will be shown. Default is False.

        **kwargs
            keyword arguments for `qutip.control.optimize_pulse_unitary`

        Returns
        -------
        tlist: array_like
            A NumPy array specifies the time of each coefficient

        coeffs: array_like
            A 2d NumPy array of the shape (len(ctrls), len(tlist)-1). Each
            row corresponds to the control pulse sequence for
            one Hamiltonian.

        Notes
        -----
        len(tlist)-1=coeffs.shape[1] since tlist gives the beginning and the
        end of the pulses
        """
        if setting_args is None:
            setting_args = {}
        if isinstance(qc, QubitCircuit):
            props = qc.propagators()
            gates = [g.name for g in qc.gates]
        elif isinstance(qc, Iterable):
            props = qc
            gates = None  # using list of Qobj, no gates name
        else:
            raise ValueError(
                "qc should be a "
                "QubitCircuit or a list of Qobj")
        if merge_gates:  # merge all gates/Qobj into one Qobj
            props = [gate_sequence_product(props)]
            gates = None

        time_record = []  # a list for all the gates
        coeff_record = []
        last_time = 0.  # used in concatenation of tlist
        for prop_ind, U_targ in enumerate(props):
            U_0 = identity(U_targ.dims[0])

            # If qc is a QubitCircuit and setting_args is not empty,
            # we update the kwargs for each gate.
            # keyword arguments in setting_arg have priority
            if gates is not None and setting_args:
                kwargs.update(setting_args[gates[prop_ind]])

            full_drift_ham = self.drift.get_ideal_qobjevo(self.dims).cte
            full_ctrls_hams = [pulse.get_ideal_qobj(self.dims)
                               for pulse in self.pulses]
            result = cpo.optimize_pulse_unitary(
                full_drift_ham, full_ctrls_hams, U_0, U_targ, **kwargs)

            if result.fid_err > min_fid_err:
                warnings.warn(
                    "The fidelity error of gate {} is higher "
                    "than required limit. Use verbose=True to see"
                    "the more detailed information.".format(prop_ind))

            time_record.append(result.time[1:] + last_time)
            last_time += result.time[-1]
            coeff_record.append(result.final_amps.T)

            if verbose:
                print("********** Gate {} **********".format(prop_ind))
                print("Final fidelity error {}".format(result.fid_err))
                print("Final gradient normal {}".format(
                                                result.grad_norm_final))
                print("Terminated due to {}".format(result.termination_reason))
                print("Number of iterations {}".format(result.num_iter))

        tlist = np.hstack([[0.]] + time_record)
        for i in range(len(self.pulses)):
            self.pulses[i].tlist = tlist
        coeffs = np.vstack([np.hstack(coeff_record)])
        self.coeffs = coeffs

        return tlist, coeffs
