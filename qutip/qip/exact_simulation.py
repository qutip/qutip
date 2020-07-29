# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in sourc e and binary forms, with or without
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

from qutip.qip.operations.gates import gate_sequence_product
from qutip.qip.circuit import Measurement


__all__ = ['Result', 'ExactSimulator']


class Result:

    def __init__(self, states, probabilities):
        """
        Store result of ExactSimulator.

        Parameters
        ----------
        states: list of Qobj.
            List of output kets or density matrices.

        probabilities: list of float.
            List of probabilities of obtaining each output state.
        """

        if isinstance(states, Qobj):
            self.states = [states]
            self.probabilities = [probabilities]
        else:
            self.states = states
            self.probabilities = probabilities

    def get_states(self):
        """
        Return list of output states.

        Returns
        ----------
        states: list of Qobj.
            List of output kets or density matrices.
        """

        if isinstance(self.states, list):
            return self.states
        else:
            return self.states[0]

    def get_results(self, index=None):
        """
        Return list of output states and corresponding probabilities

        Parameters
        ----------
        index: int
            Indicates i-th output, probability pair to be returned.

        Returns
        -------
        states: Qobj or list of Qobj
            Possible output states.

        probabilities: float or list of float
            Probabilities associated with each output state.

        """

        if index is not None:
            return self.states[index], self.probabilities[index]
        return self.states, self.probabilities


class ExactSimulator:

    def __init__(self, qc, state, cbits=None, U_list=None, measure_results=None,
                 mode="state_vector_simulator", precompute_unitary=False):
        """
        Store result of ExactSimulator.

        Parameters
        ----------
        qc: :class:`.QubitCircuit`
            Quantum Circuit to be simulated.

        state: ket or oper
            ket or density matrix

        cbits: list of int, optional
            initial value of classical bits

        U_list: list of Qobj, optional
            list of predefined unitaries corresponding to circuit.

        measure_results : tuple of ints, optional
            optional specification of each measurement result to enable
            post-selection. If specified, the measurement results are
            set to the tuple of bits (sequentially) instead of being
            chosen at random.

        mode: Boolean, optional
            Specify if input state (and therefore computation) is in
            state-vector mode or in density matrix mode. If in density matrix
            mode and given a state vector input, the output must be assumed to
            be a density matrix.

        precompute_unitary: Boolean, optional
            Specify if computation is done by pre-computing and aggregating
            gate unitaries. Possibly a faster method in the case of large number
            of repeat runs with different state inputs.
        """

        self.qc = qc
        self.mode = mode
        self.precompute_unitary = precompute_unitary

        if U_list:
            self.U_list = U_list
        elif precompute_unitary:
            self.U_list = qc.propagators(expand=False)
        else:
            self.U_list = qc.propagators()

        self.ops = []
        self.inds_list = []

        if precompute_unitary:
            self._process_ops_precompute()
        else:
            self._process_ops()

        self.initialize_run(state, cbits, measure_results)

    def _add_ops(self):
        '''
        Process list of gates (including measurements), and stores
        them in self.ops (as unitaries) for further computation.
        '''

        U_list_index = 0

        for operation in self.qc.gates:
            if isinstance(operation, Measurement):
                self.ops.append(operation)
            elif isinstance(operation, Gate):
                if operation.classical_controls:
                    self.ops.append((operation, self.U_list[U_list_index]))
                else:
                    self.ops.append(self.U_list[U_list_index])
                U_list_index += 1

    def _process_ops_precompute(self):
        '''
        Process list of gates (including measurements), aggregate
        gate unitaries (by multiplying) and store them in self.ops
        for further computation.
        '''

        prev_index = 0
        U_list_index = 0

        for gate in self.qc.gates:
            if isinstance(gate, Measurement):
                continue
            else:
                self.inds_list.append(gate.get_inds(self.qc.N))

        for operation in self.qc.gates:
            if isinstance(operation, Measurement):
                if U_list_index > prev_index:
                    self.ops.append(self._compute_unitary(
                                    self.U_list[prev_index:U_list_index],
                                    self.inds_list[prev_index:U_list_index]))
                    prev_index = U_list_index
                self.ops.append(operation)

            elif isinstance(operation, Gate):
                if operation.classical_controls:
                    if U_list_index > prev_index:
                        self.ops.append(
                            self._compute_unitary(
                                    self.U_list[prev_index:U_list_index],
                                    self.inds_list[prev_index:U_list_index]))
                        prev_index = U_list_index
                    self.ops.append((operation, self.U_list[prev_index]))
                    prev_index += 1
                    U_list_index += 1
                else:
                    U_list_index += 1

        if U_list_index > prev_index:
            self.ops.append(self._compute_unitary(
                            self.U_list[prev_index:U_list_index],
                            self.inds_list[prev_index:U_list_index]))
            prev_index = U_list_index + 1
            U_list_index = prev_index

    def initialize_run(self, state, cbits=None, measure_results=None):
        '''
        Reset Simulator state variables to start a new run.

        Parameters
        ----------
        state: ket or oper
            ket or density matrix

        cbits: list of int, optional
            initial value of classical bits

        U_list: list of Qobj, optional
            list of predefined unitaries corresponding to circuit.

        measure_results : tuple of ints, optional
            optional specification of each measurement result to enable
            post-selection. If specified, the measurement results are
            set to the tuple of bits (sequentially) instead of being
            chosen at random.
        '''

        if cbits and len(cbits) == self.qc.num_cbits:
            self.cbits = cbits
        else:
            self.cbits = [0] * self.qc.num_cbits

        if state.shape[0] != 2 ** self.qc.N:
            raise ValueError("dimension of state is incorrect")

        self.state = state
        self.probability = 1
        self.op_index = 0
        self.measure_results = measure_results
        self.measure_ind = 0

    def _compute_unitary(self, U_list, inds_list):
        '''
        Compute unitary corresponding to a product of unitaries in U_list
        and expand it to size of circuit.

        Parameters
        ----------
        U_list: list of Qobj
            list of predefined unitaries.

        inds_list: list of list of int
            list of qubit indices corresponding to each unitary in U_list

        Returns
        -------
        U: Qobj
            resultant unitary
        '''

        U_overall, overall_inds = gate_sequence_product(U_list,
                                                        inds_list=inds_list,
                                                        expand=True)
        # TODO: fix when this is the case!
        if len(overall_inds) != self.qc.N:
            U_overall = expand_operator(U_overall,
                                        N=self.qc.N,
                                        targets=overall_inds)
        return U_overall

    def run(self, state, cbits=None, measure_results=None):
        '''
        Calculate the result of one instance of circuit run.

        Parameters
        ----------
        state : ket or oper
                state vector or density matrix input.
        cbits : List of ints, optional
                initialization of the classical bits.
        measure_results : tuple of ints, optional
                optional specification of each measurement result to enable
                post-selection. If specified, the measurement results are
                set to the tuple of bits (sequentially) instead of being
                chosen at random.

        Returns
        -------
        result : Result
            returns the Result object containing output information.
        '''

        self.initialize_run(state, cbits, measure_results)
        for _ in range(len(self.ops)):
            self.step()
        return Result(self.state, self.probability)

    def step(self):
        '''
        Return state after one step of circuit evolution
        (gate or measurement).

        Returns
        -------
        state : ket or oper
            state after one evolution step.
        '''

        op = self.ops[self.op_index]
        if isinstance(op, Measurement):
            self._apply_measurement(op)
        elif isinstance(op, tuple):
            operation, U = op
            apply_gate = all([self.cbits[i] for i
                              in operation.classical_controls])
            if apply_gate:
                if self.precompute_unitary:
                    U = expand_operator(U, self.qc.N,
                                        operation.get_inds(self.qc.N))
                self._evolve_state(U)
        else:
            self._evolve_state(op)

        self.op_index += 1
        return self.state

    def _evolve_state(self, U):
        '''
        Applies unitary to state.

        Parameters
        ----------
        U: Qobj
            unitary to be applied.
        '''

        if self.mode == "state_vector_simulator":
            self._evolve_ket(U)
        elif self.mode == "density_matrix_simulator":
            self._evolve_dm(U)
        else:
            raise NotImplementedError(
                "mode {} is not available.".format(self.mode))

    def _evolve_ket(self, U):
        '''
        Applies unitary to ket state.

        Parameters
        ----------
        U: Qobj
            unitary to be applied.
        '''

        self.state = U * self.state

    def _evolve_dm(self, U):
        '''
        Applies unitary to density matrix state.

        Parameters
        ----------
        U: Qobj
            unitary to be applied.
        '''

        self.state = U * self.state * U.dag()

    def _apply_measurement(self, operation):
        '''
        Applies measurement gate specified by operation to current state.

        Parameters
        ----------
        operation: :class:`.Measurement`
            Measurement gate in a circuit object.
        '''

        states, probabilities = operation.measurement_comp_basis(self.state)
        if self.measure_results:
            i = int(self.measure_results[self.measure_ind])
            self.measure_ind += 1
        else:
            i = np.random.choice([0, 1],
                                 p=[probabilities[0], 1 - probabilities[0]])
            self.probability *= probabilities[i]
            self.state = states[i]
            if operation.classical_store is not None:
                self.cbits[operation.classical_store] = i
