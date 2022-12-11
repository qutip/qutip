.. _qip_simulator:

*********************************
Operator-level circuit simulation
*********************************

.. note::

   New in QuTiP 4.6

Run a quantum circuit
---------------------

Let's start off by defining a simple circuit which we use to demonstrate a few
examples of circuit evolution.
We take `a circuit from OpenQASM 2 <https://github.com/Qiskit/openqasm/blob/OpenQASM2.x/examples/W-state.qasm>`_

.. testcode::

    from qutip.qip.circuit import QubitCircuit, Gate
    from qutip.qip.operations import controlled_gate, hadamard_transform
    def controlled_hadamard():
        # Controlled Hadamard
        return controlled_gate(
            hadamard_transform(1), 2, control=0, target=1, control_value=1)
    qc = QubitCircuit(N=3, num_cbits=3)
    qc.user_gates = {"cH": controlled_hadamard}
    qc.add_gate("QASMU", targets=[0], arg_value=[1.91063, 0, 0])
    qc.add_gate("cH", targets=[0,1])
    qc.add_gate("TOFFOLI", targets=[2], controls=[0, 1])
    qc.add_gate("X", targets=[0])
    qc.add_gate("X", targets=[1])
    qc.add_gate("CNOT", targets=[1], controls=0)

It corresponds to the following circuit:

.. image:: /figures/qip/quantum_circuit_w_state.png

We will add the measurement gates later. This circuit prepares the W-state :math:`(\ket{001} + \ket{010} + \ket{100})/\sqrt{3}`.
The simplest way to carry out state evolution through a quantum circuit is
providing a input state to the :meth:`~qutip.qip.circuit.QubitCircuit.run`
method.

.. testcode::

  from qutip import tensor
  zero_state = tensor(basis(2, 0), basis(2, 0), basis(2, 0))
  result = qc.run(state=zero_state)
  wstate = result

  print(wstate)

**Output**:

.. testoutput::
  :options: +NORMALIZE_WHITESPACE

  Quantum object: dims = [[2, 2, 2], [1, 1, 1]], shape = (8, 1), type = ket
  Qobj data =
  [[0.        ]
   [0.57734961]
   [0.57734961]
   [0.        ]
   [0.57735159]
   [0.        ]
   [0.        ]
   [0.        ]]


As expected, the state returned is indeed the required W-state.

As soon as we introduce measurements into the circuit, it can lead to multiple outcomes
with associated probabilities.  We can also carry out circuit evolution in a manner such that it returns all the possible state
outputs along with their corresponding probabilities. Suppose, in the previous circuit,
we measure each of the three qubits at the end.

.. testcode::

  qc.add_measurement("M0", targets=[0], classical_store=0)
  qc.add_measurement("M1", targets=[1], classical_store=1)
  qc.add_measurement("M2", targets=[2], classical_store=2)

To get all the possible output states along with the respective probability of observing the
outputs, we can use the :meth:`~qutip.qip.circuit.QubitCircuit.run_statistics` function:

.. testcode::

    result = qc.run_statistics(state=tensor(basis(2, 0), basis(2, 0), basis(2, 0)))
    states = result.get_final_states()
    probabilities = result.get_probabilities()

    for state, probability in zip(states, probabilities):
        print("State:\n{}\nwith probability {}".format(state, probability))

**Output**:

.. testoutput::
  :options: +NORMALIZE_WHITESPACE

    State:
    Quantum object: dims = [[2, 2, 2], [1, 1, 1]], shape = (8, 1), type = ket
    Qobj data =
    [[0.]
    [1.]
    [0.]
    [0.]
    [0.]
    [0.]
    [0.]
    [0.]]
    with probability 0.33333257054168813
    State:
    Quantum object: dims = [[2, 2, 2], [1, 1, 1]], shape = (8, 1), type = ket
    Qobj data =
    [[0.]
    [0.]
    [1.]
    [0.]
    [0.]
    [0.]
    [0.]
    [0.]]
    with probability 0.33333257054168813
    State:
    Quantum object: dims = [[2, 2, 2], [1, 1, 1]], shape = (8, 1), type = ket
    Qobj data =
    [[0.]
    [0.]
    [0.]
    [0.]
    [1.]
    [0.]
    [0.]
    [0.]]
    with probability 0.33333485891662384

The function returns a :class:`~qutip.qip.circuit.CircuitResult` object which contains
the output states.
The method :meth:`~qutip.qip.circuit.QubitCircuit.run_statistics` can be used to obtain the
possible states and probabilities.
Since the state created by the circuit is the W-state, we observe the states
:math:`\ket{001}`,  :math:`\ket{010}` and :math:`\ket{100}` with equal probability.


Circuit simulator
-----------------

.. _simulator_class:

The :meth:`~qutip.qip.circuit.QubitCircuit.run` and :meth:`~qutip.qip.circuit.QubitCircuit.run_statistics` functions
make use of the :class:`~qutip.qip.circuit.CircuitSimulator` which enables exact simulation with more
granular options. The simulator object takes a quantum circuit as an argument. It can optionally
be supplied with an initial state. There are two modes in which the exact simulator can function. The default mode is the
"state_vector_simulator" mode. In this mode, the state evolution proceeds maintaining the ket state throughout the computation.
For each measurement gate, one of the possible outcomes is chosen probabilistically
and computation proceeds. To demonstrate, we continue with our previous circuit:


.. testcode::

  from qutip.qip.circuit import CircuitSimulator

  sim = CircuitSimulator(qc, state=zero_state)

This initializes the simulator object and carries out any pre-computation
required. There are two ways to carry out state evolution with the simulator.
The primary way is to use the :meth:`~qutip.qip.circuit.CircuitSimulator.run` and
:meth:`~qutip.qip.circuit.CircuitSimulator.run_statistics` functions just like before (only
now with the :class:`~qutip.qip.circuit.CircuitSimulator` class).

The :class:`~qutip.qip.circuit.CircuitSimulator` class also enables stepping through the circuit:

.. testcode::

  print(sim.step())

**Output**:

.. testoutput::
  :options: +NORMALIZE_WHITESPACE

  Quantum object: dims = [[2, 2, 2], [1, 1, 1]], shape = (8, 1), type = ket
  Qobj data =
  [[0.57735159]
   [0.        ]
   [0.        ]
   [0.        ]
   [0.81649565]
   [0.        ]
   [0.        ]
   [0.        ]]

This only excutes one gate in the circuit and
allows for a better understanding of how the state evolution takes place.
The method steps through both the gates and the measurements.

Precomputing the unitary
------------------------

By default, the :class:`~qutip.qip.circuit.CircuitSimulator` class is initialized such that
the circuit evolution is conducted by applying each unitary to the state interactively.
However, by setting the argument ``precompute_unitary=True``, :class:`~qutip.qip.circuit.CircuitSimulator`
precomputes the product of the unitaries (in between the measurements):

.. testcode::

  sim = CircuitSimulator(qc, precompute_unitary=True)

  print(sim.ops)

.. testoutput::
  :options: +NORMALIZE_WHITESPACE

  [Quantum object: dims = [[2, 2, 2], [2, 2, 2]], shape = (8, 8), type = oper, isherm = False
    Qobj data =
    [[ 0.          0.57734961  0.         -0.57734961  0.          0.40824922
       0.         -0.40824922]
     [ 0.57734961  0.         -0.57734961  0.          0.40824922  0.
      -0.40824922  0.        ]
     [ 0.57734961  0.          0.57734961  0.          0.40824922  0.
       0.40824922  0.        ]
     [ 0.          0.57734961  0.          0.57734961  0.          0.40824922
       0.          0.40824922]
     [ 0.57735159  0.          0.          0.         -0.81649565  0.
       0.          0.        ]
     [ 0.          0.57735159  0.          0.          0.         -0.81649565
       0.          0.        ]
     [ 0.          0.          0.57735159  0.          0.          0.
      -0.81649565  0.        ]
     [ 0.          0.          0.          0.57735159  0.          0.
       0.         -0.81649565]],
       Measurement(M0, target=[0], classical_store=0),
       Measurement(M1, target=[1], classical_store=1),
       Measurement(M2, target=[2], classical_store=2)]


Here, ``sim.ops`` stores all the circuit operations that are going to be applied during
state evolution. As observed above, all the unitaries of the circuit are compressed into
a single unitary product with the precompute optimization enabled.
This is more efficient if one runs the same circuit one multiple initial states.
However, as the number of qubits increases, this will consume more and more memory
and become unfeasible.

Density Matrix Simulation
-------------------------

By default, the state evolution is carried out in the "state_vector_simulator" mode
(specified by the **mode** argument) as described before.
In the "density_matrix_simulator" mode, the input state can be either a ket or a density
matrix. If it is a ket, it is converted into a density matrix before the evolution is
carried out. Unlike the "state_vector_simulator" mode, upon measurement, the state
does not collapse to one of the post-measurement states. Rather, the new state is now
the density matrix representing the ensemble of post-measurement states.
In this sense, we measure the qubits and forget all the results.

To demonstrate this consider the original W-state preparation circuit which is followed
just by measurement on the first qubit:

.. testcode::

    qc = QubitCircuit(N=3, num_cbits=3)
    qc.user_gates = {"cH": controlled_hadamard}
    qc.add_gate("QASMU", targets=[0], arg_value=[1.91063, 0, 0])
    qc.add_gate("cH", targets=[0,1])
    qc.add_gate("TOFFOLI", targets=[2], controls=[0, 1])
    qc.add_gate("X", targets=[0])
    qc.add_gate("X", targets=[1])
    qc.add_gate("CNOT", targets=[1], controls=0)
    qc.add_measurement("M0", targets=[0], classical_store=0)
    qc.add_measurement("M0", targets=[1], classical_store=0)
    qc.add_measurement("M0", targets=[2], classical_store=0)
    sim = CircuitSimulator(qc, mode="density_matrix_simulator")
    print(sim.run(zero_state).get_final_states()[0])

.. testoutput::
  :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[2, 2, 2], [2, 2, 2]], shape = (8, 8), type = oper, isherm = True
    Qobj data =
    [[0.         0.         0.         0.         0.         0.
      0.         0.        ]
     [0.         0.33333257 0.         0.         0.         0.
      0.         0.        ]
     [0.         0.         0.33333257 0.         0.         0.
      0.         0.        ]
     [0.         0.         0.         0.         0.         0.
      0.         0.        ]
     [0.         0.         0.         0.         0.33333486 0.
      0.         0.        ]
     [0.         0.         0.         0.         0.         0.
      0.         0.        ]
     [0.         0.         0.         0.         0.         0.
      0.         0.        ]
     [0.         0.         0.         0.         0.         0.
      0.         0.        ]]

We are left with a mixed state.

Import and export quantum circuits
----------------------------------

QuTiP supports importation and exportation of quantum circuit in the `OpenQASM 2 format <https://github.com/Qiskit/openqasm/tree/OpenQASM2.x>`_
through the functions :func:`~qutip.qip.qasm.read_qasm` and :func:`~qutip.qip.qasm.save_qasm`.
We demonstrate this using the w-state generation circuit.
The following code is in OpenQASM format:

.. code-block::

    // Name of Experiment: W-state v1

    OPENQASM 2.0;
    include "qelib1.inc";


    qreg q[4];
    creg c[3];
    gate cH a,b {
    h b;
    sdg b;
    cx a,b;
    h b;
    t b;
    cx a,b;
    t b;
    h b;
    s b;
    x b;
    s a;
    }

    u3(1.91063,0,0) q[0];
    cH q[0],q[1];
    ccx q[0],q[1],q[2];
    x q[0];
    x q[1];
    cx q[0],q[1];

    measure q[0] -> c[0];
    measure q[1] -> c[1];
    measure q[2] -> c[2];

One can save it in a ``.qasm`` file and import it using the following code:

.. testcode::

  from qutip.qip.qasm import read_qasm
  qc = read_qasm("guide/qip/w-state.qasm")
