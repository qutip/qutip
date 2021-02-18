.. QuTiP
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _qip_simulator:

****************
Exact Simulation
****************

Let's start off by defining a simple circuit which we use to demonstrate a few
examples of circuit evolution. First, we read in the circuit from a **.qasm** file using
the :func:`qutip.qip.qasm.read_qasm()` function.

.. testcode::

  from qutip.qip.qasm import read_qasm
  from qutip import tensor, basis

  qc = read_qasm("w-state.qasm")

  qc.png

.. image:: /gallery/qutip_examples/qip/images/w_state.png

This circuit prepares the W-state :math:`\newcommand{\ket}[1]{\left|{#1}\right\rangle} \frac{\ket{001} + \ket{010} + \ket{100}}{\sqrt{3}}`.
The simplest way to carry out state evolution through a quantum circuit is
providing a input state to the :func:`qutip.qip.QubitCircuit.run()`
function.

.. testcode::

  zero_state = tensor(basis(2, 0), basis(2, 0), basis(2, 0))
  result = qc.run(state=zero_state)
  wstate = result.get_states()[0]

  print(wstate)

**Output**:

.. testoutput::
  :options: +NORMALIZE_WHITESPACE

  Quantum object: dims = [[2, 2, 2], [1, 1, 1]], shape = (8, 1), type = ket
  Qobj data =
  [[0.+0.j        ]
   [0.+0.57734961j]
   [0.+0.57734961j]
   [0.+0.j        ]
   [0.+0.57735159j]
   [0.+0.j        ]
   [0.+0.j        ]
   [0.+0.j        ]]


As expected, the state returned is indeed the required W-state.
The function returns a :class:`qutip.qip.Result` object which contains
the output state which can be obtained using the :func:`qutip.qip.Result.get_states()`
function (Infact, an array of output states is returned).

As soon as we introduce measurements into the circuit, it can lead to multiple outcomes
with associated probabilities.  We can also carry out circuit evolution in a manner such that it returns all the possible state
outputs along with their corresponding probabilities. Suppose, in the previous circuit,
we measure each of the three qubits at the end.

.. testcode::

  qc.add_measurement("M0", targets=[0], classical_store=0)
  qc.add_measurement("M1", targets=[1], classical_store=1)
  qc.add_measurement("M2", targets=[2], classical_store=2)

To get all the possible output states along with the respective probability of observing the
outputs, we can use the :func:`qutip.qip.QubitCircuit.run_statistics()` function:

.. testcode::

  result = qc.run_statistics(state=tensor(basis(2, 0), basis(2, 0), basis(2, 0)))
  states, probabilities = result.get_results()

  for state, probability in zip(states, probabilities):
    print("State:\n{}\nwith probability {}".format(state, probability))

**Output**:

.. testoutput::
  :options: +NORMALIZE_WHITESPACE

  State:
  Quantum object: dims = [[2, 2, 2], [1, 1, 1]], shape = (8, 1), type = ket
  Qobj data =
  [[0.+0.j]
   [0.+1.j]
   [0.+0.j]
   [0.+0.j]
   [0.+0.j]
   [0.+0.j]
   [0.+0.j]
   [0.+0.j]]
  with probability 0.33333257054168797
  State:
  Quantum object: dims = [[2, 2, 2], [1, 1, 1]], shape = (8, 1), type = ket
  Qobj data =
  [[0.+0.j]
   [0.+0.j]
   [0.+1.j]
   [0.+0.j]
   [0.+0.j]
   [0.+0.j]
   [0.+0.j]
   [0.+0.j]]
  with probability 0.33333257054168797
  State:
  Quantum object: dims = [[2, 2, 2], [1, 1, 1]], shape = (8, 1), type = ket
  Qobj data =
  [[0.+0.j]
   [0.+0.j]
   [0.+0.j]
   [0.+0.j]
   [0.+1.j]
   [0.+0.j]
   [0.+0.j]
   [0.+0.j]]
  with probability 0.3333348589166237

The function :func:`qutip.qip.Result.get_results()` can be used to obtain the
possible states and probabilities.
Since the state created by the circuit is the W-state, we observe the states
:math:`\newcommand{\ket}[1]{\left|{#1}\right\rangle} \ket{001}`,  :math:`\newcommand{\ket}[1]{\left|{#1}\right\rangle} \ket{010}` and :math:`\newcommand{\ket}[1]{\left|{#1}\right\rangle} \ket{100}` with equal probability.



.. _simulator_class:

The :func:`qutip.qip.QubitCircuit.run()` and :func:`qutip.qip.QubitCircuit.run_statistics()` functions
make use of the :class:`qutip.qip.ExactSimulator` which enables exact simulation with more
granular options. The simulator object takes a quantum circuit as an argument. It can optionally
be supplied with an initial state. There are two modes in which the exact simulator can function. The default mode is the
"state_vector_simulator" mode. In this mode, the state evolution proceeds maintaining the ket state throughout the computation.
For each measurement gate, one of the possible outcomes is chosen probabilistically
and computation proceeds. To demonstrate, we continue with our previous circuit:


.. testcode::

  from qutip.qip.circuit import ExactSimulator

  sim = ExactSimulator(qc, state=zero_state)

This initializes the simulator object and carries out any pre-computation
required. There are two ways to carry out state evolution with the simulator.
The primary way is to use the :func:`qutip.qip.ExactSimulator.run()` and
:func:`qutip.qip.ExactSimulator.run_statistics()` functions just like before (only
now with the :class:`qutip.qip.ExactSimulator` class).

The :class:`qutip.qip.ExactSimulator` class also enables stepping through the circuit:

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

This allows for a better understanding of how the state evolution takes place.
The function steps through both the gates and the measurements.

Optimization
------------

By default, the :class:`qutip.qip.ExactSimulator` class is initialized such that
the circuit evolution is conducted by applying each unitary to the state interactively.
However, by setting the argument **precompute_unitary=True**, :class:`qutip.qip.ExactSimulator`
precomputes the product of the unitaries (in between the measurements):

.. testcode::

  sim = ExactSimulator(qc, precompute_unitary=True)

  print(sim.ops)

.. testoutput::
  :options: +NORMALIZE_WHITESPACE

  [Quantum object: dims = [[2, 2, 2], [2, 2, 2]], shape = (8, 8), type = oper, isherm = False
  Qobj data =
  [[0.+0.j         0.+0.57734961j 0.+0.j         0.-0.57734961j
    0.+0.j         0.+0.40824922j 0.+0.j         0.-0.40824922j]
   [0.+0.57734961j 0.+0.j         0.-0.57734961j 0.+0.j
    0.+0.40824922j 0.+0.j         0.-0.40824922j 0.+0.j        ]
   [0.+0.57734961j 0.+0.j         0.+0.57734961j 0.+0.j
    0.+0.40824922j 0.+0.j         0.+0.40824922j 0.+0.j        ]
   [0.+0.j         0.+0.57734961j 0.+0.j         0.+0.57734961j
    0.+0.j         0.+0.40824922j 0.+0.j         0.+0.40824922j]
   [0.+0.57735159j 0.+0.j         0.+0.j         0.+0.j
    0.-0.81649565j 0.+0.j         0.+0.j         0.+0.j        ]
   [0.+0.j         0.+0.57735159j 0.+0.j         0.+0.j
    0.+0.j         0.-0.81649565j 0.+0.j         0.+0.j        ]
   [0.+0.j         0.+0.j         0.+0.57735159j 0.+0.j
    0.+0.j         0.+0.j         0.-0.81649565j 0.+0.j        ]
   [0.+0.j         0.+0.j         0.+0.j         0.+0.57735159j
    0.+0.j         0.+0.j         0.+0.j         0.-0.81649565j]],
    Measurement(M0, target=[0], classical_store=0),
    Measurement(M1, target=[1], classical_store=1),
    Measurement(M2, target=[2], classical_store=2)]


Here, **sim.ops** stores all the circuit operations that are going to be applied during
state evolution. As observed above, all the unitaries of the circuit are compressed into
a single unitary product with the precompute optimization enabled.
This has the effect of state evolution running slightly faster on each input:

.. testcode::

  result = sim.run_statistics(zero_state)

  states, probabilities = result.get_results()

  for state, probability in zip(states, probabilities):
    print("State:\n{}\nwith probability {}".format(state, probability))

.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    State:
    Quantum object: dims = [[2, 2, 2], [1, 1, 1]], shape = (8, 1), type = ket
    Qobj data =
    [[0.+0.j]
     [0.+1.j]
     [0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [0.+0.j]]
    with probability 0.33333257054168797
    State:
    Quantum object: dims = [[2, 2, 2], [1, 1, 1]], shape = (8, 1), type = ket
    Qobj data =
    [[0.+0.j]
     [0.+0.j]
     [0.+1.j]
     [0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [0.+0.j]]
    with probability 0.33333257054168797
    State:
    Quantum object: dims = [[2, 2, 2], [1, 1, 1]], shape = (8, 1), type = ket
    Qobj data =
    [[0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [0.+0.j]
     [0.+1.j]
     [0.+0.j]
     [0.+0.j]
     [0.+0.j]]
    with probability 0.3333348589166237

Density Matrix Simulation
-------------------------

By default, the state evolution is carried out in the "state_vector_simulator" mode
(specified by the **mode** argument) as described before.
In the "density_matrix_simulator" mode, the input state can be either a ket or a density
matrix. If it is a ket, it is converted into a density matrix before the evolution is
carried out. Unlike the "state_vector_simulator" mode, upon measurement, the state
does not collapse to one of the post-measurement states. Rather, the new state is now
the density matrix representing the ensemble of post-measurement states.

To demonstrate this consider the original W-state preparation circuit which is followed
just by measurement on the first qubit:

.. testcode::

  qc = read_qasm("w-state.qasm")
  qc.add_measurement("M0", targets=0)
  sim = sim = ExactSimulator(qc, mode="density_matrix_simulator")

  print(sim.run(zero_state).get_states()[0])

.. testoutput::
  :options: +NORMALIZE_WHITESPACE

  Quantum object: dims = [[2, 2, 2], [2, 2, 2]], shape = (8, 8), type = oper, isherm = True
  Qobj data =
  [[0.         0.         0.         0.         0.         0.
    0.         0.        ]
   [0.         0.33333257 0.33333257 0.         0.         0.
    0.         0.        ]
   [0.         0.33333257 0.33333257 0.         0.         0.
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


The output state is always a single density matrix.
