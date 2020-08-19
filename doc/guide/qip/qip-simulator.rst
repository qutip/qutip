.. QuTiP
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _qip_simulator:

****************
Exact Simulation
****************

Let's start off by defining a simple circuit which we use to demonstrate a few
examples of circuit evolution.

.. testcode::

  from qutip.qip.circuit import QubitCircuit, Gate
  from qutip import tensor, basis

  qc = QubitCircuit(N=2, num_cbits=1)
  swap_gate = Gate(name="SWAP", targets=[0, 1])

  qc.add_gate(swap_gate)
  qc.add_measurement("M0", targets=[1], classical_store=0)
  qc.add_gate("CNOT", controls=0, targets=1)
  qc.add_gate("X", targets=0, classical_controls=[0])
  qc.add_gate(swap_gate)

The simplest way to carry out state evolution through a quantum circuit is
providing a input state to the :func:`qutip.qip.circuit.QubitCircuit.run`
function.

.. testcode::

  result = qc.run(state=tensor(basis(2, 0), basis(2, 1)))
  print(result.get_states(index=0))

**Output**:

.. testoutput::

  Quantum object: dims = [[2, 2], [1, 1]], shape = (4, 1), type = ket
  Qobj data =
  [[0.]
  [0.]
  [0.]
  [1.]]

The function returns a :class:`qutip.qip.circuit.Result` object which contains
the output state which can be obtained using the :func:`qutip.qip.circuit.Result.get_states`
function (here the argument is supplied to indicate that the only state at index 0 must be returned).

We can also carry out circuit evolution in a manner such that it returns all the possible state
outputs along with their corresponding probabilities. This is especially useful in the case
where we have measurement gates in our circuits which can lead to probabilistic outcomes.

.. testcode::

  result = qc.run_statistics(state=tensor(basis(2, 0), basis(2, 1)))
  states, probabilities = result.get_results()

  for state, probability in zip(states, probabilities):
      print("State:\n{}\nwith probability {}".format(state, probability))

**Output**:

.. testoutput::

  State:
  Quantum object: dims = [[2, 2], [1, 1]], shape = (4, 1), type = ket
  Qobj data =
  [[0.]
  [0.]
  [0.]
  [1.]]
  with probability 1.0

In this case, the only possible state is the state we obtained in the previous run
which is obtained with probability 1.

.. _simulator_class:

The :class:`qutip.qip.circuit.ExactSimulator` enables exact simulation with more
granular options. There are two modes in which the exact simulator can function:

- In the **state_vector_simulator** mode, the input is necessarily a
