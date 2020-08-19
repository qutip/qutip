.. QuTiP
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _qip_intro:

*********************************************
Quantum Information Processing
*********************************************

Introduction
============

The Quantum Information Processing (QIP) module aims at providing basic tools for quantum computing simulation both for simple quantum algorithm design and for experimental realization. It offers two different approaches, one with :class:`qutip.qip.QubitCircuit` calculating unitary evolution under quantum gates by matrix product, another called :class:`qutip.qip.device.Processor` using open system solver in QuTiP to simulate noisy quantum device.

.. _quantum_circuits:

Quantum Circuit
===============

The most common model for quantum computing is the quantum circuit model. In QuTiP, we use :class:`qutip.qip.QubitCircuit` to represent a quantum circuit. Each quantum gate is saved as a class object :class:`qutip.qip.operations.Gate` with information such as gate name, target qubits and arguments.
To get the matrix representation of each gate, we can call the class method :meth:`qutip.qip.QubitCircuit.propagators()`. Carrying out the matrices product, one gets the matrix representation of the whole evolution. This process is demonstrated in the following example.

We can also carry out measurements on individual qubits (both in the middle and at the end of the circuit). Each measurement is saved as a class object :class:`qutip.qip.Measurement` with parameters such as target, the target qubit on which the measurement will be carried out and classical_store, the index of the classical register on which stores the result of the measurement.

Finally, once we have constructed the circuit, we can use the
`qutip.qip.QubitCircuit.run()` function to carry out one run of the circuit from start to finish which will return the final state.
Moreover, we can run the circuit multiple times and obtain the various resulting states along with their respective observed frequencies.

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

  print(qc.gates)

**Output**:

.. testoutput::

  [Gate(SWAP, targets=[0, 1], controls=None, classical controls=None, control_value=None), Measurement(M0, target=[1], classical_store=0), Gate(CNOT, targets=[1], controls=[0], classical controls=None, control_value=None), Gate(X, targets=[0], controls=None, classical controls=[0], control_value=None), Gate(SWAP, targets=[0, 1], controls=None, classical controls=None, control_value=None)]

Unitaries
=========

There are a few useful functions associated with the circuit object. For example,
the `qutip.qip.QubitCircuit.propagators()` method returns a list of the unitaries associated
with the sequence of gates in the circuit. By default, the unitaries are expanded to the
full dimension of the circuit:

.. testcode::

  U_list = qc.propagators()
  print(U_list)

**Output**:

.. testoutput::

  [Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = True
  Qobj data =
  [[1. 0. 0. 0.]
   [0. 0. 1. 0.]
   [0. 1. 0. 0.]
   [0. 0. 0. 1.]], Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = True
  Qobj data =
  [[1. 0. 0. 0.]
   [0. 1. 0. 0.]
   [0. 0. 0. 1.]
   [0. 0. 1. 0.]], Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = True
  Qobj data =
  [[0. 0. 1. 0.]
   [0. 0. 0. 1.]
   [1. 0. 0. 0.]
   [0. 1. 0. 0.]], Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = True
  Qobj data =
  [[1. 0. 0. 0.]
   [0. 0. 1. 0.]
   [0. 1. 0. 0.]
   [0. 0. 0. 1.]]]

Another option is to only return the unitaries in their original dimension. This
can be achieved with the argument **expand=False** specified to the propagators function.

.. testcode::

  U_list = qc.propagators(expand=False)
  print(U_list)

**Output**:

.. testoutput::

  [Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = True
  Qobj data =
  [[1. 0. 0. 0.]
   [0. 0. 1. 0.]
   [0. 1. 0. 0.]
   [0. 0. 0. 1.]], Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = True
  Qobj data =
  [[1. 0. 0. 0.]
   [0. 1. 0. 0.]
   [0. 0. 0. 1.]
   [0. 0. 1. 0.]], Quantum object: dims = [[2], [2]], shape = (2, 2), type = oper, isherm = True
  Qobj data =
  [[0. 1.]
   [1. 0.]], Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = True
  Qobj data =
  [[1. 0. 0. 0.]
   [0. 0. 1. 0.]
   [0. 1. 0. 0.]
   [0. 0. 0. 1.]]]

.. _quantum_gates:

Gates
=====

The pre-defined gates for the class :class:`qutip.qip.Gate` are shown in the table below:

====================  ========================================
Gate name                           Description
====================  ========================================
"RX"                  Rotation around x axis
"RY"                  Rotation around y axis
"RZ"                  Rotation around z axis
"X"                   Pauli-X gate
"Y"                   Pauli-Y gate
"Z"                   Pauli-Z gate
"S"                   Single-qubit rotation or Z90
"T"                   Square root of S gate
"SQRTNOT"             Square root of NOT gate
"SNOT"                Hardmard gate
"PHASEGATE"           Add a phase one the state 1
"CRX"                 Controlled rotation around x axis
"CRY"                 Controlled rotation around y axis
"CRZ"                 Controlled rotation around z axis
"CX"                  Controlled X gate
"CY"                  Controlled Y gate
"CZ"                  Controlled Z gate
"CS"                  Controlled S gate
"CT"                  Controlled T gate
"CPHASE"              Controlled phase gate
"CNOT"                Controlled NOT gate
"CSIGN"               Same as CPHASE
"QASMU"               U rotation gate used as a primitive in the QASM standard
"BERKELEY"            Berkeley gate
"SWAPalpha"           SWAPalpha gate
"SWAP"                Swap the states of two qubits
"ISWAP"               Swap gate with additional phase for 01 and 10 states
"SQRTSWAP"            Square root of the SWAP gate
"SQRTISWAP"           Square root of the ISWAP gate
"FREDKIN"             Fredkin gate
"TOFFOLI"             Toffoli gate
"GLOBALPHASE"         Global phase
====================  ========================================

For some of the gates listed above, :class:`qutip.qip.QubitCircuit` also has a primitive :meth:`qutip.qip.QubitCircuit.resolve_gates()` method that decomposes them into elementary gate sets such as CNOT or SWAP with single-qubit gates. However, this method is not fully optimized. It is very likely that the depth of the circuit can be further reduced by merging quantum gates. It is required that the gate resolution be carried out before the measurements to the circuit are added.

In addition to these pre-defined gates, QuTiP also allows the user to define their own gate. The following example shows how to define a customized gate.

.. note::

   Available from QuTiP 4.4

.. testcode::

      from qutip.qip.circuit import Gate
      from qutip.qip.operations import rx

      def user_gate1(arg_value):
           # controlled rotation X
           mat = np.zeros((4, 4), dtype=np.complex)
           mat[0, 0] = mat[1, 1] = 1.
           mat[2:4, 2:4] = rx(arg_value)
           return Qobj(mat, dims=[[2, 2], [2, 2]])


      def user_gate2():
           # S gate
           mat = np.array([[1.,   0],
                           [0., 1.j]])
           return Qobj(mat, dims=[[2], [2]])

      qc = QubitCircuit(2)
      qc.user_gates = {"CTRLRX": user_gate1,
                       "S"     : user_gate2}

      # qubit 0 controls qubit 1
      qc.add_gate("CTRLRX", targets=[0,1], arg_value=np.pi/2)

      # qubit 1 controls qubit 0
      qc.add_gate("CTRLRX", targets=[1,0], arg_value=np.pi/2)

      # we also add a gate using a predefined Gate object
      g_T = Gate("S", targets=[1])
      qc.add_gate(g_T)
      props = qc.propagators()

      print(props[0])

**Output**:

.. testoutput::

      Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = False
      Qobj data =
      [[1.    +0.j     0.    +0.j     0.    +0.j     0.    +0.j    ]
      [0.    +0.j     1.    +0.j     0.    +0.j     0.    +0.j    ]
      [0.    +0.j     0.    +0.j     0.7071+0.j     0.    -0.7071j]
      [0.    +0.j     0.    +0.j     0.    -0.7071j 0.7071+0.j    ]]

.. testcode::

      print(props[1])

**Output**:

.. testoutput::

      Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = False
      Qobj data =
      [[1.    +0.j     0.    +0.j     0.    +0.j     0.    +0.j    ]
      [0.    +0.j     0.7071+0.j     0.    +0.j     0.    -0.7071j]
      [0.    +0.j     0.    +0.j     1.    +0.j     0.    +0.j    ]
      [0.    +0.j     0.    -0.7071j 0.    +0.j     0.7071+0.j    ]]

.. testcode::

      print(props[2])

**Output**:

.. testoutput::

      Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = False
      Qobj data =
      [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
      [0.+0.j 0.+1.j 0.+0.j 0.+0.j]
      [0.+0.j 0.+0.j 1.+0.j 0.+0.j]
      [0.+0.j 0.+0.j 0.+0.j 0.+1.j]]

.. _quantum_circuit_plots:

Plotting a Quantum Circuit
===================================

A quantum circuit (described above) can directly be plotted using the QCircuit library ( https://github.com/CQuIC/qcircuit ).
QCiruit is a quantum circuit drawing application and is implemented directly into QuTiP. As QCircuit uses LaTex for plotting you need to have several tools installed to use the plotting function within QuTip:
*pdflatex*, *pdfcrop* and *imagemagick (to convert pdf to png)*.
An example code for plotting the example quantum circuit from above is given:

.. testcode::

    from qutip.qip.circuit import QubitCircuit, Gate
    # create the quantum circuit
    qc = QubitCircuit(N=2, num_cbits=1)
    qc.add_gate("CNOT", controls=0, targets=1)
    qc.add_gate("H", targets=1)
    qc.add_measurement("M0", targets=1, classical_store=0)
    # plot the quantum circuit
    qc.png

.. image:: /gallery/qutip_examples/qip/images/qc_example.png



Simulation
==========

There are two different ways to simulate the action of quantum circuits using QuTiP:

- The first method utilizes unitary application through matrix products on the input states.
  This method simulates circuits exactly in a deterministic manner. This is achieved through
  :class:`qutip.qip.circuit.ExactSimulator`. A short guide to exact simulation can be
  found at :ref:`qip_simulator`. The teleportation notebook is also useful as an example.

- A different method of circuit simulation employs driving Hamiltonians with the ability to
  simulate circuits in the presence of noise. This can be achieved through the various classes
  in :class:`qutip.qip.device`. A short guide to processors for QIP simulation can be found at :ref:`qip_processor`.
