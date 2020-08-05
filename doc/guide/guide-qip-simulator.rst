.. QuTiP
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _qip_simulator:

*********************************************
Quantum Information Processing
*********************************************

Introduction
============

The Quantum Information Processing (QIP) module aims at providing basic tools
for quantum computing simulation both for simple quantum algorithm design and
for experimental realization. It offers two different approaches,
one with :class:`qutip.qip.QubitCircuit` calculating unitary evolution under
quantum gates by matrix product, another called
:class:`qutip.qip.device.Processor`
using open system solver in QuTiP to simulate noisy quantum device.

Quantum Circuit
===============

The most common model for quantum computing is the quantum circuit model. In QuTiP, we use :class:`qutip.qip.QubitCircuit` to represent a quantum circuit. Each quantum gate is saved as a class object :class:`qutip.qip.operations.Gate` with information such as gate name, target qubits and arguments.
To get the matrix representation of each gate, we can call the class method :meth:`qutip.qip.QubitCircuit.propagators()`. Carrying out the matrices product, one gets the matrix representation of the whole evolution. This process is demonstrated in the following example.

We can also carry out measurements on individual qubits (both in the middle and at the end of the circuit).
Each measurement is saved as a class object :class:`qutip.qip.Measurement` with parameters such as `target`,
the target qubit on which the measurement will be carried out and `classical_store`,
the index of the classical register which stores the measurement result.

Finally, once we have constructed the circuit, we can use the
`qutip.qip.QubitCircuit.run()` function to carry out one run of the circuit from start to finish which will return the final state as well as the probability of that state being outputs.
Moreover, `qutip.qip.QubitCircuit.run_statistics()` function can return all the possible output states along with the respective
probabilities.

.. code-block:: python

  >>> from qutip.qip.circuit import QubitCircuit, Gate
  >>> from qutip import tensor, basis
  >>> qc = QubitCircuit(N=2, num_cbits=1)
  >>> swap_gate = Gate(name="SWAP", targets=[0, 1])
  >>> qc.add_gate(swap_gate)
  >>> qc.add_measurement("M0", targets=[1], classical_store=0)
  >>> qc.add_gate("CNOT", controls=0, targets=1)
  >>> qc.add_gate("X", targets=0, classical_controls=[0])
  >>> qc.add_gate(swap_gate)
  >>> print(qc.gates)
  [Gate(SWAP, targets=[0, 1], controls=None, classical controls=None), Gate(CNOT, targets=[1], controls=[0], classical controls=None), Gate(X, targets=[0], controls=None, classical controls=[0]), Gate(SWAP, targets=[0, 1], controls=None, classical controls=None)]
  >>> U_list = qc.propagators()
  >>> print(print(gate_sequence_product(U_list)))
  Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = True
  Qobj data =
  [[1. 0. 0. 0.]
  [0. 0. 0. 1.]
  [0. 0. 1. 0.]
  [0. 1. 0. 0.]]
  >>> probability, final_state = qc.run(state=tensor(basis(2, 0), basis(2, 1)))
  >>> print(final_state)
  Quantum object: dims = [[2, 2], [1, 1]], shape = (4, 1), type = ket
  Qobj data =
  [[0.]
  [0.]
  [0.]
  [1.]]
  >>> qc.run_statistics(state=tensor(basis(2, 0), basis(2, 1)))
  ([Quantum object: dims = [[2, 2], [1, 1]], shape = (4, 1), type = ket
  Qobj data =
  [[0.]
   [0.]
   [0.]
   [1.]], [1.0])

The pre-defined gates for the class :class:`qutip.qip.Gate` are shown in the table below:

====================  ========================================
Gate name                           Description
====================  ========================================
"RX"                  Rotation around x axis
"RY"                  Rotation around y axis
"RZ"                  Rotation around z axis
"SQRTNOT"             Square root of NOT gate
"SNOT"                Hardmard gate
"PHASEGATE"           Add a phase one the state 1
"CRX"                 Controlled rotation around x axis
"CRY"                 Controlled rotation around y axis
"CRZ"                 Controlled rotation around z axis
"CPHASE"              Controlled phase gate
"CNOT"                Controlled NOT gate
"CSIGN"               Same as CPHASE
"BERKELEY"            Berkeley gate
"SWAPalpha"           SWAPalpha gate
"SWAP"                Swap the states of two qubits
"ISWAP"               Swap gate with additional phase for 01 and 10 states
"SQRTSWAP"            Square root of the SWAP gate
"SQRTISWAP"           Square root of the ISWAP gate
"FREDKIN"             Fredkin gate
"TOFFOLI"             Toffoli gate
"GLOBALPHASE"         Global phase
"QASM_U_GATE"         IBM's U gate with 3 arguments: RZRYRZ.
====================  ========================================

For some of the gates listed above, :class:`qutip.qip.QubitCircuit`
also has a primitive :meth:`qutip.qip.QubitCircuit.resolve_gates()`
method that decomposes them into elementary gate sets
such as CNOT or SWAP with single-qubit gates.
However, this method is not fully optimized.
It is very likely that the depth of the circuit can
be further reduced by merging quantum gates.
It is required that the gate resolution be carried out
before the measurements to the circuit are added.

In addition to these pre-defined gates,
QuTiP also allows the user to define their own gate.
The following example shows how to define a customized gate.

.. note::

   Available from QuTiP 4.4

.. code-block::

      >>> from qutip.qip.circuit import Gate
      >>> from qutip.qip.operations import rx
      >>> from qutip import Qobj
      >>> import numpy as np
      >>> def user_gate1(arg_value):
      ...     # controlled rotation X
      ...     mat = np.zeros((4, 4), dtype=np.complex)
      ...     mat[0, 0] = mat[1, 1] = 1.
      ...     mat[2:4, 2:4] = rx(arg_value)
      ...     return Qobj(mat, dims=[[2, 2], [2, 2]])
      ...
      >>> def user_gate2():
      ...     # S gate
      ...     mat = np.array([[1.,   0],
      ...                     [0., 1.j]])
      ...     return Qobj(mat, dims=[[2], [2]])
      ...
      >>>
      >>> qc = QubitCircuit(2)
      >>> qc.user_gates = {"CTRLRX": user_gate1,
      ...                  "S"     : user_gate2}
      >>>
      >>> # qubit 0 controlls qubit 1
      ... qc.add_gate("CTRLRX", targets=[0,1], arg_value=np.pi/2)
      >>> # qubit 1 controlls qubit 0
      ... qc.add_gate("CTRLRX", targets=[1,0], arg_value=np.pi/2)
      >>> # we also add a gate using a predefined Gate object
      ... g_T = Gate("S", targets=[1])
      >>> qc.add_gate(g_T)
      >>> props = qc.propagators()
      >>> props[0]
      Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = False
      Qobj data =
      [[1.    +0.j     0.    +0.j     0.    +0.j     0.    +0.j    ]
      [0.    +0.j     1.    +0.j     0.    +0.j     0.    +0.j    ]
      [0.    +0.j     0.    +0.j     0.7071+0.j     0.    -0.7071j]
      [0.    +0.j     0.    +0.j     0.    -0.7071j 0.7071+0.j    ]]
      >>> props[1]
      Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = False
      Qobj data =
      [[1.    +0.j     0.    +0.j     0.    +0.j     0.    +0.j    ]
      [0.    +0.j     0.7071+0.j     0.    +0.j     0.    -0.7071j]
      [0.    +0.j     0.    +0.j     1.    +0.j     0.    +0.j    ]
      [0.    +0.j     0.    -0.7071j 0.    +0.j     0.7071+0.j    ]]
      >>> props[2]
      Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = False
      Qobj data =
      [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
      [0.+0.j 0.+1.j 0.+0.j 0.+0.j]
      [0.+0.j 0.+0.j 1.+0.j 0.+0.j]
      [0.+0.j 0.+0.j 0.+0.j 0.+1.j]]



Plotting a Quantum Circuit
===================================
A quantum circuit (described above) can directly be plotted using the QCircuit library ( https://github.com/CQuIC/qcircuit ).
QCiruit is a quantum circuit drawing application and is implemented directly into QuTiP. As QCircuit uses LaTex for plotting you need to have several tools installed to use the plotting function within QuTip:
*pdflatex*, *pdfcrop* and *imagemagick (to convert pdf to png)*.
An example code for plotting the example quantum circuit from above is given:

.. code-block::

   >>> from qutip.qip.circuit import QubitCircuit, Gate
   >>> # create the quantum circuit
   >>> qc = QubitCircuit(N=2)
   >>> swap_gate = Gate(name="SWAP", targets=[0, 1])
   >>> qc.add_gate(swap_gate)
   >>> qc.add_gate("CNOT", controls=0, targets=1)
   >>> qc.add_gate(swap_gate)
   >>> # plot the quantum circuit
   >>> qc.png
