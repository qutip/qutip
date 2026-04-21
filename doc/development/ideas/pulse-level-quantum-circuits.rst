*******************************************
Pulse level description of quantum circuits
*******************************************

.. contents:: Contents
    :local:
    :depth: 3

The aim of this proposal is to enhance QuTiP quantum-circuit compilation
features with regard to quantum information processing. While QuTiP core modules
deal with dynamics simulation, there is also a module for quantum circuits
simulation. The two subsequent Google Summer of Code projects, in 2019 and 2020,
enhanced them in capabilities and features, allowing the simulation both at the
level of gates and at the level of time evolution. To connect them, a compiler
is implemented to compile quantum gates into the Hamiltonian model. We would
like to further enhance this feature in QuTiP and the connection with other
libraries.

Expected outcomes
=================

* APIs to import and export pulses to other libraries. Quantum compiler is a
  current research topic in quantum engineering. Although QuTiP has a simple
  compiler, many may want to try their own compiler which is more compatible
  with their quantum device. Allowing importation and exportation of control
  pulses will make this much easier. This will include a study of existing
  libraries, such as `qiskit.pulse` and `OpenPulse` [1]_, comparing them with
  `qutip.qip.pulse` module and building a more general and comprehensive
  description of the pulse.

* More examples of quantum system in the `qutip.qip.device` module. The circuit
  simulation and compilation depend strongly on the physical system. At the
  moment, we have two models: spin chain and cavity QED. We would like to
  include some other commonly used planform such as Superconducting system [2]_,
  Ion trap system [3]_ or silicon system. Each model will need a new set of
  control Hamiltonian and a compiler that finds the control pulse of a quantum
  gate. More involved noise models can also be added based on the physical
  system. This part is going to involve some physics and study of commonly used
  hardware platforms. The related code can be found in `qutip.qip.device` and
  `qutip.qip.compiler`.

Skills
======

* Git, Python and familiarity with the Python scientific computing stack
* quantum information processing and quantum computing (quantum circuit formalism)

Difficulty
==========

* Medium

Mentors
=======

* Boxi Li (etamin1201@gmail.com) [QuTiP GSoC 2019 graduate]
* Nathan Shammah (nathan.shammah@gmail.com)
* Alex Pitchford (alex.pitchford@gmail.com)

References
==========

.. [1] McKay D C, Alexander T, Bello L, et al. Qiskit backend specifications for openqasm and openpulse experiments[J]. arXiv preprint arXiv:1809.03452, 2018.

.. [2] Häffner H, Roos C F, Blatt R, **Quantum computing with trapped ions**, Physics reports, 2008, 469(4): 155-203.

.. [3] Krantz P, Kjaergaard M, Yan F, et al. **A quantum engineer's guide to superconducting qubits**, Applied Physics Reviews, 2019, 6(2): 021318.
