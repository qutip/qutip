************************
Quantum Error Mitigation
************************

.. contents:: Contents
    :local:
    :depth: 3

From the QuTiP 4.5 release, the qutip.qip module now contains the noisy quantum
circuit simulator (which was a GSoC project) providing enhanced features for a
pulse-level description of quantum circuits and noise models. A new class
`Processor` and several subclasses are added to represent different platforms
for quantum computing. They can transfer a quantum circuit into the
corresponding control sequence and simulate the dynamics with QuTiP solvers.
Different noise models can be added to `qutip.qip.noise` to simulate noise in a
quantum device.

This module is still young and many features can be improved, including new
device models, new noise models and integration with the existing general
framework for quantum circuits (`qutip.qip.circuit`). There are also possible
applications such as error mitigation techniques ([1]_, [2]_, [3]_).

The tutorial notebooks can be found in the Quantum information processing
section of https://qutip.org/qutip-tutorials/index-v5.html. A
recent presentation on the FOSDEM conference may help you get an overview
(https://fosdem.org/2020/schedule/event/quantum_qutip/). See also the Github
Project page for a collection of related issues and ongoing Pull Requests.

Expected outcomes
=================

- Make an overview of existing libraries and features in error mitigation,
  similarly to a literature survey for a research article, but for a code
  project (starting from Refs. [4]_, [5]_). This is done in order to best
  integrate the features in QuTiP with existing libraries and avoid
  reinventing the wheel.
- Features to perform error mitigation techniques in QuTiP, such as zero-noise
  extrapolation by pulse stretching.
- Tutorials implementing basic quantum error mitigation protocols
- Possible integration with Mitiq [6]_

Skills
======

* Background in quantum physics and quantum circuits.
* Git, python and familiarity with the Python scientific computing stack

Difficulty
==========

* Medium

Mentors
=======

* Nathan Shammah (nathan.shammah@gmail.com)
* Alex Pitchford (alex.pitchford@gmail.com)
* Eric Giguère (eric.giguere@usherbrooke.ca)
* Neill Lambert (nwlambert@gmail.com)
* Boxi Li (etamin1201@gmail.com) [QuTiP GSoC 2019 graduate]

References
==========

.. [1] Kristan Temme, Sergey Bravyi, Jay M. Gambetta, **Error mitigation for short-depth quantum circuits**, Phys. Rev. Lett. 119, 180509 (2017)

.. [2] Abhinav Kandala, Kristan Temme, Antonio D. Corcoles, Antonio Mezzacapo, Jerry M. Chow, Jay M. Gambetta,
 **Extending the computational reach of a noisy superconducting quantum processor**, Nature *567*, 491 (2019)

.. [3] S. Endo, S.C. Benjamin, Y. Li, **Practical quantum error mitigation for near-future applications**, Physical Review X *8*, 031027 (2018)

.. [4] Boxi Li's blog on the GSoC 2019 project on pulse-level control, https://gsoc2019-boxili.blogspot.com/

.. [5] Video of a recent talk on the GSoC 2019 project, https://fosdem.org/2020/schedule/event/quantum_qutip/

.. [6] `Mitiq <https://mitiq.readthedocs.io/>`_
