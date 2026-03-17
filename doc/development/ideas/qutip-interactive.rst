*****************
QuTiP Interactive
*****************

.. contents:: Contents
    :local:
    :depth: 3

QuTiP is pretty simple to use at an entry level for anyone with basic Python
skills. However, *some* Python skills are necessary. A graphical user interface
(GUI) for some parts of qutip could help make qutip more accessible. This could
be particularly helpful in education, for teachers and learners.

Ideally, interactive components could be embedded in web pages. Including, but
not limited to, Jupyter notebooks.

The scope for this is broad and flexible. Ideas including, but not limited to:

Interactive Bloch sphere
------------------------

QuTiP has a Bloch sphere virtualisation for qubit states. This could be made
interactive through sliders, radio buttons, cmd buttons etc. An interactive
Bloch sphere could have sliders for qubit state angles. Buttons to add states,
toggle state evolution path. Potential for recording animations. Matplotlib has
some interactive features (sliders, radio buttons, cmd buttons) that can be used
to control parameters. that could potentially be used.

Interactive solvers
-------------------

Options to configure dynamics generators (Lindbladian / Hamiltonian args etc)
and expectation operators. Then run solver and view state evolution.

Animated circuits
-----------------

QIP circuits could be animated. Status lights showing evolution of states during
the processing. Animated Bloch spheres for qubits.

Expected outcomes
=================

* Interactive graphical components for demonstrating quantum dynamics
* Web pages for qutip.org or Jupyter notebooks introducing quantum dynamics
  using the new components

Skills
======

* Git, Python and familiarity with the Python scientific computing stack
* elementary understanding of quantum dynamics

Difficulty
==========

* Variable

Mentors
=======

* Nathan Shammah (nathan.shammah@gmail.com)
* Alex Pitchford (alex.pitchford@gmail.com)
* Simon Cross (hodgestar@gmail.com)
* Boxi Li (etamin1201@gmail.com) [QuTiP GSoC 2019 graduate]
