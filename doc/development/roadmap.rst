.. _development_roadmap:

*************************
QuTiP Development Roadmap
*************************

Preamble
========

This document outlines plan and ideas for the current and future development of
QuTiP. The document is maintained by the QuTiP Admim team. Contributuions from
the QuTiP Community are very welcome.

In particular this document outlines plans for the next major release of qutip,
which will be version 5. And also plans and dreams beyond the next major
version.

There is lots of development going on in QuTiP that is not recorded in here.
This a just an attempt at coordinated stragetgy and ideas for the future.

.. _what-is-qutip:

What is QuTiP?
--------------

The name QuTiP refers to a few things. Most famously, qutip is a Python library
for simulating quantum dynamics. To support this, the library also contains
various software tools (functions and classes) that have more generic
applications, such as linear algebra components and visualisation utilities, and
also tools that are specifically quantum related, but have applications beyond
just solving dynamics (for instance partial trace computation).

QuTiP is also an organisation, in the Github sense, and in the sense of a group
of people working collaboratively towards common objectives, and also a web
presence `qutip.org <https://qutip.org/>`_. The QuTiP Community includes all the
people who have supported the project since in conception in 2010, including
manager, funders, developers, maintainers and users.

These related, and overlapping, uses of the QuTiP name are of little consequence
until one starts to consider how to organise all the software packages that are
somehow related to QuTiP, and specifically those that are maintained by the
QuTiP Admim Team. Herin QuTiP will refer to the project / organisation and qutip
to the library for simulating quantum dyanmics.

Should we be starting again from scratch, then we would probably chose another
name for the main qutip library, such as qutip-quantdyn. However, qutip is
famous, and the name will stay.


Library package structure
=========================

With a name as general as Quantum Toolkit in Python, the scope for new code
modules to be added to qutip is very wide. The library was becoming increasingly
difficult to maintain, and in c. 2020 the QuTiP Admim Team decided to limit the
scope of the 'main' (for want of a better name) qutip package. This scope is
restricted to components for the simulation (solving) of the dynamics of quantum
systems. The scope includes utilities to support this, including analysis and
visualisation of output.

At the same time, again with the intention of easing maintence, a decision to
limit dependences was agreed upon. Main qutip runtime code components should
depend only upon Numpy and Scipy. Installation (from source) requires Cython,
and some optional components also require Cython at runtime. Unit testing
requires Pytest. Visualisation (optional) components require Matplotlib.

Due to the all encompassing nature of the plan to abstract the linear algebra
data layer, this enhancement (developed as part of a GSoC project) was allowed
the freedom (potential for non-backward compatibility) of requiring a major
release. The timing of such allows for a restructuring of the qutip compoments,
such that some that could be deemed out of scope could be packaged in a
different way -- that is, not installed as part of the main qutip package. Hence
the proposal for different types of package described next. With reference to
the :ref:`discussion above <what-is-qutip>` on the name QuTiP/qutip, the planned
restructuring suffers from confusing naming, which seems unavoidable without
remaining either the organisation or the main package (neither of which are
desirable).

QuTiP family packages
  The main qutip package already has sub-packages,
  which are maintained in the main qutip repo. Any packages maitained by the
  QuTiP organisation will be called QuTiP 'family' packages. Sub-packages within
  qutip main will be called 'integrated' sub-packages. Some packages will be
  maintained in their own repos and installed separately within the main qutip
  folder structure to provide backwards compatibility, these are (will be)
  called qutip optional sub-packages. Others will be installed in their own
  folders, but (most likely) have qutip as a dependency -- these will just be
  called 'family' packages.

QuTiP affilliated packages
  Other packages have been developed by others
  outside of the QuTiP organisation that work with, and are complementary to,
  qutip. The plan is to give some recognition to those that we deem worthy of
  such [this needs clarification]. These packages will not be maintained by the
  QuTiP Team.


Family packages
---------------

.. _qmain:

qutip main
^^^^^^^^^^

* **current package status**: family package `qutip`
* **planned package status**: family package `qutip`

The in-scope components of the main qutip package all currently reside in the
base folder. The plan is to move some components into integrated subpackages as
follows:

- `core` quantum objects and operations
- `solver` quantum dynamics solvers

What will remain in the base folder will be miscellaneous modules. There may be
some opportunity for grouping some into a `visualisation` subpackage. There is
also some potential for renaming, as some module names have underscores, which
is unconventional.

Qtrl
^^^^

* **current package status**: integrated sub-package `qutip.control`
* **planned package status**: family package `qtrl`

There are many OSS Python packages for quantum control optimisation. There are
also many different algorithms. The current `control` integrated subpackage
provides the GRAPE and CRAB algorithms. It is too ambitious for QuTiP to attempt
(or want) to provide for all options. Control optimisation has been deemed out
of scope and hence these components will be separated out into a family package
called Qtrl.

Potentially Qtrl may be replaced by separate packages for GRAPE and CRAB, based
on the QuTiP Control Framework.

QIP
^^^

* **current package status**: integrated sub-package `qutip.qip`
* **planned package status**: family package `qutip-qip`

The QIP subpackage has been deemed out of scope (feature-wise). It also depends
on `qutip.control` and hence would be out of scope for dependency reasons. A
separate repository has already been made for qutip-qip.

qutip-symbolic
^^^^^^^^^^^^^^

* **current package status**: independent package `sympsi`
* **planned package status**: family package `qutip-symbolic`

Long ago Robert Johansson and Eunjong Kim developed Sympsi. It is a fairly
coomplete library for quantum computer algebra (symbolic computation). It is
primarily a quantum wrapper for `Sympy <https://www.sympy.org>`_.

It has fallen into unmaintained status. The latest version on the `sympsi repo
<https://github.com/sympsi/sympsi>`_ does not work with recent versions of
Sympy. Alex Pitchford has a `fork <https://github.com/ajgpitch/sympsi>`_ that
does 'work' with recent Sympy versions -- unit tests pass, and most examples
work. However, some (important) examples fail, due to lack of respect for
non-commuting operators in Sympy simplifcation functions (note this was true as
of Nov 2019, may be fixed now).

There is a [not discussed with RJ & EK] plan to move this into the QuTiP family
to allow the Admin Team to maintain, develop and promote it. The 'Sympsi' name
is cute, but a little abstract, and qutip-symbolic is proposed as an
alternative, as it is plainer and more distinct from Sympy.


Affilliated packages
--------------------

qucontrol-krotov
^^^^^^^^^^^^^^^^

* **code repository**: https://github.com/qucontrol/krotov

A package for quantum control optimisation using Krotov, developed mainly by
Michael Goerz.

Generally accepted by the Admin Team as well developed and maintained. A solid
candiate for affilliation.


Development Projects
====================

.. _solve-dl:

Solver data layer integration
-----------------------------

:tag: solve-dl
:status: development ongoing
:admin lead: `Eric <https://github.com/Ericgig>`_
:main dev: `Eric <https://github.com/Ericgig>`_

The new data layer gives opportunity for significantly improving performance of
the qutip solvers. Eric has been revamping the solvers by deploying `QobjEvo`
(the time-dependent quantum object) that he developed. `QobjEvo` will exploit
the data layer, and the solvers in turn exploit `QobjEvo`.

.. _qtrl-mig:

Qtrl migration
--------------

:tag: qtrl-mig
:status: conceptualised
:admin lead: `Alex <https://github.com/ajgpitch>`_
:main dev: TBA

The components currently packaged as an integrated subpackage of qutip main will
be moved to separate package called Qtrl. This is the original codename of the
package before it was integrated into qutip. Also changes to exploit the new
data layer will be implemented.

.. _ctrl-fw:

QuTiP control framework
-----------------------

:tag: ctrl-fw
:status: conceptualised
:admin lead: `Alex <https://github.com/ajgpitch>`_
:main dev: TBA

Create new package qutip-ctrlfw "QuTiP Control Framework". The aim is provide a
common framework that can be adopted by control optimisation packages, such that
different packages (algorithms) can be applied to the same problem.

Classes for defining a controlled system:

- named control parameters. Scalar and n-dim. Continuous and discrete variables
- mapping of control parameters to dynamics generator args
- masking for control parameters to be optimised

Classes for time-dependent variable parameterisation

- piecewise constant
- piecewise linear
- Fourier basis
- more

Classes for defining an optimisation problem:

- single and multiple objectives

.. _qutip-optim:

QuTiP optimisation
------------------

:tag: qutip-optim
:status: conceptualised
:admin lead: `Alex <https://github.com/ajgpitch>`_
:main dev: TBA

A wrapper for multi-variable optimisation functions. For instance those in
`scipy.optimize` (Nelder-Mead, BFGS), but also others, such as Bayesian
optimisation and other machine learning based approaches. Initially just
providing a common interface for quantum control optimisation, but applicable
more generally.

.. _sympsi-mig:

Sympsi migration
----------------

:tag: sympsi-mig
:status: conceptualised
:admin lead: `Alex <https://github.com/ajgpitch>`_
:main dev: TBA

Create a new family package qutip-symbolic from ajgpitch fork of Sympy. Must
gain permission from Robert Johansson and Eunjong Kim. Extended Sympy simplify
to respect non-commuting operators. Produce user documentation.

.. _status-mig:

Status messaging and recording
------------------------------

:tag: status-msg
:status: conceptualised
:admin lead: `Alex <https://github.com/ajgpitch>`_
:main dev: TBA

QuTiP has various ways of recording and reporting status and progress.

- `ProgressBar` used by some solvers
- Python logging used in qutip.control
- `Dump` used in qutip.control
- heom records `solver.Stats`

Some consolidation of these would be good.

Some processes (some solvers, correlation, control optimisation) have many
stages and many layers. `Dump` was initially developed to help with debugging,
but it is also useful for recording data for analysis. qutip.logging_utils has
been criticised for the way it uses Python logging. The output goes to stderr
and hence the output looks like errors in Jupyter notebooks.

Clearly, storing process stage data is costly in terms of memory and cpu time,
so any implementation must be able to be optionally switched on/off, and avoided
completely in low-level processes (cythonized components).

Required features:

- optional recording (storing) of process stage data (states, operators etc)
- optionally write subsets to stdout
- maybe other graphical representations
- option to save subsets to file
- should ideally replace use of `ProgressBar`, Python logging, `control.Dump`, `solver.Stats`

.. _qutip-gui:

qutip Interactive
-----------------

:status: conceptualised
:tag: qutip-gui
:admin lead: `Alex <https://github.com/ajgpitch>`_
:main dev: TBA

QuTiP is pretty simple to use at an entry level for anyone with basic Python
skills. However, *some* Python skills are necessary. A graphical user interface
(GUI) for some parts of qutip could help make qutip more accessible. This could
be particularly helpful in education, for teachers and learners.

This would make an good GSoC project. It is independent and the scope is
flexible.

The scope for this is broad and flexible. Ideas including, but not limited to:

Interactive Bloch sphere
^^^^^^^^^^^^^^^^^^^^^^^^

Matplotlib has some interactive features (sliders, radio buttons, cmd buttons)
that can be used to control parameters. They are a bit clunky to use, but they
are there. Could maybe avoid these and develop our own GUI. An interactive Bloch
sphere could have sliders for qubit state angles. Buttons to add states, toggle
state evolution path.

Interactive solvers
^^^^^^^^^^^^^^^^^^^

Options to configure dynamics generators (Lindbladian / Hamiltonian args etc)
and expectation operators. Then run solver and view state evolution.

Animated circuits
^^^^^^^^^^^^^^^^^

QIP circuits could be animated. Status lights showing evolution of states during
the processing. Animated Bloch spheres for qubits.


Completed Development Projects
==============================

.. _dl-abs:

data layer abstraction
----------------------

:tag: dl-abs
:status: completed
:admin lead: `Eric <https://github.com/Ericgig>`_
:main dev: `Jake Lishman <https://github.com/jakelishman>`_

Development completed as a GSoC project. Fully implemented in the dev.major
branch. Currently being used by some research groups.

Abstraction of the linear algebra data from code qutip components, allowing
for alternatives, such as sparse, dense etc. Difficult to summarize. Almost
every file in qutip affected in some way. A major milestone for qutip.
Significant performance improvements throughout qutip.

Some developments tasks remain, including providing full control over how the
data-layer dispatchers choose the most appropriate output type.

.. _qmain-reorg:

qutip main reorganization
-------------------------

:tag: qmain-reorg
:status: completed
:admin lead: `Eric <https://github.com/Ericgig>`_
:main dev: `Jake Lishman <https://github.com/jakelishman>`_

Reorganise qutip main components to the structure :ref:`described above <qmain>`.

.. _qmain-docs:

qutip user docs migration
-------------------------

:tag: qmain-docs
:status: completed
:admin lead: `Jake Lishman <https://github.com/jakelishman>`_
:main dev: `Jake Lishman <https://github.com/jakelishman>`_

The qutip user documentation build files are to be moved to the qutip/qutip
repo. This is more typical for an OSS package.

As part of the move, the plan is to reconstruct the Sphinx structure from
scratch. Historically, there have been many issues with building the docs.
Sphinx has come a long way since qutip docs first developed. The main source
(rst) files will remain [pretty much] as they are, although there is a lot of
scope to improve them.

The qutip-doc repo will afterwards just be used for documents, such as this one,
pertaining to the QuTiP project.

.. _qip-mig:

QIP migration
-------------

:tag: qip-mig
:status: completed
:admin lead: `Boxi <https://github.com/BoxiLi>`_
:main dev: `Sidhant Saraogi <https://github.com/sarsid>`_

A separate package for qutip-qip was created during Sidhant's GSoC project.
There is some fine tuning required, especially after qutip.control is migrated.

.. _heom-revamp:

HEOM revamp
-----------

:tag: heom-revamp
:status: completed
:admin lead: `Neill <https://github.com/nwlambert>`_
:main dev: `Simon Cross <https://github.com/hodgestar>`_, `Tarun Raheja <https://github.com/tehruhn>`_

An overhaul of the HEOM solver, to incorporate the improvements pioneered in BoFiN.

.. _release roadmap:

QuTiP major release roadmap
===========================

QuTiP v.5
---------

These Projects need to be completed for the qutip v.5 release.

- :ref:`dl-abs` (completed)
- :ref:`qmain-reorg` (completed)
- :ref:`qmain-docs` (completed)
- :ref:`solve-dl` (in-progress)
- :ref:`qip-mig` (completed)
- :ref:`qtrl-mig`
- :ref:`heom-revamp` (completed)

The planned timeline for the release is:

- **alpha version, December 2022**. Core features packaged and available for
  experienced users to test.
- **beta version, January 2023**. All required features and documentation complete,
  packaged and ready for community testing.
- **full release, April 2023**. Full tested version released.

Planned supported environment:

- python 3.8 .. 3.11
- numpy 1.20 .. 1.23
- scipy 1.5 .. 1.8
