# QuTiP Development Roadmap

## Preamble
This document outlines plan and ideas for the current and future development of QuTiP.
The document is maintained by the QuTiP admim team. Contributuions from the QuTiP Community are very welcome.

### [What is QuTiP and `qutip`?](#what-is-qutip)
The name QuTiP refers to a few things. 
Most famously, `qutip` is a Python library for simulating quantum dynamics. 
To support this, the library also contains various software tools (functions and classes) that have more generic applications, such as linear algebra components and visualisation utilities, and also tools that are specifically quantum related, but have applications beyond just solving dynamics (for instance partial trace computation).

QuTiP is also an organisation, in the Github sense, and in the sense of a group of people working collaboratively towards common objectives, and also a web presence [qutip.org](http://qutip.org/). The QuTiP Community includes all the people who have supported the project since in conception in 2010, including manager, funders, developers, maintainers and users.

These related, and overlapping, uses of the QuTiP name are of little consequence until one starts to consider how to organise all the software packages that are somehow related to QuTiP, and specifically those that are maintained by the QuTiP Admim Team. Herin QuTiP will refer to the project / organisation and `qutip` to the library for simulating quantum dyanmics.

## Library structure

With a name as general as Quantum Toolkit in Python, the scope for new code modules to be added to `qutip` is very wide. The library was becoming increasingly difficult to maintain, and in c. 2020 the QuTiP Admim Team decided to limit the scope of the 'main' (for want of a better name) qutip package. 
This scope is restricted to components for the simulation (solving) of the dynamics of quantum systems.
The scope includes utilities to support this, including analysis and visualisation of output.

At the same time, again with the intention of easing maintence, a decision to limit dependences was agreed upon.
Main `qutip` runtime code components should depend only upon Numpy and Scipy. 
Installation (from source) requires Cython, and some optional components also require Cython at runtime.
Unit testing requires Pytest.


