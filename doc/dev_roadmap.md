# QuTiP Development Roadmap

## Preamble
This document outlines plan and ideas for the current and future development of QuTiP.
The document is maintained by the QuTiP admim team. Contributuions from the QuTiP Community are very welcome.

In particular this document outlines plans for the next major release of qutip, which will be version 5. 
And also plans and dreams beyond the next major version.

### What is QuTiP?
The name QuTiP refers to a few things. 
Most famously, qutip is a Python library for simulating quantum dynamics. 
To support this, the library also contains various software tools (functions and classes) that have more generic applications, such as linear algebra components and visualisation utilities, and also tools that are specifically quantum related, but have applications beyond just solving dynamics (for instance partial trace computation).

QuTiP is also an organisation, in the Github sense, and in the sense of a group of people working collaboratively towards common objectives, and also a web presence [qutip.org](http://qutip.org/). The QuTiP Community includes all the people who have supported the project since in conception in 2010, including manager, funders, developers, maintainers and users.

These related, and overlapping, uses of the QuTiP name are of little consequence until one starts to consider how to organise all the software packages that are somehow related to QuTiP, and specifically those that are maintained by the QuTiP Admim Team. Herin QuTiP will refer to the project / organisation and qutip to the library for simulating quantum dyanmics.

## Library package structure

With a name as general as Quantum Toolkit in Python, the scope for new code modules to be added to qutip is very wide. The library was becoming increasingly difficult to maintain, and in c. 2020 the QuTiP Admim Team decided to limit the scope of the 'main' (for want of a better name) qutip package. 
This scope is restricted to components for the simulation (solving) of the dynamics of quantum systems.
The scope includes utilities to support this, including analysis and visualisation of output.

At the same time, again with the intention of easing maintence, a decision to limit dependences was agreed upon.
Main `qutip` runtime code components should depend only upon Numpy and Scipy. 
Installation (from source) requires Cython, and some optional components also require Cython at runtime.
Unit testing requires Pytest.

Due to the all encompassing nature of the plan to abstract the linear algebra data layer, this enhancement (developed as part of a GSoC project) was allowed the freedom (potential for non-backward compatibility) of requiring a major release. The timing of such allows for a restructuring of the qutip compoments, such that some that could be deemed out of scope could be packaged in a different way -- that is, not installed as part of the main qutip package. Hence the proposal for different types of package described next. With reference to the [discussion above](#what-is-qutip) on the name QuTiP/qutip, the planned restructering suffers from confusing naming, which seems unavoidable without remaining either the organisation or the main package (neither of which are desirable).

 - QuTiP family packages. The main qutip package already has sub-packages, which are maintained in the main qutip repo. Any packages maitained by the QuTiP organisation will be called QuTiP 'family' packages. Sub-packages within qutip main will be called 'integrated' sub-packages. Some packages will be maintained in their own repos and installed separately within the main qutip folder structure to provide backwards compatibility, these are (will be) called qutip optional sub-packages. Others will be installed in their own folders, but (most likely) have qutip as a dependency -- these will just be called 'family' packages.
 - QuTiP affilliated packages. Other packages have been developed by others outside of the QuTiP organisation that work with, and are complementary to, qutip. The plan is to give some recognition to those that we deserve worthy of such [this needs clarification]. These packages will not be maintained by the QuTiP Team.
 
 ### Family packages
 #### qutip main
 **current package status**: family package `qutip`
 **planned package status**: family package `qutip`
 
 #### QIP
 **current package status**: integrated sub-package `qutip.qip`
 **planned package status**: optional sub-package `qutip.qip`
 
  #### Qtrl
 **current package status**: integrated sub-package `qutip.control`
 **planned package status**: family package `qtrl`
 
 #### Qsymbolic
 **current package status**: independent package `sympsi`
 **planned package status**: family package `qsymbolic`
 
 ### Affilliated packages
 
 


