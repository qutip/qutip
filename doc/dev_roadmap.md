# QuTiP Development Roadmap

## Preamble
This document outlines plan and ideas for the current and future development of QuTiP.
The document is maintained by the QuTiP admim team. Contributuions from the QuTiP Community are very welcome.

In particular this document outlines plans for the next major release of qutip, which will be version 5. 
And also plans and dreams beyond the next major version.

There is lots of development going on in QuTiP that is not recorded in here. This a just an attempt at coordinated stragetgy and ideas for the future.

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
 
 &#x1F537; Do we really need optional subpackages? It seems that are a bit fiddly and as we are side-stepping bw compat with a major version, we could just make QIP a separate package.
 
 ### Family packages
 #### qutip main
 *current package status*: family package `qutip`
 
 *planned package status*: family package `qutip`
 
 The in scope components of the main qutip package all currently reside in the base folder.
 The plan is to move some components into integrated subpackages as follows.
 
 - `core` quantum objects and operations
 - `solver` quantum dynamics solvers
 
 What will remain in the base folder will be miscellaneous modules. There may be some opportunity for grouping some into a `visualisation` subpackage.
 
  #### Qtrl
 *current package status*: integrated sub-package `qutip.control`
 
 *planned package status*: family package `qtrl`
 
 There are many OSS Python packages for quantum control optimisation. There are also many different algorithms. 
 The current `control` integrated subpackage provides the GRAPE and CRAB algorithms. 
 It is too ambitious for QuTiP to attempt (or want) to provide for all options.
 Control optimisation has been deemed out of scope and hence these components will be separated out into a family package called Qtrl.
 
 Potentially Qtrl may be replaced by separate packages for GRAPE and CRAB, based on the QuTiP Control Framework.
 
 #### QIP
 *current package status*: integrated sub-package `qutip.qip`
 
 *planned package status*: optional sub-package `qutip.qip`
 
&#x1F537; Is it really necessary for this to be a sub-package? It could just be a separate package

The QIP subpackage has been deemed out of scope (feature-wise). It also depends on `qutip.control` and hence would be out of scope for dependency reasons.
A separate repository has already been made for qutip-qip.
 
 #### qutip-symbolic
 *current package status*: independent package `sympsi`
 
 *planned package status*: family package `qutip-symbolic`
 
 Long ago Robert Johansson developed Sympsi. It is a fairly coomplete library for quantum computer algebra (symbolic computation). 
 It is primarily a quantum wrapper for [Sympy](https://www.sympy.org).
 
 It has fallen into unmaintained status. The latest version on the [sympsi repo](https://github.com/sympsi/sympsi) does not work with recent versions of Sympy.
 Alex Pitchford has a [fork](https://github.com/ajgpitch/sympsi) that does 'work' with recent Sympy versions -- unit tests pass, and most examples. 
 However, some (important) examples fail, due to lack of respect for non-commuting operators in Sympy simplifcation functions (note this was true as of Nov 2019, may be fixed now).
 
 There is [not discussed with RJ) to move this into the QuTiP family to allow the Admin Team to maintain, develop and promote it. 
 The 'Sympsi' name is cute, but a little abstract, and qutip-symbolic is proposed as an alternative, as it is plainer.
 
 ### Affilliated packages
 

 ## Workpackages
 
 ### data layer abstraction 
 &#x1F535; tag: dl-abs
 
 ### qutip main reorganization
 
 ### QIP migration
 
 ### Qtrl migration
 
 ### Sympy non-commuting operator simplify
 
 ### QuTiP control framework
 
 ### Status messaging and recording
 
 ## qutip major release roadmap
 
 ### qutip v.5
 These workpackages need to be completed for the qutip v.5 release.
 
  - [dl-abs](data layer abstraction)
 
 


