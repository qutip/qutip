We are applying under the NumFocus umbrella. Please read the NumFocus guidelines for students. You will need to register on the Google Summer of Code website - https://summerofcode.withgoogle.com/ and make a final proposal before the deadline. You can add your draft and contact respective mentors so that we can provide feedback before the deadline.

You can look at the 2019 students blogs projects to get an idea of the work:

- `qutip.qip.noise`: [Noise in quantum circuits](https://gsoc2019-boxili.blogspot.com/)

- `qutip.lattice`: [1D Lattice dynamics](https://latticemodelfunctions.blogspot.com/)

# Learn about QuTiP
The best way to learn about QuTiP is to study the [lectures and tutorials](http://qutip.org/tutorials.html) on our [website](http://qutip.org). If you need help, then look first in the [documentation](http://qutip.org/docs/latest/index.html), then the [Help Group](https://groups.google.com/forum/#!forum/qutip) - post a message if you need to.

# Applications
In order to demonstrate your communication, technical skills, and or, scientific knowledge, please complete one or more of these tasks before applying. Be sure to refer to them clearly in your application. Note that we have three other associated repositories.
* [Notebooks](https://github.com/qutip/qutip-notebooks)
* [Documentation](https://github.com/qutip/qutip-doc)
* [Website configuration](https://github.com/qutip/qutip.github.io)

All have open issues, those labelled 'help wanted' are a good place to look. Understanding of quantum dynamics, technical (programming) skills, and being helpful to users, are all highly valued by QuTiP. Make sure to read the
[Contributing to QuTiP Development](https://github.com/qutip/qutip-doc/blob/master/qutip_dev_contrib.md) guidelines if you are submitting a Pull Request.

## Tasks before applying
* Submit a pull request in QuTiP solving some issue from the issues page.
* Make some helpful suggestion of how to resolve some issue or enhance the toolkit. The more specific the better.
* Help some user who is having technical or usage problems on the [Help Group](https://groups.google.com/forum/#!forum/qutip)

# Projects

## 1. Error mitigation in QuTiP

From QuTiP 4.5 release, the `qutip.qip` module now contains the `qutip.qip.noise` toolbox (which was a GSoC project), to provide enhanced features for pulse-level description of quantum circuits and noise models.

Many features for noise model and efficient quantum circuit compilation, can be improved.

Many devices can be added as backends, such as ion traps and specific models, as well as integration with existing general framework for quantum circuits.

One of the interesting possibilities is to include error mitigation techniques [1-3].

See also the [Github Project](https://github.com/qutip/qutip/projects/4) page for a collection of related issues and ongoing Pull Requests.
### Expected outcomes
* More devices defined in the `qutip.qip` module
* Features to perform error mitigation techniques in QuTiP, such as zero-error extrapolation
* APIs to other quantum circuits libraries and quantum assembly language (qasm)
### Skills
* Background in quantum physics and quantum circuits.
* Git, python and familiarity with the Python scientific computing stack
### Mentors
* Nathan Shammah (nathan.shammah@gmail.com)
* Alex Pitchford (alex.pitchford@gmail.com)
* Eric Giguère (eric.giguere@usherbrooke.ca)
* Neill Lambert (nwlambert@gmail.com)
* Boxi Li (etamin1201@gmail.com) [QuTiP GSoC 2019 graduate]


### Difficulty
Medium

### References
[1] Kristan Temme, Sergey Bravyi, Jay M. Gambetta, **Error mitigation for short-depth quantum circuits**, Phys. Rev. Lett. 119, 180509 (2017)

[2] Abhinav Kandala, Kristan Temme, Antonio D. Corcoles, Antonio Mezzacapo, Jerry M. Chow, Jay M. Gambetta,
 **Extending the computational reach of a noisy superconducting quantum processor**, Nature *567*, 491 (2019)

[3] S. Endo, S.C. Benjamin, Y. Li, **Practical quantum error mitigation for near-future applications**, Physical Review X *8*, 031027 (2018)

## 2. Abstraction of the quantum object class (`qutip.qobj.Qobj`)

QuTiP's `Qobj` class uses sparse matrices (csr) to store data by default. Recently, we have had some issues due to using int32 for the sparse matrix indices in QuTiP (see #845, #842, #828, #853). Also, in smaller problems, using a sparse matrix for storing data is not optimal, (see the detailed discussion by @agpitch in #437). Therefore there needs to be an abstraction of the quantum object class such that one can use any structure to store the underlying data. A starting point would be the possibility to switch between dense/sparse/int32/int64 and then to determine what other parts of the code are affected by this change. The disentangling of the matrix representation of the data has several benefits which can allow us to use other types of linear algebra tools (Numba, TensorFlow). This project would be challenging as the components are integral to the library and hence changes would have wide-reaching implications. Even beyond GSoC, the abstraction of the quantum object class can lead to some very interesting directions for QuTiP.

But as a first goal, enabling int32/int64 indices for sparse along with a switch for dense/sparse in a consistent manner should be within the timeline for GSoC 2019.

Read the relevant discussions -
* [#850](https://github.com/qutip/qutip/issues/850), [#437](https://github.com/qutip/qutip/issues/437), [#845](https://github.com/qutip/qutip/issues/845), [#842](https://github.com/qutip/qutip/issues/842), [#828](https://github.com/qutip/qutip/issues/828), [#853](https://github.com/qutip/qutip/issues/853)

* [Ericgig's implementation of int64 indices for sparse matrices](https://github.com/Ericgig/qutip/tree/long)

### Expected outcomes
* An encapsulation of the quantum object class which can switch between dense/sparse matrices with the possibility of int32/int64 for indices
* Updating of other parts of the code which assume default sparse behavior of `Qobj`
* Performance analysis.
### Skills
* Git, python and familiarity with the Python scientific computing stack
### Mentors
* Eric Giguère (eric.giguere@usherbrooke.ca)
* Alex Pitchford (alex.pitchford@gmail.com)

### Difficulty
Hard

## 3. Modernize distribution, testing and installation

QuTiP deployment is currently suffering from increasing issues arising in setup, platform-specific deployment, broken gcc compilation, and lacks novel interactive ways to reach out to the community.

Installation issues: 1.a) Fix MacOS freed object error https://github.com/qutip/qutip/issues/963 1.b) Overhaul of our setup.py and installation instructions. We believe that there could be a lot of redundant stuff in there. Related, but not exclusively https://github.com/qutip/qutip/pull/961, also compile flags https://github.com/qutip/qutip/issues/951 1.c) Automation of pip wheels https://github.com/qutip/qutip/issues/933
QuTiP users are experiencing broken installations on several MacOS and QuTiP versions. We would also like to update the setup procedure. Pypinfo data shows that over 8k users used “pip install qutip” in 2018. QuTiP does not currently support pip wheels, and this could be a key factor in smoothing out the installation process.

Testing issues: 2.a) Migrate tests to pytest or nose2 https://github.com/qutip/qutip/issues/958 2.b) Setup MS Windows CI tests https://github.com/qutip/qutip/issues/959
QuTiP prizes itself of implementing some of open-source best practices in code writing and deployment since its inception. Over time, some of the open-source software used for testing has been surpassed by newer standards and features. Nose for example, is not currently maintained. Switching to an actively maintained alternative, such as pytest, would be beneficial also for debugging during continuous integration. We would like to extend continuous integration to Windows, which is currently not officially supported even for installation. Data from pypinfo shows that over 25% of QuTiP users installing via pip are on Windows machines, a popularity also highlighted by threads on the Google Help group (177 posts, https://groups.google.com/forum/#!topic/qutip/DaxVxT8SWuI%5B1-25%5D, latest on March 28th, 2019).

Distribution enhancements: 3.a) Migrate docs to new Sphinx build https://github.com/qutip/qutip-doc/pull/70 3.b) Move example Jupyter notebooks to live server (updating MyBinder or using Colab, setting up a JupyterHub environment)
This last point would greatly help enhance the engagement of the user community with QuTiP. Right now, building the documentation is clumsy due to the way Sphinx is built in QuTiP. An additional point of great benefit for the community would be to find interactive options for the over 60 Jupyter notebooks hosted at http://qutip.org/tutorials.html, a true treasure for the practitioners and students of quantum mechanics and in particular of open quantum systems.

### Expected outcomes
* More stable `pip` and `conda` release versions
* Better documentation
* Simpler pipeline for QuTiP releases
### Skills
* Git, python and familiarity with the Python scientific computing stack
### Mentors
* Nathan Shammah (nathan.shammah@gmail.com)
* Eric Giguère (eric.giguere@usherbrooke.ca)
* Alex Pitchford (alex.pitchford@gmail.com)
* Boxi Li (etamin1201@gmail.com) [QuTiP GSoC 2019 graduate]

### Difficulty
Medium


## 4. Improve Quantum Circuits Efficiency and Portability

QuTiP has been downloaded on conda-forge 200k times, of which 150k only in the past year. The popularity of this quantum physics library, focused on quantum dynamics simulation, demands that its quantum circuit simulation capabilities get up to date with a new ecosystem that has emerged in open-source quantum computing.

The aim of this proposal is to enhance QuTiP features with regard to quantum information processing. While QuTiP core modules deal with dynamics simulation, there is a module for quantum circuits that has recently been enhanced in capabilities and features during one of the Google Summer of Code projects 2019 (GSoC).

### Expected outcomes
* Faster quantum circuits, benchmarking against other available solutions (e.g., [yao.jl](yao.jl)).

Import/export of quantum circuits to standard format and to other libraries:
* Develop functions that allow to export objects of the qutip.QuantumCircuit class to an object that are compatible with other implementations intermediate representation of quantum circuits.
* Write functions Quantum Assembly Language (QASM) an intermediate representation for quantum instructions.
* Write specific functions and API to import and export circuits from Qiskit (IBM Research’s popular library that provides access to a quantum computer in the backend) and Cirq (Google Research’s quantum circuit library).
* Further development the quantum information noise simulation module developed in GSoC 2019 project, extending it to stochastic dynamics.
* Extend the quantum information processing and quantum circuit capabilities.

### Skills
* Git, python and familiarity with the Python scientific computing stack; quantum information processing and quantum computing (quantum circuit formalism)
### Mentors
* Nathan Shammah (nathan.shammah@gmail.com)
* Eric Giguère (eric.giguere@usherbrooke.ca)
* Alex Pitchford (alex.pitchford@gmail.com)

### Difficulty
Medium

