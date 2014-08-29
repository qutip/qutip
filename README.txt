################################
# QuTiP README FILE FOR CPC
# Version: 2.2.0
# Released: 2013/03/01
# P.D. Nation and J.R. Johansson
################################


USERS GUIDE
-----------        
An in-depth user guide, including installation instructions, 
may be found at the QuTiP homepage:

http://code.google.com/p/qutip/ 
        
        
DEMO SCRIPTS
------------
QuTiP contains a collection of built-in demo scripts that 
may be called from the Python command line via:
    
>>> from qutip import *
>>> demos()


FILES LIST
----------
COPYING.txt                         # Copy of GPL-3.
INSTALL.txt                         # Installation instructions.
README.txt                          # This file for Comp. Phys. Comm.
RELEASE.txt                         # Release notes for QuTiP.
setup.py                            # QuTiP Python installation script.

qutip/
        __init__.py                 # Initialize qutip modules.
        _reset.py                   # Resets global QuTiP settings
        _version.py                 # Holds qutip version information.
        about.py                    # Calls about box.
        bloch-redfield.py           # Bloch-Redfield master equation.
        Bloch.py                    # Bloch sphere plot.
        Bloch3d.py                  # 3D Bloch sphere plot.
        continuous_variables.py     # Continuous variables routines.
        correlation.py              # Two-time correlation functions.    
        demos.py                    # Runs demos GUI or command line.
        entropy.py                  # Various entropy measures.
        eseries.py                  # Exponential series object.
        essolve.py                  # Exponential series ODE solver.
        expect.py                   # Calculates expectation values.
        fileio.py                   # Saving and loading data.
        floquet.py                  # Floquet-Markov master equation.
        gates.py                    # Quantum computation gates.
        graph.py                    # Visualization scripts.
        ipynbtools.py               # Tools for QuTiP in IPython notebooks.
        istests.py                  # Properties of a Qobj object.
        mcsolve.py                  # Monte-Carlo trajectory solver.
        mesolve.py                  # Lindblad master equation solver.        
        metrics.py                  # Density matrix metrics.
        odechecks.py                # Checks for ODE inputs.
        odeconfig.py                # Holds ODE data.
        Odeoptions.py               # Class of options for ODE solvers.
        Odedata.py                  # ODE output data class.
        operators.py                # Commonly used quantum operators.
        orbital.py                  # Angular wave function on a sphere.
        parfor.py                   # Parallel for-loop
        partial_transpose.py        # Partial transpose routine.
        propagator.py               # Propagator U(t) for density matrix.
        ptrace.py                   # Partial trace of a composite object.
        Qobj.py                     # Main quantum object class.
        qstate.py                   # Generates coupled states of qubits. 
        random_objects.py           # Random quantum operators and states.
        rhs_generate.py             # Generates Cython code at runtime.
        sesolve.py                  # Module for Schrodinger solvers.
        simdiag.py                  # Simultaneous diagonalization.
        sparse.py                   # Sparse eigensolvers.
        sphereplot.py               # Plots spherical wave functions.
        states.py                   # State vectors and density matrices.
        steady.py                   # Steady state evolution.
        stochastic.py               # Module for SDE solvers.
        superoperator.py            # Superoperators for Louvillian.
        tensor.py                   # Generates composite quantum objects.
        testing.py                  # Module for running QuTiP unit tests.
        three_level_atom.py         # Operators for three-level atoms.
        tomography.py               # Quantum process tomography functions.
        utilities.py                # Collection of utility functions.
        visualization.py            # Various visualization functions.
        wigner.py                   # Wigner function and Q functions.
    
        cyQ/
            __init__.py             # Initialize cyQ modules.
            codegen.py              # Class to auto gen. Cython files.
            spmatfuncs.c            # C file gen. from spmatfuncs.pyx.
            spmatfuncs.pyx          # Cython base matrix utils.
            setup.py                # Setup file for *.pyx -> *.c.
    
    examples/
            __init__.py             # Initialize examples modules.
            ex_10.py                # Demo script.
            ex_11.py                # Demo script.
            ex_12.py                # Demo script.
            ex_13.py                # Demo script.
            ex_14.py                # Demo script.
            ex_15.py                # Demo script.
            ex_16.py                # Demo script.
            ex_17.py                # Demo script.
            ex_18.py                # Demo script.
            ex_19.py                # Demo script.
            ex_20.py                # Demo script.
            ex_21.py                # Demo script.
            ex_22.py                # Demo script.
            ex_23.py                # Demo script.
            ex_24.py                # Demo script.
            ex_25.py                # Demo script.
            ex_26.py                # Demo script.
            ex_27.py                # Demo script.
            ex_30.py                # Demo script.
            ex_31.py                # Demo script.
            ex_32.py                # Demo script.
            ex_33.py                # Demo script.
            ex_34.py                # Demo script.
            ex_40.py                # Demo script.
            ex_41.py                # Demo script.
            ex_42.py                # Demo script.
            ex_43.py                # Demo script.
            ex_50.py                # Demo script.
            examples_text.py        # List of demos
            exconfig.py             # Examples config file.
    
    gui/
            __init__.py             # Initialize GUI modules.
            AboutBox.py             # QuTiP About box.
            Examples.py             # Demos GUI or command line. 
            icon.png                # Icon for GUI Windows.
            logo.png                # QuTiP logo used in About box.
            ProgressBar.py          # Progress bar GUI.
            syntax.py               # Syntax highlighting.

    tests/
            __init__.py             # Initialize testing modules.
            test_basis_trans.py     # Basis transformation test scripts.
            test_eigenstates.py     # Eigenstate/eigenvalue test scripts.
            test_entropy.py         # Entropy test scripts.
            test_fileio.py          # Read & write file test scripts.
            test_mcsolve.py         # Monte-Carlo solver tests scripts.
            test_mesolve.py         # Lindlad master equation tests.
            test_odechecks.py       # ODE configuration test scripts.
            test_operators.py       # Quantum operator test scripts.
            test_Qobj.py            # Qobj test scripts.
            test_qubit_evolve.py    # Test script for qubit dynamics.  
            test_rand.py            # Random operator/state test scripts.
            test_sp_eigs.py         # Sparse/Dense eigenvalue/vector test.
            test_states.py          # Quantum state test scripts.
            test_steadystate.py     # Steadystate solver test scripts.
            test_superoperator.py   # Quantum superoperator test scripts.
            test_wigner.py          # Wigner function test scripts.

