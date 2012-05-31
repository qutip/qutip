################################
# QuTiP README FILE FOR CPC
# Version: 2.0.0
# P.D. Nation and J.R. Johansson
################################


USERS GUIDE
-----------        
An in-depth user guide, including installation instructions, may be found at the QuTiP homepage:

http://code.google.com/p/qutip/ 
        
        
DEMO SCRIPTS
------------
QuTiP contains a collection of built-in demo scripts that may be called from the Python command line via:
	
>>> from qutip import *
>>> demos()


FILES LIST
----------
COPYING.txt                         # Copy of GPL-3.
INSTALL.txt                         # Installation instructions.
README.txt                          # This file for Comp. Phys. Comm.
RELEASE.txt                         # Release notes for each version of QuTiP.
setup.py                            # QuTiP Python installation script.
    
	cyQ/
        __init__.py             # Initialize cyQ modules.
        codegen.py		    	# Class for automatically generating Cython files for time-dependent problems.
    	cy_mc_funcs.c           # Cython generated C-code for Monte-Carlo solver.
        cy_mc_funcs.pyx         # Cython source code for cy_mc_funcs.c.
    	ode.rhs.c		    	# Cython generated C-code for ODE RHS multiplication.
        ode.rhs.pyx		    	# Cython source code for ode_rhs.c.
    	setup.py                # Setup file for generating *.c files from *.pyx.
    
	examples/
        __init__.py             # Initialize examples modules.
  		ex_10.py				# Demo script.
		ex_11.py				# Demo script.
		ex_12.py				# Demo script.
		ex_13.py				# Demo script.
		ex_14.py				# Demo script.
		ex_15.py				# Demo script.
		ex_16.py				# Demo script.
		ex_17.py				# Demo script.
		ex_18.py				# Demo script.
		ex_19.py				# Demo script.
		ex_20.py				# Demo script.
		ex_21.py				# Demo script.
		ex_22.py				# Demo script.
		ex_23.py				# Demo script.
		ex_24.py				# Demo script.
		ex_25.py				# Demo script.
		ex_26.py				# Demo script.
		ex_27.py				# Demo script.
		ex_30.py				# Demo script.
		ex_31.py				# Demo script.
		ex_32.py				# Demo script.
		ex_33.py				# Demo script.
		ex_34.py				# Demo script.
		ex_40.py				# Demo script.
		ex_41.py				# Demo script.
		ex_42.py				# Demo script.
		ex_43.py				# Demo script.
		ex_50.py				# Demo script.
		examples_text.py		# List of demo names and descriptions loaded by demos GUI.
		exconfig.py				# Module containing parameters for launching demo based on users command.
    
	gui/
        __init__.py             # Initialize GUI modules.
        AboutBox.py             # Class for generating QuTiP About box.
        Examples.py             # Class for demos GUI or command line. 
        logo.png                # QuTiP logo used in About box.
        ProgressBar.py          # Class for Monte-Carlo progress bar GUI.
		syntax.py				# Class for syntax highlighting used in demos GUI.
	
	qutip/
        __init__.py                 # Initialize qutip modules.
        _reset.py					# Resets global QuTiP settings and odeconfig data
		_version.py                 # Holds qutip version information.
		about.py                    # Calls about box.
        bloch-redfield.py           # Bloch-Redfield master equation solver.
        Bloch.py                    # Class generating a Bloch sphere plot.
        clebsch.py                  # Calculates Clebsch-Gordon coefficients.
        correlation.py              # Calculates two-time correlation functions <A(tau)B(0)>.
        demos.py                    # Runs the demos GUI or command line script.
        entropy.py                  # Module of functions used for calculating various entropy measures.
        eseries.py                  # Class defining the exponential series object.
        essolve.py                  # Evolution of a state vector or density matrix with ODE expressed as eseries object.
        expect.py                   # Calculates expectation values.
        fileio.py                   # Module for saving and loading Qobj objects and data sets.
		floquet.py                  # Floquet-Markov master equation solver.
        gates.py                    # A Module of select gates for use in quantum computation calculations.
        graph.py                    # Draws a Hinton diagram for visualizing a density matrix.
        istests.py                  # A collection of tests for determining the properties of a Qobj object.
        mcsolve.py                  # Monte-Carlo trajectory solver.
		mesolve.py                  # Lindblad master equation solver.        
		metrics.py                  # A collection of density matrix metrics (distance measures).
		odechecks.py				# Routine for determining which ode solver to use based on user input.
        odeconfig.py				# Holds data arrays for use by ode solvers.
		Odeoptions.py               # Class of options for ODE solvers.
		Odedata.py                  # Class for holding output data from ODE solvers.
        operators.py                # A collection of commonly used quantum operators.
        orbital.py                  # Calculates an angular wave function on a sphere.
        parfor.py                   # Runs a for-loop in parallel for a given single-variable function. 
        propagator.py               # Calculate the propagator U(t) for the density matrix or wave function.
        ptrace.py                   # Performs a partial trace of a given composite quantum object.
        Qobj.py                     # The main quantum object class.  Defines the key properties of the quantum object in QuTiP.
        qstate.py                   # Generates coupled states of qubits where each qubit is in the |up> or |down> state. 
        rand.py						# A collection of routines for generating random quantum operators and states.
		rhs_generate.py				# Generates Cython code for use in simulating time-dependent problems over an array of input variables.
		simdiag.py                  # Performs simultaneous diagonalization of commuting, Hermitian operators
        sparse.py					# Sparse eigensolver for quantum objects.
		sphereplot.py               # Plots spherical wave functions generated by orbital.py.
        states.py                   # A collection of commonly used state vectors and density matrices
        steady.py                   # Calculated the steady state evolution for a given Hamiltonian
        superoperator.py            # Module of superoperators used for converting a Hamiltonian into a Louvillian.
        tensor.py                   # Generates a composite quantum object from two or more state vectors or density matricies.
        three_level_atom.py         # A collection of commonly used states and operators for three-level atoms.
        wigner.py                   # Generates the Wigner function and Q function for a given state vector or density matrix.

        
        
        
        
        
        
        
        
        
        
        
        
        
        
    