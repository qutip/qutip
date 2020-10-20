#####################
Steady-state Solver
#####################

Once either a bosonic or fermionic problem has been defined, one can solve the steady-state. By default this is done by direct SPLU decomposition (combined with the standard imposition of normalization on the system density operator).

Typical usage is:

.. code-block:: python

    HEOMMats = BosonicHEOMSolver(Hsys, Q2, ckAR, ckAI, vkAR, vkAI, NC, options=options)
    #get steady state
    rho_ss,full_ss=resultHEOM.steady_state() 

rho_ss is standard QuTiP quantum object for the system state alone. full_ss includes that, and every auxiliary density operator, encoding the environment, but in numpy array format (not converted to quantum objects).  
To determine which ADO is which one can use the enr_state_dictionaries() built into QuTiP, and which are used to generate the ordering of all the HEOM operators in our code.  An example of how this is done is presented in example notebook 4b, where it is used to get the electronic current from the ADOs.

We will supplement this with further examples for the bosonic case in the future.
   
Additional options for steady_state() include use_mkl = True which will use MKL parralization for the problem, if available.  Boson-fast also includes approx=True option for use of an iterative lgmres approach.  