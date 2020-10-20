#####################
Dynamics (ODE) solver
#####################

Once either a bosonic or fermionic problem has been defined, one can solve the dynamics for a particular initial condition. This was already shown in the bosonic class example, but in more detail, given a defined BosonicHEOMSolver instance,
one can call:

.. code-block:: python

    HEOMMats = BosonicHEOMSolver(Hsys, Q2, ckAR, ckAI, vkAR, vkAI, NC, options=options)
    #Run ODE solver
    resultMats = HEOMMats.run(rho0, tlist) 

By default this takes only two parameters, rho0, a quantum object state or density matrix defining in the initial condition for the system, and tlist, a set of timesteps for which to return results.  The object returned
is a standard QuTiP results object. Most importantly, this contains the system state at each time-step in resultmMats.states
 
Note that at this time this auxiliary density operators are not returned directly to the user.  This will be modified in the near future to be an option (currently they are returned by default in the steadystate() solver).

 
 The boson-fast solver contains some additional options to use Intel MKL parralization, that can help speed-up the ODE solution.  