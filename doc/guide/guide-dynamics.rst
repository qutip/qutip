.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _guide-dynamics:


An Overview of the Quantum Dynamics Solvers in QuTiP
****************************************************

Unitary evolution
-----------------

The Schrödinger equation, which governs the time-evolution of closed quantum systems, is defined by its Hamiltonian and state vector. In the previous section, [GuideComposite Creating and manipulating composite objects with tensor and ptrace], we showed how Hamiltonians and state vectors are constructed in QuTiP. Given a Hamiltonian, we can calculate the unitary (non-dissipative) time-evolution of an arbitrary state vector :math:`\psi_0` (``psi0``) using the QuTiP function :func:`qutip.odesolve`. It evolves the state vector and evaluates the expectation values for a set of operators ``expt_op_list`` at the points in time in the list ``tlist``, using an ordinary differential equation solver. Alternatively, we can use the function :func:`qutip.essolve`, which uses the exponential-series technique to calculate the time evolution of a system. The :func:`qutip.odesolve` and :func:`qutip.essolve` functions take the same arguments and it is therefore easy switch between the two solvers. 

For example, the time evolution of a quantum spin-1/2 system with tunneling rate 0.1 that initially is in the up state is calculated, and the  expectation values of the :math:`\sigma_z` operator evaluated, with the following code::

    >>> H = 2 * pi * 0.1 * sigmax()
    >>> psi0 = basis(2, 0)
    >>> tlist = linspace(0.0, 10.0, 20.0)
    >>> odesolve(H, psi0, tlist, [], [sigmaz()])
    array([[ 1.00000000+0.j,  0.78914229+0.j,  0.24548596+0.j, -0.40169696+0.j,
            -0.87947669+0.j, -0.98636356+0.j, -0.67728166+0.j, -0.08257676+0.j,
             0.54695235+0.j,  0.94582040+0.j,  0.94581706+0.j,  0.54694422+0.j,
            -0.08258520+0.j, -0.67728673+0.j, -0.98636329+0.j, -0.87947111+0.j,
            -0.40168898+0.j,  0.24549302+0.j,  0.78914528+0.j,  0.99999927+0.j]])

The brackets in the fourth argument is an empty list of collapse operators,  since we consider unitary evolution in this example. See the next section for examples on how dissipation is included by defining a list of collapse operators.

The function returns an array of expectation values for the operators that are included in the list in the fifth argument. Adding operators to this list results in a larger output array returned by the function (one list of numbers, corresponding to the times in tlist, for each operator)::

    >>> odesolve(H, psi0, tlist, [], [sigmaz(), sigmay()])
    array([[  1.00000000e+00+0.j,   7.89142292e-01+0.j,   2.45485961e-01+0.j,
             -4.01696962e-01+0.j,  -8.79476686e-01+0.j,  -9.86363558e-01+0.j,
             -6.77281655e-01+0.j,  -8.25767574e-02+0.j,   5.46952346e-01+0.j,
              9.45820404e-01+0.j,   9.45817056e-01+0.j,   5.46944216e-01+0.j,
             -8.25852032e-02+0.j,  -6.77286734e-01+0.j,  -9.86363287e-01+0.j,
             -8.79471112e-01+0.j,  -4.01688979e-01+0.j,   2.45493023e-01+0.j,
              7.89145284e-01+0.j,   9.99999271e-01+0.j],
           [  0.00000000e+00+0.j,  -6.14214010e-01+0.j,  -9.69403055e-01+0.j,
             -9.15775807e-01+0.j,  -4.75947716e-01+0.j,   1.64596791e-01+0.j,
              7.35726839e-01+0.j,   9.96586861e-01+0.j,   8.37166184e-01+0.j,
              3.24695883e-01+0.j,  -3.24704840e-01+0.j,  -8.37170685e-01+0.j,
             -9.96585195e-01+0.j,  -7.35720619e-01+0.j,  -1.64588257e-01+0.j,
              4.75953748e-01+0.j,   9.15776736e-01+0.j,   9.69398541e-01+0.j,
              6.14206262e-01+0.j,  -8.13905967e-06+0.j]])
  
The resulting list of expectation values can easily be visualized using matplotlib's plotting functions::

    >>> tlist = linspace(0.0, 10.0, 100)
    >>> expt_list = odesolve(H, psi0, tlist, [], [sigmaz(), sigmay()])
    >>> 
    >>> from pylab import *
    >>> plot(tlist, real(expt_list[0]))
    >>> plot(tlist, real(expt_list[1]))
    >>> xlabel('Time')
    >>> ylabel('Expectation values')
    >>> legend(("Simga-Z", "Sigma-Y"))
    >>> show()

.. figure:: guide-dynamics-qubit.png
    :align: center

If an empty list of operators is passed as fifth parameter, the :func:`qutip.odesolve` function returns a list of state vectors for the times specified in ``tlist``::

    >>> tlist = [0.0, 1.0]
    >>> odesolve(H, psi0, tlist, [], [])
    [
    Quantum object: dims = [[2], [1]], shape = [2, 1], type = ket
    Qobj data = 
    [[ 1.+0.j]
     [ 0.+0.j]]
    , Quantum object: dims = [[2], [1]], shape = [2, 1], type = ket
    Qobj data = 
    [[ 0.80901765+0.j        ]
     [ 0.00000000-0.58778584j]]
    , Quantum object: dims = [[2], [1]], shape = [2, 1], type = ket
    Qobj data = 
    [[ 0.3090168+0.j        ]
     [ 0.0000000-0.95105751j]]
    , Quantum object: dims = [[2], [1]], shape = [2, 1], type = ket
    Qobj data = 
    [[-0.30901806+0.j        ]
     [ 0.00000000-0.95105684j]]
    ]

Non-unitary evolution
---------------------

Master equation
+++++++++++++++

For non-unitary evolution of a quantum systems, i.e., evolution that includes
incoherent processes such as relaxation and dephasing, it is common to use
master equations. In QuTiP, the same function (:func:`qutip.odesolve`) is used for 
evolution both according to the Schrödinger equation and to the master equation,
even though these two equations of motion are very different. The :func:`qutip.odesolve`
function automatically determines if it is sufficient to use the Schrödinger 
equation (if no collapse operators were given) or if it has to use the
master equation (if collapse operators were given). Note that to calculate
the time evolution according to the Schrödinger equation is easier and much
faster (for large systems) than using the master equation, so if possible the
solver will fall back on using the Schrödinger equation.

What is new in the master equation compared to the Schrödinger equation are
processes that describe dissipation in the quantum system due to its interaction
with an environment. These environmental interactions are defined by the
operators through which the system couples to the environment, and rates that
describe the strength of the processes.

In QuTiP, the product of the square root of the rate and the operator that 
describe the dissipation process is called a collapse operator. A list of 
collapse operators (``c_op_list``) is passed as the fourth argument to the 
:func:`qutip.odesolve` function in order to define the dissipation processes in the master
eqaution. When the ``c_op_list`` isn't empty, the :func:`qutip.odesolve` function will use
the master equation instead of the unitary Schröderinger equation.

Using the example with the spin dynamics from the previous section, we can
easily add a relaxation process (describing the dissipation of energy from the
spin to its environment), by adding ``sqrt(0.05) * sigmax()`` to
the previously empty list in the fourth parameter to the :func:`qutip.odesolve` function::

    >>> tlist = linspace(0.0, 10.0, 100)
    >>> expt_list = odesolve(H, psi0, tlist, [sqrt(0.05) * sigmax()], [sigmaz(), sigmay()])
    >>> 
    >>> from pylab import *
    >>> plot(tlist, real(expt_list[0]))
    >>> plot(tlist, real(expt_list[1]))
    >>> xlabel('Time')
    >>> ylabel('Expectation values')
    >>> legend(("Sigma-Z", "Sigma-Y"))
    >>> show()

Here, 0.05 is the rate and the operator :math:`\sigma_x` (:func:`qutip.sigmax`) describes the dissipation 
process.

.. figure:: guide-qubit-dynamics-dissip.png
    :align: center

Now a slightly more complex example: Consider a two-level atom coupled to a leaky single-mode cavity through a dipole-type interaction, which supports a coherent exchange of quanta between the two systems. If the atom initially is in its groundstate and the cavity in a 5-photon fock state, the dynamics is calculated with the lines following code::

    >>> tlist = linspace(0.0, 10.0, 200)
    >>> psi0 = tensor(fock(2,0), fock(10, 5))
    >>> a  = tensor(qeye(2), destroy(10))
    >>> sm = tensor(destroy(2), qeye(10))
    >>> H = 2*pi * a.dag() * a + 2 * pi * sm.dag() * sm + 2*pi * 0.25 * (sm*a.dag() + sm.dag() * a)
    >>> expt_list = odesolve(H, psi0, tlist, ntraj, [sqrt(0.1)*a], [a.dag()*a, sm.dag()*sm])
    >>> 
    >>> from pylab import *
    >>> plot(tlist, real(expt_list[0]))
    >>> plot(tlist, real(expt_list[1]))
    >>> xlabel('Time')
    >>> ylabel('Expectation values')
    >>> legend(("cavity photon number", "atom excitation probability"))
    >>> show()

.. figure:: guide-dynamics-jc.png
    :align: center

Monte-Carlo evolution
+++++++++++++++++++++

Quantum trajectory Monte-Carlo is an alternative approach for calculating the
time-evolution of dissipative quantum systems. Unlike the master equation, 
the Monte-Carlo method is based on the unitary evolution and uses the state
vector instead of density matrix to describe the state of the system.
Dissipation is introduced into the dynamics by stochastic quantum jumps,
whose rate and effect on the state of the system is described by the same
collapse operators that are used to define the master equation. The average of
a large number of such stochastic trajectories describes the dissipative 
dynamics of the system, and has been shown to give identical results as the
master equation. 

In QuTiP, Monto-Carlo evolution is implemented with the
:func:`qutip.mcsolve` function. It takes nearly the same arguments as the :func:`qutip.odesolve`
function for master-equation evolution, expect for one additional parameter
``ntraj`` (fourth parameter), which define the number of stochastic trajectories
that should be averaged. This number should usually be in the range 100 - 500 to
give a smooth results (although the optimal number for ``ntraj`` can vary from
case to case).

To illustrate the use of the Monte-Carlo evolution of quantum systems in QuTiP,
let's again consider the case of a two-level atom coupled to a leaky cavity. The 
only differences to the master-equation treatment is that in this case we 
invoke the :func:`qutip.mcsolve` function instead of :func:`qutip.odesolve`, and a new parameter 
``ntraj = 250`` has been defined::

    >>> tlist = linspace(0.0, 10.0, 200)
    >>> psi0 = tensor(fock(2,0), fock(10, 5))
    >>> a  = tensor(qeye(2), destroy(10))
    >>> sm = tensor(destroy(2), qeye(10))
    >>> H = 2*pi * a.dag() * a + 2 * pi * sm.dag() * sm + 2*pi * 0.25 * (sm*a.dag() + sm.dag() * a)
    >>> ntraj = 250
    >>> expt_list = mcsolve(H, psi0, tlist, ntraj, [sqrt(0.1)*a], [a.dag()*a, sm.dag()*sm])
    >>> 
    >>> from pylab import *
    >>> plot(tlist, real(expt_list[0]))
    >>> plot(tlist, real(expt_list[1]))
    >>> title('Monte-Carlo time evolution')
    >>> xlabel('Time')
    >>> ylabel('Expectation values')
    >>> legend(("cavity photon number", "atom excitation probability"))
    >>> show()

.. figure:: guide-dynamics-mc.png
    :align: center

The advantage of the Monte-Carlo method over the master equation approach is that only the state vector is required to be kept in the computer memory (as opposed to the entire density matrix). For large quantum system this becomes a significant advantage and the Monte-Carlo is therefore generally recommended for such systems. But for small systems, on the other hand, the added overhead of averaging a large number of stochastic trajectories to obtain the open system dynamics outweigh the benefits of the (small) memory saving, and master equations are therefore generally more efficient.

The return value(s) from the Monte-Carlo solver depend on the presence of collapse and expectation operators in the :func:`qutip.mcsolve` function, as well as how many outputs are requested by the user.  The last example had both collapse and expectation value operators::

    >>> out=mcsolve(H, psi0, tlist, ntraj, [sqrt(0.1)*a], [a.dag()*a, sm.dag()*sm])

and the user requested a single output ``out``.  In this case, the monte-carlo solver returns the average over all trajectories for the expectation values generated by the requested operators.  If we remove the collapse operators::

    >>> out=mcsolve(H, psi0, tlist, ntraj, [], [a.dag()*a, sm.dag()*sm])

then we will also get expectation values for the output.  Now, if we add back in the collapse operators, but remove the expectation value operators::

    >>> out=mcsolve(H, psi0, tlist, ntraj, [sqrt(0.1)*a], [])

then the output of :func:`qutip.mcsolve` *is not* a list of expectation values but rather a list of state vector Qobjs calculated for each time, and trajectory.  This a huge output and should be avoided unless you want to see the jumps associated with the collapse operators for individual trajectories.  For example::
    
    >>> out[0]
    
will be a list of state vector Qobjs evaluated at the times in ``tlist``.

In addition, when collapse operators are specified, the monte-carlo solver will also keep track of when a collapse occurs, and which operator did the collapse.  To obtain this information, the user must specify multiple return values from the :func:`qutip.mcsolve` function.  For example, to get the times at which collapses occurred for the trajectories we can do::

    >>> expt,times=mcsolve(H, psi0, tlist, ntraj, [sqrt(0.1)*a], [a.dag()*a, sm.dag()*sm])
    
where we have requested a second output `times`.  Again the first operator corresponds to the expectation values.  To get the information on which operator did the collapse we add a third return value::

    >>> expt,times,which=mcsolve(H, psi0, tlist, ntraj, [sqrt(0.1)*a], [a.dag()*a, sm.dag()*sm])

If no expectation values are specified then the first output will be a list of state vectors.  A example demonstrating the use of multiple return values may be found at :ref:`examples_collapsetimesmonte`.  To summarize, the table below gives the output of the monte-carlo solver for a given set of input and output conditions:

+--------------------+-----------------------+-----------------------------+------------------------------------+
| Collapse operators | Expectation operators | Number of requested outputs | Return value(s)                    |
+====================+=======================+=============================+====================================+
| NO                 | NO                    | 1                           | List of state vectors              |
+--------------------+-----------------------+-----------------------------+------------------------------------+
| NO                 | YES                   | 1                           | List of expectation values         |
+--------------------+-----------------------+-----------------------------+------------------------------------+
| YES                | NO                    | 1                           | List of state vectors for each     |
|                    |                       |                             | trajectory.                        |
+--------------------+-----------------------+-----------------------------+------------------------------------+
| YES                | NO                    | 2                           | List of state vectors for each     |
|                    |                       |                             | trajectory + List of collapse times|
|                    |                       |                             | for each trajectory.               |
+--------------------+-----------------------+-----------------------------+------------------------------------+
| YES                | NO                    | 3                           | List of state vectors for each     |
|                    |                       |                             | trajectory + List of collapse times|
|                    |                       |                             | for each trajectory + List of which|
|                    |                       |                             | operator did collapse for each     |
|                    |                       |                             | trajectory.                        |
+--------------------+-----------------------+-----------------------------+------------------------------------+
| YES                | YES                   | 1                           | List of expectation values for each|
|                    |                       |                             | trajectory.                        |
+--------------------+-----------------------+-----------------------------+------------------------------------+
| YES                | YES                   | 2                           | List of expectation values for each|
|                    |                       |                             | trajectory + List of collapse times|
|                    |                       |                             | for each trajectory.               |
+--------------------+-----------------------+-----------------------------+------------------------------------+
| YES                | YES                   | 3                           | List of expectation values for each|
|                    |                       |                             | trajectory + List of collapse times|
|                    |                       |                             | for each trajectory + List of which|
|                    |                       |                             | operator did collapse for each     |
|                    |                       |                             | trajectory.                        |
+--------------------+-----------------------+-----------------------------+------------------------------------+


Which solver should I use?
--------------------------

In general, the choice of solver is determined by the size of your system, as well as your desired output.  The computational resources required by the master equation solver scales as :math:`N^2`, where :math:`N` is the dimensionality of the Hilbert space.  For small systems, the master equation method is very efficient. In contrast, the monte-carlo solver scales as :math:`N`, but requires running multiple trajectories to average over to get the desired expectation values.  Therefore, if your system is too large, and you run out of memory using :func:`qutip.odesolve`, then the only option available will be :func:`qutip.mcsolve`.  On the other hand, the monte-carlo method cannot return the full density matrix as a function of time and you need to use :func:`qutip.odesolve` if this is required.

If your system is intermediate in size (you are not bound by memory) then it is interesting to calculate the crossover point where the monte-carlo solver begins to perform better than the master equation method.  The exact point at which one solver is better than the other will depend on the system of interest and number of processors. However as a guideline, below we have plotted the time required to solve for the evolution of coupled dissipative harmonic oscillators as a function of Hilbert space size.

.. figure:: guide-dynamics-solver-performance.png
    :align: center

Here, the number of trajectories used in :func:`qutip.mcsolve` is ``250`` and the number of processors (which determines the slope of the monte-carlo line) is ``4``.  Here we see that the monte-carlo solver begins to be more efficient than the corresponding master-equation method at a Hilbert space size of :math:`N\sim40`.  Therefore, if your system size is greater than :math:`N\sim40` and you do not need the full density matrix, then it is recommended to try the :func:`qutip.mcsolve` function. 

Time-dependent Hamiltonians (unitary and non-unitary)
-----------------------------------------------------

In the previous examples of quantum system evolution, we assumed that the systems under consideration were described by a time-independent Hamiltonian. The two main evolution solvers in QuTiP, :func:`qutip.odesolve` and :func:`qutip.mcsolve`, can also handle time-dependent Hamiltonians. If a callback function is passed as first parameter to the solver function (instead of :class:`qutip.Qobj` Hamiltonian), then this function is called at each time step and is expected to return the :class:`qutip.Qobj` Hamiltonian for that point in time. The callback function takes two arguments: the time `t` and list additional Hamiltonian arguments ``H_args``. This list of additional arguments is the same object as is passed as the sixth parameter to the solver function (only used for time-dependent Hamiltonians).

For example, let's consider a two-level system with energy splitting 1.0, and subject to a time-dependent field that couples to the :math:`\sigma_x` operator with amplitude 0.1. Furthermore, to make the example a little bit more interesting, let's also assume that the two-level system is subject to relaxation, with relaxation rate 0.01. The following code calculates the dynamics of the system in the absence and in the presence of the time-dependent driving signal::

    >>> def hamiltonian_t(t, args):
    >>>     H0 = args[0]
    >>>     H1 = args[1]
    >>>     w  = args[2]
    >>>     return H0 + H1 * sin(w * t)
    >>> 
    >>> H0 = - 2*pi * 0.5  * sigmaz()
    >>> H1 = - 2*pi * 0.05 * sigmax() 
    >>> H_args = (H0, H1, 2*pi*1.0)
    >>> psi0 = fock(2, 0)                   # intial state |0>
    >>> c_op_list = [sqrt(0.01) * sigmam()] # relaxation
    >>> tlist = arange(0.0, 50.0, 0.01)
    >>>
    >>> expt_sz    = odesolve(H0, psi0, tlist, c_op_list, [sigmaz()])
    >>> expt_sz_td = odesolve(hamiltonian_t, psi0, tlist, c_op_list, [sigmaz()], H_args)
    >>>
    >>> #expt_sz_td = mcsolve(hamiltonian_t, psi0, tlist,250, c_op_list, [sigmaz()], H_args) #monte-carlo
    >>>
    >>> from pylab import *
    >>> plot(tlist, expt_sz[0],    'r')
    >>> plot(tlist, expt_sz_td[0], 'b')
    >>> ylabel("Expectation value of Sigma-Z")
    >>> xlabel("time")
    >>> legend(("H = H0", "H = H0 + H1 * sin(w*t)"), loc=4)
    >>> show()

.. figure:: guide-dynamics-td.png
    :align: center
    
   
Setting ODE solver options
--------------------------

Occasionally it is necessary to change the built in parameters of the ODE solvers used by both the odesolve and mcsolve functions.  The ODE options for either of these functions may be changed by calling the Odeoptions class::

    opts=Odeoptions()

the properties and default values of this class can be view via the `print` command::

    >>> print opts
    Odeoptions properties:
    ----------------------
    atol:        1e-08
    rtol:        1e-06
    method:      adams
    order:       12
    nsteps:      1000
    first_step:  0
    min_step:    0
    max_step:    0
    tidy:        True
    num_cpus:    8
    parallel:    False

These properties are detailed in the following table.  Assuming ``opts=Odeoptions()``:

+-----------------+-----------------+----------------------------------------------------------------+
| Property        | Default setting | Description                                                    |
+=================+=================+================================================================+
| opts.atol       | 1e-8            | Absolute tolerance                                             |
+-----------------+-----------------+----------------------------------------------------------------+
| opts.rtol       | 1e-6            | Relative tolerance                                             |
+-----------------+-----------------+----------------------------------------------------------------+
| opts.method     | 'adams'         | Solver method.  Can be 'adams' (non-stiff) or 'bdf' (stiff)    |
+-----------------+-----------------+----------------------------------------------------------------+
| opts.order      | 12              | Order of solver.  Must be <=12 for 'adams' and <=5 for 'bdf'   |
+-----------------+-----------------+----------------------------------------------------------------+
| opts.nsteps     | 1000            | Max. number of steps to take for each interval                 |
+-----------------+-----------------+----------------------------------------------------------------+
| opts.first_step | 0               | Size of initial step.  0 = determined automatically by solver. |
+-----------------+-----------------+----------------------------------------------------------------+
| opts.min_step   | 0               | Minimum step size.  0 = determined automatically by solver.    |
+-----------------+-----------------+----------------------------------------------------------------+
| opts.max_step   | 0               | Maximum step size.  0 = determined automatically by solver.    |
+-----------------+-----------------+----------------------------------------------------------------+
| opts.tidy       | True            | Whether to run tidyup function on time-independent Hamiltonian.| 
+-----------------+-----------------+----------------------------------------------------------------+
| opts.num_cpus   | # of processors | Number of cpu's used by mcsolve.                               | 
+-----------------+-----------------+----------------------------------------------------------------+
| opts.parallel   | False           | Whether to use parallel sparse matrix-vector multiplication.   | 
+-----------------+-----------------+----------------------------------------------------------------+

As an example, let us consider relaxing the conditions on the ODE solver::

    >>> opts.atol=1e-10
    >>> opts.rtol=1e-8
    >>> opts.nsteps=500
    >>> print opts

    Odeoptions properties:
    ----------------------
    atol:        1e-10
    rtol:        1e-08
    method:      adams
    order:       12
    nsteps:      500
    first_step:  0
    min_step:    0
    max_step:    0
    tidy:        True
    num_cpus:    8
    parallel:    False

To use these new settings we can use the keyword argument `options` in either the `odesolve` or `mcsolve` function.  We can modify the last example as::

    >>> odesolve(H0, psi0, tlist, c_op_list, [sigmaz()],options=opts)
    >>> odesolve(hamiltonian_t, psi0, tlist, c_op_list, [sigmaz()], H_args,options=opts)

or::
    
    >>> mcsolve(H0, psi0, tlist, ntraj,c_op_list, [sigmaz()],options=opts)
    >>> mcsolve(hamiltonian_t, psi0, tlist, ntraj, c_op_list, [sigmaz()], H_args,options=opts)


Performance (version 1.1.1)
---------------------------

Here we compare the performance of the master-equation and monte-Carlo solvers to their quantum optics toolbox counterparts.

In this example, we calculate the time-evolution of the density matrix for a coupled oscillator system using the odesolve function, and compare it to the quantum optics toolbox (qotoolbox).  Here, we see that the QuTiP solver out performs it's qotoolbox counterpart by a substantial margin as the system size increases.

.. figure:: guide-dynamics-odesolve-performance.png
    :align: center

To test the Monte-Carlo solvers, here we simulate a trilinear Hamiltonian over a range of Hilbert space sizes.  Since QuTiP uses multiprocessing, we can measure the performance gain when using several CPU's.  In contrast, the qotoolbox is limited to a single-processor only.  In the legend, we show the speed-up factor in the parenthesis, which should ideally be equal to the number of processors.  Finally, we have included the results using hyperthreading, written here as 4+(x) where x is the number of hyperthreads, found in some newer Intel processors.  We see however that the performance benefit is marginal at best.


.. figure:: guide-dynamics-mcsolve-performance.png
    :align: center

