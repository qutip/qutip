.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _master:

*********************************
Lindblad Master Equation Solver
*********************************

.. _master-unitary:

Unitary evolution
====================
The dynamics of a closed (pure) quantum system is governed by the Schrödinger equation

.. math::
   :label: schrodinger

	i\hbar\frac{\partial}{\partial t}\Psi = \hat H \Psi,

where :math:`\Psi` is the wave function, :math:`\hat H` the Hamiltonian, and :math:`\hbar` is Planck's constant. In general, the Schrödinger equation is a partial differential equation (PDE) where both :math:`\Psi` and :math:`\hat H` are functions of space and time. For computational purposes it is useful to expand the PDE in a set of basis functions that span the Hilbert space of the Hamiltonian, and to write the equation in matrix and vector form

.. math::
   
   i\hbar\frac{d}{dt}\left|\psi\right> = H \left|\psi\right>

where :math:`\left|\psi\right>` is the state vector and :math:`H` is the matrix representation of the Hamiltonian. This matrix equation can, in principle, be solved by diagonalizing the Hamiltonian matrix :math:`H`. In practice, however, it is difficult to perform this diagonalization unless the size of the Hilbert space (dimension of the matrix :math:`H`) is small. Analytically, it is a formidable task to calculate the dynamics for systems with more than two states. If, in addition, we consider dissipation due to the inevitable interaction with a surrounding environment, the computational complexity grows even larger, and we have to resort to numerical calculations in all realistic situations. This illustrates the importance of numerical calculations in describing the dynamics of open quantum systems, and the need for efficient and accessible tools for this task.

The Schrödinger equation, which governs the time-evolution of closed quantum systems, is defined by its Hamiltonian and state vector. In the previous section, :ref:`tensor`, we showed how Hamiltonians and state vectors are constructed in QuTiP. Given a Hamiltonian, we can calculate the unitary (non-dissipative) time-evolution of an arbitrary state vector :math:`\left|\psi_0\right>` (``psi0``) using the QuTiP function :func:`qutip.mesolve`. It evolves the state vector and evaluates the expectation values for a set of operators ``expt_op_list`` at the points in time in the list ``tlist``, using an ordinary differential equation solver. Alternatively, we can use the function :func:`qutip.essolve`, which uses the exponential-series technique to calculate the time evolution of a system. The :func:`qutip.mesolve` and :func:`qutip.essolve` functions take the same arguments and it is therefore easy switch between the two solvers. 

For example, the time evolution of a quantum spin-1/2 system with tunneling rate 0.1 that initially is in the up state is calculated, and the  expectation values of the :math:`\sigma_z` operator evaluated, with the following code::

    >>> H = 2 * pi * 0.1 * sigmax()
    >>> psi0 = basis(2, 0)
    >>> tlist = linspace(0.0, 10.0, 20.0)
    >>> mesolve(H, psi0, tlist, [], [sigmaz()])
    array([[ 1.00000000+0.j,  0.78914229+0.j,  0.24548596+0.j, -0.40169696+0.j,
            -0.87947669+0.j, -0.98636356+0.j, -0.67728166+0.j, -0.08257676+0.j,
             0.54695235+0.j,  0.94582040+0.j,  0.94581706+0.j,  0.54694422+0.j,
            -0.08258520+0.j, -0.67728673+0.j, -0.98636329+0.j, -0.87947111+0.j,
            -0.40168898+0.j,  0.24549302+0.j,  0.78914528+0.j,  0.99999927+0.j]])

The brackets in the fourth argument is an empty list of collapse operators,  since we consider unitary evolution in this example. See the next section for examples on how dissipation is included by defining a list of collapse operators.

The function returns an array of expectation values for the operators that are included in the list in the fifth argument. Adding operators to this list results in a larger output array returned by the function (one list of numbers, corresponding to the times in tlist, for each operator)::

    >>> mesolve(H, psi0, tlist, [], [sigmaz(), sigmay()])
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
    >>> expt_list = mesolve(H, psi0, tlist, [], [sigmaz(), sigmay()])
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
   :width: 4in


If an empty list of operators is passed as fifth parameter, the :func:`qutip.mesolve` function returns a list of state vectors for the times specified in ``tlist``::

    >>> tlist = [0.0, 1.0]
    >>> mesolve(H, psi0, tlist, [], [])
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

.. _master-nonunitary:

Non-unitary evolution
=======================

While the evolution of the state vector in a closed quantum system is deterministic, open quantum systems are stochastic in nature. The effect of an environment on the system of interest is to induce stochastic transitions between energy levels, and to introduce uncertainty in the phase difference between states of the system. The state of an open quantum system is therefore described in terms of ensemble averaged states using the density matrix formalism. A density matrix :math:`\rho` describes a probability distribution of quantum states :math:`\left|\psi_n\right>`, in a matrix representation :math:`\rho = \sum_n p_n \left|\psi_n\right>\left<\psi_n\right|`, where :math:`p_n` is the classical probability that the system is in the quantum state :math:`\left|\psi_n\right>`. The time evolution of a density matrix :math:`\rho` is the topic of the remaining portions of this section.

.. _master-master:

The Lindblad Master equation
=============================

The standard approach for deriving the equations of motion for a system interacting with its environment is to expand the scope of the system to include the environment. The combined quantum system is then closed, and its evolution is governed by the von Neumann equation

.. math::
   :label: neumann_total
   
   \dot \rho_{\rm tot}(t) = -\frac{i}{\hbar}[H_{\rm tot}, \rho_{\rm tot}(t)],

the equivalent of the Schrödinger equation (:eq:`schrodinger`) in the density matrix formalism. Here, the total Hamiltonian 

.. math::

 	H_{\rm tot} = H_{\rm sys} + H_{\rm env} + H_{\rm int},

includes the original system Hamiltonian :math:`H_{\rm sys}`, the Hamiltonian for the environment :math:`H_{\rm env}`, and a term representing the interaction between the system and its environment :math:`H_{\rm int}`. Since we are only interested in the dynamics of the system, we can at this point perform a partial trace over the environmental degrees of freedom in Eq.~(:eq:`neumann_total`), and thereby obtain a master equation for the motion of the original system density matrix. The most general trace-preserving and completely positive form of this evolution is the Lindblad master equation for the reduced density matrix :math:`\rho = {\rm Tr}_{\rm env}[\rho_{\rm tot}]` 

.. math::
	:label: master_equation

	\dot\rho(t)=-\frac{i}{\hbar}[H(t),\rho(t)]+\sum_n \frac{1}{2} \left[2 C_n \rho(t) C_n^{+} - \rho(t) C_n^{+} C_n - C_n^{+} C_n \rho(t)\right]

where the :math:`C_n = \sqrt{\gamma_n} A_n` are collapse operators, and :math:`A_n` are the operators through which the environment couples to the system in :math:`H_{\rm int}`, and :math:`\gamma_n` are the corresponding rates.  The derivation of Eq.~(:eq:`master_equation`) may be found in several sources, and will not be reproduced here.  Instead, we emphasize the approximations that are required to arrive at the master equation in the form of Eq.~(:eq:`master_equation`), and hence perform a calculation in QuTiP:

- **Separability:** At :math:`t=0` there are no correlations between the system and its environment such that the total density matrix can be written as a tensor product :math:`\rho^I_{\rm tot}(0) = \rho^I(0) \otimes \rho^I_{\rm env}(0)`.

- **Born approximation:** Requires: (1) that the state of the environment does not significantly change as a result of the interaction with the system;  (2) The system and the environment remain separable throughout the evolution. These assumptions are justified if the interaction is weak, and if the environment is much larger than the system. In summary, :math:`\rho_{\rm tot}(t) \approx \rho(t)\otimes\rho_{\rm env}`.

- **Markov approximation** The time-scale of decay for the environment :math:`\tau_{\rm env}` is much shorter than the smallest time-scale of the system dynamics :math:`\tau_{\rm sys} \gg \tau_{\rm env}`. This approximation is often deemed a "short-memory environment" as it requires that environmental correlation functions decay on a time-scale fast compared to those of the system.

- **Secular approximation** Stipulates that elements in the master equation corresponding to transition frequencies satisfy :math:`|\omega_{ab}-\omega_{cd}| \ll 1/\tau_{\rm sys}`, i.e., all fast rotating terms in the interaction picture can be neglected. It also ignores terms that lead to a small renormalization of the system energy levels. This approximation is not strictly necessary for all master-equation formalisms (e.g., the Block-Redfield master equation), but it is required for arriving at the Lindblad form (:eq:`master_equation`) which is used in QuTiP.


For systems with environments satisfying the conditions outlined above, the Lindblad master equation (:eq:`master_equation`) governs the time-evolution of the system density matrix, giving an ensemble average of the system dynamics. In order to ensure that these approximations are not violated, it is important that the decay rates :math:`\gamma_n` be smaller than the minimum energy splitting in the system Hamiltonian. Situations that demand special attention therefore include, for example, systems strongly coupled to their environment, and systems with degenerate or nearly degenerate energy levels. 


For non-unitary evolution of a quantum systems, i.e., evolution that includes
incoherent processes such as relaxation and dephasing, it is common to use
master equations. In QuTiP, the same function (:func:`qutip.mesolve`) is used for 
evolution both according to the Schrödinger equation and to the master equation,
even though these two equations of motion are very different. The :func:`qutip.mesolve`
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
:func:`qutip.mesolve` function in order to define the dissipation processes in the master
eqaution. When the ``c_op_list`` isn't empty, the :func:`qutip.mesolve` function will use
the master equation instead of the unitary Schröderinger equation.

Using the example with the spin dynamics from the previous section, we can
easily add a relaxation process (describing the dissipation of energy from the
spin to its environment), by adding ``sqrt(0.05) * sigmax()`` to
the previously empty list in the fourth parameter to the :func:`qutip.mesolve` function::

    >>> tlist = linspace(0.0, 10.0, 100)
    >>> expt_list = mesolve(H, psi0, tlist, [sqrt(0.05) * sigmax()], [sigmaz(), sigmay()])
    >>> 
    >>> from pylab import *
    >>> plot(tlist, real(expt_list[0]))
    >>> plot(tlist, real(expt_list[1]))
    >>> xlabel('Time')
    >>> ylabel('Expectation values')
    >>> legend(("Sigma-Z", "Sigma-Y"))
    >>> show()

Here, 0.05 is the rate and the operator :math:`\sigma_x` (:func:`qutip.operators.sigmax`) describes the dissipation 
process.

.. figure:: guide-qubit-dynamics-dissip.png
   :align: center
   :width: 4in


Now a slightly more complex example: Consider a two-level atom coupled to a leaky single-mode cavity through a dipole-type interaction, which supports a coherent exchange of quanta between the two systems. If the atom initially is in its groundstate and the cavity in a 5-photon fock state, the dynamics is calculated with the lines following code::

    >>> tlist = linspace(0.0, 10.0, 200)
    >>> psi0 = tensor(fock(2,0), fock(10, 5))
    >>> a  = tensor(qeye(2), destroy(10))
    >>> sm = tensor(destroy(2), qeye(10))
    >>> H = 2*pi * a.dag() * a + 2 * pi * sm.dag() * sm + 2*pi * 0.25 * (sm*a.dag() + sm.dag() * a)
    >>> expt_list = mesolve(H, psi0, tlist, ntraj, [sqrt(0.1)*a], [a.dag()*a, sm.dag()*sm])
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
   :width: 4in






