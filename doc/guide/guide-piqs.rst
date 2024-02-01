.. _master-piqs:

*********************************
Permutational Invariance
*********************************

.. _master-unitary-piqs:

Permutational Invariant Quantum Solver (PIQS)
=============================================
The *Permutational Invariant Quantum Solver (PIQS)* is a QuTiP module that allows to study the dynamics of an open quantum system consisting of an ensemble of identical qubits that can dissipate through local and collective baths according to a Lindblad master equation.

The Liouvillian of an ensemble of :math:`N` qubits, or two-level systems (TLSs), :math:`\mathcal{D}_{TLS}(\rho)`, can be built using only polynomial – instead of exponential – resources.
This has many applications for the study of realistic quantum optics models of many TLSs and in general as a tool in cavity QED.

Consider a system evolving according to the equation

.. math::
    \dot{\rho} = \mathcal{D}_\text{TLS}(\rho)=-\frac{i}{\hbar}\lbrack H,\rho \rbrack
    +\frac{\gamma_\text{CE}}{2}\mathcal{L}_{J_{-}}[\rho]
    +\frac{\gamma_\text{CD}}{2}\mathcal{L}_{J_{z}}[\rho]
    +\frac{\gamma_\text{CP}}{2}\mathcal{L}_{J_{+}}[\rho]

    +\sum_{n=1}^{N}\left(
    \frac{\gamma_\text{E}}{2}\mathcal{L}_{J_{-,n}}[\rho]
    +\frac{\gamma_\text{D}}{2}\mathcal{L}_{J_{z,n}}[\rho]
    +\frac{\gamma_\text{P}}{2}\mathcal{L}_{J_{+,n}}[\rho]\right)


where :math:`J_{\alpha,n}=\frac{1}{2}\sigma_{\alpha,n}` are SU(2) Pauli spin operators, with :math:`{\alpha=x,y,z}` and :math:`J_{\pm,n}=\sigma_{\pm,n}`. The collective spin operators are :math:`J_{\alpha} = \sum_{n}J_{\alpha,n}` . The Lindblad super-operators are :math:`\mathcal{L}_{A} = 2A\rho A^\dagger - A^\dagger A \rho - \rho A^\dagger A`.

The inclusion of local processes in the dynamics lead to using a Liouvillian space of dimension :math:`4^N`. By exploiting the permutational invariance of identical particles [2-8], the Liouvillian :math:`\mathcal{D}_\text{TLS}(\rho)` can be built as a block-diagonal matrix in the basis of Dicke states :math:`|j, m \rangle`.

The system under study is defined by creating an object of the
:class:`~qutip.piqs.piqs.Dicke` class, e.g. simply named
:code:`system`, whose first attribute is

- :code:`system.N`, the number of TLSs of the system :math:`N`.

The rates for collective and local processes are simply defined as

- :code:`collective_emission` defines :math:`\gamma_\text{CE}`, collective (superradiant) emission
- :code:`collective_dephasing` defines :math:`\gamma_\text{CD}`, collective dephasing
- :code:`collective_pumping` defines :math:`\gamma_\text{CP}`, collective pumping.
- :code:`emission` defines :math:`\gamma_\text{E}`, incoherent emission (losses)
- :code:`dephasing` defines :math:`\gamma_\text{D}`, local dephasing
- :code:`pumping`  defines :math:`\gamma_\text{P}`, incoherent pumping.

Then the :code:`system.lindbladian()` creates the total TLS Lindbladian superoperator matrix. Similarly, :code:`system.hamiltonian` defines the TLS hamiltonian of the system :math:`H_\text{TLS}`.

The system's Liouvillian can be built using :code:`system.liouvillian()`.
The properties of a Piqs object can be visualized by simply calling :code:`system`.
We give two basic examples on the use of *PIQS*.
In the first example the incoherent emission of N driven TLSs is considered.

.. code-block:: python

    from qutip import piqs
    N = 10
    system = piqs.Dicke(N, emission = 1, pumping = 2)
    L = system.liouvillian()
    steady = steadystate(L)

For more example of use, see the "Permutational Invariant Lindblad Dynamics" section in the tutorials section of the website, `https://qutip.org/tutorials.html <https://qutip.org/tutorials.html>`_.

.. list-table:: Useful PIQS functions.
   :widths: 25 25 50
   :header-rows: 1

   * - Operators
     - Command
     - Description
   * - Collective spin algebra :math:`J_x,\ J_y,\ J_z`
     - ``jspin(N)``
     - The collective spin algebra  :math:`J_x,\ J_y,\ J_z` for :math:`N` TLSs
   * - Collective spin :math:`J_x`
     - ``jspin(N, "x")``
     - The collective spin operator :math:`Jx`. Requires :math:`N` number of TLSs
   * - Collective spin :math:`J_y`
     - ``jspin(N, "y")``
     - The collective spin operator :math:`J_y`. Requires :math:`N` number of TLSs
   * - Collective spin :math:`J_z`
     - ``jspin(N, "z")``
     - The collective spin operator :math:`J_z`. Requires :math:`N` number of TLSs
   * - Collective spin :math:`J_+`
     - ``jspin(N, "+")``
     - The collective spin operator :math:`J_+`.
   * - Collective spin :math:`J_-`
     - ``jspin(N, "-")``
     - The collective spin operator :math:`J_-`.
   * - Collective spin :math:`J_z` in uncoupled basis
     - ``jspin(N, "z", basis='uncoupled')``
     - The collective spin operator :math:`J_z` in the uncoupled basis of dimension :math:`2^N`.
   * - Dicke state :math:`|j,m\rangle` density matrix
     - ``dicke(N, j, m)``
     - The density matrix for the Dicke state given by :math:`|j,m\rangle`
   * - Excited-state density matrix  in Dicke basis
     - ``excited(N)``
     - The excited state in the Dicke basis
   * - Excited-state density matrix  in uncoupled basis
     - ``excited(N, basis="uncoupled")``
     - The excited state in the uncoupled basis
   * - Ground-state density matrix  in Dicke basis
     - ``ground(N)``
     - The ground state in the Dicke basis
   * - GHZ-state density matrix in the Dicke basis
     - ``ghz(N)``
     - The GHZ-state density matrix in the Dicke (default) basis for N number of TLS
   * - Collapse operators of the ensemble
     - ``Dicke.c_ops()``
     - The collapse operators for the ensemble can be called by the `c_ops` method of the Dicke class.

Note that the mathematical object representing the density matrix of the full system
that is manipulated (or obtained from `steadystate`) in the Dicke-basis formalism
used here is a *representative of the density matrix*. This *representative object*
is of linear size N^2, whereas the full density matrix is defined over a 2^N Hilbert
space. In order to calculate nonlinear functions of such density matrix, such as the
Von Neumann entropy or the purity, it is necessary to take into account the degeneracy
of each block of such block-diagonal density matrix. Note that as long as one calculates
expected values of operators, being Tr[A*rho] a *linear* function of `rho`, the
*representative density matrix* give straightforwardly the correct result. When a
*nonlinear* function of the density matrix needs to be calculated, one needs to
weigh each degenerate block correctly; this is taken care by the `dicke_function_trace`
in :obj:`.piqs`, and the user can use it to define general nonlinear functions that
can be described as the trace of a Taylor expandable function. Two nonlinear functions
that use `dicke_function_trace` and are already implemented are `purity_dicke`, to
calculate the purity of a density matrix in the Dicke basis, and `entropy_vn_dicke`,
which can be used to calculate the Von Neumann entropy.

More functions relative to the :obj:`qutip.piqs` module can be found at :ref:`apidoc`.
Attributes to the :class:`.piqs.Dicke` and :class:`.piqs.Pim` class can also be found there.
