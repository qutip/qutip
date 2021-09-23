.. _steady:

*************************************
Solving for Steady-State Solutions
*************************************

.. _steady-intro:

Introduction
============

For time-independent open quantum systems with decay rates larger than the corresponding excitation rates, the system will tend toward a steady state as :math:`t\rightarrow\infty` that satisfies the equation

.. math::
    \frac{d\hat{\rho}_{ss}}{dt}=\mathcal{L}\hat{\rho}_{ss}=0.

Although the requirement for time-independence seems quite resitrictive, one can often employ a transformation to the interaction picture that yields a time-independent Hamiltonian.  For many these systems, solving for the asymptotic density matrix :math:`\hat{\rho}_{ss}` can be achieved using direct or iterative solution methods faster than using master equation or Monte Carlo simulations.  Although the steady state equation has a simple mathematical form, the properties of the Liouvillian operator are such that the solutions to this equation are anything but straightforward to find.

Steady State solvers in QuTiP
=============================

In QuTiP, the steady-state solution for a system Hamiltonian or Liouvillian is given by :func:`qutip.steadystate.steadystate`.  This function implements a number of different methods for finding the steady state, each with their own pros and cons, where the method used can be chosen using the ``method`` keyword argument.

.. cssclass:: table-striped

.. list-table::
   :widths: 10 15 30
   :header-rows: 1

   * - Method
     - Keyword
     - Description
   * - Direct (default)
     - 'direct'
     - Direct solution solving :math:`Ax=b` via sparse LU decomposition.
   * - Eigenvalue
     - 'eigen'
     - Iteratively find the zero eigenvalue of :math:`\mathcal{L}`.
   * - Inverse-Power
     - 'power'
     - Solve using the inverse-power method.
   * - GMRES
     - 'iterative-gmres'
     - Solve using the GMRES method and optional preconditioner.
   * - LGMRES
     - 'iterative-lgmres'
     - Solve using the LGMRES method and optional preconditioner.
   * - BICGSTAB
     - 'iterative-bicgstab'
     - Solve using the BICGSTAB method and optional preconditioner.
   * - SVD
     - 'svd'
     - Steady-state solution via the **dense** SVD of the Liouvillian.


The function :func:`qutip.steadystate.steadystate` can take either a Hamiltonian and a list of collapse operators as input, generating internally the corresponding Liouvillian super operator in Lindblad form, or alternatively, a Liouvillian passed by the user. When possible, we recommend passing the Hamiltonian and collapse operators to :func:`qutip.steadystate.steadystate`, and letting the function automatically build the Liouvillian (in Lindblad form) for the system.

As of QuTiP 3.2, the ``direct`` and ``power`` methods can take advantage of the Intel Pardiso LU solver in the Intel Math Kernel library that comes with the Anacoda (2.5+) and Intel Python distributions.  This gives a substantial increase in performance compared with the standard SuperLU method used by SciPy.  To verify that QuTiP can find the necessary libraries, one can check for ``INTEL MKL Ext: True`` in the QuTiP about box (:func:`qutip.about`).


.. _steady-usage:

Using the Steadystate Solver
=============================

Solving for the steady state solution to the Lindblad master equation for a general system with :func:`qutip.steadystate.steadystate` can be accomplished using::

>>> rho_ss = steadystate(H, c_ops)

where ``H`` is a quantum object representing the system Hamiltonian, and ``c_ops`` is a list of quantum objects for the system collapse operators. The output, labeled as ``rho_ss``, is the steady-state solution for the systems.  If no other keywords are passed to the solver, the default 'direct' method is used, generating a solution that is exact to machine precision at the expense of a large memory requirement.  The large amount of memory need for the direct LU decomposition method stems from the large bandwidth of the system Liouvillian and the correspondingly large fill-in (extra nonzero elements) generated in the LU factors.  This fill-in can be reduced by using bandwidth minimization algorithms such as those discussed in :ref:`steady-args`.  However, in most cases, the default fill-in reducing algorithm is nearly optimal.  Additional parameters may be used by calling the steady-state solver as:

.. code-block:: python

   rho_ss = steadystate(H, c_ops, method='power', use_rcm=True)

where ``method='power'`` indicates that we are using the inverse-power solution method, and ``use_rcm=True`` turns on a bandwidth minimization routine.


Although it is not obvious, the ``'direct'``, ``eigen``, and ``'power'`` methods all use an LU decomposition internally and thus suffer from a large memory overhead.  In contrast, iterative methods such as the ``'iterative-gmres'``, ``'iterative-lgmres'``, and ``'iterative-bicgstab'`` methods do not factor the matrix and thus take less memory than these previous methods and allowing, in principle, for extremely large system sizes. The downside is that these methods can take much longer than the direct method as the condition number of the Liouvillian matrix is large, indicating that these iterative methods require a large number of iterations for convergence.  To overcome this, one can use a preconditioner :math:`M` that solves for an approximate inverse for the (modified) Liouvillian, thus better conditioning the problem, leading to faster convergence.  The use of a preconditioner can actually make these iterative methods faster than the other solution methods.  The problem with precondioning is that it is only well defined for Hermitian matrices.  Since the Liouvillian is non-Hermitian, the ability to find a good preconditioner is not guaranteed.  And moreover, if a preconditioner is found, it is not guaranteed to have a good condition number. QuTiP can make use of an incomplete LU preconditioner when using the iterative ``'gmres'``, ``'lgmres'``, and ``'bicgstab'`` solvers by setting ``use_precond=True``. The preconditioner optionally makes use of a combination of symmetric and anti-symmetric matrix permutations that attempt to improve the preconditioning process.  These features are discussed in the :ref:`steady-args` section.  Even with these state-of-the-art permutations, the generation of a successful preconditoner for non-symmetric matrices is currently a trial-and-error process due to the lack of mathematical work done in this area.  It is always recommended to begin with the direct solver with no additional arguments before selecting a different method.

Finding the steady-state solution is not limited to the Lindblad form of the master equation. Any time-independent Liouvillian constructed from a Hamiltonian and collapse operators can be used as an input::

>>> rho_ss = steadystate(L)

where ``L`` is the Louvillian.  All of the additional arguments can also be used in this case.


.. _steady-args:

Additional Solver Arguments
=============================

The following additional solver arguments are available for the steady-state solver:

.. cssclass:: table-striped

.. list-table::
   :widths: 10 30 60
   :header-rows: 1

   * - Keyword
     - Options (default listed first)
     - Description
   * - method
     - 'direct', 'eigen', 'power', 'iterative-gmres','iterative-lgmres', 'svd'
     - Method used for solving for the steady-state density matrix.
   * - sparse
     - True, False
     - Use sparse version of direct solver.
   * - weight
     - None
     - Allows the user to define the weighting factor used in the ``'direct'``, ``'GMRES'``, and ``'LGMRES'`` solvers.
   * - permc_spec
     - 'COLAMD', 'NATURAL'
     - Column ordering used in the sparse LU decomposition.
   * - use_rcm
     - False, True
     - Use a Reverse Cuthill-Mckee reordering to minimize the bandwidth of the modified Liouvillian used in the LU decomposition.  If ``use_rcm=True`` then the column ordering is set to ``'Natural'`` automatically unless explicitly set.
   * - use_precond
     - False, True
     - Attempt to generate a preconditioner when using the ``'iterative-gmres'`` and ``'iterative-lgmres'`` methods.
   * - M
     - None, sparse_matrix, LinearOperator
     - A user defined preconditioner, if any.
   * - use_wbm
     - False, True
     - Use a Weighted Bipartite Matching algorithm to attempt to make the modified Liouvillian more diagonally dominate, and thus for favorable for preconditioning.  Set to ``True`` automatically when using a iterative method, unless explicitly set.
   * - tol
     - 1e-9
     - Tolerance used in finding the solution for all methods expect ``'direct'`` and ``'svd'``.
   * - maxiter
     - 10000
     - Maximum number of iterations to perform for all methods expect ``'direct'`` and ``'svd'``.
   * - fill_factor
     - 10
     - Upper-bound on the allowed fill-in for the approximate inverse preconditioner.  This value may need to be set much higher than this in some cases.
   * - drop_tol
     - 1e-3
     - Sets the threshold for the relative magnitude of preconditioner elements that should be dropped.  A lower number yields a more accurate approximate inverse at the expense of fill-in and increased runtime.
   * - diag_pivot_thresh
     - None
     - Sets the threshold between :math:`[0,1]` for which diagonal elements are considered acceptable pivot points when using a preconditioner.
   * - ILU_MILU
     - 'smilu_2'
     - Selects the incomplete LU decomposition method algorithm used.

Further information can be found in the :func:`qutip.steadystate.steadystate` docstrings.


.. _steady-example:

Example: Harmonic Oscillator in Thermal Bath
============================================

A simple example of a system that reaches a steady state is a harmonic oscillator coupled to a thermal environment.  Below we consider a harmonic oscillator, initially in the :math:`\left|10\right>` number state, and weakly coupled to a thermal environment characterized by an average particle expectation value of :math:`\left<n\right>=2`.  We calculate the evolution via master equation and Monte Carlo methods, and see that they converge to the steady-state solution.  Here we choose to perform only a few Monte Carlo trajectories so we can distinguish this evolution from the master-equation solution.

.. plot:: guide/scripts/ex_steady.py
   :include-source:
