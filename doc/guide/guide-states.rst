.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _guide-states:

Manipulating States and Operators
*********************************

Introduction
============

In the previous guide section :ref:`guide-basics`, we saw how to create operators and states, using the functions built into QuTiP.  In this portion of the guide, we will look at performing basic operations with states and operators.  For more detailed demonstrations on how to use and manipulate these objects, see the :ref:`examples` chapter.

State vectors
==============

Here we begin by creating a Fock :func:`qutip.basis` vacuum state vector :math:`\left|0\right>` with in a Hilbert space with 5 number states, 0 -> 4:

>>> vec=basis(5,0)
>>> print vec
Quantum object: dims = [[5], [1]], shape = [5, 1], type = ket
Qobj data = 
[[ 1.]  #<-- |0>
 [ 0.]  #<-- |1>
 [ 0.]  #<-- |2>
 [ 0.]  #<-- |3>
 [ 0.]] #<-- |4>

and then create a lowering operator :math:`\left(\hat{a}\right)` corresponding to 5 number states using the :func:`qutip.destroy` function:

>>> a=destroy(5)
>>> print a
Quantum object: dims = [[5], [5]], shape = [5, 5], type = oper, isHerm = False
Qobj data = 
[[ 0.          1.          0.          0.          0.        ]
 [ 0.          0.          1.41421356  0.          0.        ]
 [ 0.          0.          0.          1.73205081  0.        ]
 [ 0.          0.          0.          0.          2.        ]
 [ 0.          0.          0.          0.          0.        ]]


Now lets apply the destruction operator to our vacuum state ``vec``,

>>> a*vec
Quantum object: dims = [[5], [1]], shape = [5, 1], type = ket
Qobj data = 
[[ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]]


We see that, as expected, the vacuum is transformed to the zero vector.  A more interesting example comes from using the adjoint of the lowering operator, the raising operator :math:`\hat{a}^\dagger`:

>>> a.dag()*vec
Quantum object: dims = [[5], [1]], shape = [5, 1], type = ket
Qobj data = 
[[ 0.]
 [ 1.]  #<-- |1>
 [ 0.]
 [ 0.]
 [ 0.]]

The raising operator has in indeed raised the state `vec` from the vacuum to the :math:`\left| 1\right>` state.  Instead of using the dagger ``dag()`` command to raise the state, we could have also used the built in :func:`qutip.create` function to make a raising operator:

>>> c=create(5)
>>> c*vec()
Quantum object: dims = [[5], [1]], shape = [5, 1], type = ket
Qobj data = 
[[ 0.]
 [ 1.]
 [ 0.]
 [ 0.]
 [ 0.]]

which obviously does the same thing.  We can of course raise the vacuum state more than once:

>>> c*c*vec
Quantum object: dims = [[5], [1]], shape = [5, 1], type = ket
Qobj data = 
[[ 0.        ]
 [ 0.        ]
 [ 1.41421356] #<-- |2>
 [ 0.        ]
 [ 0.        ]]

or just taking the square of the raising operator :math:`\left(\hat{a}^\dagger\right)^{2}`:

>>> c**2*vec
Quantum object: dims = [[5], [1]], shape = [5, 1], type = ket
Qobj data = 
[[ 0.        ]
 [ 0.        ]
 [ 1.41421356]
 [ 0.        ]
 [ 0.        ]]

Applying the raising operator twice gives the expected :math:`\sqrt (n+1)` dependence.  We can use the product of :math:`c*a` to also apply the number operator to the state vector ``vec``:

>>> c*a*vec
Quantum object: dims = [[5], [1]], shape = [5, 1], type = ket
Qobj data = 
[[ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]]

or on the :math:`\left| 1\right>` state:

>>> c*a*(c*vec)
Quantum object: dims = [[5], [1]], shape = [5, 1], type = ket
Qobj data = 
[[ 0.]
 [ 1.]
 [ 0.]
 [ 0.]
 [ 0.]]

or the :math:`\left| 2\right>` state:

>>> c*a*(c**2*vec)
Quantum object: dims = [[5], [1]], shape = [5, 1], type = ket
Qobj data = 
[[ 0.        ]
 [ 0.        ]
 [ 2.82842712]
 [ 0.        ]
 [ 0.        ]]

Notice how in this last example, application of the number operator does not give the expected value :math:`n=2`, but rather :math:`2\sqrt{2}`.  This is because this last state is not normalized to unity as :math:`c\left| n\right>=\sqrt{n+1}\left| n+1\right>`.  Therefore, we should normalize our vector first:

>>> c*a*(c**2*vec).unit()
Quantum object: dims = [[5], [1]], shape = [5, 1], type = ket
Qobj data = 
[[ 0.]
 [ 0.]
 [ 2.]
 [ 0.]
 [ 0.]]

Since we are giving a demonstration of using states and operators, we have done a lot more work than we should have.  For example, we do not need to operate on the vacuum state to generate a higher number fock state.  Instead we can use the :func:`qutip.basis` (or :func:`qutip.fock`) function to directly obtain the required state:

>>> vec=basis(5,2)
>>> print vec
Quantum object: dims = [[5], [1]], shape = [5, 1], type = ket
Qobj data = 
[[ 0.]
 [ 0.]
 [ 1.]
 [ 0.]
 [ 0.]]

Notice how it is automatically normalized.  We can also use the built in :func:`qutip.num` operator:

>>> n=num(5)
>>> print n
Quantum object: dims = [[5], [5]], shape = [5, 5], type = oper, isHerm = True
Qobj data = 
[[0 0 0 0 0]
 [0 1 0 0 0]
 [0 0 2 0 0]
 [0 0 0 3 0]
 [0 0 0 0 4]]

Therefore, instead of ``c*a*(c**2*vec).unit()`` we have:

>>> n*vec
Quantum object: dims = [[5], [1]], shape = [5, 1], type = ket
Qobj data = 
[[ 0.]
 [ 0.]
 [ 2.]
 [ 0.]
 [ 0.]]

We can also create superpositions of states:

>>> vec=(basis(5,0)+basis(5,1)).unit()
>>> print vec
Quantum object: dims = [[5], [1]], shape = [5, 1], type = ket
Qobj data = 
[[ 0.70710678]
 [ 0.70710678]
 [ 0.        ]
 [ 0.        ]
 [ 0.        ]]

where we have used the :func:`qutip.Qobj.unit` function to again normalize the state.  Operating with the number function again:

>>> n*vec
Quantum object: dims = [[5], [1]], shape = [5, 1], type = ket
Qobj data = 
[[ 0.        ]
 [ 0.70710678]
 [ 0.        ]
 [ 0.        ]
 [ 0.        ]]

We can also create coherent states and squeezed states by applying the :func:`qutip.displace` and :func:`qutip.squeez` functions to the vacuum state:

>>> vec=basis(5,0)
>>> d=displace(5,1j)
>>> s=squeez(5,0.25+0.25j)
>>> d*vec
Quantum object: dims = [[5], [1]], shape = [5, 1], type = ket
Qobj data = 
[[ 0.60655682+0.j        ]
 [ 0.00000000+0.60628133j]
 [-0.43038740+0.j        ]
 [ 0.00000000-0.24104351j]
 [ 0.14552147+0.j        ]]

>>> d*s*vec
Quantum object: dims = [[5], [1]], shape = [5, 1], type = ket
Qobj data = 
[[ 0.65893786+0.08139381j]
 [ 0.10779462+0.51579735j]
 [-0.37567217-0.01326853j]
 [-0.02688063-0.23828775j]
 [ 0.26352814+0.11512178j]]

Of course, displacing the vacuum gives a coherent state, which can also be generated using the built in :func:`qutip.coherent` function.

Density matrices
=================

The main purpose of QuTiP is to explore the dynamics of **open** quantum systems, where the most general state of a system is not longer a state vector, but rather a density matrix.  Since operations on density matrices operate identically to those of vectors, we will just briefly highlight creating and using these structures.

The simplest density matrix is created by forming the outer-product :math:\left\psi\right>\left<\psi\right|` of a ket vector:

>>> vec=basis(5,2)
>>> vec*vec.dag()
Quantum object: dims = [[5], [5]], shape = [5, 5], type = oper, isHerm = True
Qobj data = 
[[ 0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.]
 [ 0.  0.  1.  0.  0.]
 [ 0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.]]

A similar task can also be accomplished via the :func:`qutip.fock_dm` or :func:`qutip.ket2dm` functions:

>>> fock_dm(5,2)
Quantum object: dims = [[5], [5]], shape = [5, 5], type = oper, isHerm = True
Qobj data = 
[[ 0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.]
 [ 0.  0.  1.  0.  0.]
 [ 0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.]]

>>> ket2dm(vec)
Quantum object: dims = [[5], [5]], shape = [5, 5], type = oper, isHerm = True
Qobj data = 
[[ 0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.]
 [ 0.  0.  1.  0.  0.]
 [ 0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.]]

If we want to create a density matrix with equal classical probability of being found in the :math:`\left|2\right>` or :math:`\left|4\right>` number states we can do the following:

>>> 0.5*ket2dm(basis(5,4))+0.5*ket2dm(basis(5,2))
Quantum object: dims = [[5], [5]], shape = [5, 5], type = oper, isHerm = True
Qobj data = 
[[ 0.   0.   0.   0.   0. ]
 [ 0.   0.   0.   0.   0. ]
 [ 0.   0.   0.5  0.   0. ]
 [ 0.   0.   0.   0.   0. ]
 [ 0.   0.   0.   0.   0.5]]

or use ``0.5*fock_dm(5,2)+0.5*fock_dm(5,4)``.  There are also several other built in functions for creating predefined density matrices, for example :func:`qutip.coherent_dm` and :func:`qutip.thermal_dm` which create coherent state and thermal state density matrices, respectively.

>>> coherent_dm(5,1.25)
Quantum object: dims = [[5], [5]], shape = [5, 5], type = oper, isHerm = True
Qobj data = 
[[ 0.20980701  0.26141096  0.23509686  0.15572585  0.13390765]
 [ 0.26141096  0.32570738  0.29292109  0.19402805  0.16684347]
 [ 0.23509686  0.29292109  0.26343512  0.17449684  0.1500487 ]
 [ 0.15572585  0.19402805  0.17449684  0.11558499  0.09939079]
 [ 0.13390765  0.16684347  0.1500487   0.09939079  0.0854655 ]]

>>> thermal_dm(5,1.25)
Quantum object: dims = [[5], [5]], shape = [5, 5], type = oper, isHerm = True
Qobj data = 
[[ 0.44444444  0.          0.          0.          0.        ]
 [ 0.          0.24691358  0.          0.          0.        ]
 [ 0.          0.          0.13717421  0.          0.        ]
 [ 0.          0.          0.          0.0762079   0.        ]
 [ 0.          0.          0.          0.          0.04233772]]
}}}

QuTiP also provides a set of distance metrics for determining how close two density matrix distributions are to each other.  Included are the trace distance :func:`qutip.tracedist` and the fidelity :func:`qutip.fidelity`.

>>> x=coherent_dm(5,1.25)
>>> y=coherent_dm(5,1.25j) #<-- note the 'j'
>>> z=thermal_dm(5,0.125)
>>> fidelity(x,x)
1.0000000051410474
>>> tracedist(y,y)
0.0

We also know that for two-pure states, the trace distance (T) and the fidelity (F) are related by :math:`T=\sqrt{1-F^{2}}`.

>>> tracedist(y,x)
0.9771565838870081

>>> sqrt(1-fidelity(y,x)**2)
0.97715657039974568

For a pure state and a mixed state, :math:`1-F^{2}\le T` which can also be verified:

>>> 1-fidelity(x,z)**2
0.7784456314854065

>>> tracedist(x,z)
0.8563182215236257

Qubit (two-level) systems
=========================

Having spent a fair amount of time on basis states that represent harmonic oscillator states, we now move on to qubit, or two-level spin systems.  To create a state vector corresponding to a qubit system, we use the same :func:`qutip.basis`, or :func:`qutip.fock`, function with only two levels:

>>> spin=basis(2,0)

Now at this point one may ask how this state is different than that of a harmonic oscillator in the vacuum state truncated to two energy levels?

>>> vec=basis(2,0)

At this stage, there is no difference.  This should not be surprising as we called the exact same function twice.  The difference between the two comes from the action of the spin operators :func:`qutip.sigmax`, :func:`qutip.sigmay`, :func:`qutip.sigmaz`, :func:`qutip.sigmap`, and :func:`qutip.sigmam` on these two-level states.  For example, if ``vec`` corresponds to the vacuum state of a harmonic oscillator, then, as we have already seen, we can use the raising operator to get the :math:`\left|1\right>` state:

>>> vec
Quantum object: dims = [[2], [1]], shape = [2, 1], type = ket
Qobj data = 
[[ 1.]
 [ 0.]]

>>> c=create(2)
>>> c*vec
Quantum object: dims = [[2], [1]], shape = [2, 1], type = ket
Qobj data = 
[[ 0.]
 [ 1.]]

For a spin system, the operator analogous to the raising operator is the sigma-plus operator :func:`qutip.sigmap`.  Operating on the ``spin`` state gives:

>>> spin
Quantum object: dims = [[2], [1]], shape = [2, 1], type = ket
Qobj data = 
[[ 1.]
 [ 0.]]

>>> sigmap()*spin
Quantum object: dims = [[2], [1]], shape = [2, 1], type = ket
Qobj data = 
[[ 0.]
 [ 0.]]

Now we see the difference!  The :func:`qutip.sigmap` operator acting on the ``spin`` state returns the zero vector.  Why is this?  To see what happened, let us use the :func:`qutip.sigmaz` operator:

>>> sigmaz()
Quantum object: dims = [[2], [2]], shape = [2, 2], type = oper, isHerm = True
Qobj data = 
[[ 1.  0.]
 [ 0. -1.]]

>>> sigmaz()*spin
Quantum object: dims = [[2], [1]], shape = [2, 1], type = ket
Qobj data = 
[[ 1.]
 [ 0.]]

>>> spin2=basis(2,1)
>>> spin2
Quantum object: dims = [[2], [1]], shape = [2, 1], type = ket
Qobj data = 
[[ 0.]
 [ 1.]]

>>> sigmaz()*spin2
Quantum object: dims = [[2], [1]], shape = [2, 1], type = ket
Qobj data = 
[[ 0.]
 [-1.]]

The answer is now apparent.  Since the QuTiP :func:`qutip.sigmaz` function uses the standard z-basis representation of the sigma-z spin operator, the ``spin`` state corresponds to the :math:`\left|\mathrm{up}\right>` state of a two-level spin system while ``spin2`` gives the :math:`\left|\mathrm{down}\right>` state.  Therefore, in our previous example ``sigmap()*spin``, we raised the qubit state out of the truncated two-level Hilbert space resulting in the zero state.  

While at first glance this convention might seem somewhat odd, it is in fact quite handy.  For one, the spin operators remain in the conventional form.  Second, when the spin system is in the :math:`\left|\mathrm{up}\right>` state:

>>> sigmaz()*spin
Quantum object: dims = [[2], [1]], shape = [2, 1], type = ket
Qobj data = 
[[ 1.] #<--- zeroth element of matrix
 [ 0.]]

the non-zero component is the zeroth-element of the underlying matrix (remember that python uses c-indexing, and matrices start with the zeroth element).  The :math:`\left|\mathrm{down}\right>` state therefore has a non-zero entry in the first index position.  This corresponds nicely with the quantum information definitions of qubit states, where the excited :math:`\left|\mathrm{up}\right>` state is label as :math:`\left|0\right>`, and the :math:`\left|\mathrm{up}\right>` state by :math:`\left|1\right>`.

If one wants to create spin operators for higher spin systems, then the :func:`qutip.operators.jmat` function comes in handy. 

Expectation values
===================

Some of the most important information about quantum systems comes from calculating the expectation value of operators, both Hermitian and non-Hermitian, as the state or density matrix of the system varies in time.  Therefore, in this section we demonstrate the use of the :func:`qutip.expect` function.  Further examples of using the :func:`qutip.expect` function may be found at :ref:`examples_drivencavitysteady` and :ref:`examples_thermalmonte`.  To begin:

>>> vac=basis(5,0)
>>> one=basis(5,1)
>>> c=create(5)
>>> N=num(5)
>>> expect(N,vac)
0.0

>>> expect(N,one)
1.0

>>> coh=coherent_dm(5,1.0j)
>>> expect(N,coh)
0.997055574581 #should be equal to 1, small diff. due to truncated Hilbert space

>>> cat=(basis(5,4)+1.0j*basis(5,3)).unit()
>>> expect(c,cat)
1j

The :func:`qutip.expect` function also accepts lists or arrays of state vectors or density matrices for the second input:

>>> states=[(c**k*vac).unit() for k in range(5)] #must normalize
>>> expect(N,states)
[ 0.  1.  2.  3.  4.]

>>> cat_list=[(basis(5,4)+x*basis(5,3)).unit() for x in [0,1.0j,-1.0,-1.0j]]
>>> expect(c,cat_list)
[ 0.+0.j  0.+1.j -1.+0.j  0.-1.j]

Notice how in this last example, all of the return values are complex numbers.  Yet if we calculate just the first expectation value,

>>> expect(c,basis(5,4))
0.0

we get a real number.  This is because the :func:`qutip.expect` function looks to see whether the operator is Hermitian or not.  If the operator is Hermitian, than the output will always be real.  In the case of non-Hermitian operators, the return values may be complex.  Therefore, the expect function will return a array of complex values for non-Hermitian operators when the input is a list/array of states or density matrices.

Of course, the expect function works for spin states and operators:

>>> up=basis(2,0)
>>> down=basis(2,1)
>>> expect(sigmaz(),up)
1.0
>>>expect(sigmaz(),down)
-1.0

as well as the composite objects discussed in the next section :ref:`guide-tensor`:

>>> spin1=basis(2,0)
>>> spin2=basis(2,1)
>>>two_spins=tensor(spin1,spin2)
>>> sz1=tensor(sigmaz(),qeye(2))
>>> sz2=tensor(qeye(2),sigmaz())

>>> expect(sz1,two_spins)
1.0

>>> expect(sz2,two_spins)
-1.0

