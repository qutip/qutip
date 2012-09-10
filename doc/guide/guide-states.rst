.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson


.. _states:

*************************************
Manipulating States and Operators
*************************************

.. ipython::
   :suppress:

   In [1]: from qutip import *

.. _states-intro:

Introduction
=================

In the previous guide section :ref:`basics`, we saw how to create operators and states, using the functions built into QuTiP.  In this portion of the guide, we will look at performing basic operations with states and operators.  For more detailed demonstrations on how to use and manipulate these objects, see the :ref:`examples` section.


.. _states-vectors:

State Vectors (kets or bras)
==============================

Here we begin by creating a Fock :func:`qutip.basis` vacuum state vector :math:`\left|0\right>` with in a Hilbert space with 5 number states, 0 -> 4:

.. ipython::
 
    In [1]: vec=basis(5,0)
    
    In [2]: print(vec)


and then create a lowering operator :math:`\left(\hat{a}\right)` corresponding to 5 number states using the :func:`qutip.destroy` function:

.. ipython::

    In [1]: a=destroy(5)
	
    In [2]: print(a)


Now lets apply the destruction operator to our vacuum state ``vec``,


.. ipython::

    In [1]: a*vec


We see that, as expected, the vacuum is transformed to the zero vector.  A more interesting example comes from using the adjoint of the lowering operator, the raising operator :math:`\hat{a}^{+}`:

.. ipython::

    In [1]: a.dag()*vec


The raising operator has in indeed raised the state `vec` from the vacuum to the :math:`\left| 1\right>` state.  Instead of using the dagger ``dag()`` command to raise the state, we could have also used the built in :func:`qutip.create` function to make a raising operator:

.. ipython::

    In [1]: c=create(5)
	
    In [2]: c*vec


which obviously does the same thing.  We can of course raise the vacuum state more than once:

.. ipython::

    In [1]: c*c*vec


or just taking the square of the raising operator :math:`\left(\hat{a}^{+}\right)^{2}`:

.. ipython::

    In [1]: c**2*vec


Applying the raising operator twice gives the expected :math:`\sqrt{n+1}` dependence.  We can use the product of :math:`c*a` to also apply the number operator to the state vector ``vec``:

.. ipython::

    In [1]: c*a*vec


or on the :math:`\left| 1\right>` state:

.. ipython::

    In [1]: c*a*(c*vec)


or the :math:`\left| 2\right>` state:

.. ipython::

    In [1]: c*a*(c**2*vec)


Notice how in this last example, application of the number operator does not give the expected value :math:`n=2`, but rather :math:`2\sqrt{2}`.  This is because this last state is not normalized to unity as :math:`c\left| n\right>=\sqrt{n+1}\left| n+1\right>`.  Therefore, we should normalize our vector first:

.. ipython::

    In [1]: c*a*(c**2*vec).unit()


Since we are giving a demonstration of using states and operators, we have done a lot more work than we should have.  For example, we do not need to operate on the vacuum state to generate a higher number Fock state.  Instead we can use the :func:`qutip.basis` (or :func:`qutip.fock`) function to directly obtain the required state:

.. ipython::

    In [1]: vec=basis(5,2)
   
    In [2]: print(vec)


Notice how it is automatically normalized.  We can also use the built in :func:`qutip.num` operator:

.. ipython::

    In [1]: n=num(5)
   
    In [2]: print(n)


Therefore, instead of ``c*a*(c**2*vec).unit()`` we have:

.. ipython::

    In [1]: n*vec


We can also create superpositions of states:

.. ipython::

    In [1]: vec=(basis(5,0)+basis(5,1)).unit()
   
    In [2]: print(vec)


where we have used the :func:`qutip.Qobj.unit` function to again normalize the state.  Operating with the number function again:

.. ipython::

    In [1]: n*vec


We can also create coherent states and squeezed states by applying the :func:`qutip.displace` and :func:`qutip.squeez` functions to the vacuum state:

.. ipython::

    In [1]: vec=basis(5,0)
  
    In [2]: d=displace(5,1j)
   
    In [3]: s=squeez(5,0.25+0.25j)
   
    In [4]: d*vec


.. ipython::

    In [1]: d*s*vec


Of course, displacing the vacuum gives a coherent state, which can also be generated using the built in :func:`qutip.coherent` function.


.. _states-dm:

Density matrices
=================

The main purpose of QuTiP is to explore the dynamics of **open** quantum systems, where the most general state of a system is not longer a state vector, but rather a density matrix.  Since operations on density matrices operate identically to those of vectors, we will just briefly highlight creating and using these structures.

The simplest density matrix is created by forming the outer-product :math:\left\psi\right>\left<\psi\right|` of a ket vector:

.. ipython::

    In [1]: vec=basis(5,2)
   
    In [2]: vec*vec.dag()

A similar task can also be accomplished via the :func:`qutip.fock_dm` or :func:`qutip.ket2dm` functions:

.. ipython::

    In [1]: fock_dm(5,2)


.. ipython::

    In [1]: ket2dm(vec)


If we want to create a density matrix with equal classical probability of being found in the :math:`\left|2\right>` or :math:`\left|4\right>` number states we can do the following:

.. ipython::

    In [1]: 0.5*ket2dm(basis(5,4))+0.5*ket2dm(basis(5,2))


or use ``0.5*fock_dm(5,2)+0.5*fock_dm(5,4)``.  There are also several other built in functions for creating predefined density matrices, for example :func:`qutip.coherent_dm` and :func:`qutip.thermal_dm` which create coherent state and thermal state density matrices, respectively.


.. ipython::

    In [1]: coherent_dm(5,1.25)


.. ipython::

    In [1]: thermal_dm(5,1.25)


QuTiP also provides a set of distance metrics for determining how close two density matrix distributions are to each other.  Included are the trace distance :func:`qutip.tracedist` and the fidelity :func:`qutip.fidelity`.

.. ipython::

    In [1]: x=coherent_dm(5,1.25)
	
    In [2]: y=coherent_dm(5,1.25j) #<-- note the 'j'
	
    In [3]: z=thermal_dm(5,0.125)
	
    In [4]: fidelity(x,x)
	
    In [5]: tracedist(y,y)


We also know that for two-pure states, the trace distance (T) and the fidelity (F) are related by :math:`T=\sqrt{1-F^{2}}`.

.. ipython::

    In [1]: tracedist(y,x)

.. ipython::

	In [1]: sqrt(1-fidelity(y,x)**2)


For a pure state and a mixed state, :math:`1-F^{2}\le T` which can also be verified:

.. ipython::

    In [1]: 1-fidelity(x,z)**2

.. ipython::

    In [1]: tracedist(x,z)


.. _states-qubit:

Qubit (two-level) systems
=========================

Having spent a fair amount of time on basis states that represent harmonic oscillator states, we now move on to qubit, or two-level spin systems.  To create a state vector corresponding to a qubit system, we use the same :func:`qutip.basis`, or :func:`qutip.fock`, function with only two levels:


.. ipython::

    In [1]: spin=basis(2,0)

Now at this point one may ask how this state is different than that of a harmonic oscillator in the vacuum state truncated to two energy levels?

.. ipython::
    
	In [1]: vec=basis(2,0)

At this stage, there is no difference.  This should not be surprising as we called the exact same function twice.  The difference between the two comes from the action of the spin operators :func:`qutip.sigmax`, :func:`qutip.sigmay`, :func:`qutip.sigmaz`, :func:`qutip.sigmap`, and :func:`qutip.sigmam` on these two-level states.  For example, if ``vec`` corresponds to the vacuum state of a harmonic oscillator, then, as we have already seen, we can use the raising operator to get the :math:`\left|1\right>` state:

.. ipython::
    
	In [1]: vec

.. ipython::
    
	In [1]: c=create(2)
	
	In [2]: c*vec


For a spin system, the operator analogous to the raising operator is the sigma-plus operator :func:`qutip.sigmap`.  Operating on the ``spin`` state gives:

.. ipython::
    
	In [1]: spin
    
	In [2]: sigmap()*spin

Now we see the difference!  The :func:`qutip.sigmap` operator acting on the ``spin`` state returns the zero vector.  Why is this?  To see what happened, let us use the :func:`qutip.sigmaz` operator:

.. ipython::
    
	In [1]: sigmaz()
	
	In [2]: sigmaz()*spin
	
	In [3]: spin2=basis(2,1)
	
	In [4]: spin2
	
	In [5]: sigmaz()*spin2


The answer is now apparent.  Since the QuTiP :func:`qutip.sigmaz` function uses the standard z-basis representation of the sigma-z spin operator, the ``spin`` state corresponds to the :math:`\left|\mathrm{up}\right>` state of a two-level spin system while ``spin2`` gives the :math:`\left|\mathrm{down}\right>` state.  Therefore, in our previous example ``sigmap()*spin``, we raised the qubit state out of the truncated two-level Hilbert space resulting in the zero state.  

While at first glance this convention might seem somewhat odd, it is in fact quite handy.  For one, the spin operators remain in the conventional form.  Second, when the spin system is in the :math:`\left|\mathrm{up}\right>` state:

.. ipython::
    
	In [1]: sigmaz()*spin

the non-zero component is the zeroth-element of the underlying matrix (remember that python uses c-indexing, and matrices start with the zeroth element).  The :math:`\left|\mathrm{down}\right>` state therefore has a non-zero entry in the first index position.  This corresponds nicely with the quantum information definitions of qubit states, where the excited :math:`\left|\mathrm{up}\right>` state is label as :math:`\left|0\right>`, and the :math:`\left|\mathrm{down}\right>` state by :math:`\left|1\right>`.

If one wants to create spin operators for higher spin systems, then the :func:`qutip.operators.jmat` function comes in handy. 

.. _states-expect:

Expectation values
===================

Some of the most important information about quantum systems comes from calculating the expectation value of operators, both Hermitian and non-Hermitian, as the state or density matrix of the system varies in time.  Therefore, in this section we demonstrate the use of the :func:`qutip.expect` function.  To begin:

.. ipython::
    
	In [1]: vac=basis(5,0)
	
	In [2]: one=basis(5,1)
	
	In [3]: c=create(5)
	
	In [4]: N=num(5)
	
	In [5]: expect(N,vac)
	
	In [6]: expect(N,one)


.. ipython::
    
	In [1]: coh=coherent_dm(5,1.0j)
	
	In [2]: expect(N,coh)

.. ipython::
    
	In [1]: cat=(basis(5,4)+1.0j*basis(5,3)).unit()
	
	In [2]: expect(c,cat)

The :func:`qutip.expect` function also accepts lists or arrays of state vectors or density matrices for the second input:

.. ipython::
    
	In [1]: states=[(c**k*vac).unit() for k in range(5)] #must normalize
	
	In [2]: expect(N,states)

.. ipython::
    
	In [1]: cat_list=[(basis(5,4)+x*basis(5,3)).unit() for x in [0,1.0j,-1.0,-1.0j]]
	
	In [2]: expect(c,cat_list)

Notice how in this last example, all of the return values are complex numbers.  This is because the :func:`qutip.expect` function looks to see whether the operator is Hermitian or not.  If the operator is Hermitian, than the output will always be real.  In the case of non-Hermitian operators, the return values may be complex.  Therefore, the expect function will return a array of complex values for non-Hermitian operators when the input is a list/array of states or density matrices.

Of course, the expect function works for spin states and operators:


.. ipython::
    
	In [1]: up=basis(2,0)
	
	In [2]: down=basis(2,1)
	
	In [3]: expect(sigmaz(),up)
	
	In [4]: expect(sigmaz(),down)


as well as the composite objects discussed in the next section :ref:`tensor`:

.. ipython::
    
	In [1]: spin1=basis(2,0)
	
	In [2]: spin2=basis(2,1)
	
	In [3]: two_spins=tensor(spin1,spin2)
	
	In [4]: sz1=tensor(sigmaz(),qeye(2))
	
	In [5]: sz2=tensor(qeye(2),sigmaz())
	
	In [6]: expect(sz1,two_spins)
	
	In [7]: expect(sz2,two_spins)


