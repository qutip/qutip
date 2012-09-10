.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _parfor:

******************************************
Using QuTiP's Built-in Parallel for-loop
******************************************

.. ipython::
   :suppress:

   In [1]: from qutip import *

Often one is interested in the output of a given function as a single-parameter is varied.  For instance, in the Driven steady-state cavity example, we calculate the steady-state response as the driving frequency is varied.  In cases such as this, where each iteration is independent of the others, we may speedup the calculations by performing the iterations in parallel.  In QuTiP, parallel computations may be performed using the :func:`qutip.parfor` (parallel-for-loop) function.

To use the parfor function we need to define a function of a single-variable, and the range over which this variable is to be iterated.  For example:


.. ipython::

   In [1]: def func1(x): return x,x**2,x**3
   
   In [2]: [a,b,c]=parfor(func1,range(10))
   
   In [3]: print(a)
   
   In [4]: print(b)
   
   In [5]: print(c)

One can also use a single output variable as:

.. ipython::

   In [1]: x=parfor(func1,range(10))
   
   In [2]: print(x[0])
   
   In [3]: print(x[1])
   
   In [4]: print(x[2])

The :func:`qutip.parfor` function is not limited to just numbers, but also works for a variety of outputs:

.. ipython::

   In [1]: def func2(x): return x,Qobj(x),'a'*x
   
   In [2]: [a,b,c]=parfor(func2,range(5))
   
   In [3]: print(a)
   
   In [4]: print(b)
   
   In [5]: print(c)

Although :func:`qutip.parfor` allows functions with only one input, we can in fact pass more an a single variable by using a list of lists. Sounds confusing, but it is quite easy.


.. ipython::

	In [1]: def func1(args): index,x=args; print(index); return x,x**2,x**3
   
	In [2]: args=[[k,2*k] for k in range(10)] #<-- create list of lists with more than one variable

	In [3]: args
	
	In [4]: [a,b,c]=parfor(func1,args)
	
	In [5]: print(a)


This example also highlights the fact that the parfor function does not evaluate the sequence of elements in order.  Therefore, passing an index variable, as done in the previous example, is useful if one needs to keep track of individual function evaluations, for example when plotting.  Parfor is also useful for repeated tasks such as generating plots corresponding to the dynamical evolution of your system, or simultaneously simulating different parameter configurations.

