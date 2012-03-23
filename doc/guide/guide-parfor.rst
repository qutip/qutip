.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _parfor:

******************************************
Using QuTiP's Built-in Parallel for-loop
******************************************

Often one is interested in the output of a given function as a single-parameter is varied.  For instance, in the Driven steady-state cavity example, we calculate the steady-state response as the driving frequency is varied.  In cases such as this, where each iteration is independent of the others, we may speedup the calculations by performing the iterations in parallel.  In QuTiP, parallel computations may be performed using the :func:`qutip.parfor` (parallel-for-loop) function.

To use the parfor function we need to define a function of a single-variable, and the range over which this variable is to be iterated.  For example:

>>> def func1(x):
...     return x,x**2,x**3

>>> [a,b,c]=parfor(func1,range(10))
>>> print a
[0 1 2 3 4 5 6 7 8 9]
>>> print b
[ 0 1 4 9 16 25 36 49 64 81]
>>> print c
[ 0 1 8 27 64 125 216 343 512 729]


One can also use a single output variable as:

>>> x=parfor(func1,range(10))
>>> print x[0]
[0 1 2 3 4 5 6 7 8 9]
>>> print x[1]
[ 0  1  4  9 16 25 36 49 64 81]
>>> print x[2]
[  0   1   8  27  64 125 216 343 512 729]


The :func:`qutip.parfor` function is not limited to just numbers, but also works for a variety of outputs:


>>> def func2(x):
...     return x,Qobj(x),'a'*x

>>> print [a,b,c]=parfor(func2,range(5))
>>> print a
[0 1 2 3 4]
>>> print b
Quantum object: dims = [[1], [1]], shape = [1, 1], type = ket
Qobj data = 
[[0]]
Quantum object: dims = [[1], [1]], shape = [1, 1], type = ket
Qobj data = 
[[1]]
Quantum object: dims = [[1], [1]], shape = [1, 1], type = ket
Qobj data = 
[[2]]
Quantum object: dims = [[1], [1]], shape = [1, 1], type = ket
Qobj data = 
[[3]]
Quantum object: dims = [[1], [1]], shape = [1, 1], type = ket
Qobj data = 
[[4]]
>>> print c
['' 'a' 'aa' 'aaa' 'aaaa']



Although :func:`qutip.parfor` allows functions with only one input, we can in fact pass more an a single variable by using a list of lists. Sounds confusing, but it is quite easy.


>>> def func1(args):
...     index,x=args #<--sets the index variable to args[0], and x to args[1]
...     print index #<-- print which element in sequence is being calculated
...     return x,x**2,x**3

>>> args=[[k,2*k] for k in range(10)] #<-- create list of lists with more than one variable
>>> print args
[[0, 0], [1, 2], [2, 4], [3, 6], [4, 8], [5, 10], [6, 12], [7, 14], [8, 16], [9, 18]]
>>> [a,b,c]=parfor(func1,args)
0
1
2
3
4
5
6
8
9
7
>>> print a
[ 0  2  4  6  8 10 12 14 16 18]


This example also highlights the fact that the parfor function does not evaluate the sequence of elements in order.  Therefore, passing an index variable, as done in the previous example, is useful if one needs to keep track of individual function evaluations, for example when plotting.  Parfor is also useful for repeated tasks such as generating plots corresponding to the dynamical evolution of your system, or simultaneously simulating different parameter configurations.

