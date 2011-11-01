.. QuTiP 
   Copyright (C) 2011, Paul D. Nation & Robert J. Johansson

.. _guide-basics:

Performing Basic operations on Quantum Objects
**********************************************

First things first
==================

.. warning::
   Do not run QuTiP from the installation directory.

To load the qutip modules, we must first call the import statement:

>>> from qutip import *


The quantum object class
========================

Intro
+++++

The key difference between classical and quantum mechanics lies in the use of operators instead of numbers as variables.  Moreover, we need to specify state vectors and their properties. Therefore, in computing the dynamics of quantum systems we need a data structure that is capable of encapsulating the properties of a quantum operator and ket/bra vectors.  The quantum object class, :func:`qutip.Qobj`, accomplishes this using matrix representation.

To begin, let us create a blank Qobj (in Python, Class objects differ from functions in the use of a beginning capital letter)

>>> Qobj() 
Quantum object: dims = [[1], [1]], shape = [1, 1]
Qobj data = 
[[0]]

where we see the blank Qobj object with dimensions, shape, and data.  Here the data corresponds to a 1x1-dimensional matrix consisting of a single zero entry.  We can create a Qobj with a user defined data set by passing a list or array of data into the Qobj:

>>> Qobj([1,2,3,4,5])
Quantum object: dims = [[1], [5]], shape = [1, 5]
Qobj data = 
[[1 2 3 4 5]]

>>> x=array([[1],[2],[3],[4],[5]])
>>> print Qobj(x)
Quantum object: dims = [[5], [1]], shape = [5, 1]
Qobj data = 
[[1]
 [2]
 [3]
 [4]
 [5]]

>>> r=random((4,4))
>>> print Qobj(r)
Quantum object: dims = [[4], [4]], shape = [4, 4]
Qobj data = 
[[ 0.76799998  0.06936066  0.10970546  0.13724402]
 [ 0.70644984  0.15371775  0.90649545  0.15349102]
 [ 0.69515726  0.13609801  0.52707457  0.6484309 ]
 [ 0.78328543  0.87295996  0.58964046  0.3998962 ]]

Notice how both the dims and shape change according to the input data.  Although dims and shape appear to have the same function, the difference will become quite clear in the section on tensor products and partial traces.  (If you are running QuTiP from a python script you must use the :func:`print` function to view the Qobj properties.)


States and operators
++++++++++++++++++++

Now, unless you have lots of free time, specifying the data for each object is inefficient.  Even more so when most objects correspond to commonly used types such as the ladder operators of a harmonic oscillator,the Pauli spin operators for a two-level system, or state vectors such as Fock states.  Therefore, QuTiP includes predefined objects for a variety of states:

+--------------------------+----------------------------+----------------------------------------+
| States                   | Command (# means optional) | Inputs                                 |
+==========================+============================+========================================+
| Fock state ket vector    | basis(N,#m) / fock(N,#m)   | N = number of levels in Hilbert space, |
|                          |                            | m = level containing excitation        |
|                          |                            | (0 if no m given)                      | 
+--------------------------+----------------------------+----------------------------------------+
| Fock density matrix      | fock_dm(N,#p)              | same as basis(N,m) / fock(N,m)         |
| (outer product of basis) |                            |                                        |
+--------------------------+----------------------------+----------------------------------------+
| Coherent state           | coherent(N,alpha)          | alpha = complex number (eigenvalue)    |
|                          |                            | for requested coherent state           |
+--------------------------+----------------------------+----------------------------------------+
| Coherent density matrix  | coherent_dm(N,alpha)       | same as coherent(N,alpha)              |
| (outer product)          |                            |                                        |
+--------------------------+----------------------------+----------------------------------------+
| Thermal density matrix   | thermal_dm(N,n)            | n = particle number expectation value  |
| (for n particles)        |                            |                                        |
+--------------------------+----------------------------+----------------------------------------+

and operators:

+--------------------------+----------------------------+----------------------------------------+
| Operators                | Command (# means optional) | Inputs                                 |
+==========================+============================+========================================+
| Identity                 | qeye(N)                    | N = number of levels in Hilbert space. |
+--------------------------+----------------------------+----------------------------------------+
| Lowering (destruction)   | destroy(N)                 | same as above                          |
| operator                 |                            |                                        |
+--------------------------+----------------------------+----------------------------------------+
| Raising (creation)       | create(N)                  | same as above                          |
| operator                 |                            |                                        |
+--------------------------+----------------------------+----------------------------------------+
| Number operator          | num(N)                     | same as above                          |
+--------------------------+----------------------------+----------------------------------------+
| Single-mode              | displace(N,alpha)          | N=number of levels in Hilbert space,   |
| displacement operator    |                            | alpha = complex displacement amplitude.|
+--------------------------+----------------------------+----------------------------------------+
| Single-mode              | squeez(N,sp)               | N=number of levels in Hilbert space,   |
| squeezing operator       |                            | sp = squeezing parameter.              |
+--------------------------+----------------------------+----------------------------------------+
| Sigma-X                  | sigmax()                   |                                        |
+--------------------------+----------------------------+----------------------------------------+
| Sigma-Y                  | sigmay()                   |                                        |
+--------------------------+----------------------------+----------------------------------------+
| Sigma-Z                  | sigmaz()                   |                                        |
+--------------------------+----------------------------+----------------------------------------+
| Sigma plus               | sigmap()                   |                                        |
+--------------------------+----------------------------+----------------------------------------+
| Sigma minus              | sigmam()                   |                                        |
+--------------------------+----------------------------+----------------------------------------+
| Higher spin operators    | jmat(j,#s)                 | j = integer or half-integer            |
|                          |                            | representing spin, s = 'x', 'y', 'z',  |
|                          |                            | '+', or '-'                            |
+--------------------------+----------------------------+----------------------------------------+


As an example, we give the output for a few of these objects:

>>> print basis(5,3)
Quantum object: dims = [[5], [1]], shape = [5, 1]
Qobj data = 
[[ 0.]
 [ 0.]
 [ 0.]
 [ 1.]
 [ 0.]]

>>> print coherent(5,0.5-0.5j)
Quantum object: dims = [[5], [1]], shape = [5, 1]
Qobj data = 
[[  7.78801702e-01 -5.63234865e-20j]
 [  3.89391417e-01 -3.89391417e-01j]
 [  7.59246032e-19 -2.75458952e-01j]
 [ -7.89861710e-02 -7.89861710e-02j]
 [ -4.31427083e-02 +3.46944695e-18j]]

>>> destroy(4)
Quantum object: dims = [[4], [4]], shape = [4, 4]
Qobj data = 
[[ 0.          1.          0.          0.        ]
 [ 0.          0.          1.41421356  0.        ]
 [ 0.          0.          0.          1.73205081]
 [ 0.          0.          0.          0.        ]]

>>> sigmaz()
Quantum object: dims = [[2], [2]], shape = [2, 2]
Qobj data = 
[[ 1.  0.]
 [ 0. -1.]]

>>> jmat(5/2.0,'+')
Quantum object: dims = [[6], [6]], shape = [6, 6]
Qobj data = 
[[ 0.          2.23606798  0.          0.          0.          0.        ]
 [ 0.          0.          2.82842712  0.          0.          0.        ]
 [ 0.          0.          0.          3.          0.          0.        ]
 [ 0.          0.          0.          0.          2.82842712  0.        ]
 [ 0.          0.          0.          0.          0.          2.23606798]
 [ 0.          0.          0.          0.          0.          0.        ]]


Qobj properties
+++++++++++++++

We have seen that a quantum object has three internal attributes, the data, dims, and shape properties.  These can be accessed in the following way:

>>> q=destroy(4)
>>> print q.dims
[[4], [4]]

>>> q.shape
[4, 4]  

In general, the properties of a Qobj object (or any Python class) can be retrieved using the `Q.property` notation.  In addition to the properties shown with the `print` function, the Qobj class also has the following:

+---------------+---------------+----------------------------------------+
| Property      | Command       | Description                            |
+===============+===============+========================================+
| Data          | Q.data        | Matrix representing state or operator  |
+---------------+---------------+----------------------------------------+
| Dimensions    | Q.dims        | List keeping track of shapes for       |
|               |               | individual components of a             |
|               |               | multipartite system (for tensor        |
|               |               | products and partial traces).          |
+---------------+---------------+----------------------------------------+
| Shape         | Q.shape       | Dimensions of underlying data matrix.  |
+---------------+---------------+----------------------------------------+
| is Hermitian? | Q.isherm      | Is the operator Hermitian or not?      |
+---------------+---------------+----------------------------------------+
| Type          | Q.type        | Is object of type 'ket, 'bra',         |
|               |               | 'oper', or 'super'?                    |
+---------------+---------------+----------------------------------------+

.. _about: 
.. figure:: quide-basics-qobj-box.png
   :align: center


For the destruction operator above:

>>> q.type
'oper'

>>> q.isherm
False

>>> q.data
<4x4 sparse matrix of type '<type 'numpy.complex128'>'
	with 3 stored elements in Compressed Sparse Row format>

The data property returns a message stating that the data is a sparse matrix.  All Qobj's store their data as a sparse matrix to save memory.  To access the underlying matrix one needs to use the :func:`qutip.Qobj.full` function as described in the functions section.

Qobj Math
+++++++++++

The rules for mathematical operations on Qobj's are similar to standard matrix arithmetic:

>>> q=destroy(4)
>>> x=sigmax()
>>> print q+5
Quantum object: dims = [[4], [4]], shape = [4, 4]
Qobj data = 
[[ 5.          6.          5.          5.        ]
 [ 5.          5.          6.41421356  5.        ]
 [ 5.          5.          5.          6.73205081]
 [ 5.          5.          5.          5.        ]]

>>> print x*x
Quantum object: dims = [[2], [2]], shape = [2, 2]
Qobj data = 
[[ 1.  0.]
 [ 0.  1.]]

>>> print q**3
Quantum object: dims = [[4], [4]], shape = [4, 4]
Qobj data = 
[[ 0.          0.          0.          2.44948974]
 [ 0.          0.          0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [ 0.          0.          0.          0.        ]]

>>> print x/sqrt(2)
Quantum object: dims = [[2], [2]], shape = [2, 2]
Qobj data = 
[[ 0.          0.70710678]
 [ 0.70710678  0.        ]]

of course, like matrices, multiplying two objects of incompatible shape throws an error:

>>> q*x
TypeError: Incompatible Qobj shapes

In addition, the logic operators is equal `==` and is not equal `!=` are also supported.

Functions operating on Qobj class
==================================

Like properties, the quantum object class has defined functions which operate only on members of the Qobj class.  For a general quantum object `Q`:

+-----------------+-----------------+----------------------------------------+
| Function        | Command         | Description                            |
+=================+=================+========================================+
| Dagger (adjoint)| Q.dag()         | Returns adjoint (dagger) of object.    |
+-----------------+-----------------+----------------------------------------+
| Diagonal        | Q.diag()        | Returns the diagonal elements.         |
+-----------------+-----------------+----------------------------------------+
| Eigenstates     | Q.eigenstates() | Returns eigenstates and eigenvectors.  |
+-----------------+-----------------+----------------------------------------+
| Exponential     | Q.expm()        | Matrix exponential of operator.        |
+-----------------+-----------------+----------------------------------------+
| Full            | Q.full()        | Returns full (not sparse) array of     |
|                 |                 | Q's data property.                     |
+-----------------+-----------------+----------------------------------------+
| Norm            | Q.norm()        | Returns L2 norm for states,            |
|                 |                 | trace norm for operators.              |
+-----------------+-----------------+----------------------------------------+
| Sqrt            | Q.sqrtm()       | Matrix sqrt of operator.               |
+-----------------+-----------------+----------------------------------------+
| Trace           | Q.tr()          | Returns trace of quantum object.       |
+-----------------+-----------------+----------------------------------------+
| Unit            | Q.unit()        | Returns normalized (unit)              |
|                 |                 | vector Q/Q.norm().                     |  
+-----------------+-----------------+----------------------------------------+


>>> basis(5,3)
Quantum object: dims = [[5], [1]], shape = [5, 1], type = ket
Qobj data = 
[[ 0.]
 [ 0.]
 [ 0.]
 [ 1.]
 [ 0.]]

>>> basis(5,3).dag()
Quantum object: dims = [[1], [5]], shape = [1, 5], type = bra
Qobj data = 
[[ 0.  0.  0.  1.  0.]]

>>> coherent_dm(5,1)
Quantum object: dims = [[5], [5]], shape = [5, 5], type = oper, isHerm = True
Qobj data = 
[[ 0.36791117  0.36774407  0.26105441  0.14620658  0.08826704]
 [ 0.36774407  0.36757705  0.26093584  0.14614018  0.08822695]
 [ 0.26105441  0.26093584  0.18523331  0.10374209  0.06263061]
 [ 0.14620658  0.14614018  0.10374209  0.05810197  0.035077  ]
 [ 0.08826704  0.08822695  0.06263061  0.035077    0.0211765 ]]

>>> coherent_dm(5,1).diag()
array([ 0.36791117,  0.36757705,  0.18523331,  0.05810197,  0.0211765 ])

>>> coherent_dm(5,1).full()
array([[ 0.36791117,  0.36774407,  0.26105441,  0.14620658,  0.08826704],
       [ 0.36774407,  0.36757705,  0.26093584,  0.14614018,  0.08822695],
       [ 0.26105441,  0.26093584,  0.18523331,  0.10374209,  0.06263061],
       [ 0.14620658,  0.14614018,  0.10374209,  0.05810197,  0.035077  ],
       [ 0.08826704,  0.08822695,  0.06263061,  0.035077  ,  0.0211765 ]])

>>> coherent_dm(5,1).norm()
1.0

>>> coherent_dm(5,1).sqrtm()
Quantum object: dims = [[5], [5]], shape = [5, 5], type = oper, isHerm = False
Qobj data = 
[[ 0.36791117 +6.66013801e-09j  0.36774407 -2.87612199e-09j
   0.26105441 -4.24323387e-09j  0.14620658 -1.21628628e-09j
   0.08826704 -1.21357197e-09j]
 [ 0.36774407 -3.87481342e-09j  0.36757705 +1.66576107e-09j
   0.26093584 +2.50548614e-09j  0.14614018 +7.07508704e-10j
   0.08822695 +6.28805009e-10j]
 [ 0.26105441 -2.75065517e-09j  0.26093584 +1.15201146e-09j
   0.18523331 +1.92733313e-09j  0.10374209 +5.01775972e-10j
   0.06263061 +1.34247407e-10j]
 [ 0.14620658 -1.54053667e-09j  0.14614017 +6.89127552e-10j
   0.10374209 +8.65055761e-10j  0.05810198 +2.81704042e-10j
   0.03507700 +5.25048476e-10j]
 [ 0.08826704 -9.30044364e-10j  0.08822695 +4.99516749e-10j
   0.06263061 +1.14878928e-10j  0.03507700 +1.71358232e-10j
   0.02117650 +1.17185351e-09j]]

>>> coherent_dm(5,1).tr()
1.0

>>> (basis(4,2)+basis(4,1)).unit()
Quantum object: dims = [[4], [1]], shape = [4, 1], type = ket
Qobj data = 
[[ 0.        ]
 [ 0.70710678]
 [ 0.70710678]
 [ 0.        ]]


