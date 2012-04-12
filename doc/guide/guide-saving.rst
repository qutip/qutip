.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _saving:

**********************************
Saving QuTiP Objects and Data Sets
**********************************

With time-consuming calculations it is often necessary to store the results to files on disk, so it can be post-processed and archived. In QuTiP there are two facilities for storing data: Quantum objects can be stored to files and later read back as python pickles, and numerical data (vectors and matrices) can be exported as plain text files in for example CSV (comma-separated values), TSV (tab-separated values), etc. The former method is prefered when further calculations will be performed with the data, and the latter when the calculations are completed and data is to be imported into a post-processing tool (e.g. for generating figures).

Storing and loading QuTiP objects
=================================

To store and load arbitrary QuTiP related objects (:class:`Qobj`, :class:`Odedata`, etc.) there are two functions: :func:`qutip.qsave` and :func:`qutip.qload`. The function :func:`qutip.qsave` takes an arbitrary object as first parameter and an optional filename as second parameter (default filename is `qutip_data.qu`). The filename extension is always `.qu`. The function :func:`qutip.qload` takes a mandatory filename as first argument and loads and returns the objects in the file.

To illustrate how these functions can be used, consider a simple calculation of the steadystate of the harmonic oscillator:

>>> a = destroy(10); H = a.dag() * a ; c_ops = [sqrt(0.5) * a, sqrt(0.25) * a.dag()]
>>> rho_ss = steadystate(H, c_ops)

The steadystate density matrix `rho_ss` is an instance of :class:`qutip.Qobj`. It can be stored to a file `steadystate.qu` using 

>>> qsave(rho_ss, 'steadystate')
>>> ls *.qu
steadystate.qu

and it can later be loaded again, and used in further calculations:

>>> rho_ss_loaded = qload('steadystate')
Loaded Qobj object:
Quantum object: dims = [[10], [10]], shape = [10, 10], type = oper, isHerm = True
>>> a = destroy(10)
>>> expect(a.dag() * a, rho_ss_loaded)
array(0.9902248289344705)

The nice thing about the :func:`qutip.qsave` and :func:`qutip.qload` functions is that almost any object can be stored and load again later on. We can for example store a list of density matrices as returned by :func:`qutip.mesolve`:

>>> a = destroy(10); H = a.dag() * a ; c_ops = [sqrt(0.5) * a, sqrt(0.25) * a.dag()]
>>> psi0 = rand_ket(10)
>>> tlist = linspace(0, 10, 10)
>>> dm_list = mesolve(H, psi0, tlist, c_ops, [])
>>> qsave(dm_list, 'density_matrix_vs_time')

And it can then be loaded and used again, for example in an other program:

>>> dm_list_loaded = qload('density_matrix_vs_time')
Loaded list object.
>>> # use it in some calculation
>>> a = destroy(10)
>>> array([expect(a.dag() * a, dm) for dm in dm_list_loaded])
array([ 4.30052873,  3.41114025,  2.78257234,  2.32509271,  1.98722684,
        1.73608258,  1.54875697,  1.4087477 ,  1.30396859,  1.22548884])


Storing and loading datasets
============================

The :func:`qutip.qsave` and :func:`qutip.qload` are great, but the file format used is only understood by QuTiP (python) programs. When data must be exported to other programs the prefered method is to store the data in the commonly used plain-text file formats. With the QuTiP functions :func:`qutip.file_data_store` and :func:`qutip.file_data_read` we can store and load **numpy** arrays and matrices to files on disk using a deliminator-separated value format (for example comma-separated values CSV). Almost any program can handle this file format.

The :func:`qutip.file_data_store` takes two mandatory and three optional arguments: 

>>> file_data_store(filename, data, numtype="complex", numformat="decimal", sep=",")

where `filename` is the name of the file, `data` is the data to be written to the file (must be a *numpy* array), `numtype` (optional) is a flag indicating numerical type that can take values `complex` or `real`, `numformat` (optional) specifies the numerical format that can take the values `exp` for the format `1.0e1` and `decimal` for the format `10.0`, and `sep` (optional) is an arbitrary single-character field separator (usually a tab, space, comma, semicolon, etc.). 

A common use for the :func:`qutip.file_data_store` function is to store the expectation values of a set of operatators for a sequence of times, e.g., as returned by the :func:`qutip.mesolve` function, which is what the following example does:

>>> a = destroy(10); H = a.dag() * a ; c_ops = [sqrt(0.5) * a, sqrt(0.25) * a.dag()]
>>> psi0 = rand_ket(10)
>>> tlist = linspace(0, 100, 100)
>>> expt_values = mesolve(H, psi0, tlist, c_ops, [a.dag() * a, a+a.dag(), -1j(a-a.dag())])
>>> shape(expt_values)
(3, 100)
>>> shape(tlist)
(100,1)
>>> output_data = vstack((tlist, expt_values))   # join time and expt data
>>> file_data_store('expect.dat', output_data.T) # Note the .T for transpose !
>>> ls *.dat
expect.dat
>>> !head expect.dat
# Generated by QuTiP: 100x4 complex matrix in decimal format [',' separated values].
0.0000000000+0.0000000000j,4.2424841416+0.0000000000j,2.3200046262+0.0000000000j,0.1937084248+0.0000000000j
0.1010101010+0.0000000000j,4.1379975175+0.0000000000j,2.2803919588+0.0000000000j,-0.0289188081+0.0000000000j
0.2020202020+0.0000000000j,4.0424499915+0.0000000000j,2.2243729051+0.0000000000j,-0.2486713739+0.0000000000j
0.3030303030+0.0000000000j,3.9527450545+0.0000000000j,2.1495725421+0.0000000000j,-0.4605913338+0.0000000000j
0.4040404040+0.0000000000j,3.8674018874+0.0000000000j,2.0562127752+0.0000000000j,-0.6616078826+0.0000000000j
0.5050505051+0.0000000000j,3.7856377679+0.0000000000j,1.9454779063+0.0000000000j,-0.8495893849+0.0000000000j
0.6060606061+0.0000000000j,3.7069902520+0.0000000000j,1.8189312038+0.0000000000j,-1.0229166838+0.0000000000j
0.7070707071+0.0000000000j,3.6311546972+0.0000000000j,1.6783060144+0.0000000000j,-1.1802965424+0.0000000000j
0.8080808081+0.0000000000j,3.5579106202+0.0000000000j,1.5254272690+0.0000000000j,-1.3206823679+0.0000000000j

In this case we didn't really need to store both the real and imaginary parts, so instead we could use the `numtype="real"` option:

>>> file_data_store('expect.dat', output_data.T, numtype="real")
>>> !head -n5 expect.dat
# Generated by QuTiP: 100x4 real matrix in decimal format [',' separated values].
0.0000000000,4.2424841416,2.3200046262,0.1937084248
0.1010101010,4.1379975175,2.2803919588,-0.0289188081
0.2020202020,4.0424499915,2.2243729051,-0.2486713739
0.3030303030,3.9527450545,2.1495725421,-0.4605913338

and if we prefer scientific notation we can request that using the `numformat="exp"` option

>>> file_data_store('expect.dat', output_data.T, numtype="real", numformat="exp")
>>> !head -n 5 expect.dat
# Generated by QuTiP: 100x4 real matrix in exp format [',' separated values].
0.0000000000e+00,4.2424841416e+00,2.3200046262e+00,1.9370842484e-01
1.0101010101e-01,4.1379975175e+00,2.2803919588e+00,-2.8918808147e-02
2.0202020202e-01,4.0424499915e+00,2.2243729051e+00,-2.4867137392e-01
3.0303030303e-01,3.9527450545e+00,2.1495725421e+00,-4.6059133382e-01


Loading data previously stored using :func:`qutip.file_data_store` (or some other software) is a even easier. Regardless of which deliminator was used, if data was stored as complex or real numbers, if it is in decimal or exponential form, the data can be loaded using the :func:`qutip.file_data_read`, which only takes the filename as mandatory argument.

>>> input_data = file_data_read('expect.dat')
>>> shape(input_data)
(100, 4)
>>> # do something with the data, e.g.
>>> plot(input_data[:,0],input_data[:,1]); show()

(If a particularly obscure choice of deliminator was used it might be necessary to use the optional second argument, for example `sep="_"` if _ is the deliminator).

