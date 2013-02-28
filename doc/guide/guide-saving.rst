.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _saving:

**********************************
Saving QuTiP Objects and Data Sets
**********************************

.. ipython::
   :suppress:

   In [1]: from qutip import *


With time-consuming calculations it is often necessary to store the results to files on disk, so it can be post-processed and archived. In QuTiP there are two facilities for storing data: Quantum objects can be stored to files and later read back as python pickles, and numerical data (vectors and matrices) can be exported as plain text files in for example CSV (comma-separated values), TSV (tab-separated values), etc. The former method is preferred when further calculations will be performed with the data, and the latter when the calculations are completed and data is to be imported into a post-processing tool (e.g. for generating figures).

Storing and loading QuTiP objects
=================================

To store and load arbitrary QuTiP related objects (:class:`qutip.Qobj`, :class:`qutip.Odedata`, etc.) there are two functions: :func:`qutip.fileio.qsave` and :func:`qutip.fileio.qload`. The function :func:`qutip.fileio.qsave` takes an arbitrary object as first parameter and an optional filename as second parameter (default filename is `qutip_data.qu`). The filename extension is always `.qu`. The function :func:`qutip.fileio.qload` takes a mandatory filename as first argument and loads and returns the objects in the file.

To illustrate how these functions can be used, consider a simple calculation of the steadystate of the harmonic oscillator:

.. ipython::

   	In [1]: a = destroy(10); H = a.dag() * a ; c_ops = [sqrt(0.5) * a, sqrt(0.25) * a.dag()]
   
   	In [2]: rho_ss = steadystate(H, c_ops)

The steadystate density matrix `rho_ss` is an instance of :class:`qutip.Qobj`. It can be stored to a file `steadystate.qu` using 

.. ipython::

   	In [1]: qsave(rho_ss, 'steadystate')
   
   	In [2]: ls *.qu


and it can later be loaded again, and used in further calculations:

.. ipython::

   	In [1]: rho_ss_loaded = qload('steadystate')
   
   	In [2]: a = destroy(10)
	
	In [3]: expect(a.dag() * a, rho_ss_loaded)

The nice thing about the :func:`qutip.fileio.qsave` and :func:`qutip.fileio.qload` functions is that almost any object can be stored and load again later on. We can for example store a list of density matrices as returned by :func:`qutip.mesolve`:

.. ipython::

   	In [1]: a = destroy(10); H = a.dag() * a ; c_ops = [sqrt(0.5) * a, sqrt(0.25) * a.dag()]
   
   	In [2]: psi0 = rand_ket(10)
	
	In [3]: tlist = linspace(0, 10, 10)
	
	In [4]: dm_list = mesolve(H, psi0, tlist, c_ops, [])
	
	In [5]: qsave(dm_list, 'density_matrix_vs_time')

And it can then be loaded and used again, for example in an other program:

.. ipython::

   	In [1]: dm_list_loaded = qload('density_matrix_vs_time')
   
   	In [2]: a = destroy(10)
	
	In [3]: expect(a.dag() * a, dm_list_loaded.states)


Storing and loading datasets
============================

The :func:`qutip.fileio.qsave` and :func:`qutip.fileio.qload` are great, but the file format used is only understood by QuTiP (python) programs. When data must be exported to other programs the preferred method is to store the data in the commonly used plain-text file formats. With the QuTiP functions :func:`qutip.file_data_store` and :func:`qutip.file_data_read` we can store and load **numpy** arrays and matrices to files on disk using a deliminator-separated value format (for example comma-separated values CSV). Almost any program can handle this file format.

The :func:`qutip.file_data_store` takes two mandatory and three optional arguments: 

>>> file_data_store(filename, data, numtype="complex", numformat="decimal", sep=",")

where `filename` is the name of the file, `data` is the data to be written to the file (must be a *numpy* array), `numtype` (optional) is a flag indicating numerical type that can take values `complex` or `real`, `numformat` (optional) specifies the numerical format that can take the values `exp` for the format `1.0e1` and `decimal` for the format `10.0`, and `sep` (optional) is an arbitrary single-character field separator (usually a tab, space, comma, semicolon, etc.). 

A common use for the :func:`qutip.file_data_store` function is to store the expectation values of a set of operators for a sequence of times, e.g., as returned by the :func:`qutip.mesolve` function, which is what the following example does:

.. ipython::

   	In [1]: a = destroy(10); H = a.dag() * a ; c_ops = [sqrt(0.5) * a, sqrt(0.25) * a.dag()]
   
   	In [2]: psi0 = rand_ket(10)
	
	In [3]: tlist = linspace(0, 100, 100)
	
	In [4]: medata = mesolve(H, psi0, tlist, c_ops, [a.dag() * a, a + a.dag(), -1j * (a - a.dag())])
	
	In [5]:	shape(medata.expect)
	
	In [6]: shape(tlist)
	
	In [7]: output_data = vstack((tlist, medata.expect))   # join time and expt data
	
	In [8]: file_data_store('expect.dat', output_data.T) # Note the .T for transpose!
	
	In [9]: ls *.dat
	
	In [10]: !head expect.dat


In this case we didn't really need to store both the real and imaginary parts, so instead we could use the `numtype="real"` option:

.. ipython::

   	In [1]: file_data_store('expect.dat', output_data.T, numtype="real")
   
   	In [2]: !head -n5 expect.dat


and if we prefer scientific notation we can request that using the `numformat="exp"` option

.. ipython::

   	In [1]: file_data_store('expect.dat', output_data.T, numtype="real", numformat="exp")
   
   	In [2]: !head -n 5 expect.dat

Loading data previously stored using :func:`qutip.file_data_store` (or some other software) is a even easier. Regardless of which deliminator was used, if data was stored as complex or real numbers, if it is in decimal or exponential form, the data can be loaded using the :func:`qutip.file_data_read`, which only takes the filename as mandatory argument.

.. ipython::

   	In [1]: input_data = file_data_read('expect.dat')
   
   	In [2]: shape(input_data)
	
	In [4]: from pylab import *
	
	@savefig saving_ex.png width=4in align=center
	In [3]: plot(input_data[:,0], input_data[:,1]);  # plot the data


(If a particularly obscure choice of deliminator was used it might be necessary to use the optional second argument, for example `sep="_"` if _ is the deliminator).

