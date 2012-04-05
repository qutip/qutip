.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _saving:

**********************************
Saving QuTiP Objects and Data Sets
**********************************

With time-consuming calculations it is often necessary to store the results to files on disk, so it can be post-processed and archived. In QuTiP there are two facilities for storing data: Quantum objects can be stored to files and later read back as python pickles, and numerical data (vectors and matrices) can be exported as plain text files in for example CSV (comma-separated values), TSV (tab-separated values), etc. The former method is prefered when further calculations will be performed with the data, and the latter when the calculations are completed and data is to be imported into a post-processing tool (e.g. for generating figures).

Storing and loading QuTiP objects
=================================

To store and load an arbitrary QuTiP related objects (:class:`Qobj`, :class:`Odedata`, etc.) the two functions :func:`qutip.qsave` and :func:`qutip.qload`. The :func:`qutip.qsave` takes an arbitrary object as first parameter and an optional filename as second parameter (default filename is `qutip_data.qu`). The filename extension is always `.qu`. The :func:`qutip.qload` a mandatory filename as first argument and loads and returns the objects in the file.

To illustrate how these functions can be used, consider a simple calculation of the steadystate of the 

>>> a = destroy(10); H = a.dag() * a ; c_ops = [sqrt(0.5) * a, sqrt(0.25) * a.dag()]
>>> rho_ss = steadystate(H, c_ops)

The steadystate density matrix `rho_ss` is an instance of :class:`Qobj`. It can be stored to a file `steadystate.qu` using 

>>> qsave(rho_ss, 'steadystate')
>>> ls *.qu
steadystate.qu

and it can later be loaded again, and used for further calculations:

>>> rho_ss_loaded = qload('steadystate')
Loaded Qobj object:
Quantum object: dims = [[10], [10]], shape = [10, 10], type = oper, isHerm = True
>>> a = destroy(10)
>>> expect(a.dag() * a, rho_ss_loaded)
array(0.9902248289344705)

The nice thing about the :func:`qutip.qsave` and :func:`qutip.qload` functions is that almost any object can be stored and load again later on. We can for example store a list of density matrices as returned by :func:`mesolve`:

>>> a = destroy(10); H = a.dag() * a ; c_ops = [sqrt(0.5) * a, sqrt(0.25) * a.dag()]
>>> psi0 = rand_ket(10)
>>> tlist = linspace(0, 10, 10)
>>> dm_list = mesolve(H, psi0, tlist, c_ops, [])
>>> qsave(dm_list, 'density_matrix_vs_time')

And it can be loaded used again later on, for example in an other program.

>>> dm_list_loaded = qload('density_matrix_vs_time')
Loaded list object.
>>> # use it in some calculation
>>> a = destroy(10)
>>> array([expect(a.dag() * a, dm) for dm in dm_list_loaded])
array([ 4.30052873,  3.41114025,  2.78257234,  2.32509271,  1.98722684,
        1.73608258,  1.54875697,  1.4087477 ,  1.30396859,  1.22548884])



Storing and loading datasets
============================

