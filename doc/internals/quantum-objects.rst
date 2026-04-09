Quantum Objects
###############

The primary user-facing type is :obj:`~qutip.Qobj`, which all users will have
interacted with as it represents any quantum object in QuTiP.
:obj:`~qutip.Qobj` consists of two major components: the underlying data
representation of the object, and information about what object is being
represented.  In fact, :obj:`~qutip.Qobj` itself is only really responsible for
managing this "auxiliary" information, ensuring that all mathematical operations
on the data make sense, and neatly wrapping the data-layer functions to be more
convenient for the user.  All the numeric heavy lifting is handled by the data
layer.


Data Store
==========

The data is actually stored in the :attr:`~qutip.Qobj.data` attribute of a
:obj:`~qutip.Qobj`.  This is always a "data-layer type", which will be covered
in more detail in the coming sections.  The different available types store the
data in different manners, such as full two-dimensional dense matrices (like
NumPy arrays) or compressed sparse matrix formats, as different formats have
different advantages.  :obj:`~qutip.Qobj` in general does not care what form its
data is stored in, as all numerical operations are handed off to the data-layer
dispatchers, which will ensure the correct specialised methods are called.

Typically these data layer types do not expose buffers to Python-space
programmes directly, though individual types may have an :meth:`as_array`,
:meth:`as_scipy`, or similar method.  This is for speed and memory-safety.  Once
Python can access the underlying data buffers directly, all references to them
must be managed by the garbage collector to ensure that nothing goes stale, even
if only partial slices onto the data are taken.  To access a copy of the data,
:obj:`~qutip.Qobj` exposes the :meth:`~qutip.Qobj.full` method, which will
always return a two-dimensional :obj:`numpy.ndarray`.  If your object is too
large to reasonably have a dense-matrix representation, you will need to use the
specific methods on the data-layer type that is used.

The amount of information guaranteed to be stored by data-layer types is very
small---sufficient to know whether a linear algebra operation *can* take place.
This is currently just the shape of the data being represented; there is no
information on Hilbert space (tensor product) dimensions, what type of object
the data represents, or anything else.  This is all managed by
:obj:`~qutip.Qobj`.


Auxiliary Information
=====================

:obj:`~qutip.Qobj` contains several pieces of auxiliary information about the
object being represented.  The major parts are:

:attr:`qutip.Qobj.dims`
   The tensor-product structure that this quantum object lives in.  This is a
   list of two elements---the "left" dimensions and the "right" dimensions.  For
   example, an operator on a Hilbert space formed of two qubits has
   :attr:`~qutip.Qobj.dims` of ``[[2, 2], [2, 2]]``.  A state (a "ket") in the
   same space has dimensions of ``[[2, 2], [1, 1]]``.  The data structure
   storing these two objects would not keep this information; the
   :attr:`~qutip.core.data.Data.shape` attributes would respectively be simply
   ``(4, 4)`` and ``(4, 1)``.

:attr:`qutip.Qobj.type`
   The type of object represented by this :obj:`~qutip.Qobj`.  This is typically
   derived from :attr:`~qutip.Qobj.dims`.  This is a string, containing a
   human-readable description of what type the object is, such as ``"oper"``,
   ``"ket"`` or ``"bra"``.

:attr:`qutip.Qobj.superrep`
   If this object is a super-operator, then this attribute is a string
   describing the particular representation used, such as the default
   ``"super"`` or something more specific like ``"choi"``.


Methods
=======

As :obj:`~qutip.Qobj` is the primary Python user-facing type, it provides all
the Python niceties and "magic" methods, such as :meth:`~qutip.Qobj.__add__`,
:meth:`~qutip.Qobj.__mul__` and so forth.  :obj:`~qutip.Qobj` will check that
the operation makes sense, returning :data:`NotImplemented` on failure, but a
successful output will typically require passing off to the data layer.  The
data layer types themselves generally do *not* have these magic methods, as they
are not designed to be manipulated by user code in Python.

In addition to the operator-overloading methods, :obj:`~qutip.Qobj` also
provides many quantum-specific mathematical operations, such as
:meth:`~qutip.Qobj.dag`, :meth:`~qutip.Qobj.norm`, :meth:`~qutip.Qobj.proj` and
:meth:`~qutip.Qobj.eigenstates`.  Typically these are pass-throughs to the data
layer, as :obj:`~qutip.Qobj` accesses its own data using the abstract interface
defined there.

:obj:`~qutip.Qobj` adds in its own :meth:`~qutip.Qobj.__repr__` methods for nice
output in REPLs, and includes IPython and Jupyter integration.  When running in
a rich environment, the :meth:`~qutip.Qobj._repr_latex_` method will be called
to better format the output.


Derived and Related Types
=========================

:obj:`~qutip.Qobj` is not the only type representing quantum objects in QuTiP,
but it is by far the most common from the user's perspective.  Sometimes, for
efficiency, other classes are necessary which either derive from or wrap
:obj:`~qutip.Qobj`.  These are often more specialised, and used only in some
parts of the code base.


QobjEvo
-------

The most wide-spread of these related types is :obj:`~qutip.QobjEvo`, which
represents a time-dependent quantum objects.  Essentially, it represents an
object :math:`A(t)` which can be described by

.. math::
   A(t) = \sum_k f_k(t) A_k

for some scalar, time-dependent functions :math:`f_k(t)` and some time-\
*independent* quantum objects in compatible Hilbert spaces :math:`A_k`.  These
:math:`A_k` will be instances of :obj:`~qutip.Qobj`.

This class is mostly for usage in solvers and optimisers which work on
time-dependent objects, as it can transpile the scalar time-dependence down to
C code and compile this to run natively.  The classes may be instantiated
directly by users when they want to reuse the results of compilation, as this
process typically takes several seconds.

:obj:`~qutip.QobjEvo` does not store its own data, but uses the underlying
:obj:`~qutip.Qobj` instances until its :meth:`~qutip.QobjEvo.compile` method is
called.  This method produces a C extension type :obj:`~qutip.CQobjEvo`, which
*does* store its own data.
