Converting Between Types: :obj:`~qutip.core.data.to`
####################################################

The backbone of the data layer is the ability to freely convert between two
given types.  As long as we have this, we know we can always change our inputs
into a set of types which we have a known-good function to handle.

The object :obj:`~qutip.core.data.to` is the true source of all types on the
data layer.  If a type is known to :obj:`to`, it will work with any mathematical
operation in QuTiP.  For a type to be known, at a minimum it needs one
conversion *to* it from a known data type, and one conversion *from* it to a
known data type.

Some examples of usage:

- simple conversion from one type to another ::

   >>> matrix = data.dense.identity(5)
   >>> matrix
   Dense(shape=(5, 5), fortran=True)
   >>> data.to(data.CSR, matrix)
   CSR(shape=(5, 5), nnz=5)

- get a callable function for a particular conversion from one specified
  type to another ::

   >>> data.to[data.CSR, data.Dense]
   <converter to CSR from Dense>

- get a callable function into a particular type from any data-layer type ::

   >>> data.to[data.Dense]
   <converter to Dense>

- add a new data layer type to the conversion, and have everything else filled
  in automatically ::

   >>> class NewDataType(data.Data):
   ...     # [...]
   >>> def new_from_dense(matrix: data.Dense) -> NewDataType:
   ...     # [...]
   >>> def dense_from_new(matrix: NewDataType) -> data.Dense:
   ...     # [...]
   >>> data.to.add_conversions([
   ...     (NewDataType, data.Dense, new_from_dense),
   ...     (data.Dense, NewDataType, dense_from_new),
   ... ])
   >>> data.to[data.CSR, NewDataType]
   <converter to CSR from NewDataType>


Basic Usage
===========

Convert data into a different type.  This object is the knowledge source for
every allowable data-layer type in QuTiP, and provides the conversions between
all of them.

The base use is to call this object as a function with signature ::

    (type, data) -> converted_data

where ``type`` is a type object (such as :obj:`~qutip.core.data.CSR`, or that
obtained by calling ``type(matrix)``) and ``data`` is data in a data-layer type.
If you want to create a data-layer type from non-data-layer data, use
:obj:`qutip.core.data.create` instead.

You can get individual converters by using the key-lookup syntax.  For example,
the item ::

    to[CSR, Dense]

is a callable which accepts arguments of type :obj:`Dense` and returns the
equivalent item of type :obj:`CSR`.  You can also get a generic converter to a
particular data type if only one type is specified, so ::

    to[Dense]

is a callable which accepts all known (at the time of the lookup) data-layer
types, and converts them to ``Dense``.  See the `Efficiency Notes`_ section
below for more detail.

Internally, the conversion process may go through several steps if new
data-layer types have been defined with few conversions specified between them
and the pre-existing converters.  The first-class QuTiP data types :obj:`Dense`
and :obj:`CSR` will typically have the fastest connectivity.


Adding New Types
================

You can add new data-layer types by calling the
:obj:`~qutip.core.data._to.add_conversions` method of this object, which will
also rebuild all of the mathematical dispatchers.  You must specify one function
which converts a known data type *into* your new type, and one that converts
*from* your new type into a known type.

Because all the dispatchers automatically handle missing specialisations for all
types known by using :obj:`~qutip.core.data.to`, this is completely sufficient
to add an object to QuTiP.


How It Works
============

At its simplest, the problem is how to convert from every type to every other
type without requiring the developer to write a function for every possible
input and output, which is quadratic complexity.  This is a directed-graph
traversal problem; the types are the vertices, and the functions converting from
one type to another are the edges.  In general, a conversion from one type to
another is the function composition of the edges of the shortest path.

We use the Floyd--Warshall algorithm
(:obj:`scipy.sparse.csgraph.floyd_warshall`) evaluate the predecessor matrix.
We build up a :obj:`~qutip.core.convert._converter` object for every pair of
types from this matrix; we do not expect a large number of types, so we are not
concerned with the additional memory usage of this method, but we want to
eliminate as much run-time cost as possible.

The graph view of this problem also allows us to associate a weight with every
specialised conversion function.  This means we can penalise certain edges, such
as making the dense-to-sparse conversion less desirable than one which converts
between different dense representations.

Adding a new type in this model is simple; the graph remains completely
connected when a new vertex is added, provided there is an inbound edge from
inside the current graph and an outbound edge to the same graph.

There is no particular reason to prefer the Floyd--Warshall algorithm over
Dijkstra or Bellman--Ford.  We forbid negative weights and the number of
vertices should be relatively small, so any of these would be suitable.


Implementation
==============

The function :obj:`to` is a singleton instance of the class
:obj:`qutip.core.data.convert._to`.  Its state is effectively global state of
the QuTiP module.  We use a class with attributes instead of module-level
variables for two reasons:

#. it allows us to have both the ``__getitem__`` syntax and the call syntax on
   the same object
#. it's more convenient to have :meth:`~qutip.core.data.to.add_conversions` as a
   method attched directly to the function, rather than it being somewhere
   totally separate

Because of its global-stateful nature, we refer to :obj:`to` as the knowledge
source of data-layer types.  This means that all the dispatchers depend on it,
and all dispatchers store a reference to themselves *in* :obj:`to` so that they
can be updated when new data types are added.


Efficiency Notes
================

We generally prefer to use more memory to make speed gains in the conversion
(and dispatching) operations.  The amount of additional memory used is trivial
for the number of types defined in the data layer, but any speed penalty must be
paid on every single call.

The entire :obj:`~qutip.core.data.to` object and all subsidiary
:obj:`_converter` and :obj:`_partial_converter` objects are pickleable, and so
can be sent across a pipe.

The converters returned by single-key access (e.g. ``data.to[data.Dense]``) are
constructed individually on a call to ``__getitem__``, and are instances of the
private type :obj:`qutip.core.data.convert._partial_converter`, which internally
stores a reference to every "full" converter, and dispatches to the correct one
when called.  There is no efficiency gain from using one of these objects, they
are provided only for convenience.

Internally, ``to(to_type, data)`` effectively calls ``to[to_type, type(data)]``,
so storing the object elides the creation of a single tuple and a dict lookup,
but the cost of this is generally less than 100ns so it is generally not
necessary to do it unless you will be making millions of calls to fast
operations in a tight loop.
