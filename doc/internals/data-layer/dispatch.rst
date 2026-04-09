Dispatch Operations: :obj:`~qutip.core.data.Dispatcher`
#######################################################

The user most commonly interacts with the data layer by calling mathematical
operations on :obj:`Qobj` instances, which verify that the operation makes
mathematical sense (e.g. the Hilbert space dimensions are equal) and then call
the data-layer dispatchers to handle the particular storage formats in use.

The multiple-dispatch system implemented by :obj:`Dispatcher` lets all
mathematical operations in QuTiP use every possible combination of concrete
input types, even if no developer or user actually wrote a function for that
specific combination.

A basic example of adding two matrices together, and controlling the output type
using the dispatch mechanism: ::

   >>> import scipy.sparse
   >>> import numpy as np
   >>> a = data.CSR(scipy.sparse.csr_matrix(np.random.rand(5, 5)))
   >>> b = data.Dense(np.random.rand(5, 5))
   >>> data.add(a, b)
   Dense(shape=(5, 5), fortran=True)
   >>> data.add(a, b, out=data.CSR)
   CSR(shape=(5, 5), nnz=25)


Using key-lookup syntax to get a callable object which says whether it is a
concrete, "built-in" function (direct) or an ersatz one formed by converting the
inputs and outputs with :obj:`~qutip.core.data.to`: ::

   >>> data.add[data.CSR, data.Dense]
   <indirect specialisation (CSR, Dense, Dense) of add>
   >>> data.add[data.CSR, data.CSR, data.CSR]
   <direct specialisation (CSR, CSR, CSR) of add>


Basic Usage
===========

A :obj:`Dispatcher` provides a single mathematical function for *all*
combinations of types known by :obj:`~qutip.core.data.to`, regardless of whether
the particular specialisation has been defined for the input data types.  In the
first example above, the operator :obj:`~qutip.core.data.add` currently only
knows two specialisations; it knows how to add ``CSR + CSR -> CSR`` and ``Dense
+ Dense -> Dense`` directly, but it is still able to produce the correct result
when asked to do ``CSR + Dense -> CSR`` and similar.  The type of the output can
be, but does not need to be, specified.  The :obj:`~qutip.core.data.Dispatcher`
will choose a suitable output type if one is not given.

For example, the objects :obj:`qutip.data.add`, :obj:`qutip.data.pow` and
:obj:`qutip.data.matmul` are some examples of dispatchers in the data layer.
Respectively, these have the signatures ::

   data.add(left: Data, right: Data, scale: complex = 1) -> Data
   data.pow(matrix: Data, n: integer) -> Data
   data.matmul(left: Data, right: Data) -> Data

These are callable functions, so the base use is to call them.

Just like :obj:`to`, key-lookup syntax can be used to get a single callable
object representing a single specialisation.  The callable object has an
attribute ``direct`` which is ``True`` if no type conversions would need to take
place, and ``False`` is at least one would have to happen.  Just like in the
regular call, you can either specify or not specify the type of the output, but
the types of the inputs must always be given. ::

   >>> data.pow[data.CSR]
   <direct specialisation (CSR, CSR) of pow>
   >>> data.pow[data.CSR].direct
   True
   >>> data.pow[data.CSR, data.Dense].direct
   False

The returned object is callable with the same signature as the dispatcher
(except the ``out`` keyword argument is no longer there), and requires that the
inputs match the types stated.


Adding New Specialisations
==========================

New specialisations can be added to a pre-existing dispatcher with the
:meth:`qutip.data.Dispatcher.add_specialisations` method.  This is very similar
in form to :meth:`qutip.data.to.add_conversions`; it takes lists of tuples,
where the first elements of the tuple define the types in the specialisation,
and the last is the specialised function itself.

For example, a user might need to multiply ``Dense @ CSR`` frequently and get a
``Dense`` output.  Currently, there is no direct specialisation for this: ::

   >>> data.matmul[Dense, CSR, Dense]
   <indirect specialisation (Dense, CSR, Dense) of matmul>

The user may then choose to define their own specialisation to handle this case
efficiently: ::

   >>> def matmul_1(left: Dense, right: CSR) -> Dense:
   ...     # [...]
   ...     return out

They would give this to :obj:`~qutip.data.matmul` by calling ::

   >>> data.matmul.add_specialisations([
   ...     (Dense, CSR, Dense, matmul_1),
   ... ])

Now we find ::

   >>> data.matmul[Dense, CSR, Dense]
   <direct specialisation (Dense, CSR, Dense) of matmul>

Additionally, the whole lookup table will be rebuilt taking this new
specialisation into account, which means the indirect specialisation
``matmul(Dense, CSR) -> CSR`` will now make use of this new method, because it
has a low conversion weight.


Adding New Types
================

Now let's say the user wants to add a new ``NewDataType`` type all across QuTiP.
The only action they *must* take is to tell :obj:`qutip.data.to` about this new
type.  Let's say they define it like this: ::

   >>> class NewDataType:
   ...     # [...]
   >>> def new_from_dense(matrix: data.Dense) -> NewDataType:
   ...     # [...]
   >>> def dense_from_new(matrix: NewDataType) -> data.Dense:
   ...     # [...]
   >>> data.to.add_conversions([
   ...     (NewDataType, data.Dense, new_from_dense),
   ...     (data.Dense, NewDataType, dense_from_new),
   ... ])

As we saw in the previous section, this is enough to define conversions between
all pairs of types.  What's more, this is *also* enough to define all operations
in the data layer as well: ::

   >>> data.matmul[NewDataType, data.CSR]
   <indirect specialisation (NewDataType, CSR, CSR) of matmul>

All of the data layer will now work seamlessly with the new type, even though
this is actually achieved by conversion to and from a known data type.  There
was no need to call anything other than :meth:`qutip.data.to.add_conversions`.
Internally, this is achieved by :meth:`qutip.data.Dispatcher.__init__` storing a
reference to itself in :obj:`~qutip.data.to`, and :obj:`~qutip.data.to` calling
:meth:`~qutip.data.Dispatcher.rebuild_lookup` as part of
:meth:`~qutip.data._to.add_conversions`.

Now the user only needs to add in the specialisations that they actually need
for the bottle-neck parts of their application, and leave the dispatcher to
handle all other minor components by automatic conversion.  As in the previous
subsection, they do this by calling
:meth:`~qutip.data.Dispatcher.add_specialisations` on the relevant operations.


Creating a New Dispatcher
=========================

In most user-defined functions which operate on :attr:`qutip.Qobj.data` it will
be completely sufficient for them to simply call
``data.to(desired_type, input_data)`` on entry to the function, and then they
can guarantee that they are always working with the type of data they support.

However, in some cases they may want to support dispatched operations in the
same way that we do within the library code.  For this reason, the data layer
exports :obj:`~qutip.data.Dispatcher` as a public symbol.  The minimal amount of
work that needs to be done is to call the initialiser, and then call
:meth:`~qutip.data.Dispatcher.add_specialisations`.  For example, let's say the
user has defined two specialisations for their simple new function
``add_square``: ::

   >>> def add_square_csr(left, right):
   ...     return data.add_csr(left, data.matmul_csr(right, right))
   ...
   >>> def add_square_dense(left, right):
   ...     return data.add_dense(left, data.matmul_dense(right, right))
   ...

(Ignore for now that this would be better achieved by just using the dispatchers
:obj:`~qutip.data.add` and :obj:`~qutip.data.matmul` directly.)  Now they create
the dispatcher simply by doing ::

   >>> add_square = data.Dispatcher(add_square_csr, inputs=('left', 'right'), name='add_square', out=True)
   >>> add_square.add_specialisations([
   ...     (data.CSR, data.CSR, data.CSR, add_square_csr),
   ...     (data.Dense, data.Dense, data.Dense, add_square_dense),
   ... ])

This is enough for :obj:`~qutip.data.Dispatcher` to have extracted the signature
and satisfied all of the specialisations.  Note that the ``inputs`` argument
does not provide the signature, it tells the dispatcher which arguments are
data-layer types it should dispatch on, e.g. for :obj:`~qutip.data.pow` as
defined above ``inputs = ('matrix',)``, but the signature is
``(matrix, n) -> out``.  See that the specialisations are now complete: ::

   >>> add_square
   <dispatcher: add_square(left, right)>
   >>> add_square[data.Dense, data.CSR, data.CSR]
   <indirect specialisation (Dense, CSR, CSR) of add_square>

In the initialisation, the function ``add_square_csr`` is passed as an example
from which :obj:`~qutip.data.Dispatcher` extracts the call signature, the module
name and the docstring (if it exists).  It is not actually added as a
specialisation until ``add_square.add_specialisations`` is called afterwards.

If desired, the user can set or override the docstring for the resulting
dispatcher by directly writing to the ``__doc__`` attribute of the object.  We
*always* do this within the library.

.. note::
   Within the Cython components of the library, we manually construct the
   signature and pass it into the :obj:`qutip.data.Dispatcher` constructor
   because Cython-compiled functions do not embed their signature in a manner in
   which :obj:`inspect.signature` can extract it (even with the
   :obj:`cython.embedsignature` directive).  We also use this to cut out some
   arguments in the call signatures which would not work with the dispatch
   mechanism (like ``out`` parameters).


Other Features
==============

The :obj:`~qutip.data.Dispatcher` can operate on a function with any call
signature (except ones which use ``*args`` or ``**kwargs``), even if not all of
the arguments are data-layer types.  At definition, the creator of the
:obj:`~qutip.data.Dispatcher` says which input arguments are meant to be
dispatched on, and whether the output should be dispatched on, and all other
arguments are passed through like normal.


How It Works
============

Calling a :obj:`~qutip.data.Dispatcher` happens in five stages:

#. bind parameters to the call signature
#. resolve all dispatch rules into the best known specialisation
#. convert dispatch parameters to match the chosen specialisation
#. call the specialisation
#. if necessary, convert the output to the correct type

First, the arguments have to be bound from the generic signature ``(*args,
**kwargs)`` to match the actual call signature of the underlying functions so
that the arguments which need dispatching are identified.  This is done by the
internal class :obj:`~qutip.data.dispatch._bind`, which is constructed from the
signature of the specialisations when the :obj:`~qutip.data.Dispatcher` is
instantiated.

Once we have the arguments, we have to resolve all the dispatch rules to know
which specialisation we should use.  In the most common scenario, the only
active dispatch rules will be either *global* rules defined in the QuTiP
options, or *instance* rules defined on this particular
:obj:`~qutip.data.Dispatcher`---there will not be any *local* rules passed in
the function call.  If this is the case, the :obj:`~qutip.data.Dispatcher` will
already have a pre-built lookup table, and choosing the correct specialisation
is simply a hash lookup on the types.  This is linear complexity in the number
of types dispatched on, whereas the naïve ``if/elif/else`` structure would go as
:math:`\mathcal O(n^m)`, where :math:`n` is the number of known
data-layer types, and :math:`m` is the number of dispatch parameters.

The next most common case would be that the only local rule is to fix the output
type.  In this case, the secondary lookup table is used to avoid having to
compute a proper set of weights.

The final case for resolving rules is the most general one: we have been given
some proper dispatch rules that we have to handle at call time to resolve which
specialisation we must choose.  Here, we have no choice but to recalculate the
correct specialisation, which comes with some (hopefully small) runtime cost.

The last three steps of the dispatch operation are simple.  Now that we have the
specialisation we are going to use, we simply use :obj:`~qutip.data.to` to
convert the given inputs to the correct types, we call the function, and we
return the answer (possible converting it too).

Building the lookup table is a relatively simple process, but asymptotically
it has very poor complexity.  This is why as much as possible is done ahead of
time, at library initialisation or when a new type is added.  To build the
tables, we have to compare every possible set of inputs to every known
specialisation, and then choose the specialisation which has the least "weight"
given the defined rules.


Implementation Details
======================

The backing specialisations can be found in
:attr:`qutip.Dispatcher._specialisations`, and the complete lookup table is in
:attr:`qutip.Dispatcher._lookup`.  These are marked as private, because messing
around with them will almost certainly cause the dispatcher to stop working.

Only one specialisation needs to be defined for a dispatcher to work with *all*
data types known by :obj:`~qutip.data.to`.  We achieve this because
:obj:`~qutip.data.to` guarantees that all possible conversions between data
types will exist, so :obj:`~qutip.data.Dispatcher` can always convert its inputs
into those which will match one of its known specialisations.

Within the initialisation of the data layer, we use a "magic" ``_defer`` keyword
argument to :meth:`~qutip.data.Dispatcher.add_specialisations` to break a
circular dependency.  This is because the "type" modules :obj:`qutip.data.csr`
and :obj:`qutip.data.dense` depend on some mathematical modules (e.g.
:obj:`~qutip.data.add` and :obj:`~qutip.data.matmul`) to provide the ``__add__``
and similar methods on the types.  For ease of development we want the
dispatchers to be defined in the same modules that all the specialisations are
(though this is not at all necessary), but the dispatchers require
:obj:`~qutip.data.to` to be populated with the types before specialisations can
be added.  The ``_defer`` keyword here just defers the building of the lookup
table until an explicit call to :meth:`~qutip.data.Dispatcher.rebuild_lookup`,
breaking the cycle.  The user will never need to do this, because by the time
they receive the :obj:`~qutip.Dispatcher` object, :obj:`~qutip.data.to` is
already initialised to a minimum degree.


Efficiency Notes
================

The specialisations returned by the ``__getitem__`` lookups are not
significantly faster than just calling the dispatcher directly, because the bulk
of the heavy lifting is done when
:meth:`~qutip.data.Dispatcher.add_specialisations` or
:meth:`~qutip.data.Dispatcher.rebuild_lookup` is called.  On call, the generic
signature ``(*args, **kwargs)`` has to be bound to the actual signature of the
underlying operation, regardless of whether the specialisation has already been
found.  At the Cython level there is short-circuit access to the call machinery
in the specialisations themselves, but this cannot be safely exposed outside of
the :obj:`qutip.data.Dispatcher` class.
