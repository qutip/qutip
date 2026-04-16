Data Layer
##########

The bulk of mathematical heavy lifting in QuTiP is handled by functions on the
"data layer".  The term "data layer" is used to refer to all linear algebra
types which QuTiP uses to represent low-level data, operations which take place
on these types, and the dispatch logic necessary to ensure that the correct
operations are called when given two arbitrary, known types.  Crucially, this
will work even if we have not defined a function which handles those particular
two types, so we do not need to rewrite the whole library eight times over; we
only write multiple specialisations for components that are speed bottlenecks.

All data types on the data layer inherit from :obj:`qutip.core.data.Data`,
although this is itself an abstract type which cannot be instantiated.
Dispatch functions are instances of the type :obj:`qutip.core.data.Dispatch`,
which provide a Python-callable interface.

The data layer is primarily written in Cython, and compiled to C++ before being
compiled fully into CPython extension types.

There are three main operations of the data layer:

#. conversion between data-layer types;
#. implementations of mathematical operations particular to a certain set of
   inputs and an output;
#. a multiple-dispatch system which chooses the "best" specialisation for
   a given mathematical operation, based on the input parameters it
   receives, when it might not have a complete set of specialisations.

Typically the user interacts indirectly with the data layer; various
:obj:`~qutip.Qobj` operations will invoke it, but the user will not need to
worry about what it is doing.  All they will see is that QuTiP can use the best
data structures to represent their data, and everything will *just work*.  From
a developer's perspective, it means that high-level QuTiP functions do not need
to concern themselves with what representation the data is stored in; they can
use low-level mathematical operations, and everything will work out.


.. toctree::
   :maxdepth: 2
   :caption: Sections

   terminology
   motivation
   type-conversion
   dispatch
   types
