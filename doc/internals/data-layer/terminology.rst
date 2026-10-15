Terminology
===========

This is a short terminology glossary for the data layer and operations on it.
See the rest of the guide for more in depth information.

abstract type
   An instance of a data-layer type, but we have no more information than that.
   If we knew exactly which type it was, we would have a *concrete type*.

concrete type
   One particular type, for example :obj:`CSR`.  The "opposite" of this is an
   *abstract type*.

conversion
   A function which takes in one type and outputs another.  For example, a
   function ``csr_from_dense(dense_matrix)`` would be a *specialised*
   conversion, and a function ``csr_from_anything(matrix)`` would be a general
   or *abstract* conversion.

dispatcher
   A instance of :obj:`~qutip.core.data.Dispatcher`.  This is a callable
   function that takes in *abstract* data-layer types and calls the closest
   *concrete specialisation* that it knows about after converting its inputs
   into the correct types.

specialisation
   A defined function which takes in a particular, concrete set of data-layer
   types and returns another concrete data-layer type.  The user will typically
   never call these directly, they are only used directly within Cython code or
   indirectly from a dispatcher.
