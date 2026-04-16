Motivation
##########

We always want to use a suitable data-storage format when representing operators
and states, even if this is not always the *same* format.  Even if there are
only two different formats, this poses a large problem for code duplication in
the library; functions like :obj:`~qutip.core.data.add` and
:obj:`~qutip.core.data.matmul` take two inputs and return one output, giving
eight different type specialisations that need to be written and maintained for
full coverage, or every mathematical function turns into something nightmarish
like ::

   def add(left, right, out=None):
      if isinstance(left, CSR):
         if isinstance(right, CSR):
            if out is None or isinstance(out, CSR):
               return add_csr(left, right, out)
            elif isinstance(out, Dense):
               return add_csr_csr_dense(left, right, out)
            else:
               raise TypeError
         elif isinstance(right, Dense):
            # ...
         # ...
      # ...
      else:
         raise TypeError

This is obviously completely unsustainable for even two different types, let
alone more than that, and offers very little user customisation.

Instead, we want a unified, centrally controlled system where the developer
simply calls ``add(left, right)``, and the correct specialisation is determined.


Why Not Just Use NumPy?
=======================

NumPy is a fantastic tool for representing numerical data, but it is limited to
dense matrices, while many operators in quantum mechanics are often much more
suited to a sparse representation.

For cases which *are* well-described by dense matrices, the data-layer type
:obj:`~qutip.core.data.Dense` is very similar to a NumPy array underneath (and
in fact can be directly viewed as one using its
:meth:`~qutip.core.data.Dense.as_ndarray` method), but is guaranteed to hold
exactly two dimensions, of which one is stored contiguously.  These additional
internal guarantees help speed in the tightest loops, and the type can be
constructed very quickly from an :obj:`~numpy.ndarray` that is already in the
correct format.

For the large number of cases where the underlying data is much sparser, we use
the :obj:`qutip.core.data.CSR` type, which is a form of compressed sparse row
matrix very similar to SciPy's :obj:`scipy.sparse.csr_matrix`.  There are a few
reasons for not wanting to use SciPy's implementation:

#. Instantiation of :obj:`~scipy.sparse.csr_matrix` is very slow.
#. :obj:`~scipy.sparse.csr_matrix` can use different types as integer indices
   in its index arrays, but this can make it more difficult to interface with C
   code underneath.
#. QuTiP has many parts where very low-level C access is required, and having
   to always deal with Python types means that we must often hold the GIL and
   pay non-trivial overhead penalties when accessing Python attributes.
#. We need to add enough specialised routines that it's not such a big deal if
   we replicate some functionality as well.

Older versions of QuTiP used to reduce these issues by using a
:class:`fast_csr_matrix` type which derived from
:obj:`~scipy.sparse.csr_matrix` and overrode its :meth:`!__init__` method to
remove the slow index-checking code and ensured that only data of the correct
types was stored.  In C-level code, a secondary struct :c:struct:`CSR_Matrix`
was defined, which led to various parts of the code have several entry points,
depending on how many of the arguments had been converted to the structure
representation, and there was still a lot of overhead in converting back to
Python-level code at the end.

The new :obj:`~qutip.core.data.CSR` type stores data in conceptually the same
manner as SciPy, but is defined purely at the Cython level.  This means that it
pays almost no overhead when switching between Python and C access, and code
working with the types need not hold the GIL.  Further, the internal storage
makes similar guarantees to the :obj:`~qutip.core.data.Dense` format about the
data storage, simplifying mathematical code within QuTiP.  It can also be
viewed as a SciPy object when it needs to be used from within Python.

Previous versions of QuTiP also *only* supported the :class:`fast_csr_matrix`
type as the backing data store.  There are many cases where this is a deeply
unsuitable type: in small systems, sparse matrices require large overheads and
stymie data caching, while even in large systems many operations produce
outputs which are nearly 100% dense such as time-evolution operators and matrix
exponentials.  For optimal control applications, the majority of the time spent
was just in dealing with the sparse overheads.  Allowing multiple types to
represent data lets us use the right tool for each job, but it does mean that
further care is taken to ensure that all the mathematical parts of the library
can function without needing to produce an exponential number of new
mathematical functions whenever a type or new operation is added.

Further, even if we were to use NumPy and SciPy objects, we would still be
faced with the problem of handling multiple dispatch.  As soon as QuTiP needed
to add any new functionality that was not already a function in
:obj:`scipy.sparse` or :obj:`scipy.linalg`, particularly one that takes two
matrices as arguments, we would have had to implement the same dispatch system
anyway.
