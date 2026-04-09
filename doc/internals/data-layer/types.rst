Type Descriptions
#################

There are currently two first-class data types defined in QuTiP, but the
generic nature of the dispatch operations means that it is relatively
straightforward to add new types for specific use-cases.

Abstract Base: :obj:`~qutip.core.data.Data`
===========================================

The base :obj:`~qutip.core.data.Data` requires very little information to be
stored---only the two-dimensional shape of the matrix.  This is common to all
data types, and readable (but not writeable) from Python.


Compressed Sparse Row: :obj:`~qutip.core.data.CSR`
==================================================

The `compressed sparse row format`_ has historically always been QuTiP's format
of choice.  Only non-zero data entries are stored, and information is kept
detailing how many stored entries are in each row, and which columns they
appear in.  This is one of the most common sparse matrix formats, having
minimal storage requirements for arbitrary sparse matrices, and perhaps most
importantly for linear algebra, it is especially suited for taking
matrix--vector products.

QuTiP's implementation stores all indexing types as the centrally defined
:c:type:`~qutip.core.data.idxint` type, which is fixed at compile time.
Typically this will be a 32- or 64-bit integer, and we generally use signed
arithmetic to be consistent with Python indexing (although we do actually allow
negative indexing into C arrays).  All variables which are used to index into
an array should follow this type within C or Cython code.

:obj:`~qutip.core.data.CSR` can be instantiated from Python in similar ways to
SciPy's :obj:`~scipy.sparse.csr_matrix`, but it also provides fast-path
initialisation from Python or C using the type's
:meth:`~qutip.core.data.CSR.copy` method, or the low-level constructors
:obj:`~qutip.core.data.csr.empty`, :obj:`~qutip.core.data.csr.zeroes`,
:obj:`~qutip.core.data.csr.identity`, and 
:obj:`~qutip.core.data.csr.copy_structure`.

.. _compressed sparse row format: https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)


Access From Python
------------------

We do not expose the underlying memory buffers to the user in Python space by
default.  This is to avoid needing to acquire the GIL every time one of our
objects is created, especially when C code creates several of them in a
function which otherwise would not need to speak to the Python interpreter at
all.

Instead, we expose a method :meth:`~qutip.core.data.CSR.as_scipy`, which
returns a :obj:`~scipy.sparse.csr_matrix`.  So that the Python-space user can
work with the data if they desire, this output is simply a "view" onto the same
underlying data buffers.  This has some memory management implications that
will be discussed in the next section.

The problem of :obj:`~scipy.sparse.csr_matrix` having a slow constructor still
persists, however.  We do not want to have to define a whole new derived class
(like the old :class:`fast_csr_matrix`) just to override :meth:`!__init__`,
mostly because it's unnecessary and bloats our own code, but it also may have
annoying knock-on effects for users with imperfect polymorphic code and it adds
overhead to method resolution.  Instead, we simply allocate space for a
:obj:`~scipy.sparse.csr_matrix` with its
:meth:`~scipy.sparse.csr_matrix.__new__` method, call the first reasonable
method in the initialisation chain, and fill in the rest in Cython code.
Because of the guarantees about the :obj:`~qutip.core.data.CSR` type, we know
that our data will already be in the correct format.

We then store a reference to this object within :obj:`~qutip.core.data.CSR` so
that subsequent calls do not need to pay the initialisation penalty.  This also
helps with memory management.


Memory Management
-----------------

When constructed from Python, :obj:`~qutip.core.data.CSR` does not take
ownership of its memory since we know we already have to be dealing with
refcounting and the GIL.  We use NumPy's access methods to construct new
arrays, and let NumPy handle management of the data.

However, when constructed from Cython code, including Cython functions called
by Python, there is no need to interface with NumPy or create Python objects
other the very last instance when we have to return it to the user in Python
space.  Here we use low-level C memory management, and rely on the general
principle of low-level QuTiP development that *you must not store references to
other objects' data*.  Other libraries allow this, but instead require that you
suitably increment the relevant refcounts.  We do not keep track of anything
like this, and simply do not permit references in this manner within our code.
Cython operations like :meth:`~qutip.core.data.CSR.copy` will always copy the
data, *never* return a view.

Sometimes, however, the user will need to access the data directly from Python
space.  In these cases, we must ensure that the data buffer cannot be freed
while the user holds a reference to it.  We allow the user to use the
:meth:`~qutip.core.data.CSR.as_scipy` method to view the data, and as part of
this process, we create new a :obj:`~numpy.ndarray` for each buffer, and set
the :c:data:`NPY_ARRAY_OWNDATA` flag to force NumPy to manage reference
counting for us.

Since we have just passed on ownership of our data to another entity, we always
keep a (strong) reference to the created object within our own type.  This was
we can guarantee that NumPy will not deallocate our storage before we are done
with it, and NumPy's memory management will also ensure that the memory *is*
deallocated safely once all Python views onto it are gone.

It is important when allocating buffers which may become the backing of a
:obj:`~qutip.core.data.CSR` type that you *always* use
:c:func:`PyDataMem_NEW` (or others in the ``PyDataMem`` family) and
:c:func:`PyDataMem_FREE` to allocate and free memory.  Doing
otherwise may cause segfaults or other complete interpreter crashes, as it may
not use the same allocator that NumPy does.  In particular, the Windows runtime
can easily result in this happening if raw ``malloc`` or ``calloc`` are used,
and the CPython allocator :c:func:`!cpython.mem.PyMem_Malloc` will tend to
allocate small requests into an internal reserved buffer on its stack, which
cannot be freed from NumPy.


Dense: :obj:`~qutip.core.data.Dense`
====================================

The :obj:`~qutip.core.data.Dense` format is the most "traditional" storage
format for linear algebra operators and vectors.  This simply explicitly stores
a value for every single element in the matrix, even if it is zero.  There is no
need to store any other information, other than the matrix's shape, so for small
dense matrices, this can actually result in less memory usage than sparse
formats.

We guarantee that the data is contiguous in either row-major (C) or column-major
(Fortran) format.  This is useful in several places when interfacing with LAPACK
and BLAS functions, and generally keeps memory access fast and cache-friendly.
This is in contrast to NumPy, where taking strided views onto data can return a
new array whose memory is *not* contiguous.  NumPy makes this decision so that
strided views and slices can return faster as they do not need to copy memory,
but these operations are exceptionally rare within the QuTiP core, so we do not
optimise our data structures to support them.


Python Access and Memory Management
-----------------------------------

Similarly to :obj:`~qutip.core.data.CSR`, :obj:`~qutip.core.data.Dense` provides
the :meth:`~qutip.core.data.Dense.as_ndarray` method to view its data as a NumPy
array.  This view can be used to modify the data of the object in-place.

As with :obj:`~qutip.core.data.CSR`, no Python reference is created until the
Python user specifically requests the NumPy array view.  The memory management
is handled in the same way; when instantiated from Cython code, the backing will
be a raw C pointer which is deallocated when the Python instance of
:obj:`~qutip.core.data.Dense` goes out-of-scope and is garbage collected.  If
the NumPy view is used, then ownership of the pointer is transferred to NumPy,
and a reference to the :obj:`~numpy.ndarray` is stored within the instance to
ensure it always outlives us.
