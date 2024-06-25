#cython: language_level=3

"""
The conversion machinery between different data-layer types, and creation
routines from arbitrary data.  The classes `_to` and `_create` are not intended
to be exported names, but are the backing machinery of the functions `data.to`
and `data.create`, which are built up as the last objects in the `__init__.py`
initialisation of the `data` module.
"""

# This module is compiled by Cython because it's the core of the entire
# dispatch table, and having it compiled to a C extension saves about 1µs per
# call.  This is not much at all, and there's very little which benefits from
# Cython compiliation, but such core functionality is called millions of times
# even in a simple interactive QuTiP session, and it all adds up.

import numbers

import numpy as np
from scipy.sparse import dok_matrix, csgraph
cimport cython
from qutip.core.data.base cimport Data

__all__ = ['to', 'create']


class _Epsilon:
    """
    Constant for an small weight non-null weight.
    Use to set `Data` specialisation just over direct specialisation.
    """
    def __repr__(self):
        return "EPSILON"

    def __eq__(self, other):
        if isinstance(other, _Epsilon):
            return True
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, _Epsilon):
            return self
        return other

    def __radd__(self, other):
        if isinstance(other, _Epsilon):
            return self
        return other

    def __lt__(self, other):
        """ positive number > _Epsilon > 0 """
        if isinstance(other, _Epsilon):
            return False
        return other > 0.

    def __gt__(self, other):
        if isinstance(other, _Epsilon):
            return False
        return other <= 0.


EPSILON = _Epsilon()


def _raise_if_unconnected(dtype_list, weights):
    unconnected = {}
    for i, type_ in enumerate(dtype_list):
        missing = [dtype_list[j].__name__
                   for j, weight in enumerate(weights[:, i])
                   if weight == np.inf]
        if missing:
            unconnected[type_.__name__] = missing
    if unconnected:
        message = "Conversion graph not connected.  Cannot reach:\n * "
        message += "\n * ".join(to + " from (" + ", ".join(froms) + ")"
                                for to, froms in unconnected.items())
        raise NotImplementedError(message)


cdef class _converter:
    """Callable which converts objects of type `x.from_` to type `x.to`."""

    cdef list functions
    cdef Py_ssize_t n_functions
    cdef readonly type to
    cdef readonly type from_

    def __init__(self, functions, to_type, from_type):
        self.functions = list(functions)
        self.n_functions = len(self.functions)
        self.to = to_type
        self.from_ = from_type

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __call__(self, arg):
        if not isinstance(arg, self.from_):
            raise TypeError(str(arg) + " is not of type " + str(self.from_))
        cdef Py_ssize_t i
        for i in range(self.n_functions):
            arg = self.functions[i](arg)
        return arg

    def __repr__(self):
        return ("<converter to "
                + self.to.__name__
                + " from " + self.from_.__name__
                + ">")


def identity_converter(arg):
    return arg


cdef class _partial_converter:
    """Convert from any known data-layer type into the type `x.to`."""

    cdef object converter
    cdef readonly type to

    def __init__(self, converter, to_type):
        self.converter = converter
        self.to = to_type

    def __call__(self, arg):
        try:
            return self.converter[self.to, type(arg)](arg)
        except KeyError:
            raise TypeError("unknown type of input: " + str(arg)) from None

    def __repr__(self):
        return "<converter to " + self.to.__name__ + ">"


# While `_to` and `_create` are defined as objects here, they are actually
# exported by `data.__init__.py` as singleton function objects of their
# respective types (without the leading underscore).

cdef class _to:
    """
    Convert data into a different type.  This object is the knowledge source
    for every allowable data-layer type in QuTiP, and provides the conversions
    between all of them.

    The base use is to call this object as a function with signature
        (type, data) -> converted_data
    where `type` is a type object (such as `data.CSR`, or that obtained by
    calling `type(matrix)`) and `data` is data in a data-layer type.  If you
    want to create a data-layer type from non-data-layer data, use `create`
    instead.

    You can get individual converters by using the key-lookup syntax.  For
    example, the item
        to[CSR, Dense]
    is a callable which accepts arguments of type `Dense` and returns the
    equivalent item of type `CSR`.  You can also get a generic converter to a
    particular data type if only one type is specified, so
        to[Dense]
    is a callable which accepts all known (at the time of the lookup)
    data-layer types, and converts them to `Dense`.  See the `Efficiency notes`
    section below for more detail.

    Internally, the conversion process may go through several steps if new
    data-layer types have been defined with few conversions specified between
    them and the pre-existing converters.  The first-class QuTiP data types
    `Dense` and `CSR` will typically have the best connectivity.


    Adding new types
    ----------------
    You can add new data-layer types by calling the `add_conversions` method of
    this object, and then rebuilding all of the mathematical dispatchers.  See
    the docstring of that method for more information.


    Efficiency notes
    ----------------
    From an efficiency perspective, there is very little benefit to using the
    key-lookup syntax.  Internally, `to(to_type, data)` effectively calls
    `to[to_type, type(data)]`, so storing the object elides the creation of a
    single tuple and a dict lookup, but the cost of this is generally less than
    500ns.  Using the one-argument lookup (e.g. `to[Dense]`) is no more
    efficient than the general call at all, but can be used in cases where a
    single callable is required and is more efficient than `functools.partial`.
    """

    cdef readonly set dtypes
    cdef readonly list dispatchers
    cdef dict _direct_convert
    cdef dict _convert
    cdef readonly dict weight
    cdef readonly dict _str2type

    def __init__(self):
        self._direct_convert = {}
        self._convert = {}
        self.dtypes = set()
        self.weight = {}
        self.dispatchers = []
        self._str2type = {}

    def add_conversions(self, converters):
        """
        Add conversion functions between different data types.  This is an
        advanced function, and is only intended for the QuTiP user who wants to
        add a new underlying data type to QuTiP.

        Any new data type must have at least one converter function given to
        produce the new data type from an existing data type, and at least one
        which produces an existing data type from the new one.  You need not
        specify any more than this, although for efficiency reasons, you may
        want to specify direct conversions for all types you expect the new
        type to interact with frequently.

        Parameters
        ----------
        converters : iterable of (to_type, from_type, converter, [weight])
            An iterable of 3- or 4-tuples describing all the new conversions.
            Each element can individually be a 3- or 4-tuple; they do not need
            to be all one or the other.

            Elements
            ........
            to_type : type
                The data-layer type that is output by the converter.

            from_type : type
                The data-layer type to be input to the converter.

            converter : callable (Data -> Data)
                The converter function.  This should take a single argument
                (the input data-layer function) and output a data-layer object
                of `to_type`.  The converter may assume without error checking
                that its input will always be of `to_type`.  It is safe to
                specify the same conversion function for multiple inputs so
                long as the function handles them all safely, but it must
                always return a single output type.

            weight : positive real, optional (1)
                The weight associated with this conversion.  This must be > 0,
                and defaults to `1` if not supplied (which is fixed to be the
                cost of conversion to `Dense` from `CSR`).  It is generally
                safe just to leave this blank; it is always at best an
                approximation.  The currently defined weights are accessible in
                the `weights` attribute of this object.
                Weight of ~0.001 are should be used in case when no conversion
                is needed or ``converter = lambda mat : mat``.
        """
        for arg in converters:
            if len(arg) == 3:
                to_type, from_type, converter = arg
                weight = 1
            elif len(arg) == 4:
                to_type, from_type, converter, weight = arg
            else:
                raise TypeError("unknown converter specification: " + str(arg))
            if not isinstance(to_type, type):
                raise TypeError(repr(to_type) + " is not a type object")
            if not isinstance(from_type, type):
                raise TypeError(repr(from_type) + " is not a type object")
            if not isinstance(weight, numbers.Real) or weight <= 0:
                raise TypeError("weight " + repr(weight) + " is not valid")
            self.dtypes.add(from_type)
            self.dtypes.add(to_type)
            self._direct_convert[(to_type, from_type)] = (converter, weight)
        # Two-way mapping to convert between the type of a dtype and an integer
        # enumeration value for it.
        order, index = [], {}
        for i, dtype in enumerate(self.dtypes):
            order.append(dtype)
            index[dtype] = i
        # Treat the conversion problem as a shortest-path graph problem.  We
        # build up the graph description as a matrix, then solve the
        # all-pairs-shortest-path problem.  We forbid negative weights and
        # there are unlikely to be many data types, so the choice of algorithm
        # is unimportant (Dijkstra's, Floyd--Warshall, Bellman--Ford, etc).
        graph = dok_matrix((len(order), len(order)))
        for (to_type, from_type), (_, weight) in self._direct_convert.items():
            graph[index[from_type], index[to_type]] = weight
        weights, predecessors =\
            csgraph.floyd_warshall(graph.tocsr(), return_predecessors=True)
        _raise_if_unconnected(order, weights)
        # Build the whole shortest path conversion lookup.  We directly store
        # all complete shortest paths, even though this is not the most memory
        # efficient, because we expect that there will generally be a small
        # number of available data types, and we care more about minimising the
        # number of lookups required.
        self.weight = {}
        self._convert = {}
        for i, from_t in enumerate(order):
            for j, to_t in enumerate(order):
                convert = []
                weight = 0
                cur_t = to_t
                pred_i = predecessors[i, j]
                while pred_i >= 0:
                    pred_t = order[pred_i]
                    _convert, _weight = self._direct_convert[(cur_t, pred_t)]
                    convert.append(_convert)
                    weight += _weight
                    cur_t = pred_t
                    pred_i = predecessors[i, pred_i]
                self.weight[(to_t, from_t)] = weight
                self._convert[(to_t, from_t)] =\
                    _converter(convert[::-1], to_t, from_t)
        for dtype in self.dtypes:
            self.weight[(dtype, Data)] = 1.
            self.weight[(Data, dtype)] = EPSILON
            self._convert[(dtype, Data)] = _partial_converter(self, dtype)
            self._convert[(Data, dtype)] = identity_converter
        for dispatcher in self.dispatchers:
            dispatcher.rebuild_lookup()

    def register_aliases(self, aliases, layer_type):
        """
        Register a user frendly name for a data-layer type to be recognized by
        the :meth:`parse` method.

        Parameters
        ----------
        aliases : str or list of str
            Name of list of names to be understood to represent the layer_type.

        layer_type : type
            Data-layer type, must have been registered with
            :meth:`add_conversions` first.
        """
        if layer_type not in self.dtypes:
            raise ValueError(
                "Type is not a data-layer type: " + repr(layer_type))
        if isinstance(aliases, str):
            aliases = [aliases]
        for alias in aliases:
            if type(alias) is not str:
                raise TypeError("The alias must be a str : " + repr(alias))
            self._str2type[alias] = layer_type

    def parse(self, dtype):
        """
        Return a data-layer type object given its name or the type itself.

        Parameters
        ----------
        dtype : type, str
            Either the name of a data-layer type or a type itself.

        Returns
        -------
        type
            A data-layer type.

        Raises
        ------
        TypeError
            If ``dtype`` is neither a string nor a type.
        ValueError
            If ``dtype`` is a name, but no data-layer type of that name is
            registered, or if ``dtype`` is a type, but not a known data-layer
            type.
        """
        if type(dtype) is type:
            if dtype not in self.dtypes and dtype is not Data:
                raise ValueError(
                    "Type is not a data-layer type: " + repr(dtype))
            return dtype
        elif type(dtype) is str:
            try:
                return self._str2type[dtype]
            except KeyError:
                raise ValueError(
                    "Type name is not known to the data-layer: " + repr(dtype)
                    ) from None

        raise TypeError(
            "Invalid dtype is neither a type nor a type name: " + repr(dtype))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __getitem__(self, arg):
        if type(arg) is not tuple:
            arg = (arg,)
        if not arg or len(arg) > 2:
            raise KeyError(arg)
        to_t = self.parse(arg[0])
        if len(arg) == 1:
            return _partial_converter(self, to_t)
        from_t = self.parse(arg[1])
        return self._convert[to_t, from_t]

    def __call__(self, to_type, data):
        to_type = self.parse(to_type)
        from_type = self.parse(type(data))
        if to_type == from_type:
            return data
        return self._convert[to_type, from_type](data)


cdef class _create:
    cdef readonly list _creators

    def __init__(self):
        self._creators = []

    def add_creators(self, creators):
        """
        Add creation functions to make a data-layer object from an arbitrary
        Python object.

        Parameters
        ----------
        creators : iterable of (condition, creator, [priority])
            An iterable of 2- or 3-tuples describing the new data layer
            creation functions.
            Each element can individually be a 2- or 3-tuple; they do not need
            to be all one or the other.

            Elements
            ........
            condition : callable (object) -> bool
                Function determining if the given object can be converted to a
                data-layer type using this creator.

            creator function: callable (object, shape, copy=True) -> Data
                The creator function. It should take an object and a shape and
                return a data-layer type instance. The object may be any object
                for which the condition function returned ``True`` when tested.
                The ``object`` and ``shape`` parameters are passed positionally,
                and the ``copy`` parameter is passed by keyword.

            priority : real, optional (0)
                The priority associated with this creator. Higher priority
                conditions will be tested first and the first valid creator
                (i.e. for which ``condition(object) == True``) will handle
                the creation.

        Notes
        -----
        Default creators are added with the following priorities:

        * Objects that are instances of data-layer types are
          converted using ``.copy`` with priority 100.
        * Objects that have a direct equivalent such as ``numpy.ndarray``
          or ``scipy.sparse.csr_matrix`` are converted with priority 80.
        * Objects for which ``scipy.sparse.issparse`` is ``True``
          are converted using an internal CSR converter with
          priority 20.
        * If no condition are meet, ``numpy.array`` is used to try convert
          the input to an array (priority -inf).
        """
        for condition, creator, *priority in creators:
            if not callable(condition):
                raise TypeError(repr(condition) + " is not a callable")
            if not callable(create):
                raise TypeError(repr(create) + " is not a callable")
            if len(priority) >= 2:
                raise ValueError("Too many values to unpack for a creator, " +
                                 "expected 2 or 3, got "+ str(2+len(priority)))
            priority = float(priority[0] if priority else 0)
            self._creators.append((condition, creator, priority))
        self._creators.sort(key=lambda creator: creator[2], reverse=True)

    def __call__(self, arg, shape=None, copy=True):
        """
        Build a :class:`.Data` object from arg.

        Parameters
        ----------
        arg : object
            Object to be converted to `qutip.data.Data`. Anything that can be
            converted to a numpy array are valid input.

        shape : tuple, optional
            The shape of the output as (``rows``, ``columns``).
        """
        for condition, create, _ in self._creators:
            if condition(arg):
                return create(arg, shape, copy=copy)
        raise TypeError(f"arg `{repr(arg)}` cannot be converted to "
                        "qutip data format")


to = _to()
create = _create()
