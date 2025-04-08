#cython: language_level=3
#cython: boundscheck=False

import functools
import inspect
import itertools
import warnings

from .convert import to as _to
from .convert import EPSILON

cimport cython
from libc cimport math
from libcpp cimport bool
from qutip.core.data.base cimport Data
from qutip import settings

__all__ = ['Dispatcher']


cdef object _conversion_weight(tuple froms, tuple tos, dict weight_map, bint out):
    """
    Find the total weight of conversion if the types in `froms` are converted
    element-wise to the types in `tos`.  `weight_map` is a mapping of
    `(to_type, from_type): real`; it should almost certainly be
    `data.to.weight`.

    Specialisations that support any types input should use ``Data``.
    """
    cdef object weight = 0.0
    cdef Py_ssize_t i, n=len(froms)
    if len(tos) != n:
        raise ValueError(
            "number of arguments not equal: " + str(n) + " and " + str(len(tos))
        )
    if out:
        n = n - 1
        weight = weight + weight_map[froms[n], tos[n]]
    for i in range(n):
        weight = weight + weight_map[tos[i], froms[i]]
    return weight


cdef class _constructed_specialisation:
    """
    Callable object providing the specialisation of a data-layer operation for
    a particular set of types (`self.types`).  This may or may not involve
    conversion of the input types and the output to match a known
    specialisation; if it has no conversions, `self.direct` will be `True`,
    otherwise it will be `False`.

    See `self.__signature__` or `self.__text_signature__` for the call
    signature of this object.
    """
    cdef readonly bint _output
    cdef object _call
    cdef readonly Py_ssize_t _n_inputs, _n_dispatch
    cdef readonly tuple types
    cdef readonly tuple _converters
    cdef readonly str _short_name
    cdef public str __doc__
    cdef public str __name__
    # cdef public str __module__
    cdef public object __signature__
    cdef readonly str __text_signature__

    def __init__(self, base, Dispatcher dispatcher, types, converters, out):
        self.__doc__ = inspect.getdoc(dispatcher)
        self._short_name = dispatcher.__name__
        self.__name__ = (
            self._short_name
            + "_"
            + "_".join([x.__name__ for x in types])
        )
        # self.__module__ = dispatcher.__module__
        self.__signature__ = dispatcher.__signature__
        self.__text_signature__ = dispatcher.__text_signature__
        self._output = out
        self._call = base
        self.types = types
        self._converters = converters
        self._n_dispatch = len(converters)
        self._n_inputs = len(converters) - out

    @cython.wraparound(False)
    def __call__(self, *args, **kwargs):
        cdef int i
        cdef list _args = list(args)
        for i in range(self._n_inputs):
            _args[i] = self._converters[i](args[i])
        out = self._call(*_args, **kwargs)
        if self._output:
            out = self._converters[self._n_dispatch - 1](out)
        return out

    def __repr__(self):
        if len(self.types) == 1:
            spec = self.types[0].__name__
        else:
            spec = "(" + ", ".join(x.__name__ for x in self.types) + ")"
        return "".join([
            "<indirect specialisation ", spec, " of ", self._short_name, ">"
        ])


cdef class _group_specialisation:
    """
    Callable object providing the specialisation of a data-layer operation for
    a particular set of types (`self.types`) and a group output type.

    See `self.__signature__` or `self.__text_signature__` for the call
    signature of this object.
    """
    cdef object _call
    cdef readonly Py_ssize_t _n_inputs, _n_dispatch
    cdef readonly tuple types
    cdef readonly object group
    cdef readonly str _short_name
    cdef public str __doc__
    cdef public str __name__
    cdef public object __signature__
    cdef readonly str __text_signature__

    def __init__(self, base, Dispatcher dispatcher, types, group):
        self.__doc__ = inspect.getdoc(dispatcher)
        self._short_name = dispatcher.__name__
        self.__name__ = (
            self._short_name
            + "_"
            + "_".join([x.__name__ for x in types])
        )
        self.__signature__ = dispatcher.__signature__
        self.__text_signature__ = dispatcher.__text_signature__
        self._call = base
        self.types = types
        self.group = group

    @cython.wraparound(False)
    def __call__(self, *args, **kwargs):
        return _to(self.group, self._call(*args, **kwargs))

    def __repr__(self):
        if len(self.types) == 0:
            spec = self.group.name
        else:
            spec = "(" + ", ".join(x.__name__ for x in self.types) + f" {self.group.name})"
        return "".join([
            f"<indirect specialisation {spec} of ", self._short_name, ">"
        ])


cdef class Dispatcher:
    """
    Dispatcher for a data-layer operation.  This object can be called with the
    signature shown in `self.__signature__` or `self.__text_signature__`, where
    the arguments listed in `self.inputs` can be any data-layer types (i.e.
    ones that have valid conversions in `data.to`).

    You can define additional specialisations for this dispatcher by calling
    its `add_specialisations` method.  New data types must be added to
    `data.to` before they can be added as specialisations to a dispatcher.

    You can get a callable object representing a single set of dispatcher types
    by using the key-lookup syntax
        Dispatcher[type1, type2, ...]
    where `type1`, `type2`, etc are the dispatched arguments (with the output
    type on the end, if this is a dispatcher over the output type.
    """
    cdef readonly dict _specialisations
    cdef readonly Py_ssize_t _n_dispatch, _n_inputs
    cdef readonly dict _lookup
    cdef readonly set _dtypes
    cdef readonly bint _pass_on_dtype
    cdef readonly tuple inputs
    cdef readonly bint output
    cdef public str __doc__
    cdef public str __name__
    # cdef public str __module__
    cdef public object __signature__
    cdef readonly str __text_signature__

    def __init__(self, signature_source, inputs, bint out=False,
                 str name=None, str module=None):
        """
        Create a new data layer dispatching operator.

        Parameters
        ----------
        signature_source : callable or inspect.Signature
            An object from which the call signature of operation can be
            determined.  You can pass any callable defined in Python space, and
            the signature will be extracted.  Note that the callable will not
            be added as a specialisation by this; you will still have to call
            `add_specialisations`.

            If you cannot provide a callable with an extractable signature
            (e.g. Cython extension methods), you can instead directly provide
            an instance of `inspect.Signature`, which will be used instead.

        inputs : iterable of str
            The parameters which should be dispatched over.  These must be
            positional arguments, and must feature in the signature provided.

        out : bool, optional (False)
            Whether to dispatch on the output of the function.  Defaults to
            `False`.

        name : str, optional
            If given, the `__name__` parameter of the dispatcher is set to
            this.  If not given and `signature_source` is _not_ an instance of
            `inspect.Signature`, then we will attempt to read `__name__` from
            there instead.

        module : str, optional
            If given, the `__module__` parameter of the dispatcher is set to
            this.  If not given and `signature_source` is _not_ an instance of
            `inspect.Signature`, then we will attempt to read `__module__` from
            there instead.

            .. note::

                Commented for now because of a bug in cython 3 (cython#5472)
        """
        if isinstance(inputs, str):
            inputs = (inputs,)
        inputs = tuple(inputs)
        if inputs == () and out is False:
            warnings.warn(
                "No parameters to dispatch on."
                " Maybe you meant to specify 'inputs' or 'out'?"
            )
        self.inputs = inputs
        if isinstance(signature_source, inspect.Signature):
            self.__signature__ = signature_source
        else:
            self.__signature__ = inspect.signature(signature_source)
        for input in self.inputs:
            if (
                self.__signature__._parameters[input].kind
                != inspect.Parameter.POSITIONAL_ONLY
            ):
                raise ValueError("inputs parameters must be positional only.")
            if list(self.__signature__._parameters).index(input) >= len(inputs):
                raise ValueError("inputs must be the first positional parameters.")
        if name is not None:
            self.__name__ = name
        elif not isinstance(signature_source, inspect.Signature):
            self.__name__ = signature_source.__name__
        else:
            self.__name__ = 'dispatcher'
        # if module is not None:
        #     self.__module__ = module
        # elif not isinstance(signature_source, inspect.Signature):
        #     self.__module__ = signature_source.__module__
        self.__text_signature__ = self.__name__ + str(self.__signature__)
        if not isinstance(signature_source, inspect.Signature):
            self.__doc__ = inspect.getdoc(signature_source)
        self.output = out
        self._specialisations = {}
        self._lookup = {}
        self._n_inputs = len(self.inputs)
        self._n_dispatch = len(self.inputs) + self.output
        self._pass_on_dtype = 'dtype' in self.__signature__.parameters
        # Add ourselves to the list of dispatchers to be updated.
        _to.dispatchers.append(self)

    def add_specialisations(self, specialisations, _defer=False):
        """
        Add specialisations for particular combinations of data types to this
        operation.  The data types must already be known in `data.to` before
        you try to provide them here.  All data types defined in `data.to` will
        automatically work with this dispatcher, but will involve inefficient
        conversions to and from other types unless you define a closer
        specialisation using this method.

        The lookup table will automatically be rebuilt after this method is
        called.  Specialisations defined more than once will use the most
        recent version; you can use this to override currently known
        specialisations if desired.

        Parameters
        ----------
        specialisations : iterable of tuples
            An iterable where each element specifies a new specialisation for
            this operation.  Each element of the iterable should be a tuple,
            whose items are the types (instances of `type`) which this
            specialisation takes in each of the slots defined by
            `Dispatcher.inputs`, and the output type if this is a dispatcher
            over output types.  The last element should be the callable itself.

            The callable must have exactly the same signature as
            `Dispatcher.__signature__`; it is not enough that it takes all the
            same keyword arguments, but they must come in the same order as
            well (this is a speed optimisation for the dispatching operation).

            For example, if this is a dispatcher with the signature
                add(left, right, scale=1)
            which also dispatches over its output, and we have specialisations
                add_1(left: CSR, right: Dense, scale=1) -> Dense
                add_2(left: Dense, right: CSC, scale=1) -> CSR
            then to add this, `specialisations` should look like
                [
                    (CSR, Dense, Dense, add_1),
                    (Dense, CSC, CSR, add_2),
                ]
            Type annotations present in the specialisation objects are ignored.

        _defer : bool, optional (False)
            Only intended for internal library use during initialisation. If
            `True`, then the input types are not checked, and the full lookup
            table is not built until a manual call to
            `Dispatcher.rebuild_lookup()` is made.  If you are getting errors,
            remember that you should add the data type conversions to `data.to`
            before you try to add specialisations.
        """
        for arg in specialisations:
            arg = tuple(arg)
            if len(arg) != self._n_dispatch + 1:
                raise ValueError(
                    "specialisation " + str(arg)
                    + " has wrong number of parameters: needed types for "
                    + str(self.inputs)
                    + (", an output type" if self.output else "")
                    + " and a callable"
                )
            for i in range(self._n_dispatch):
                if (
                    not _defer
                    and arg[i] not in _to.dtypes
                    and arg[i] is not Data
                ):
                    raise ValueError(str(arg[i]) + " is not a known data type")
            if not callable(arg[self._n_dispatch]):
                raise TypeError(str(arg[-1]) + " is not callable")
            self._specialisations[arg[:-1]] = arg[-1]
        if not _defer:
            self.rebuild_lookup()

    cdef object _find_specialization(
        self, tuple in_types, bint output,
        type default=None, int verbose=False
    ):
        # The complexity of building the table here is very poor, but it's a
        # cost we pay very infrequently, and until it's proved to be a
        # bottle-neck in real code, we stick with the simple algorithm.
        cdef object weight, cur
        cdef tuple types, out_types, displayed_type
        cdef object function
        cdef int n_dispatch
        weight = math.INFINITY
        types = None
        function = None
        n_dispatch = len(in_types)
        if verbose:
            print(f"Building {self.__name__}{[in_type.__name__ for in_type in in_types]}")
        for out_types, out_function in self._specialisations.items():
            cur = _conversion_weight(
                in_types, out_types[:n_dispatch], _to.weight, out=output
            )
            if verbose:
                print(f"    {out_function.__name__}: {cur}")
            if cur < weight:
                weight = cur
                types = out_types
                function = out_function

        if cur == math.INFINITY:
            raise ValueError("No valid specialisations found")

        if verbose:
            print(f"Selected: {function.__name__}")
            print()

        if weight in [EPSILON, 0.] and not (output and types[-1] is Data):
            self._lookup[in_types] = function
        elif default is not None:
            # No exact match, use default dtype
            self._lookup[in_types] = self._lookup[in_types + (default,)]
        else:
            if output:
                converters = tuple(
                    [_to[pair] for pair in zip(types[:-1], in_types[:-1])]
                    + [_to[in_types[-1], types[-1]]]
                )
            else:
                converters = tuple(_to[pair] for pair in zip(types, in_types))
            displayed_type = in_types
            if len(in_types) < len(types):
                displayed_type = displayed_type + (types[-1],)
            self._lookup[in_types] =\
                _constructed_specialisation(function, self, displayed_type,
                                            converters, output)

    def rebuild_lookup(self, verbose=False):
        """
        Manually trigger a rebuild of the lookup table for this dispatcher.
        This is called automatically when new data types are added to
        `data.to`, or when specialisations are added to this object with
        `Dispatcher.add_specialisations`.

        You most likely do not need to call this function yourself.
        """
        if not self._specialisations:
            return
        self._dtypes = _to.dtypes.copy()
        for in_types in itertools.product(self._dtypes, repeat=self._n_dispatch):
            self._find_specialization(in_types, self.output, None, verbose=verbose)
        # Now build the lookup table in the case that we dispatch on the output
        # type as well, but the user has called us without specifying it.
        if self.output and settings.core["default_dtype_scope"] == "full":
            default_dtype = _to.parse(settings.core["default_dtype"])
            for in_types in itertools.product(self._dtypes, repeat=self._n_dispatch-1):
                self._lookup[in_types] = self._lookup[in_types + (default_dtype,)]
        elif self.output:
            default_dtype = None
            if settings.core["default_dtype_scope"] == "missing":
                default_dtype = _to.parse(settings.core["default_dtype"])
            for in_types in itertools.product(self._dtypes, repeat=self._n_dispatch-1):
                self._find_specialization(in_types, False, default_dtype, verbose)
        if self.output:
            for in_types in itertools.product(self._dtypes, repeat=self._n_dispatch-1):
                for group in _to.groups:
                    self._lookup[in_types + (group,)] = _group_specialisation(
                        self._lookup[in_types], self, in_types, group
                    )

    def __getitem__(self, types):
        """
        Get the particular specialisation for the given types.  The output is a
        callable object which requires that the dispatched arguments match
        those specified in `types`.
        """
        if type(types) is not tuple:
            types = (types,)
        types = tuple(_to.parse(arg) for arg in types)
        try:
            return self._lookup[types]
        except KeyError:
            raise TypeError("specialisation not known for types: " + str(types)) from None

    def __repr__(self):
        return "<dispatcher: " + self.__text_signature__ + ">"

    def __call__(self, *args, dtype=None, **kwargs):
        cdef list dispatch = []
        cdef int i
        if self._pass_on_dtype:
            kwargs['dtype'] = dtype
        if not (self._pass_on_dtype or self.output) and dtype is not None:
            raise TypeError("unknown argument 'dtype'")
        if len(args) < self._n_inputs:
            raise TypeError(
                "All dispatched data input must be passed "
                "as positional arguments."
            )
        for i in range(self._n_inputs):
            dispatch.append(type(args[i]))

        if self.output and dtype is not None:
            dtype = _to.parse(dtype)
            dispatch.append(dtype)
        try:
            function = self._lookup[tuple(dispatch)]
        except KeyError:
            raise TypeError("unknown types to dispatch on: " + str(dispatch)) from None
        return function(*args, **kwargs)
