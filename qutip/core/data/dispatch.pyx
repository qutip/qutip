#cython: language_level=3

import functools
import inspect
import itertools
import warnings

from .convert import to as _to

cimport cython
from libc cimport math
from libcpp cimport bool

__all__ = ['Dispatcher']

cdef class _bind:
    """
    Cythonised implementation of inspect.Signature.bind, supporting faster
    binding and handling of default arguments.  On construction, the signature
    of the base function is extracted, and all parameters are parsed into
    positional or keyword slots, using positional wherever possible, and their
    defaults are stored.
    """
    # Instance of inpsect.Signature representing the function.
    cdef object signature
    cdef object inputs
    # Mapping of (str: int), where str is the name of any argument which may be
    # specified by a keyword, and int is its location in `self._pos` if
    # available, or -1 if it is keyword-only.
    cdef dict _locations
    # Default values (or inspect.Parameter.empty) for every parameter which
    # may be passed as a positional argument.
    cdef list _pos
    # For each dispatcher input which can be given as a positional argument,
    # these two lists hold the index of the input in the given tuple of input
    # names and the corresponding index into the positional arguments after
    # binding is complete.  {_pos_inputs_pos[n]: _pos_inputs_input[n]} is
    # effectively the same mapping as _kw_inputs (but for positional
    # arguments), but we use two lists rather than a dictionary for speed.
    cdef list _pos_inputs_input, _pos_inputs_pos
    # Default values (or inspect.Parameter.empty) for every parameter which
    # _must_ be passed as a keyword argument.
    cdef dict _kw
    # Mapping of (name, index into the input tuple) for each input which must
    # be specified as a keyword argument
    cdef list _kw_inputs
    # Names of the keyword arguments which have default values.
    cdef set _default_kw_names
    # Respectively, numbers of positional parameters, keyword-only parameters
    # and dispatcher inputs.
    cdef Py_ssize_t n_args, n_kwargs, n_inputs
    # Respectively, numbers of inputs which are positional and keyword-only.
    cdef Py_ssize_t _n_pos_inputs, _n_kw_inputs
    # Respectively, numbers of positional arguments without a default, and
    # number of keyword-only arguments which _have_ a default.
    cdef Py_ssize_t _n_pos_no_default, _n_kw_default


    def __init__(self, signature, tuple inputs):
        for arg in inputs:
            if arg not in signature.parameters:
                raise AttributeError("No argument matches '{}'.".format(arg))
        self.signature = signature
        self.inputs = inputs
        self._locations = {}
        self._pos = []
        self._pos_inputs_input = []
        self._pos_inputs_pos = []
        self._kw = {}
        self._kw_inputs = []
        self._default_kw_names = set()
        self._n_pos_no_default = 0
        # signature.parameters is ordered for all Python versions.
        for i, (name, parameter) in enumerate(signature.parameters.items()):
            kind = parameter.kind
            if (kind == parameter.VAR_POSITIONAL
                    or kind == parameter.VAR_KEYWORD):
                raise TypeError("Cannot dispatch with *args or **kwargs.")
            if kind == parameter.KEYWORD_ONLY:
                self._kw[name] = parameter.default
                self._locations[name] = -1
                if parameter.default is not parameter.empty:
                    self._default_kw_names.add(name)
                if name in inputs:
                    self._kw_inputs.append((name, inputs.index(name)))
            else:
                self._pos.append(parameter.default)
                if kind != parameter.POSITIONAL_ONLY:
                    self._locations[name] = i
                if parameter.default is parameter.empty:
                    self._n_pos_no_default += 1
                if name in inputs:
                    # Effectively a mapping.
                    self._pos_inputs_input.append(inputs.index(name))
                    self._pos_inputs_pos.append(len(self._pos) - 1)
        self.n_args = len(self._pos)
        self.n_kwargs = len(self._kw)
        self.n_inputs = len(inputs)
        self._n_pos_inputs = len(self._pos_inputs_input)
        self._n_kw_inputs = len(self._kw_inputs)
        self._n_kw_default = len(self._default_kw_names)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef tuple bind(self, tuple args, dict kwargs):
        """
        Cython reimplementation of inspect.Signature.bind for binding the
        collected `*args` and `**kwargs` of a generic call to a specific
        signature, checking that everything matches.

        The output is a parsed tuple (args, kwargs), where `args` is a list and
        `kwargs` is a dict, however all arguments which _can_ be passed
        positionally will be moved from `kwargs` into `args` for the output.
        The resultant `args` and `kwargs` can be unpacked into the underlying
        function call safely.  Default values are not filled in by this method.

        This is necessary to allow a generic `Dispatcher` class to work with
        all type signatures.  The result of this function can be fed to
        `_bind.convert_types` and `_bind.dispatch_types`.
        """
        cdef Py_ssize_t location, got_pos=0, got_kw=0
        cdef Py_ssize_t n_passed_args = len(args)
        if n_passed_args > self.n_args:
            raise TypeError(
                "Too many arguments: expected at most {} but received {}."
                .format(n_passed_args, self.n_args)
            )
        # _pos and _kw are filled with their defaults at initialisation, and
        # _pos is already the correct length (no-default parameters have
        # sentinel values).
        cdef list out_pos = self._pos.copy()
        cdef dict out_kw = self._kw.copy()
        got_pos = self.n_args - self._n_pos_no_default
        got_kw = self._n_kw_default
        # Positional arguments unambiguously fill the first positional slots.
        for location in range(n_passed_args):
            out_pos[location] = args[location]
        # How many arguments that we didn't have defaults for did we just get?
        # Positionals with a default are always after those without one.
        if n_passed_args > self._n_pos_no_default:
            got_pos += self._n_pos_no_default
        else:
            got_pos += n_passed_args
        # Everything else has been passed by keyword, but it may be allowed to
        # be passed positionally, which is the case we prefer.  dict.items() is
        # (relatively) expensive, so we use a boolean test for the fast path.
        cdef str kw
        cdef object arg
        if kwargs:
            for kw, arg in kwargs.items():
                try:
                    location = self._locations[kw]
                except KeyError:
                    raise TypeError("Unknown argument '{}'.".format(kw)) from None
                # _locations[kw] = -1 if kw is keyword-only, otherwise the
                # corresponding positional location.
                if location >= 0:
                    if location < n_passed_args:
                        raise TypeError("Multiple values for '{}'".format(kw))
                    out_pos[location] = arg
                    if location < self._n_pos_no_default:
                        got_pos += 1
                else:
                    out_kw[kw] = arg
                    if kw not in self._default_kw_names:
                        got_kw += 1
        if got_pos < self.n_args:
            raise TypeError("Too few positional arguments passed.")
        if got_kw < self.n_kwargs:
            raise TypeError("Not all keyword arguments were filled.")
        return out_pos, out_kw

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef list dispatch_types(self, list args, dict kwargs):
        """
        Get a list of the types which the dispatch should operate over, given
        the output of `_bind.bind`.  The return value is a list of the _input_
        types to be dispatched on, in order that they were specified at
        `Dispatcher` creation.  The output type is not returned as part of this
        list, because there is no way for `_bind` to know it.
        """
        cdef list dispatch = [None] * self.n_inputs
        cdef str kw
        cdef Py_ssize_t location, i
        for location in range(self._n_pos_inputs):
            dispatch[self._pos_inputs_input[location]]\
                = type(args[self._pos_inputs_pos[location]])
        for i in range(self._n_kw_inputs):
            kw, location = self._kw_inputs[i]
            dispatch[location] = type(kwargs[kw])
        return dispatch

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef tuple convert_types(self, list args, dict kwargs, tuple converters):
        """
        Apply the type converters `converters` to the parsed arguments `args`
        and `kwargs`.

        If there are `n` inputs which are dispatched on, `converters` should be
        a tuple whose first `n` elements are converters (such as those obtained
        from `data.to[to_type, from_type]`) to the desired types.

        `args` and `kwargs` should be the output of `_bind.bind`; the function
        will likely fail if called on unparsed arguments.
        """
        cdef str kw
        cdef Py_ssize_t location, i
        for location in range(self._n_pos_inputs):
            args[location] = converters[location](args[location])
        for i in range(self._n_kw_inputs):
            kw, location = self._kw_inputs[i]
            kwargs[kw] = converters[location](kwargs[kw])
        return args, kwargs


cdef double _conversion_weight(tuple froms, tuple tos, dict weight_map, bint out) except -1:
    """
    Find the total weight of conversion if the types in `froms` are converted
    element-wise to the types in `tos`.  `weight_map` is a mapping of
    `(to_type, from_type): real`; it should almost certainly be
    `data.to.weight`.
    """
    cdef double weight = 0.0
    cdef Py_ssize_t i, n=len(froms)
    if len(tos) != n:
        raise ValueError(
            "number of arguments not equal: " + str(n) + " and " + str(len(tos))
        )
    if out:
        n = n - 1
        weight += weight_map[froms[n], tos[n]]
    for i in range(n):
        weight += weight_map[tos[i], froms[i]]
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
    cdef _bind _parameters
    cdef bint _output
    cdef object _call
    cdef Py_ssize_t _n_dispatch
    cdef readonly tuple types
    cdef tuple _converters
    cdef str _short_name
    cdef public str __doc__
    cdef public str __name__
    cdef public str __module__
    cdef public object __signature__
    cdef readonly str __text_signature__
    cdef readonly bint direct

    def __init__(self, base, Dispatcher dispatcher, types, converters, out,
                 direct):
        self.__doc__ = inspect.getdoc(dispatcher)
        self._short_name = dispatcher.__name__
        self.__name__ = (
            self._short_name
            + "_"
            + "_".join([x.__name__ for x in types])
        )
        self.__module__ = dispatcher.__module__
        self.__signature__ = dispatcher.__signature__
        self.__text_signature__ = dispatcher.__text_signature__
        self._parameters = dispatcher._parameters
        self._output = out
        self._call = base
        self._n_dispatch = len(types)
        self.types = types
        self.direct = direct
        self._converters = converters

    cdef object prebound(self, list args, dict kwargs):
        """
        Call this specialisation with pre-parsed arguments and keyword
        arguments.  `args` and `kwargs` must be the output of the relevant
        `_bind.bind` method for this function.
        """
        self._parameters.convert_types(args, kwargs, self._converters)
        out = self._call(*args, **kwargs)
        if self._output:
            out = self._converters[self._n_dispatch - 1](out)
        return out

    def __call__(self, *args, **kwargs):
        cdef list args_
        cdef dict kwargs_
        args_, kwargs_ = self._parameters.bind(args, kwargs)
        return self.prebound(args_, kwargs_)

    def __repr__(self):
        if len(self.types) == 1:
            spec = self.types[0].__name__
        else:
            spec = "(" + ", ".join(x.__name__ for x in self.types) + ")"
        direct = ("" if self.direct else "in") + "direct"
        return "".join([
            "<", direct, " specialisation ", spec, " of ", self._short_name, ">"
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
    cdef _bind _parameters
    cdef readonly dict _specialisations
    cdef Py_ssize_t _n_dispatch
    cdef readonly dict _lookup
    cdef set _dtypes
    cdef bint _pass_on_dtype
    cdef readonly tuple inputs
    cdef readonly bint output
    cdef public str __doc__
    cdef public str __name__
    cdef public str __module__
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
            The parameters which should be dispatched over.  These can be
            positional or keyword arguments, but must feature in the signature
            provided.

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
        self._parameters = _bind(self.__signature__, inputs)
        if name is not None:
            self.__name__ = name
        elif not isinstance(signature_source, inspect.Signature):
            self.__name__ = signature_source.__name__
        else:
            self.__name__ = 'dispatcher'
        if module is not None:
            self.__module__ = module
        elif not isinstance(signature_source, inspect.Signature):
            self.__module__ = signature_source.__module__
        self.__text_signature__ = self.__name__ + str(self.__signature__)
        if not isinstance(signature_source, inspect.Signature):
            self.__doc__ = inspect.getdoc(signature_source)
        self.output = out
        self._specialisations = {}
        self._lookup = {}
        self._n_dispatch = self._parameters.n_inputs + self.output
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
                    + str(self._parameters.inputs)
                    + (", an output type" if self.output else "")
                    + " and a callable"
                )
            for i in range(self._n_dispatch):
                if (not _defer) and arg[i] not in _to.dtypes:
                    raise ValueError(str(arg[i]) + " is not a known data type")
            if not callable(arg[self._n_dispatch]):
                raise TypeError(str(arg[-1]) + " is not callable")
            self._specialisations[arg[:-1]] = arg[-1]
        if not _defer:
            self.rebuild_lookup()

    def rebuild_lookup(self):
        """
        Manually trigger a rebuild of the lookup table for this dispatcher.
        This is called automatically when new data types are added to
        `data.to`, or when specialisations are added to this object with
        `Dispatcher.add_specialisations`.

        You most likely do not need to call this function yourself.
        """
        cdef double weight, cur
        cdef tuple types, out_types
        cdef object function
        if not self._specialisations:
            return
        self._dtypes = _to.dtypes.copy()
        # The complexity of building the table here is very poor, but it's a
        # cost we pay very infrequently, and until it's proved to be a
        # bottle-neck in real code, we stick with the simple algorithm.
        for in_types in itertools.product(self._dtypes, repeat=self._n_dispatch):
            weight = math.INFINITY
            types = None
            function = None
            for out_types, out_function in self._specialisations.items():
                cur = _conversion_weight(in_types, out_types, _to.weight,
                                         out=self.output)
                if cur < weight:
                    weight = cur
                    types = out_types
                    function = out_function
            if self.output:
                converters = tuple(
                    [_to[pair] for pair in zip(types[:-1], in_types[:-1])]
                    + [_to[in_types[-1], types[-1]]]
                )
            else:
                converters = tuple(_to[pair] for pair in zip(types, in_types))
            self._lookup[in_types] =\
                _constructed_specialisation(function, self, in_types,
                                            converters, self.output,
                                            weight == 0)
        # Now build the lookup table in the case that we dispatch on the output
        # type as well, but the user has called us without specifying it.
        # TODO: option to control default output type choice if unspecified?
        if self.output:
            for in_types in itertools.product(self._dtypes, repeat=self._n_dispatch-1):
                weight = math.INFINITY
                types = None
                function = None
                for out_types, out_function in self._specialisations.items():
                    cur = _conversion_weight(in_types, out_types[:-1],
                                             _to.weight, out=False)
                    if cur < weight:
                        weight = cur
                        types = out_types
                        function = out_function
                converters = tuple(_to[pair] for pair in zip(types, in_types))
                self._lookup[in_types] =\
                    _constructed_specialisation(function, self,
                                                in_types + (types[-1],),
                                                converters, False,
                                                weight == 0)

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
        cdef list args_, dispatch
        cdef dict kwargs_
        if self._pass_on_dtype:
            kwargs['dtype'] = dtype
        if not (self._pass_on_dtype or self.output) and dtype is not None:
            raise TypeError("unknown argument 'dtype'")
        args_, kwargs_ = self._parameters.bind(args, kwargs)
        dispatch = self._parameters.dispatch_types(args_, kwargs_)
        if self.output and dtype is not None:
            dtype = _to.parse(dtype)
            dispatch.append(dtype)
        cdef _constructed_specialisation function
        try:
            function = self._lookup[tuple(dispatch)]
        except KeyError:
            raise TypeError("unknown types to dispatch on: " + str(dispatch)) from None
        return function.prebound(args_, kwargs_)
