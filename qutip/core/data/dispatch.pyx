#cython: language_level=3

import functools
import inspect
import itertools
import warnings

from .convert import to as _to

cimport cython
from libc cimport math
from libcpp cimport bool


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
        Cython reimplementation of inspect.Signature.bind which also finds the
        required dispatch parameters at the same time.  We can use the
        assumption that instances of this class are immutable to pre-compute
        various portions of the lookup code, and ensure that the majority of
        operations are done by in-bounds integer indexing into pre-constructed
        lists rather than string lookups in dicts.
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
        # be passed positionally, which is the case we prefer.
        cdef str kw
        cdef Py_ssize_t arg
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
    cdef list dispatch_types(self, args, kwargs):
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
        cdef str kw
        cdef Py_ssize_t location, i
        for location in range(self._n_pos_inputs):
            args[location] = converters[location](args[location])
        for i in range(self._n_kw_inputs):
            kw, location = self._kw_inputs[i]
            kwargs[kw] = converters[location](kwargs[kw])
        return args, kwargs


cdef double _conversion_weight(tuple froms, tuple tos, dict weight_map) except -1:
    cdef double out = 0.0
    cdef Py_ssize_t i, n=len(froms)
    if len(tos) != n:
        raise ValueError(
            "number of arguments not equal: " + str(n) + " and " + str(len(tos))
        )
    for i in range(n):
        out += weight_map[tos[i], froms[i]]
    return out


cdef class _constructed_specialisation:
    cdef _bind _parameters
    cdef bint _output
    cdef object _call
    cdef Py_ssize_t _n_dispatch
    cdef tuple _types
    cdef tuple _converters
    cdef readonly str __doc__
    cdef readonly str __name__
    cdef readonly str __module__
    cdef readonly object __signature__
    cdef readonly str __text_signature__

    def __init__(self, base, Dispatcher dispatcher, types, converters, out):
        self.__doc__ = inspect.getdoc(dispatcher)
        self.__name__ = (
            dispatcher.__name__
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
        self._types = types
        self._converters = converters

    cdef object prebound(self, list args, dict kwargs):
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


cdef class Dispatcher:
    cdef _bind _parameters
    cdef bint _output
    cdef dict _specialisations
    cdef Py_ssize_t _n_dispatch
    cdef readonly dict _lookup
    cdef set _dtypes
    cdef bint _pass_on_out
    cdef public str __doc__
    cdef public str __name__
    cdef public str __module__
    cdef public object __signature__
    cdef public str __text_signature__

    def __init__(self, signature_source, inputs, bint out=False,
                 str name=None, str module=None):
        if isinstance(inputs, str):
            inputs = (inputs,)
        inputs = tuple(inputs)
        if inputs == ():
            warnings.warn(
                "No parameters to dispatch on."
                " Maybe you meant to specify 'inputs'?"
            )
        if isinstance(signature_source, inspect.Signature):
            self.__signature__ = signature_source
        else:
            self.__signature__ = inspect.signature(signature_source)
        self._parameters = _bind(self.__signature__, inputs)
        self.__name__ = name or 'dispatcher'
        self.__text_signature__ = self.__name__ + str(self.__signature__)
        self.__module__ = module
        self._output = out
        self._specialisations = {}
        self._lookup = {}
        self._n_dispatch = self._parameters.n_inputs + self._output
        self._pass_on_out = 'out' in self.__signature__.parameters
        # Add ourselves to the list of dispatchers to be updated.
        _to.dispatchers.append(self)

    def add_specialisations(self, specialisations, _defer=False):
        for arg in specialisations:
            arg = tuple(arg)
            if len(arg) != self._n_dispatch + 1:
                raise ValueError(
                    "specialisation " + str(arg)
                    + " has wrong number of parameters: needed types for "
                    + str(self._parameters.inputs)
                    + (", an output type" if self._output else "")
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
        cdef double weight, cur
        cdef tuple types, out_types
        cdef object function
        cdef type chosen_out_type
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
                cur = _conversion_weight(in_types, out_types, _to.weight)
                if cur < weight:
                    weight = cur
                    types = out_types
                    function = out_function
            if self._output:
                converters = tuple(
                    [_to[pair] for pair in zip(types[:-1], in_types[:-1])]
                    + [_to[in_types[-1], types[-1]]]
                )
            else:
                converters = tuple(_to[pair] for pair in zip(types, in_types))
            self._lookup[in_types] =\
                _constructed_specialisation(function, self, in_types,
                                            converters, self._output)
        # Now build the lookup table in the case that we dispatch on the output
        # type as well, but the user has called us without specifying it.
        # TODO: option to control default output type choice if unspecified?
        if self._output:
            for in_types in itertools.product(self._dtypes, repeat=self._n_dispatch-1):
                weight = math.INFINITY
                types = None
                function = None
                for out_types, out_function in self._specialisations.items():
                    cur = _conversion_weight(in_types, out_types[:-1], _to.weight)
                    if cur < weight:
                        weight = cur
                        types = out_types
                        function = out_function
                converters = tuple(_to[pair] for pair in zip(types, in_types))
                self._lookup[in_types] =\
                    _constructed_specialisation(function, self, in_types, converters, False)

    def __getitem__(self, types):
        return self._lookup[types]

    def __call__(self, *args, out=None, **kwargs):
        cdef list args_, dispatch
        cdef dict kwargs_
        if self._pass_on_out:
            kwargs['out'] = out
        if not (self._pass_on_out or self._output) and out is not None:
            raise TypeError("unknown argument 'out'")
        args_, kwargs_ = self._parameters.bind(args, kwargs)
        dispatch = self._parameters.dispatch_types(args_, kwargs_)
        if self._output and out is not None:
            if self._pass_on_out:
                out = type(out)
            dispatch.append(out)
        cdef _constructed_specialisation function = self._lookup[tuple(dispatch)]
        return function.prebound(args_, kwargs_)
