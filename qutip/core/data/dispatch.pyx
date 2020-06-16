#cython: language_level=3

import functools
import inspect
import warnings

cimport cython
from libcpp cimport bool

__all__ = ['dispatch']

_MAX_INDENT = 999

def _trim_docstring(docstring):
    # Code taken near-verbatim from PEP 257.
    if not docstring:
        return ''
    # Convert tabs to spaces (following the normal Python rules) and split into
    # a list of lines:
    lines = docstring.expandtabs().splitlines()
    # Determine minimum indentation (first line doesn't count):
    indent = _MAX_INDENT
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))
    # Remove indentation (first line is special):
    trimmed = [lines[0].strip()]
    if indent < _MAX_INDENT:
        for line in lines[1:]:
            trimmed.append(line[indent:].rstrip())
    # Strip off trailing and leading blank lines:
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)
    # Return a single string:
    return '\n'.join(trimmed)


cdef class _Bind:
    """
    Cythonised implementation of inspect.Signature.bind, supporting faster
    binding and handling of default arguments.  On construction, the signature
    of the base function is extracted, and all parameters are parsed into
    positional or keyword slots, using positional wherever possible, and their
    defaults are stored.
    """

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
    # Mapping of {name: index into the input tuple} for each input which must
    # be specified as a keyword argument
    cdef dict _kw_inputs
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


    def __init__(self, function, tuple inputs):
        signature = inspect.signature(function)
        for arg in inputs:
            if arg not in signature.parameters:
                raise AttributeError("No argument matches '{}'.".format(arg))
        self._locations = {}
        self._pos = []
        self._pos_inputs_input = []
        self._pos_inputs_pos = []
        self._kw = {}
        self._kw_inputs = {}
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
                    self._kw_inputs[name] = inputs.index(name)
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
        cdef list out_inputs = [None]*self.n_inputs
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
        for location in range(self._n_pos_inputs):
            out_inputs[self._pos_inputs_input[location]]\
                = out_pos[self._pos_inputs_pos[location]]
        # We're unlikely to dispatch on any keyword-only items.  Use faster
        # integer comparison to skip the call to dict.items() if so.
        if self._n_kw_inputs != 0:
            for kw, location in self._kw_inputs.items():
                out_inputs[location] = out_kw[kw]
        return out_inputs, out_pos, out_kw


cdef class Dispatcher:
    """
    A multiple-dispatch function which performs runtime-dispatch over some of
    its arguments.
    """

    cdef readonly object generic
    cdef _Bind _parameters
    cdef tuple inputs
    cdef dict lookup
    cdef dict __dict__

    # The docstring to __init__ is also used for `dispatch`.
    def __init__(self, function, inputs=()):
        """
        Parameters
        ----------
        function: callable
            The base case to be called if a suitable specialised method is not
            found.  The function must not take a `*args` or `**kwargs`
            parameter.

        inputs: str | iterable of str
            The names of the parameters which should be used for the multiple
            dispatch.
        """
        self.generic = function
        functools.update_wrapper(self, function)
        if isinstance(inputs, str):
            inputs = (inputs,)
        inputs = tuple(inputs)
        if inputs == ():
            warnings.warn(
                "No parameters to dispatch on."
                " Maybe you meant to specify 'inputs'?"
            )
        self._parameters = _Bind(function, inputs)
        self.lookup = {}

    def register(self, types):
        """
        Use as a decorator to register a specialised function for the given
        tuple of types.  Only the types of the `input` parameters should be
        passed.

        The registered function must have exactly the same signature as the
        base case.
        """
        if isinstance(types, type):
            types = (types,)
        key = tuple(types)
        if key in self.lookup:
            signature = (
                self.generic.__name__ + "("
                + ", ".join(type.__name__ for type in types)
                + ")"
            )
            warnings.warn("Overriding previously defined specialisation {}."
                          .format(signature))

        def _register(function):
            self.lookup[key] = function
            return function
        return _register

    cdef object _get(self, list inputs, bool as_types=False):
        if as_types:
            key = tuple(inputs)
        else:
            # Use a list not a generating expression because Cython compiles it
            # to faster code.
            key = tuple([x.__class__ for x in inputs])
        return self.lookup.get(key, self.generic)

    def get(self, inputs):
        """
        Get the Python callable that would be called for the given input types.
        """
        if isinstance(inputs, type):
            inputs = [inputs]
        if not all(isinstance(x, type) for x in inputs):
            raise TypeError("'inputs' should be an iterable of types.")
        return self._get(list(inputs), as_types=True)

    def __call__(self, *args, **kwargs):
        cdef list inputs, args_
        cdef dict kwargs_
        inputs, args_, kwargs_ = self._parameters.bind(args, kwargs)
        return self._get(inputs)(*args_, **kwargs_)


@cython.binding(True)
def dispatch(function=None, **kwargs):
    # Called as a function, or as a decorator with no arguments.
    if function is not None:
        return Dispatcher(function, **kwargs)

    # Standard decorator usage with arguments.
    def decorator(function):
        return Dispatcher(function, **kwargs)
    return decorator
dispatch.__doc__ = (
    _trim_docstring("""\
    Use as either a decorator or a standard function to create a function which
    dispatches over certain specified arguments.
    """)
    + "\n\n"
    + _trim_docstring(Dispatcher.__init__.__doc__)
)
