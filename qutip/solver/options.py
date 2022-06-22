__all__ = ['SolverOptions']

known_solver = {}


def _remove_indent(line):
    """
    Remove one indent from a string.
    """
    if len(line) >= 4 and line[:4] == "    ":
        line = line[4:]
    return line


def _adjust_docstring_indent(doc):
    return "\n".join(_remove_indent(line) for line in doc.split('\n'))


class SolverOptions():
    """
    Options for solvers.

    Options can be specified either as arguments to the constructor::

        opts = Options('sesolve', progress_bar='enhanced', ...)

    or by changing the class attributes after creation::

        opts = Options('sesolve')
        opts['progress_bar'] = 'enhanced'

    .. note::
        Passing options that are not used by the solver will end up having them
        being dropped. Each integrator method has their own options and
        changing the ``method`` will change the list of supported options.

    Parameters
    ----------
    solver : str ["sesolve", "mesolve", "brmesolve", etc.]
        Solver for which this options is intended to be used with. When no
        value is passed, only options shared between most solver will be
        displayed.
    """
    def __init__(self, solver='solver', method=None,
                 _solver_feedback=None, **kwargs):
        if solver not in known_solver:
            raise ValueError(f'Unknown solver "{solver}".')

        solver_class = known_solver[solver]
        # Various spelling are supported, set the official name.
        self.solver = solver_class.name
        # Add solver's options to the doc-string
        self._doc = self.__class__.__doc__ + _adjust_docstring_indent(
            solver_class.options.__doc__
        )

        # Get a copy of the solver default options
        self._default_solver_options = solver_class.solver_options.copy()
        self.supported_integrator = solver_class.avail_integrators()

        method = method or self._default_solver_options['method']
        self._options = {'method': method}
        self._set_integrator_options(method)

        # Merge all options
        for key in kwargs:
            self.__setitem__(key, kwargs[key])

        # A function to call to inform the solver an options was updated.
        self._solver_feedback = _solver_feedback

    def _set_integrator_options(self, method):
        """
        - Get a copy of the integrator's default options.
        - Create a set of all supported keys.
        - Remove the options' items that were supported by the previous
          integrator, but a not by the new one.
        - Rewrite the doc-string to include the new integrator's options.
        """
        if method not in self.supported_integrator:
            raise ValueError(
                f'Integration method "{method}" is not supported '
                "by the solver."
            )
        integrator = self.supported_integrator[method]
        self._default_integrator_options = integrator.integrator_options.copy()
        self.supported_keys = (
             self._default_solver_options.keys()
             | self._default_integrator_options.keys()
        )
        # We drop options that the new integrator does not support.
        self._options = {
            key: val
            for key, val in self._options.items()
            if key in self.supported_keys
        }
        self.__doc__ = (
            self._doc + _adjust_docstring_indent(integrator.options.__doc__)
        )

    def __setitem__(self, key, value):
        """
        Set the options.
        Settings to ``None`` revert the the default value.
        """
        if key == 'method' and value != self._options['method']:
            self._set_integrator_options(value)

        if key not in self.supported_keys:
            raise KeyError(f"'{key}' is not a supported options.")
        elif value is not None:
            self._options[key] = value
        elif key in self._options:
            del self._options[key]

        if self._solver_feedback is not None:
            self._solver_feedback({key})

    def __getitem__(self, key):
        for dictionary in [
            self._options,
            self._default_solver_options,
            self._default_integrator_options
        ]:
            if key in dictionary:
                return dictionary[key]
        raise KeyError(f"'{key}' is not a supported options.")

    def __contains__(self, key):
        return key in self.supported_keys

    def __delitem__(self, key):
        """
        Revert an options to it's default value.
        """
        if key in self._options:
            del self._options[key]
        elif key in self.supported_keys:
            pass  # Default value, can't be erased.
        else:
            raise KeyError(f"'{key}' is not a supported options.")

    def __str__(self):
        longest_s = max(len(self[key]) for key in self._default_solver_options)
        longest_i = max(len(self[key]) for key in self._default_integrator_options)
        lines = []
        lines.append(f"Options for {self.solver}:")
        for key in self._default_solver_options:
            lines.append(f"    {key:{longest_s}} : '{self[key].__repr__()}'")
        method = self._default_solver_options['method']
        lines.append(f"Options for {method} integrator:")
        for key in self._default_integrator_options:
            lines.append(f"    {key:{longest_i}} : '{self[key].__repr__()}'")
        return "\n".join(lines)

    def __repr__(self):
        items = []
        items.append(f"SolverOptions(solver={self.solver}")
        for key, val in self.items():
            items.append(f"{key}={val.__repr__()}")
        return ", ".join(items) + ")"

    def keys(self):
        """
        Return the keys of the non-default options.
        """
        return self._options.keys()

    def values(self):
        """
        Return the values of the non-default options.
        """
        return tuple(self[key] for key in self.keys())

    def items(self):
        """
        Return the (key, value) pairs of the non-default options.
        """
        return tuple((key, self[key]) for key in self.keys())

    def copy(self):
        copy = SolverOptions(
            self.solver,
            **self,
            _solver_feedback=_self._solver_feedback
        )
        return copy

    def __bool__(self):
        return bool(self._options)
