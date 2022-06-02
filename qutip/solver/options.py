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
    def __init__(self, solver='solver', method=None, **kwargs):
        if solver not in known_solver:
            raise ValueError(f'Unknown solver "{solver}".')
        self.solver = known_solver[solver]
        self._doc = self.__doc__ + _adjust_docstring_indent(
            self.solver.options.__doc__
        )
        self.solver_options = self.solver.default_options.copy()
        if method:
            self.solver_options['method'] = method
        else:
            method = self.solver_options['method']
        self.supported_integrator = self.solver.avail_integrators()
        self.integrator_options = self._get_integrator_options(method)

        # Merge all options
        for key in kwargs:
            self.__setitem__(key, kwargs[key])

    def _get_integrator_options(self, method):
        if method not in self.supported_integrator:
            raise ValueError(
                f'Integration method "{method}" is not supported '
                "by the solver."
            )
        integrator = self.supported_integrator[method]
        integrator_options = integrator.integrator_options.copy()
        self.__doc__ = self._doc + integrator.options.__doc__
        return integrator_options

    def __setitem__(self, key, value):
        if key == 'method' and values != self.solver_options[key]:
            # This does not keep options values that are in common.
            self.integrator_options = self._get_integrator_options(value)

        if key in self.solver_options:
            self.solver_options[key] = value
        elif key in self.integrator_options:
            self.integrator_options[key] = value
        else:
            raise KeyError(key)

    def __getitem__(self, key):
        for dictionary in [
            self.solver_options,
            self.integrator_options
        ]:
            if key in dictionary:
                return dictionary[key]
        raise KeyError(key)

    def __str__(self):
        longest_s = max(len(key) for key in self.solver_options)
        longest_i = max(len(key) for key in self.integrator_options)
        lines = []
        lines.append(f"Options for {self.solver.name}:")
        for key, val in self.solver_options.items():
            lines.append(f"    {key:{longest_s}} : '{val.__repr__()}'")
        method = self.solver_options['method']
        lines.append(f"Options for {method} integrator:")
        for key, val in self.integrator_options.items():
            lines.append(f"    {key:{longest_i}} : '{val.__repr__()}'")
        return "\n".join(lines)

    def __repr__(self):
        items = []
        items.append(f"SolverOptions(solver={self.solver.name}")
        for key, val in self.solver_options.items():
            items.append(f"{key}={val.__repr__()}")
        for key, val in self.integrator_options.items():
            items.append(f"{key}='{val.__repr__()}'")
        return ", ".join(items) + ")"

    def keys(self):
        return self.solver_options.keys()| self.integrator_options.keys()

    def values(self):
        return tuple(
            list(self.solver_options.values())
            + list(self.integrator_options.values())
        )

    def items(self):
        return tuple(
            list(self.solver_options.items())
            + list(self.integrator_options.items())
        )

    def __contains__(self, key):
        return key in self.solver_options or key in self.integrator_options

    def copy(self):
        return SolverOptions(self.solver.name, **self)
