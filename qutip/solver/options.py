__all__ = ['SolverOptions']

from ..optionsclass import QutipOptions


options_docstrings = {
    "store_final_state": (
        "bool", """
        Whether or not to store the final state of the evolution in the
        result class."""
    ),

    "store_states": (
        "bool, None", """
        Whether or not to store the state vectors or density matrices.
        On `None` the states will be saved if no expectation operators are
        given."""
    ),

    "normalize_output": (
        "bool", """
        Normalize output state to hide ODE numerical errors.
        "all" will normalize both ket and dm.
        On "ket", only 'ket' output are normalized.
        Leave empty for no normalization."""
    ),

    "progress_bar": (
        "str {'text', 'enhanced', 'tqdm', ''}", """
        How to present the solver progress.
        'tqdm' uses the python module of the same name and raise an error if
        not installed. Empty string or False will disable the bar."""
    ),

    "progress_kwargs": (
        "dict", """
        kwargs to pass to the progress_bar. Qutip's bars use `chunk_size`."""
    ),

    "method": (
        "dict", """
        kwargs to pass to the progress_bar. Qutip's bars use `chunk_size`."""
    ),
}


class Options():
    """
    General class of options for any solver. Options can be specified either as
    arguments to the constructor::

        opts = Options(progress_bar='enhanced', ...)

    or by changing the class attributes after creation::

        opts = Options()
        opts['progress_bar'] = 'enhanced'

    Only the most commonly used options are listed here. See the options class
    matching the solver, such as ``McOptions``, for a full list of supported
    parameters.

    Options
    -------
    method : str {'adams', 'bdf', 'dop853', 'lsoda', ...}
        Name of the algorithm to use to solve the differential equations ot the
        system studied.

    atol : float
        Absolute tolerance.

    rtol : float
        Relative tolerance.

    store_final_state : bool
        Whether or not to store the final state of the evolution in the
        result class.

    store_states : bool
        Whether or not to store the state vectors or density matrices.
        On `None` the states will be saved if no expectation operators are
        given.

    normalize_output : str
        normalize output state to hide ODE numerical errors.
        "all" will normalize both ket and dm.
        On "ket", only 'ket' output are normalized.
        Leave empty for no normalization.

    progress_bar : str
        How to present the solver progress.
        True will result in 'text'.
        'tqdm' uses the python module of the same name and raise an error if
        not installed.
        Empty string or False will disable the bar.

    progress_kwargs : dict
        kwargs to pass to the progress_bar. Qutip's bars use `chunk_size`.
    """
    all_options = set()

    def __init__(self, base=None, **options):
        if isinstance(base, dict):
            options.update(base)
        elif isinstance(base, SolverOptions):
            options.update(base.options)
            options.update(base.ode.options)
        elif isinstance(base, (QutipOptions, Options)):
            options.update(base.options)
        self.options = {}
        for key in options:
            self[key] = options[key]

    def __setitem__(self, key, value):
        if key in self.all_options:
            self.options[key] = value

    def __getitem__(self, key):
        return self.options[key]

    def __str__(self):
        if not self.options:
            return "Options()"
        longest = max(len(key) for key in self.options)
        out = "Options({\n"
        for key, val in self.options.items():
            if isinstance(val, str):
                out += f"    {key:{longest}} : '{val}'\n"
            else:
                out += f"    {key:{longest}} : {val}\n"
        out += "})\n"
        return out

    def __repr__(self):
        return self.__str__()

    @classmethod
    def _add_options(cls, optioncls):
        cls.all_options |= set(optioncls.default)


class SolverOptions(QutipOptions):
    """
    Parent class of Options classes used by solvers.
    It add support to contain integrator options ::

    opt = Options(method='bdf', atol=1e-5, progress_bar=True)
    opt.ode == OdeBdfOptions(atol=1e-5)
    """
    default = {
        "progress_bar": "text",
        "progress_kwargs": {"chunk_size":10},
        "store_final_state": False,
        "store_states": None,
        "normalize_output": "ket",
        "operator_data_type": "",
        "state_data_type": "",
        'method': 'adams',
    }
    _ode_options = {}
    frozen = False

    def __init__(self, base=None, *,
                 ode=None, _strict=True, _frozen=False, **options):
        if isinstance(base, dict):
            options.update(base)
        elif type(base) is Options:
            _strict = False
            opt = {
                key: val
                for key, val in base.options.items()
                if val is not None
            }
            options.update(opt)
        elif (
            type(base) is self.__class__
            or (isinstance(base, SolverOptions) and not _strict)
        ):
            options.update(base.options)
            ode = base.ode
        elif isinstance(base, SolverOptions) and _strict:
            # TODO: SeOptions, MeOptions contain the same info, should we be
            # lenient? Or rename `_strict` to `force` and ask to use the flag?
            pass
            # raise TypeError("Cannot convert between different options types")

        self.options = self.default.copy()
        self._from_dict(options)
        self.ode = ode
        self.ode._from_dict(options)
        self.frozen = _frozen
        self.ode.frozen = _frozen

        if _strict and options:
            pass
            # raise KeyError("Unknown option(s): " +
            #                f"{set(options) - set(self.default)}")

    def __setitem__(self, key, value):
        # TODO: Do we support keys from the integrator options?
        if self[key] == value:
            return None
        super().__setitem__(key, value)
        if key == 'method':
            self.ode = None

    def __str__(self):
        out = super().__str__()
        out += str(self.ode)
        return out

    @property
    def ode(self):
        return self._ode

    @ode.setter
    def ode(self, new):
        if self.frozen:
            raise RuntimeError("Options associated cannot be modified, "
                               "only overwritten.")
        ode_options_class = self.ode_options(self.options['method'])
        if isinstance(new, ode_options_class):
            self._ode = new
        elif new is None:
            self._ode = ode_options_class()
        else:
            raise RuntimeError("Ode options do not match the method")

    @classmethod
    def ode_options(cls, key):
        if key in cls._ode_options:
            return cls._ode_options[key]
        return SolverOptions._ode_options[key]


Options._add_options(SolverOptions)
