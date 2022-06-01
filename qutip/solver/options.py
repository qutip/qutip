__all__ = ['SolverOptions']

from ..optionsclass import QutipOptions

known_solver = {}
known_integrator = {}

class Options():
    _doc = """
    General class of options for solvers. Options can be specified either as
    arguments to the constructor::

        opts = Options(progress_bar='enhanced', ...)

    or by changing the class attributes after creation::

        opts = Options()
        opts['progress_bar'] = 'enhanced'

    Parameters
    ----------
    solver : str ['sesolve', 'brmesolve', etc.], optional
        Which solver this options is intended too. When given, all items used
        by this solver will be included.

    method : str ['adams', 'dop853', etc.], optional
        Which integration method to use. When given, all options items used
        by this integration method will be included.
    """
    __doc__ = _doc

    def __init__(self, solver=None, method=None, **kwargs):
        self.solver = solver
        self.solver_options, solver_doc = self._get_solver_options(solver)
        method = method or _solver_options.get('method', None)
        self.integrator_options, integrator_doc = self._get_integrator_options(method)
        self.extra_options = {}
        # Merge all options
        for key in kwargs:
            self.__setitem__(ket, kwargs[key])

    def _get_solver_options(self, solver):
        if solver:
            opt = known_solver[solver].default_options
            self._doc = self._doc + known_solver[solver].options.__doc__
        else:
            opt = {}
        return opt, doc

    def _get_integrator_options(self, solver):
        if solver:
            opt = known_integrator[solver].default_options
            self.__doc__ = self._doc + known_integrator[solver].options.__doc__
        else:
            opt = {}
            self.__doc__ = self._doc
        return opt, doc

    def __setitem__(self, key, value):
        if key == 'method':
            self.integrator_options = self._get_integrator_options(value)
        if key in self.solver_options:
            self.solver_options[key] = value
        elif key in self.integrator_options:
            self.integrator_options[key] = value
        else:
            self.extra_options[key] = value

    def __getitem__(self, key):
        for dictionary in [
            self.solver_options,
            self.integrator_options,
            self.extra_options
        ]:
            if key in dictionary:
                return dictionary[key]
        raise KeyError()

    def __str__(self):
        all_options = {
            **self.solver_options,
            **self.integrator_options,
            **self.extra_options
        }
        if not all_options:
            return "<Options()>"
        longest = max(len(key) for key in all_options)
        out = "<Options({\n"
        for key, val in all_options.items():
            if isinstance(val, str):
                out += f"    {key:{longest}} : '{val}'\n"
            else:
                out += f"    {key:{longest}} : {val}\n"
        out += "})>\n"
        return out

    def __repr__(self):
        return self.__str__()


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
